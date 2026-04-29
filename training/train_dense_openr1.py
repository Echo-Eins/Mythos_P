#!/usr/bin/env python3
"""
Train/evaluate/generate with the dense OpenMythos RDT prototype on OpenR1 Math.

This script is intended to run on the Spark/GB10 environment where torch,
flash-attn, datasets, transformers, and the tokenizer stack are installed.
It is not expected to run on the local Windows machine used for code editing.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass

import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from open_mythos.dense_lm import (
    DenseLMConfig,
    OpenMythosDenseLM,
    dense_lm_config_from_dict,
)


DATASET_ID = "open-r1/OpenR1-Math-220k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense OpenMythos OpenR1 trainer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train", help="Train the dense recurrent LM")
    add_data_args(train, tokenizer_default="Qwen/Qwen2.5-1.5B")
    add_model_args(train)
    add_training_args(train)

    preflight = sub.add_parser(
        "preflight",
        help="Run a fast Spark-side data/model/train smoke check before a long run",
    )
    add_data_args(
        preflight,
        tokenizer_default="Qwen/Qwen2.5-1.5B",
        max_samples_default=64,
    )
    add_model_args(preflight)
    add_training_args(preflight)
    preflight.add_argument("--probe-batches", type=int, default=4)
    preflight.add_argument("--generate-tokens", type=int, default=4)
    preflight.add_argument("--check-save-load", action="store_true")
    preflight.add_argument(
        "--no-backward",
        dest="run_backward",
        action="store_false",
        help="Skip backward/optimizer smoke step.",
    )
    preflight.set_defaults(
        out_dir=Path("runs") / "dense_openr1_preflight",
        batch_size=2,
        grad_accum=1,
        max_steps=2,
        eval_batches=2,
        num_workers=0,
        log_every=1,
        run_backward=True,
    )

    evaluate = sub.add_parser("eval", help="Evaluate a checkpoint")
    add_data_args(evaluate, tokenizer_default=None)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-batches", type=int, default=100)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--prefetch-factor", type=int, default=2)
    evaluate.add_argument("--n-loops", type=int, default=None)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")

    exact_eval = sub.add_parser("exact-eval", help="Generate and exact-match final answers")
    add_data_args(exact_eval, tokenizer_default=None, max_samples_default=100)
    exact_eval.add_argument("--checkpoint", type=Path, required=True)
    exact_eval.add_argument("--max-new-tokens", type=int, default=512)
    exact_eval.add_argument("--n-loops", type=int, default=None)
    exact_eval.add_argument("--temperature", type=float, default=1.0)
    exact_eval.add_argument("--top-k", type=int, default=0)
    exact_eval.add_argument("--top-p", type=float, default=1.0)
    exact_eval.add_argument("--sample", action="store_true")
    exact_eval.add_argument("--log-every", type=int, default=10)
    exact_eval.add_argument("--predictions-path", type=Path, default=None)
    exact_eval.add_argument("--metrics-jsonl", type=Path, default=None)
    exact_eval.add_argument("--print-samples", type=int, default=5)
    exact_eval.add_argument("--device", default="cuda")
    exact_eval.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")

    generate = sub.add_parser("generate", help="Generate from a checkpoint")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--tokenizer", default=None)
    generate.add_argument("--prompt", default=None)
    generate.add_argument("--prompt-file", type=Path, default=None)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--n-loops", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-k", type=int, default=50)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--greedy", action="store_true")
    generate.add_argument("--device", default="cuda")
    generate.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")

    return parser.parse_args()


def add_data_args(
    parser: argparse.ArgumentParser,
    *,
    tokenizer_default: Optional[str],
    max_samples_default: Optional[int] = None,
) -> None:
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset-subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default=tokenizer_default)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=max_samples_default)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--loss-on",
        choices=("response", "all"),
        default="response",
        help=(
            "Token loss target. 'response' masks the problem/prompt tokens and "
            "trains only solution/answer tokens; 'all' keeps plain causal LM loss."
        ),
    )
    parser.add_argument(
        "--long-sample-policy",
        choices=("drop", "truncate"),
        default="drop",
        help=(
            "How to handle formatted samples longer than --max-seq-len + 1. "
            "'drop' avoids training on incomplete solutions; 'truncate' keeps "
            "the prefix without forcing a synthetic EOS."
        ),
    )
    parser.add_argument(
        "--min-response-tokens",
        type=int,
        default=16,
        help="Skip samples that produce fewer supervised response tokens.",
    )
    parser.add_argument(
        "--pack-samples",
        action="store_true",
        help=(
            "Pack multiple samples into one causal chunk. Faster, but answer "
            "tokens can attend to previous samples unless block-diagonal masks "
            "are added; keep disabled for exact supervised math runs."
        ),
    )
    parser.add_argument(
        "--pad-to-multiple",
        type=int,
        default=8,
        help=(
            "Right-pad each dynamic batch to a multiple of this length. Use 1 "
            "to disable alignment; 8 is a conservative tensor-core-friendly default."
        ),
    )
    parser.add_argument(
        "--static-padding",
        action="store_true",
        help=(
            "Pad every batch to --max-seq-len. This reproduces the old fixed-shape "
            "behavior and is mainly useful for debugging."
        ),
    )
    parser.add_argument(
        "--no-length-bucketing",
        dest="length_bucketing",
        action="store_false",
        default=True,
        help=(
            "Disable length-grouped training batches. By default, shuffled "
            "mega-batches are locally sorted by sequence length to reduce padding."
        ),
    )
    parser.add_argument(
        "--length-bucket-mult",
        type=int,
        default=32,
        help="Mega-batch size multiplier for length bucketing.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=1536)
    parser.add_argument("--n-heads", type=int, default=24)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--prelude-layers", type=int, default=2)
    parser.add_argument("--coda-layers", type=int, default=2)
    parser.add_argument("--max-loop-iters", type=int, default=4)
    parser.add_argument("--train-loops", type=int, default=None)
    parser.add_argument("--ffn-hidden-dim", type=int, default=4352)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=500_000.0)
    parser.add_argument("--ada-norm", dest="use_ada_norm", action="store_true", default=True)
    parser.add_argument("--no-ada-norm", dest="use_ada_norm", action="store_false")
    parser.add_argument("--lti-init-log-dt", type=float, default=-1.0)
    parser.add_argument("--lti-init-input-gain", type=float, default=0.3)
    parser.add_argument("--lti-init-delta-gain", type=float, default=0.35)
    parser.add_argument("--lti-max-input-gain", type=float, default=1.0)
    parser.add_argument("--lti-max-delta-gain", type=float, default=1.0)
    parser.add_argument("--recurrent-input-h-init", type=float, default=0.7)
    parser.add_argument("--recurrent-output-h-init", type=float, default=0.75)
    parser.add_argument("--use-act", action="store_true")
    parser.add_argument("--act-threshold", type=float, default=0.99)
    parser.add_argument("--act-ponder-weight", type=float, default=0.01)
    parser.add_argument("--act-min-steps", type=int, default=1)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / "dense_openr1")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--max-epochs",
        type=float,
        default=None,
        help=(
            "Optional epoch-based training target. If set, --max-steps is "
            "computed after dataset filtering as "
            "ceil(max_epochs * train_batches_per_epoch / grad_accum)."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "linear", "constant"),
        default="cosine",
    )
    parser.add_argument(
        "--decay-steps",
        type=int,
        default=None,
        help=(
            "Number of optimizer steps used for LR decay after warmup. "
            "Defaults to --max-steps. If smaller than --max-steps, LR stays "
            "at --lr * --min-lr-ratio afterwards."
        ),
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def validate_loader_args(args: argparse.Namespace) -> None:
    if args.max_seq_len <= 1:
        raise ValueError("--max-seq-len must be > 1")
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided")
    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be in (0, 1)")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if getattr(args, "prefetch_factor", 2) <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.pad_to_multiple <= 0:
        raise ValueError("--pad-to-multiple must be positive")
    if args.length_bucket_mult <= 0:
        raise ValueError("--length-bucket-mult must be positive")


def validate_training_runtime_args(args: argparse.Namespace) -> None:
    if args.grad_accum <= 0:
        raise ValueError("--grad-accum must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive")
    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.lr <= 0.0:
        raise ValueError("--lr must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if args.grad_clip <= 0.0:
        raise ValueError("--grad-clip must be positive")


def load_tokenizer(tokenizer_id: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return tokenizer


def load_openr1_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_path is not None:
        ds = load_from_disk(str(args.dataset_path))
        if isinstance(ds, DatasetDict):
            ds = ds[args.split]
        return ds
    return load_dataset(DATASET_ID, args.dataset_subset, split=args.split)


def select_best_solution(row: dict[str, Any]) -> str:
    solution = row.get("solution")
    if isinstance(solution, str) and solution.strip():
        return solution.strip()

    generations = row.get("generations")
    complete = row.get("is_reasoning_complete")
    verified = row.get("correctness_math_verify")
    if isinstance(generations, list):
        for idx, candidate in enumerate(generations):
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            ok_complete = not isinstance(complete, list) or idx >= len(complete) or bool(complete[idx])
            ok_verified = not isinstance(verified, list) or idx >= len(verified) or bool(verified[idx])
            if ok_complete and ok_verified:
                return candidate.strip()
        for candidate in generations:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return ""


@dataclass
class TokenSequence:
    ids: list[int]
    label_mask: list[int]


def format_openr1_parts(row: dict[str, Any]) -> tuple[str, str]:
    problem = str(row.get("problem") or "").strip()
    solution = select_best_solution(row)
    answer = str(row.get("answer") or "").strip()
    if not problem or not solution:
        return "", ""

    prompt = f"Problem:\n{problem}\n\nSolution:\n"
    response = solution
    if answer:
        response += f"\n\nFinal answer:\n{answer}"
    return prompt, response.strip()


def format_openr1_sample(row: dict[str, Any]) -> str:
    prompt, response = format_openr1_parts(row)
    return f"{prompt}{response}".strip()


def format_openr1_prompt(row: dict[str, Any]) -> str:
    problem = str(row.get("problem") or "").strip()
    if not problem:
        return ""
    return f"Problem:\n{problem}\n\nSolution:\n"


def reference_answer(row: dict[str, Any]) -> str:
    answer = row.get("answer")
    if answer is None:
        return ""
    if isinstance(answer, (list, tuple)):
        for item in answer:
            if item is not None and str(item).strip():
                return str(item).strip()
        return ""
    return str(answer).strip()


def _balanced_brace_content(text: str, open_idx: int) -> str:
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return ""
    depth = 0
    start = open_idx + 1
    idx = open_idx
    while idx < len(text):
        char = text[idx]
        if char == "\\":
            idx += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx]
        idx += 1
    return ""


def extract_boxed_answer(text: str) -> str:
    boxed: list[str] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*", text):
        open_idx = text.find("{", match.end())
        value = _balanced_brace_content(text, open_idx)
        if value.strip():
            boxed.append(value.strip())
    return boxed[-1] if boxed else ""


def trim_generated_solution(text: str) -> str:
    for marker in ("\nProblem:", "\n\nProblem:", "\nSolution:"):
        pos = text.find(marker)
        if pos >= 0:
            text = text[:pos]
    return text.strip()


def extract_final_answer(text: str) -> str:
    text = trim_generated_solution(text)
    if not text:
        return ""

    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed

    marker_matches = list(
        re.finditer(r"(?i)(?:final\s+answer|answer)\s*(?::|\uFF1A)", text)
    )
    if marker_matches:
        tail = text[marker_matches[-1].end() :].strip()
        for line in tail.splitlines():
            line = line.strip()
            if line:
                return line

    hash_matches = list(re.finditer(r"(?m)^\s*####\s*(.+?)\s*$", text))
    if hash_matches:
        return hash_matches[-1].group(1).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _replace_simple_frac(match: re.Match[str]) -> str:
    return f"({match.group(1)})/({match.group(2)})"


def normalize_answer(answer: str) -> str:
    s = str(answer or "").strip()
    boxed = extract_boxed_answer(s)
    if boxed:
        s = boxed

    s = s.strip().strip("`")
    s = re.sub(r"(?is)^(?:final\s+answer|answer)\s*(?::|\uFF1A)\s*", "", s).strip()
    s = re.sub(r"(?is)^the\s+answer\s+is\s+", "", s).strip()
    s = s.strip().strip(".\u3002;:, ")

    wrappers = ((r"\(", r"\)"), (r"\[", r"\]"), ("$", "$"))
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if s.startswith(left) and s.endswith(right):
                s = s[len(left) : -len(right)].strip()
                changed = True

    s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\\(?:left|right)", "", s)
    s = re.sub(r"\\(?:,|;|!|quad|qquad)", "", s)
    s = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", _replace_simple_frac, s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    s = re.sub(r"\s+", "", s.lower())

    changed = True
    while changed and len(s) >= 2:
        changed = False
        if (s[0], s[-1]) in {("(", ")"), ("{", "}"), ("[", "]")}:
            s = s[1:-1].strip()
            changed = True
    return s.strip().strip(".\u3002;:,")


def _parse_simple_fraction(answer: str) -> Optional[Fraction]:
    s = normalize_answer(answer)
    s = s.replace("(", "").replace(")", "")
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d+/\d+)", s):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def answers_exact_match(predicted: str, target: str) -> bool:
    pred_norm = normalize_answer(predicted)
    target_norm = normalize_answer(target)
    if not pred_norm or not target_norm:
        return False
    if pred_norm == target_norm:
        return True
    pred_num = _parse_simple_fraction(pred_norm)
    target_num = _parse_simple_fraction(target_norm)
    return pred_num is not None and target_num is not None and pred_num == target_num


def repeated_ngram_ratio(token_ids: list[int], n: int = 4) -> float:
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[idx : idx + n]) for idx in range(len(token_ids) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def unique_token_ratio(token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def build_token_sequences(
    ds: Dataset,
    tokenizer,
    *,
    max_seq_len: int,
    max_samples: Optional[int],
    seed: int,
    loss_on: str,
    long_sample_policy: str,
    min_response_tokens: int,
) -> tuple[list[TokenSequence], dict[str, int]]:
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.shuffle(seed=seed).select(range(max_samples))
    if min_response_tokens < 1:
        raise ValueError("--min-response-tokens must be positive")

    eos = tokenizer.eos_token_id
    sequences: list[TokenSequence] = []
    stats = {
        "raw_samples": len(ds),
        "kept_samples": 0,
        "dropped_empty": 0,
        "dropped_too_long": 0,
        "dropped_short_response": 0,
        "truncated_samples": 0,
    }
    for row in ds:
        prompt, response = format_openr1_parts(row)
        if not prompt.strip() or not response.strip():
            stats["dropped_empty"] += 1
            continue

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)
        if eos is not None:
            response_ids.append(eos)

        if len(response_ids) < min_response_tokens:
            stats["dropped_short_response"] += 1
            continue

        ids = prompt_ids + response_ids
        response_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
        if loss_on == "all":
            label_mask = [1] * len(ids)
        elif loss_on == "response":
            label_mask = response_mask.copy()
        else:
            raise ValueError(f"unknown loss target: {loss_on}")

        if len(ids) < 2:
            stats["dropped_empty"] += 1
            continue
        if len(ids) > max_seq_len + 1:
            if long_sample_policy == "drop":
                stats["dropped_too_long"] += 1
                continue
            if long_sample_policy != "truncate":
                raise ValueError(f"unknown long sample policy: {long_sample_policy}")
            ids = ids[: max_seq_len + 1]
            label_mask = label_mask[: max_seq_len + 1]
            response_mask = response_mask[: max_seq_len + 1]
            stats["truncated_samples"] += 1

        if sum(response_mask[1:]) < min_response_tokens:
            stats["dropped_short_response"] += 1
            continue

        sequences.append(TokenSequence(ids=ids, label_mask=label_mask))
        stats["kept_samples"] += 1
    return sequences, stats


class PackedCausalDataset(TorchDataset):
    """
    Stores tokenized samples or packed chunks for causal LM training.

    Each item returns:
        input_ids = chunk[:-1]
        labels    = chunk[1:], with non-supervised tokens masked to -100.

    Padding is deliberately handled by CausalBatchCollator.  Keeping the stored
    chunks variable-length avoids spending attention compute on pad-only tails.
    """

    def __init__(
        self,
        sequences: Iterable[TokenSequence],
        *,
        seq_len: int,
        pad_token_id: int,
        pack_samples: bool = False,
    ) -> None:
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.chunks: list[tuple[list[int], list[int]]] = []

        token_buffer: list[int] = []
        mask_buffer: list[int] = []

        def append_chunk(ids: list[int], mask: list[int]) -> None:
            if len(ids) < 2 or sum(mask[1:]) == 0:
                return
            if len(ids) > seq_len + 1:
                raise ValueError("sequence length exceeds packed chunk length")
            self.chunks.append((ids, mask))

        if not pack_samples:
            for seq in sequences:
                if len(seq.ids) != len(seq.label_mask):
                    raise ValueError("token ids and label mask must have the same length")
                if len(seq.ids) > seq_len + 1:
                    raise ValueError("sequence length exceeds packed chunk length")
                append_chunk(seq.ids, seq.label_mask)
            return

        def flush_buffer() -> None:
            nonlocal token_buffer, mask_buffer
            if len(token_buffer) > 1:
                append_chunk(token_buffer, mask_buffer)
            token_buffer = []
            mask_buffer = []

        for seq in sequences:
            if len(seq.ids) != len(seq.label_mask):
                raise ValueError("token ids and label mask must have the same length")
            if len(seq.ids) > seq_len + 1:
                raise ValueError("sequence length exceeds packed chunk length")
            if len(token_buffer) + len(seq.ids) > seq_len + 1:
                flush_buffer()
            if len(seq.ids) == seq_len + 1:
                append_chunk(seq.ids, seq.label_mask)
                continue
            token_buffer.extend(seq.ids)
            mask_buffer.extend(seq.label_mask)
        flush_buffer()

    def __len__(self) -> int:
        return len(self.chunks)

    def input_lengths(self) -> list[int]:
        return [len(ids) - 1 for ids, _ in self.chunks]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, mask = self.chunks[idx]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        label_mask = torch.tensor(mask[1:], dtype=torch.bool)
        labels[~label_mask] = -100
        return {"input_ids": input_ids, "labels": labels}


class CausalBatchCollator:
    """Right-pad causal LM items to the shortest safe batch-local length."""

    def __init__(
        self,
        *,
        pad_token_id: int,
        max_seq_len: int,
        pad_to_multiple: int = 8,
        static_padding: bool = False,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if pad_to_multiple <= 0:
            raise ValueError("--pad-to-multiple must be positive")
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.pad_to_multiple = pad_to_multiple
        self.static_padding = static_padding

    def _target_len(self, lengths: list[int]) -> int:
        if not lengths:
            raise ValueError("cannot collate an empty batch")
        max_len = max(lengths)
        if max_len > self.max_seq_len:
            raise ValueError("batch item exceeds max_seq_len")
        if self.static_padding:
            return self.max_seq_len
        if self.pad_to_multiple == 1:
            return max_len
        aligned = int(math.ceil(max_len / self.pad_to_multiple) * self.pad_to_multiple)
        return min(aligned, self.max_seq_len)

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        lengths = [int(item["input_ids"].numel()) for item in batch]
        target_len = self._target_len(lengths)
        batch_size = len(batch)

        input_ids = torch.full(
            (batch_size, target_len),
            self.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full((batch_size, target_len), -100, dtype=torch.long)
        seq_lens = torch.tensor(lengths, dtype=torch.long)

        for idx, item in enumerate(batch):
            length = lengths[idx]
            input_ids[idx, :length] = item["input_ids"]
            labels[idx, :length] = item["labels"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "seq_lens": seq_lens,
        }


class LengthGroupedBatchSampler:
    """
    Stochastic length-grouped batch sampler.

    Each epoch first shuffles all indices, then sorts short shuffled windows by
    sequence length before forming batches.  This keeps enough randomness for
    SFT while avoiding pathological batches that mix 80-token and 1024-token
    samples under dynamic padding.
    """

    def __init__(
        self,
        *,
        lengths: list[int],
        batch_size: int,
        drop_last: bool,
        seed: int,
        bucket_mult: int = 64,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if bucket_mult <= 0:
            raise ValueError("--length-bucket-mult must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.bucket_size = max(batch_size, batch_size * bucket_mult)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        indices = list(range(len(self.lengths)))
        rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start : batch_start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)

        rng.shuffle(batches)
        yield from batches


def split_sequences(
    sequences: list[TokenSequence],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[TokenSequence], list[TokenSequence]]:
    if not sequences:
        raise ValueError("no tokenized sequences were produced")
    rng = random.Random(seed)
    indices = list(range(len(sequences)))
    rng.shuffle(indices)
    val_count = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else 0
    val_set = set(indices[:val_count])
    train_seq = [seq for idx, seq in enumerate(sequences) if idx not in val_set]
    val_seq = [seq for idx, seq in enumerate(sequences) if idx in val_set]
    return train_seq, val_seq


def dataset_length_stats(ds: PackedCausalDataset) -> dict[str, float | int]:
    lengths = ds.input_lengths()
    if not lengths:
        return {}
    return {
        "items": len(lengths),
        "min_input_len": min(lengths),
        "max_input_len": max(lengths),
        "mean_input_len": sum(lengths) / len(lengths),
        "static_padding_tokens": len(lengths) * ds.seq_len,
        "unpadded_input_tokens": sum(lengths),
        "static_padding_efficiency": sum(lengths) / max(1, len(lengths) * ds.seq_len),
    }


def build_loaders(args: argparse.Namespace, tokenizer) -> tuple[DataLoader, DataLoader]:
    validate_loader_args(args)
    raw = load_openr1_dataset(args)
    sequences, data_stats = build_token_sequences(
        raw,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
        loss_on=args.loss_on,
        long_sample_policy=args.long_sample_policy,
        min_response_tokens=args.min_response_tokens,
    )
    args.data_stats = data_stats
    train_seq, val_seq = split_sequences(
        sequences,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    pad_id = tokenizer.pad_token_id
    train_ds = PackedCausalDataset(
        train_seq,
        seq_len=args.max_seq_len,
        pad_token_id=pad_id,
        pack_samples=args.pack_samples,
    )
    val_ds = PackedCausalDataset(
        val_seq,
        seq_len=args.max_seq_len,
        pad_token_id=pad_id,
        pack_samples=args.pack_samples,
    )
    if len(train_ds) == 0:
        raise ValueError("training dataset has zero chunks")
    if len(val_ds) == 0:
        raise ValueError("validation dataset has zero chunks")

    args.data_stats = {
        **data_stats,
        "train_length_stats": dataset_length_stats(train_ds),
        "val_length_stats": dataset_length_stats(val_ds),
        "dynamic_padding": not args.static_padding,
        "pad_to_multiple": args.pad_to_multiple,
        "length_bucketing": args.length_bucketing,
        "length_bucket_mult": args.length_bucket_mult,
        "num_workers": args.num_workers,
        "prefetch_factor": getattr(args, "prefetch_factor", None),
    }
    collator = CausalBatchCollator(
        pad_token_id=pad_id,
        max_seq_len=args.max_seq_len,
        pad_to_multiple=args.pad_to_multiple,
        static_padding=args.static_padding,
    )
    loader_kwargs: dict[str, Any] = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collator,
    }
    if args.num_workers > 0:
        if getattr(args, "prefetch_factor", 2) <= 0:
            raise ValueError("--prefetch-factor must be positive when num_workers > 0")
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = getattr(args, "prefetch_factor", 2)

    if args.length_bucketing:
        train_sampler = LengthGroupedBatchSampler(
            lengths=train_ds.input_lengths(),
            batch_size=args.batch_size,
            drop_last=True,
            seed=args.seed,
            bucket_mult=args.length_bucket_mult,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def make_model_config(args: argparse.Namespace, tokenizer) -> DenseLMConfig:
    return DenseLMConfig(
        vocab_size=len(tokenizer),
        dim=args.dim,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        prelude_layers=args.prelude_layers,
        coda_layers=args.coda_layers,
        max_loop_iters=args.max_loop_iters,
        max_seq_len=args.max_seq_len,
        ffn_hidden_dim=args.ffn_hidden_dim,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        use_ada_norm=args.use_ada_norm,
        lti_init_log_dt=args.lti_init_log_dt,
        lti_init_input_gain=args.lti_init_input_gain,
        lti_init_delta_gain=args.lti_init_delta_gain,
        lti_max_input_gain=args.lti_max_input_gain,
        lti_max_delta_gain=args.lti_max_delta_gain,
        recurrent_input_h_init=args.recurrent_input_h_init,
        recurrent_output_h_init=args.recurrent_output_h_init,
        use_act=args.use_act,
        act_threshold=args.act_threshold,
        act_ponder_weight=args.act_ponder_weight,
        act_min_steps=args.act_min_steps,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )


def make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be non-negative")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.decay_steps is not None and args.decay_steps <= 0:
        raise ValueError("--decay-steps must be positive when provided")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        raise ValueError("--min-lr-ratio must be in [0, 1]")

    warmup_steps = args.warmup_steps
    decay_steps = args.decay_steps or args.max_steps
    min_ratio = args.min_lr_ratio
    schedule = args.lr_schedule

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(0.0, float(step) / float(warmup_steps))
        if schedule == "constant":
            return 1.0

        decay_span = max(1, decay_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(decay_span)))
        if schedule == "linear":
            return min_ratio + (1.0 - min_ratio) * (1.0 - progress)
        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cosine
        raise ValueError(f"unknown lr schedule: {schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def sync_scheduler_to_step(
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
) -> None:
    """
    Force optimizer LR to match the scheduler at an absolute training step.

    This matters when resuming old checkpoints whose optimizer state may carry
    a stale LR, for example a previous cosine run that ended at exactly 0.
    """

    scheduler.last_epoch = step
    lrs = [
        base_lr * lr_lambda(step)
        for base_lr, lr_lambda in zip(scheduler.base_lrs, scheduler.lr_lambdas)
    ]
    for group, lr in zip(scheduler.optimizer.param_groups, lrs):
        group["lr"] = lr
    scheduler._last_lr = lrs


def parameter_uses_weight_decay(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad:
        return False
    if param.ndim < 2:
        return False
    no_decay_fragments = (
        "embed",
        "lm_head",
        "norm",
        "injection",
        "raw_h_gate",
    )
    return not any(fragment in name for fragment in no_decay_fragments)


def make_optimizer(
    model: OpenMythosDenseLM,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    decay_param_count = 0
    no_decay_param_count = 0
    decay_names: list[str] = []
    no_decay_names: list[str] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if parameter_uses_weight_decay(name, param):
            decay_params.append(param)
            decay_param_count += param.numel()
            if len(decay_names) < 8:
                decay_names.append(name)
        else:
            no_decay_params.append(param)
            no_decay_param_count += param.numel()
            if len(no_decay_names) < 8:
                no_decay_names.append(name)

    param_groups: list[dict[str, Any]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": args.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    stats = {
        "weight_decay": args.weight_decay,
        "decay_param_count": decay_param_count,
        "no_decay_param_count": no_decay_param_count,
        "decay_group_count": len(decay_params),
        "no_decay_group_count": len(no_decay_params),
        "decay_examples": decay_names,
        "no_decay_examples": no_decay_names,
    }
    return optimizer, stats


def save_checkpoint(
    path: Path,
    *,
    model: OpenMythosDenseLM,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    step: int,
    tokenizer_id: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model": model.state_dict(),
        "config": model.config_dict(),
        "tokenizer": tokenizer_id,
        "args": vars(args),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[OpenMythosDenseLM, dict[str, Any]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dense_lm_config_from_dict(ckpt["config"])
    model = OpenMythosDenseLM(cfg)
    load_model_state_compat(model, ckpt["model"])
    model.to(device=device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model, ckpt


def load_model_state_compat(model: OpenMythosDenseLM, state: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {
        name for name in missing if name.endswith("injection.raw_delta_gain")
    }
    allowed_unexpected = {
        name
        for name in unexpected
        if not model.cfg.use_act
        and (name.startswith("recurrent.act.") or name.startswith("recurrent.act_norm."))
    }
    real_missing = [name for name in missing if name not in allowed_missing]
    real_unexpected = [name for name in unexpected if name not in allowed_unexpected]
    if real_missing or real_unexpected:
        details = {"missing": real_missing, "unexpected": real_unexpected}
        raise RuntimeError(f"checkpoint state is incompatible: {json.dumps(details)}")
    if allowed_missing or allowed_unexpected:
        print_json(
            {
                "checkpoint_warning": "checkpoint loaded with compatible state differences",
                "missing_initialized": sorted(allowed_missing),
                "unexpected_ignored": sorted(allowed_unexpected),
            }
        )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def scalar_stat(stats: dict[str, Any], key: str) -> Optional[float]:
    value = stats.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value)
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    return value


def append_jsonl(path: Optional[Path], payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {"time": time.time(), **payload}
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(json_safe(event), ensure_ascii=False) + "\n")


def print_json(payload: dict[str, Any], *, indent: Optional[int] = None) -> None:
    print(json.dumps(json_safe(payload), ensure_ascii=False, indent=indent), flush=True)


def snapshot_tensor_outputs(value: Any) -> Any:
    """
    Clone tensor outputs before another compiled model invocation can overwrite them.

    `torch.compile(..., mode="reduce-overhead")` may use CUDA Graphs.  CUDA Graph
    output tensors can be reused by later compiled calls, so long-lived logging
    values must be detached and cloned immediately after forward.
    """

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: snapshot_tensor_outputs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_tensor_outputs(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_tensor_outputs(item) for item in value)
    return value


def epoch_metrics(
    *,
    micro_batches_seen: int,
    train_batches_per_epoch: int,
    max_steps: int,
    grad_accum: int,
) -> dict[str, float | int]:
    if train_batches_per_epoch <= 0:
        raise ValueError("train_batches_per_epoch must be positive")
    updates_per_epoch = train_batches_per_epoch / max(1, grad_accum)
    target_epochs = max_steps / max(updates_per_epoch, 1e-12)
    epochs_seen = micro_batches_seen / train_batches_per_epoch
    epoch_index = int(math.floor(epochs_seen))
    return {
        "epochs_seen": epochs_seen,
        "epoch_index": epoch_index,
        "epoch_number": epoch_index + 1,
        "current_epoch": epoch_index,
        "epoch_progress": epochs_seen - epoch_index,
        "target_epochs": target_epochs,
        "updates_per_epoch": updates_per_epoch,
    }


@torch.no_grad()
def evaluate(
    model: OpenMythosDenseLM,
    loader: DataLoader,
    *,
    device: torch.device,
    dtype: torch.dtype,
    n_loops: Optional[int],
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    objective_loss_sum = 0.0
    lm_loss_sum = 0.0
    supervised_tokens = 0
    first_stats: Optional[dict[str, Any]] = None
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        token_count = int((batch["labels"] != -100).sum().item())
        if token_count == 0:
            continue
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
            out = model(
                batch["input_ids"],
                labels=batch["labels"],
                n_loops=n_loops,
                collect_stats=(idx == 0),
        )
        if out.loss is None:
            raise RuntimeError("model returned no loss during evaluation")
        objective_loss = float(out.loss.detach().cpu())
        out_stats = snapshot_tensor_outputs(out.stats) if out.stats is not None else None
        lm_loss = scalar_stat(out_stats, "lm_loss") if out_stats is not None else None
        objective_loss_sum += objective_loss * token_count
        lm_loss_sum += (objective_loss if lm_loss is None else lm_loss) * token_count
        supervised_tokens += token_count
        if first_stats is None and out_stats is not None:
            first_stats = out_stats
    if supervised_tokens == 0:
        raise ValueError("evaluation produced zero supervised tokens")
    mean_objective_loss = objective_loss_sum / supervised_tokens
    mean_lm_loss = lm_loss_sum / supervised_tokens
    metrics: dict[str, float] = {
        "loss": mean_objective_loss,
        "lm_loss": mean_lm_loss,
        "ppl": math.exp(min(mean_lm_loss, 20.0)),
        "tokens": supervised_tokens,
    }
    if first_stats is not None:
        for key in (
            "lti_A_min",
            "lti_A_max",
            "lti_tau_min",
            "lti_tau_max",
            "lti_B_abs_max",
            "lti_input_gain_abs_max",
            "lti_delta_gain_abs_max",
            "lti_effective_input_abs_max",
            "lti_effective_delta_abs_max",
            "recurrent_input_mixer_h_gate_mean",
            "recurrent_output_bridge_h_gate_mean",
            "recurrent_coda_input_rms",
            "act_loss",
            "act_ponder_loss",
            "act_expected_steps",
            "act_hard_steps",
            "act_remainder",
            "act_halt_fraction",
            "act_halting_p_mean",
        ):
            value = scalar_stat(first_stats, key)
            if value is not None:
                metrics[key] = value
    return metrics


def validate_training_batch(
    batch: dict[str, torch.Tensor],
    tokenizer,
    *,
    vocab_size: int,
    context: str,
) -> dict[str, Any]:
    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    seq_lens = batch.get("seq_lens")
    if input_ids is None or labels is None:
        raise ValueError(f"{context}: batch must contain input_ids and labels")
    if input_ids.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"{context}: input_ids and labels must have shape (B, T)")
    if input_ids.shape != labels.shape:
        raise ValueError(f"{context}: input_ids and labels shapes differ")
    if input_ids.shape[0] <= 0 or input_ids.shape[1] <= 0:
        raise ValueError(f"{context}: empty batch shape {tuple(input_ids.shape)}")

    if input_ids.min().item() < 0 or input_ids.max().item() >= vocab_size:
        raise ValueError(f"{context}: input_ids contain ids outside tokenizer vocab")

    supervised = labels[labels != -100]
    if supervised.numel() == 0:
        raise ValueError(f"{context}: zero supervised labels")
    if supervised.min().item() < 0 or supervised.max().item() >= vocab_size:
        raise ValueError(f"{context}: supervised labels contain ids outside tokenizer vocab")

    if seq_lens is not None:
        if seq_lens.ndim != 1 or seq_lens.shape[0] != input_ids.shape[0]:
            raise ValueError(f"{context}: seq_lens must have shape (B,)")
        seq_values = [int(x) for x in seq_lens.tolist()]
        if min(seq_values) <= 0 or max(seq_values) > input_ids.shape[1]:
            raise ValueError(f"{context}: seq_lens values are outside padded batch length")
        pad_id = tokenizer.pad_token_id
        for row_idx, seq_len in enumerate(seq_values):
            if seq_len < input_ids.shape[1]:
                if not torch.equal(
                    labels[row_idx, seq_len:],
                    torch.full_like(labels[row_idx, seq_len:], -100),
                ):
                    raise ValueError(f"{context}: padded label positions are not -100")
                if pad_id is not None and not torch.equal(
                    input_ids[row_idx, seq_len:],
                    torch.full_like(input_ids[row_idx, seq_len:], int(pad_id)),
                ):
                    raise ValueError(f"{context}: padded input positions do not use pad_token_id")
    else:
        seq_values = [input_ids.shape[1]] * input_ids.shape[0]

    eos_id = tokenizer.eos_token_id
    eos_targets = int((supervised == eos_id).sum().item()) if eos_id is not None else 0
    return {
        "batch_size": int(input_ids.shape[0]),
        "padded_seq_len": int(input_ids.shape[1]),
        "min_seq_len": min(seq_values),
        "max_seq_len": max(seq_values),
        "real_input_tokens": int(sum(seq_values)),
        "padded_input_tokens": int(input_ids.numel()),
        "supervised_tokens": int(supervised.numel()),
        "supervised_eos_tokens": eos_targets,
        "eos_is_pad": bool(eos_id is not None and eos_id == tokenizer.pad_token_id),
    }


def probe_training_batches(
    loader: DataLoader,
    tokenizer,
    *,
    vocab_size: int,
    max_batches: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if max_batches <= 0:
        raise ValueError("--probe-batches must be positive")

    first_batch: Optional[dict[str, torch.Tensor]] = None
    aggregate = {
        "batches": 0,
        "real_input_tokens": 0,
        "padded_input_tokens": 0,
        "supervised_tokens": 0,
        "supervised_eos_tokens": 0,
        "min_seq_len": None,
        "max_seq_len": 0,
        "max_padded_seq_len": 0,
    }
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        if first_batch is None:
            first_batch = batch
        stats = validate_training_batch(
            batch,
            tokenizer,
            vocab_size=vocab_size,
            context=f"preflight batch {batch_idx}",
        )
        aggregate["batches"] += 1
        aggregate["real_input_tokens"] += stats["real_input_tokens"]
        aggregate["padded_input_tokens"] += stats["padded_input_tokens"]
        aggregate["supervised_tokens"] += stats["supervised_tokens"]
        aggregate["supervised_eos_tokens"] += stats["supervised_eos_tokens"]
        aggregate["min_seq_len"] = (
            stats["min_seq_len"]
            if aggregate["min_seq_len"] is None
            else min(int(aggregate["min_seq_len"]), stats["min_seq_len"])
        )
        aggregate["max_seq_len"] = max(int(aggregate["max_seq_len"]), stats["max_seq_len"])
        aggregate["max_padded_seq_len"] = max(
            int(aggregate["max_padded_seq_len"]),
            stats["padded_seq_len"],
        )

    if first_batch is None or aggregate["batches"] == 0:
        raise ValueError("preflight could not read any training batches")
    aggregate["padding_efficiency"] = aggregate["real_input_tokens"] / max(
        1,
        aggregate["padded_input_tokens"],
    )
    aggregate["supervised_eos_fraction"] = aggregate["supervised_eos_tokens"] / max(
        1,
        aggregate["supervised_tokens"],
    )
    if tokenizer.eos_token_id is not None and aggregate["supervised_eos_tokens"] == 0:
        raise RuntimeError(
            "preflight found zero supervised EOS labels. This usually means EOS "
            "is being masked out; do not start a long generation run until fixed."
        )
    return first_batch, aggregate


def preflight_cmd(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    validate_training_runtime_args(args)
    if args.tokenizer is None:
        raise ValueError("preflight requires --tokenizer")
    if args.generate_tokens < 0:
        raise ValueError("--generate-tokens must be non-negative")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_jsonl or (args.out_dir / "preflight_metrics.jsonl")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    tokenizer = load_tokenizer(args.tokenizer)
    train_loader, val_loader = build_loaders(args, tokenizer)
    first_batch, label_probe = probe_training_batches(
        train_loader,
        tokenizer,
        vocab_size=len(tokenizer),
        max_batches=args.probe_batches,
    )
    cfg = make_model_config(args, tokenizer)
    base_model = OpenMythosDenseLM(cfg).to(device)
    optimizer, optimizer_stats = make_optimizer(base_model, args)
    scheduler = make_lr_scheduler(optimizer, args)
    model = base_model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    start_payload = {
        "event": "preflight_start",
        "params": base_model.num_parameters(),
        "parameter_breakdown": base_model.parameter_breakdown(),
        "train_batches": len(train_loader),
        "val_batches": len(val_loader),
        "data": getattr(args, "data_stats", None),
        "label_probe": label_probe,
        "optimizer": optimizer_stats,
        "device": str(device),
        "dtype": args.dtype,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "config": asdict(cfg),
    }
    print_json(start_payload, indent=2)
    append_jsonl(metrics_path, start_payload)

    batch = move_batch(first_batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
        out = model(
            batch["input_ids"],
            labels=batch["labels"],
            n_loops=args.train_loops,
            collect_stats=True,
        )
    if out.loss is None:
        raise RuntimeError("preflight forward returned no loss")
    preflight_loss_tensor = out.loss.detach()
    if not bool(torch.isfinite(preflight_loss_tensor).item()):
        raise RuntimeError("preflight forward returned a non-finite loss")
    preflight_loss = float(preflight_loss_tensor.cpu().item())
    preflight_stats = snapshot_tensor_outputs(out.stats) if out.stats is not None else None

    grad_norm_value: Optional[float] = None
    if args.run_backward:
        out.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        if not bool(torch.isfinite(grad_norm.detach()).item()):
            raise RuntimeError("preflight backward produced a non-finite grad norm")
        grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        scheduler.step()

    eval_metrics = evaluate(
        model,
        val_loader,
        device=device,
        dtype=dtype,
        n_loops=args.train_loops,
        max_batches=args.eval_batches,
    )

    generated_shape: Optional[list[int]] = None
    if args.generate_tokens > 0:
        seq_len = int(batch["seq_lens"][0].item()) if "seq_lens" in batch else batch["input_ids"].shape[1]
        prompt_len = max(1, min(seq_len, cfg.max_seq_len))
        generated = model.generate(
            batch["input_ids"][:1, :prompt_len],
            max_new_tokens=args.generate_tokens,
            n_loops=args.train_loops,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_shape = [int(x) for x in generated.shape]

    checkpoint_path: Optional[Path] = None
    if args.check_save_load:
        checkpoint_path = args.out_dir / "preflight.pt"
        save_checkpoint(
            checkpoint_path,
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=1 if args.run_backward else 0,
            tokenizer_id=args.tokenizer,
            args=args,
        )
        reloaded, _ = load_checkpoint(checkpoint_path, device=device, dtype=dtype)
        reloaded.eval()

    summary = {
        "event": "preflight_ok",
        "loss": preflight_loss,
        "lm_loss": scalar_stat(preflight_stats, "lm_loss") if preflight_stats is not None else None,
        "grad_norm": grad_norm_value,
        "lr_after_step": scheduler.get_last_lr()[0],
        "eval": eval_metrics,
        "generated_shape": generated_shape,
        "checkpoint_path": checkpoint_path,
    }
    if preflight_stats is not None:
        for key in (
            "lti_A_min",
            "lti_A_max",
            "lti_tau_min",
            "lti_tau_max",
            "lti_input_gain_abs_max",
            "lti_delta_gain_abs_max",
            "lti_effective_input_abs_max",
            "lti_effective_delta_abs_max",
            "recurrent_input_mixer_h_gate_mean",
            "recurrent_output_bridge_h_gate_mean",
            "recurrent_coda_input_rms",
        ):
            value = scalar_stat(preflight_stats, key)
            if value is not None:
                summary[key] = value
    print_json(summary, indent=2)
    append_jsonl(metrics_path, summary)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    validate_training_runtime_args(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    if args.tokenizer is None:
        raise ValueError("train requires --tokenizer")
    tokenizer = load_tokenizer(args.tokenizer)
    train_loader, val_loader = build_loaders(args, tokenizer)
    train_batches_per_epoch = len(train_loader)
    if train_batches_per_epoch <= 0:
        raise ValueError("training loader has zero batches; reduce --batch-size")
    train_chunks_per_epoch = train_batches_per_epoch * args.batch_size
    if args.max_epochs is not None:
        if args.max_epochs <= 0.0:
            raise ValueError("--max-epochs must be positive when provided")
        args.max_steps = max(
            1,
            int(math.ceil(args.max_epochs * train_batches_per_epoch / args.grad_accum)),
        )

    cfg = make_model_config(args, tokenizer)
    model = OpenMythosDenseLM(cfg).to(device)
    optimizer, optimizer_stats = make_optimizer(model, args)
    scheduler = make_lr_scheduler(optimizer, args)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        load_model_state_compat(base_model, ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_step = int(ckpt.get("step", 0))
    sync_scheduler_to_step(scheduler, start_step)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics_path = args.metrics_jsonl or (args.out_dir / "metrics.jsonl")
    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    start_micro_batches_seen = start_step * args.grad_accum
    start_epoch_metrics = epoch_metrics(
        micro_batches_seen=start_micro_batches_seen,
        train_batches_per_epoch=train_batches_per_epoch,
        max_steps=args.max_steps,
        grad_accum=args.grad_accum,
    )
    startup_payload = {
        "event": "startup",
        "params": base_model.num_parameters(),
        "parameter_breakdown": base_model.parameter_breakdown(),
        "train_chunks": len(train_loader.dataset),
        "val_chunks": len(val_loader.dataset),
        "train_batches_per_epoch": train_batches_per_epoch,
        "train_chunks_per_epoch": train_chunks_per_epoch,
        "max_epochs": args.max_epochs,
        **start_epoch_metrics,
        "data": getattr(args, "data_stats", None),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch_chunks": args.batch_size * args.grad_accum,
        "loss_on": args.loss_on,
        "long_sample_policy": args.long_sample_policy,
        "min_response_tokens": args.min_response_tokens,
        "pack_samples": args.pack_samples,
        "dynamic_padding": not args.static_padding,
        "pad_to_multiple": args.pad_to_multiple,
        "length_bucketing": args.length_bucketing,
        "length_bucket_mult": args.length_bucket_mult,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "compile": args.compile,
        "compile_mode": args.compile_mode if args.compile else None,
        "dim": cfg.dim,
        "n_heads": cfg.n_heads,
        "ffn_hidden_dim": cfg.resolved_ffn_hidden_dim(),
        "prelude_layers": cfg.prelude_layers,
        "coda_layers": cfg.coda_layers,
        "max_loop_iters": cfg.max_loop_iters,
        "train_loops": args.train_loops,
        "recurrent_input_h_init": cfg.recurrent_input_h_init,
        "recurrent_output_h_init": cfg.recurrent_output_h_init,
        "use_ada_norm": cfg.use_ada_norm,
        "use_act": cfg.use_act,
        "optimizer": optimizer_stats,
        "lr_schedule": args.lr_schedule,
        "base_lr": args.lr,
        "min_lr": args.lr * args.min_lr_ratio,
        "decay_steps": args.decay_steps or args.max_steps,
        "config": asdict(cfg),
    }
    print_json(startup_payload, indent=2)
    append_jsonl(metrics_path, startup_payload)

    data_iter = iter(train_loader)
    micro_batches_seen = start_micro_batches_seen
    completed_epoch_events = int(micro_batches_seen // train_batches_per_epoch)
    running_loss = 0.0
    running_lm_loss = 0.0
    running_act_loss = 0.0
    running_tokens = 0
    running_real_input_tokens = 0
    running_padded_input_tokens = 0
    t0 = time.perf_counter()
    model.train()

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        accum_lm_loss = 0.0
        accum_act_loss = 0.0
        stats = None
        micro_batches: list[tuple[dict[str, torch.Tensor], int]] = []
        update_tokens = 0
        update_real_input_tokens = 0
        update_padded_input_tokens = 0

        for micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                completed = int(micro_batches_seen // train_batches_per_epoch)
                if completed > completed_epoch_events:
                    completed_epoch_events = completed
                    event = {
                        "event": "epoch_end",
                        "step": step + 1,
                        "completed_epochs": completed,
                        "micro_batches_seen": micro_batches_seen,
                        "chunks_seen": micro_batches_seen * args.batch_size,
                        **epoch_metrics(
                            micro_batches_seen=micro_batches_seen,
                            train_batches_per_epoch=train_batches_per_epoch,
                            max_steps=args.max_steps,
                            grad_accum=args.grad_accum,
                        ),
                    }
                    print_json(event)
                    append_jsonl(metrics_path, event)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            micro_batches_seen += 1
            token_count = int((batch["labels"] != -100).sum().item())
            if token_count == 0:
                continue
            micro_batches.append((batch, token_count))
            update_tokens += token_count
            update_real_input_tokens += int(batch["seq_lens"].sum().item())
            update_padded_input_tokens += int(batch["input_ids"].numel())

        if update_tokens == 0:
            raise RuntimeError("training update has zero supervised tokens")

        for micro, (batch, token_count) in enumerate(micro_batches):
            batch = move_batch(batch, device)
            collect_micro_stats = micro == 0 and (step + 1) % args.log_every == 0
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                out = model(
                    batch["input_ids"],
                    labels=batch["labels"],
                    n_loops=args.train_loops,
                    collect_stats=collect_micro_stats,
                )
                if out.loss is None:
                    raise RuntimeError("model returned no loss during training")
                loss = out.loss * (token_count / update_tokens)

            loss.backward()
            accum_loss += float(loss.detach().cpu())
            out_stats = snapshot_tensor_outputs(out.stats) if out.stats is not None else None
            if out_stats is not None:
                lm_loss_value = scalar_stat(out_stats, "lm_loss")
                if lm_loss_value is not None:
                    accum_lm_loss += lm_loss_value * (token_count / update_tokens)
                act_loss_value = scalar_stat(out_stats, "act_loss")
                if act_loss_value is not None:
                    accum_act_loss += act_loss_value * (token_count / update_tokens)
            if out_stats is not None and (collect_micro_stats or stats is None):
                stats = out_stats

        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step_num = step + 1
        running_loss += accum_loss
        running_lm_loss += accum_lm_loss
        running_act_loss += accum_act_loss
        running_tokens += update_tokens
        running_real_input_tokens += update_real_input_tokens
        running_padded_input_tokens += update_padded_input_tokens

        if step_num % args.log_every == 0:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            tok_s = running_tokens / elapsed
            input_tok_s = running_real_input_tokens / elapsed
            padded_tok_s = running_padded_input_tokens / elapsed
            log: dict[str, Any] = {
                "step": step_num,
                "loss": running_loss / args.log_every,
                "lm_loss": running_lm_loss / args.log_every,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tok_s": tok_s,
                "input_tok_s": input_tok_s,
                "padded_tok_s": padded_tok_s,
                "padding_efficiency": running_real_input_tokens
                / max(1, running_padded_input_tokens),
                "padded_seq_len_mean": running_padded_input_tokens
                / max(1, args.log_every * args.grad_accum * args.batch_size),
                "real_seq_len_mean": running_real_input_tokens
                / max(1, args.log_every * args.grad_accum * args.batch_size),
                "micro_batches_seen": micro_batches_seen,
                "chunks_seen": micro_batches_seen * args.batch_size,
                **epoch_metrics(
                    micro_batches_seen=micro_batches_seen,
                    train_batches_per_epoch=train_batches_per_epoch,
                    max_steps=args.max_steps,
                    grad_accum=args.grad_accum,
                ),
            }
            if running_act_loss > 0.0:
                log["act_loss"] = running_act_loss / args.log_every
            if device.type == "cuda":
                log["cuda_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if stats is not None:
                for key in (
                    "lti_A_min",
                    "lti_A_max",
                    "lti_tau_min",
                    "lti_tau_max",
                    "lti_B_abs_max",
                    "lti_input_gain_abs_max",
                    "lti_delta_gain_abs_max",
                    "lti_effective_input_abs_max",
                    "lti_effective_delta_abs_max",
                    "recurrent_input_mixer_h_gate_mean",
                    "recurrent_output_bridge_h_gate_mean",
                    "recurrent_coda_input_rms",
                    "act_ponder_loss",
                    "act_expected_steps",
                    "act_hard_steps",
                    "act_remainder",
                    "act_halt_fraction",
                    "act_halting_p_mean",
                ):
                    value = scalar_stat(stats, key)
                    if value is not None:
                        log[key] = value
                if stats.get("recurrent_loop_rms") is not None:
                    log["loop_rms"] = [
                        round(float(x), 6)
                        for x in stats["recurrent_loop_rms"].detach().cpu()
                    ]
                for stat_key, log_key in (
                    ("recurrent_mixed_rms", "mixed_rms"),
                    ("recurrent_delta_rms", "delta_rms"),
                    ("recurrent_drive_rms", "drive_rms"),
                    ("recurrent_h_e_cosine", "h_e_cosine"),
                    ("recurrent_h_delta_cosine", "h_delta_cosine"),
                ):
                    if stats.get(stat_key) is not None:
                        log[log_key] = [
                            round(float(x), 6)
                            for x in stats[stat_key].detach().cpu()
                        ]
            print_json(log)
            append_jsonl(metrics_path, {"event": "train", **log})
            running_loss = 0.0
            running_lm_loss = 0.0
            running_act_loss = 0.0
            running_tokens = 0
            running_real_input_tokens = 0
            running_padded_input_tokens = 0
            t0 = time.perf_counter()

        if step_num % args.eval_every == 0:
            metrics = evaluate(
                model,
                val_loader,
                device=device,
                dtype=dtype,
                n_loops=args.train_loops,
                max_batches=args.eval_batches,
            )
            eval_payload = {
                "event": "eval",
                "step": step_num,
                "eval": metrics,
                **epoch_metrics(
                    micro_batches_seen=micro_batches_seen,
                    train_batches_per_epoch=train_batches_per_epoch,
                    max_steps=args.max_steps,
                    grad_accum=args.grad_accum,
                ),
            }
            print_json({key: value for key, value in eval_payload.items() if key != "event"})
            append_jsonl(metrics_path, eval_payload)
            model.train()

        if step_num % args.save_every == 0:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            checkpoint_path = args.out_dir / f"step_{step_num:07d}.pt"
            save_checkpoint(
                checkpoint_path,
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step_num,
                tokenizer_id=args.tokenizer,
                args=args,
            )
            append_jsonl(
                metrics_path,
                {
                    "event": "checkpoint",
                    "step": step_num,
                    "path": checkpoint_path,
                    **epoch_metrics(
                        micro_batches_seen=micro_batches_seen,
                        train_batches_per_epoch=train_batches_per_epoch,
                        max_steps=args.max_steps,
                        grad_accum=args.grad_accum,
                    ),
                },
            )

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    save_checkpoint(
        args.out_dir / "final.pt",
        model=base_model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=args.max_steps,
        tokenizer_id=args.tokenizer,
        args=args,
    )
    append_jsonl(
        metrics_path,
        {
            "event": "checkpoint",
            "step": args.max_steps,
            "path": args.out_dir / "final.pt",
            "final": True,
            **epoch_metrics(
                micro_batches_seen=micro_batches_seen,
                train_batches_per_epoch=train_batches_per_epoch,
                max_steps=args.max_steps,
                grad_accum=args.grad_accum,
            ),
        },
    )


def eval_cmd(args: argparse.Namespace) -> None:
    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    model, ckpt = load_checkpoint(args.checkpoint, device=device, dtype=dtype)
    tokenizer_id = args.tokenizer if getattr(args, "tokenizer", None) else ckpt.get("tokenizer")
    if tokenizer_id is None:
        raise ValueError("checkpoint has no tokenizer id; pass --tokenizer")
    tokenizer = load_tokenizer(tokenizer_id)
    args.tokenizer = tokenizer_id
    train_loader, val_loader = build_loaders(args, tokenizer)
    del train_loader
    metrics = evaluate(
        model,
        val_loader,
        device=device,
        dtype=dtype,
        n_loops=args.n_loops,
        max_batches=args.max_batches,
    )
    print_json(metrics, indent=2)


@torch.no_grad()
def exact_eval_cmd(args: argparse.Namespace) -> None:
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if args.print_samples < 0:
        raise ValueError("--print-samples must be non-negative")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    model, ckpt = load_checkpoint(args.checkpoint, device=device, dtype=dtype)
    model.eval()

    tokenizer_id = args.tokenizer or ckpt.get("tokenizer")
    if tokenizer_id is None:
        raise ValueError("checkpoint has no tokenizer id; pass --tokenizer")
    tokenizer = load_tokenizer(tokenizer_id)
    args.tokenizer = tokenizer_id

    raw = load_openr1_dataset(args)
    if args.max_samples is not None:
        max_samples = min(args.max_samples, len(raw))
        raw = raw.shuffle(seed=args.seed).select(range(max_samples))

    pred_fp = None
    if args.predictions_path is not None:
        args.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        pred_fp = args.predictions_path.open("w", encoding="utf-8")

    start_payload = {
        "event": "exact_eval_start",
        "checkpoint": str(args.checkpoint),
        "tokenizer": tokenizer_id,
        "samples": len(raw),
        "prompt_format": "Problem:\\n{problem}\\n\\nSolution:\\n",
        "max_new_tokens": args.max_new_tokens,
        "n_loops": args.n_loops,
        "sample": args.sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "predictions_path": args.predictions_path,
    }
    print_json(start_payload, indent=2)
    append_jsonl(args.metrics_jsonl, start_payload)

    evaluated = 0
    correct = 0
    skipped_no_answer = 0
    skipped_no_prompt = 0
    skipped_too_long = 0
    generated_tokens = 0
    eos_count = 0
    hit_max_count = 0
    repeat_4gram_sum = 0.0
    unique_token_ratio_sum = 0.0
    printed = 0

    try:
        for row_idx, row in enumerate(raw):
            prompt = format_openr1_prompt(row)
            target = reference_answer(row)
            if not prompt:
                skipped_no_prompt += 1
                continue
            if not target:
                skipped_no_answer += 1
                continue

            input_ids = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )["input_ids"]
            prompt_tokens = int(input_ids.shape[1])
            if prompt_tokens <= 0:
                skipped_no_prompt += 1
                continue
            if prompt_tokens > model.cfg.max_seq_len:
                skipped_too_long += 1
                continue

            input_ids = input_ids.to(device)
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                n_loops=args.n_loops,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.sample,
                eos_token_id=tokenizer.eos_token_id,
            )
            gen_ids = out[0, prompt_tokens:]
            gen_list = [int(x) for x in gen_ids.detach().cpu().tolist()]
            generated_count = len(gen_list)
            eos_emitted = tokenizer.eos_token_id is not None and tokenizer.eos_token_id in gen_list
            hit_max_new_tokens = generated_count >= args.max_new_tokens and not eos_emitted
            repeat_4gram = repeated_ngram_ratio(gen_list, n=4)
            unique_ratio = unique_token_ratio(gen_list)

            generated_tokens += generated_count
            eos_count += int(eos_emitted)
            hit_max_count += int(hit_max_new_tokens)
            repeat_4gram_sum += repeat_4gram
            unique_token_ratio_sum += unique_ratio
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            prediction = extract_final_answer(generated_text)
            is_correct = answers_exact_match(prediction, target)

            evaluated += 1
            correct += int(is_correct)

            record = {
                "row_idx": row_idx,
                "exact": is_correct,
                "problem": str(row.get("problem") or "").strip(),
                "target": target,
                "prediction": prediction,
                "target_norm": normalize_answer(target),
                "prediction_norm": normalize_answer(prediction),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_count,
                "eos_emitted": eos_emitted,
                "hit_max_new_tokens": hit_max_new_tokens,
                "repeat_4gram_ratio": repeat_4gram,
                "unique_token_ratio": unique_ratio,
                "generated_text": generated_text,
            }
            if pred_fp is not None:
                pred_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            append_jsonl(
                args.metrics_jsonl,
                {
                    "event": "exact_sample",
                    "row_idx": row_idx,
                    "exact": is_correct,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_count,
                    "eos_emitted": eos_emitted,
                    "hit_max_new_tokens": hit_max_new_tokens,
                    "repeat_4gram_ratio": repeat_4gram,
                    "unique_token_ratio": unique_ratio,
                },
            )

            if printed < args.print_samples:
                print_json({"sample": record})
                printed += 1

            if args.log_every > 0 and evaluated % args.log_every == 0:
                print_json(
                    {
                        "evaluated": evaluated,
                        "correct": correct,
                        "exact_match": correct / max(1, evaluated),
                        "eos_rate": eos_count / max(1, evaluated),
                        "hit_max_new_tokens_rate": hit_max_count / max(1, evaluated),
                        "avg_repeat_4gram_ratio": repeat_4gram_sum / max(1, evaluated),
                    }
                )
    finally:
        if pred_fp is not None:
            pred_fp.close()

    metrics = {
        "n_loops": args.n_loops,
        "evaluated": evaluated,
        "correct": correct,
        "exact_match": correct / max(1, evaluated),
        "skipped_no_answer": skipped_no_answer,
        "skipped_no_prompt": skipped_no_prompt,
        "skipped_too_long": skipped_too_long,
        "avg_generated_tokens": generated_tokens / max(1, evaluated),
        "eos_rate": eos_count / max(1, evaluated),
        "hit_max_new_tokens_rate": hit_max_count / max(1, evaluated),
        "avg_repeat_4gram_ratio": repeat_4gram_sum / max(1, evaluated),
        "avg_unique_token_ratio": unique_token_ratio_sum / max(1, evaluated),
    }
    print_json({"exact_eval": metrics}, indent=2)
    append_jsonl(args.metrics_jsonl, {"event": "exact_eval", "exact_eval": metrics})


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt is not None:
        return args.prompt
    raise ValueError("pass --prompt or --prompt-file")


def generate_cmd(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    model, ckpt = load_checkpoint(args.checkpoint, device=device, dtype=dtype)
    model.eval()
    tokenizer_id = args.tokenizer or ckpt.get("tokenizer")
    if tokenizer_id is None:
        raise ValueError("checkpoint has no tokenizer id; pass --tokenizer")
    tokenizer = load_tokenizer(tokenizer_id)
    prompt = read_prompt(args)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        n_loops=args.n_loops,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True), flush=True)


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "preflight":
        preflight_cmd(args)
    elif args.cmd == "eval":
        eval_cmd(args)
    elif args.cmd == "exact-eval":
        exact_eval_cmd(args)
    elif args.cmd == "generate":
        generate_cmd(args)
    else:
        raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
