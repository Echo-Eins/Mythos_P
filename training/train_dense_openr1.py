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

    evaluate = sub.add_parser("eval", help="Evaluate a checkpoint")
    add_data_args(evaluate, tokenizer_default=None)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-batches", type=int, default=100)
    evaluate.add_argument("--num-workers", type=int, default=2)
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


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--prelude-layers", type=int, default=1)
    parser.add_argument("--coda-layers", type=int, default=1)
    parser.add_argument("--max-loop-iters", type=int, default=4)
    parser.add_argument("--train-loops", type=int, default=None)
    parser.add_argument("--ffn-hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope-theta", type=float, default=500_000.0)
    parser.add_argument("--use-act", action="store_true")
    parser.add_argument("--act-threshold", type=float, default=0.99)
    parser.add_argument("--act-ponder-weight", type=float, default=0.01)
    parser.add_argument("--act-min-steps", type=int, default=1)


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / "dense_openr1")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2000)
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
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--compile", action="store_true")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


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
    Packs variable-length tokenized samples into fixed-length causal LM chunks.

    Each item returns:
        input_ids = chunk[:-1]
        labels    = chunk[1:], with non-supervised tokens masked to -100.
    """

    def __init__(
        self,
        sequences: Iterable[TokenSequence],
        *,
        seq_len: int,
        pad_token_id: int,
    ) -> None:
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.chunks: list[tuple[list[int], list[int]]] = []

        token_buffer: list[int] = []
        mask_buffer: list[int] = []

        def append_chunk(ids: list[int], mask: list[int]) -> None:
            if len(ids) < 2 or sum(mask[1:]) == 0:
                return
            self.chunks.append((ids, mask))

        def flush_padded() -> None:
            nonlocal token_buffer, mask_buffer
            if len(token_buffer) > 1:
                pad_count = seq_len + 1 - len(token_buffer)
                padded_ids = token_buffer + [pad_token_id] * pad_count
                padded_mask = mask_buffer + [0] * pad_count
                append_chunk(padded_ids, padded_mask)
            token_buffer = []
            mask_buffer = []

        for seq in sequences:
            if len(seq.ids) != len(seq.label_mask):
                raise ValueError("token ids and label mask must have the same length")
            if len(seq.ids) > seq_len + 1:
                raise ValueError("sequence length exceeds packed chunk length")
            if len(token_buffer) + len(seq.ids) > seq_len + 1:
                flush_padded()
            if len(seq.ids) == seq_len + 1:
                append_chunk(seq.ids, seq.label_mask)
                continue
            token_buffer.extend(seq.ids)
            mask_buffer.extend(seq.label_mask)
        flush_padded()

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, mask = self.chunks[idx]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        label_mask = torch.tensor(mask[1:], dtype=torch.bool)
        labels[~label_mask] = -100
        labels[labels == self.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


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


def build_loaders(args: argparse.Namespace, tokenizer) -> tuple[DataLoader, DataLoader]:
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
    train_ds = PackedCausalDataset(train_seq, seq_len=args.max_seq_len, pad_token_id=pad_id)
    val_ds = PackedCausalDataset(val_seq, seq_len=args.max_seq_len, pad_token_id=pad_id)
    if len(train_ds) == 0:
        raise ValueError("training dataset has zero chunks")
    if len(val_ds) == 0:
        raise ValueError("validation dataset has zero chunks")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
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
    model.load_state_dict(ckpt["model"])
    model.to(device=device)
    if dtype != torch.float32:
        model.to(dtype=dtype)
    return model, ckpt


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
        lm_loss = scalar_stat(out.stats, "lm_loss") if out.stats is not None else None
        objective_loss_sum += objective_loss * token_count
        lm_loss_sum += (objective_loss if lm_loss is None else lm_loss) * token_count
        supervised_tokens += token_count
        if first_stats is None and out.stats is not None:
            first_stats = out.stats
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
            "act_loss",
            "act_ponder_loss",
            "act_expected_steps",
            "act_hard_steps",
            "act_halt_fraction",
            "act_halting_p_mean",
        ):
            value = scalar_stat(first_stats, key)
            if value is not None:
                metrics[key] = value
    return metrics


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = dtype_from_name(args.dtype)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    if args.tokenizer is None:
        raise ValueError("train requires --tokenizer")
    tokenizer = load_tokenizer(args.tokenizer)
    train_loader, val_loader = build_loaders(args, tokenizer)

    cfg = make_model_config(args, tokenizer)
    model = OpenMythosDenseLM(cfg).to(device)
    if args.compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = make_lr_scheduler(optimizer, args)

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        base_model.load_state_dict(ckpt["model"])
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

    print(
        json.dumps(
            {
                "params": (model._orig_mod if hasattr(model, "_orig_mod") else model).num_parameters(),
                "train_chunks": len(train_loader.dataset),
                "val_chunks": len(val_loader.dataset),
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
                "train_loops": args.train_loops,
                "lr_schedule": args.lr_schedule,
                "base_lr": args.lr,
                "min_lr": args.lr * args.min_lr_ratio,
                "decay_steps": args.decay_steps or args.max_steps,
            },
            indent=2,
        ),
        flush=True,
    )

    data_iter = iter(train_loader)
    running_loss = 0.0
    running_tokens = 0
    t0 = time.perf_counter()
    model.train()

    for step in range(start_step, args.max_steps):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        stats = None
        micro_batches: list[tuple[dict[str, torch.Tensor], int]] = []
        update_tokens = 0

        for micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            token_count = int((batch["labels"] != -100).sum().item())
            if token_count == 0:
                continue
            micro_batches.append((batch, token_count))
            update_tokens += token_count

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
            if out.stats is not None and (collect_micro_stats or stats is None):
                stats = out.stats

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step_num = step + 1
        running_loss += accum_loss
        running_tokens += update_tokens

        if step_num % args.log_every == 0:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            tok_s = running_tokens / elapsed
            log: dict[str, Any] = {
                "step": step_num,
                "loss": running_loss / args.log_every,
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "tok_s": tok_s,
            }
            if device.type == "cuda":
                log["cuda_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            if stats is not None:
                for key in (
                    "lti_A_min",
                    "lti_A_max",
                    "lti_B_abs_max",
                    "lm_loss",
                    "act_loss",
                    "act_ponder_loss",
                    "act_expected_steps",
                    "act_hard_steps",
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
            print(json.dumps(log, ensure_ascii=False), flush=True)
            running_loss = 0.0
            running_tokens = 0
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
            print(json.dumps({"step": step_num, "eval": metrics}, ensure_ascii=False), flush=True)
            model.train()

        if step_num % args.save_every == 0:
            base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_checkpoint(
                args.out_dir / f"step_{step_num:07d}.pt",
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step_num,
                tokenizer_id=args.tokenizer,
                args=args,
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


def eval_cmd(args: argparse.Namespace) -> None:
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


@torch.no_grad()
def exact_eval_cmd(args: argparse.Namespace) -> None:
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

    print(
        json.dumps(
            {
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
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    evaluated = 0
    correct = 0
    skipped_no_answer = 0
    skipped_no_prompt = 0
    skipped_too_long = 0
    generated_tokens = 0
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
            generated_tokens += int(gen_ids.numel())
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
                "generated_tokens": int(gen_ids.numel()),
                "generated_text": generated_text,
            }
            if pred_fp is not None:
                pred_fp.write(json.dumps(record, ensure_ascii=False) + "\n")

            if printed < args.print_samples:
                print(json.dumps({"sample": record}, ensure_ascii=False), flush=True)
                printed += 1

            if args.log_every > 0 and evaluated % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "evaluated": evaluated,
                            "correct": correct,
                            "exact_match": correct / max(1, evaluated),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        if pred_fp is not None:
            pred_fp.close()

    metrics = {
        "evaluated": evaluated,
        "correct": correct,
        "exact_match": correct / max(1, evaluated),
        "skipped_no_answer": skipped_no_answer,
        "skipped_no_prompt": skipped_no_prompt,
        "skipped_too_long": skipped_too_long,
        "avg_generated_tokens": generated_tokens / max(1, evaluated),
    }
    print(json.dumps({"exact_eval": metrics}, ensure_ascii=False, indent=2), flush=True)


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
