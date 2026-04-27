#!/usr/bin/env python3
"""
Train/evaluate/generate with the dense OpenMythos RDT prototype on OpenR1 Math.

This script is intended to run on the Spark/GB10 environment where torch,
flash-attn, datasets, transformers, and the tokenizer stack are installed.
It is not expected to run on the local Windows machine used for code editing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

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
) -> None:
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--dataset-subset", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokenizer", default=tokenizer_default)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dim", type=int, default=512)
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


def format_openr1_sample(row: dict[str, Any]) -> str:
    problem = str(row.get("problem") or "").strip()
    solution = select_best_solution(row)
    answer = str(row.get("answer") or "").strip()

    text = f"Problem:\n{problem}\n\nSolution:\n{solution}"
    if answer:
        text += f"\n\nFinal answer:\n{answer}"
    return text.strip()


def build_token_sequences(
    ds: Dataset,
    tokenizer,
    *,
    max_seq_len: int,
    max_samples: Optional[int],
    seed: int,
) -> list[list[int]]:
    if max_samples is not None:
        max_samples = min(max_samples, len(ds))
        ds = ds.shuffle(seed=seed).select(range(max_samples))

    eos = tokenizer.eos_token_id
    sequences: list[list[int]] = []
    for row in ds:
        text = format_openr1_sample(row)
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos is not None:
            ids.append(eos)
        if len(ids) < 2:
            continue
        if len(ids) > max_seq_len + 1:
            ids = ids[: max_seq_len + 1]
            if eos is not None:
                ids[-1] = eos
        sequences.append(ids)
    return sequences


class PackedCausalDataset(TorchDataset):
    """
    Packs variable-length tokenized samples into fixed-length causal LM chunks.

    Each item returns:
        input_ids = chunk[:-1]
        labels    = chunk[1:]
    """

    def __init__(
        self,
        sequences: Iterable[list[int]],
        *,
        seq_len: int,
        pad_token_id: int,
    ) -> None:
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.chunks: list[list[int]] = []

        buffer: list[int] = []
        for ids in sequences:
            buffer.extend(ids)
            while len(buffer) >= seq_len + 1:
                self.chunks.append(buffer[: seq_len + 1])
                buffer = buffer[seq_len + 1 :]
        if len(buffer) > 1:
            padded = buffer + [pad_token_id] * (seq_len + 1 - len(buffer))
            self.chunks.append(padded)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        labels[labels == self.pad_token_id] = -100
        return {"input_ids": input_ids, "labels": labels}


def split_sequences(
    sequences: list[list[int]],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[list[int]], list[list[int]]]:
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
    sequences = build_token_sequences(
        raw,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        seed=args.seed,
    )
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
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )


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
    objective_losses: list[float] = []
    lm_losses: list[float] = []
    first_stats: Optional[dict[str, Any]] = None
    autocast_enabled = device.type == "cuda" and dtype != torch.float32
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
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
        objective_losses.append(objective_loss)
        lm_loss = scalar_stat(out.stats, "lm_loss") if out.stats is not None else None
        lm_losses.append(objective_loss if lm_loss is None else lm_loss)
        if first_stats is None and out.stats is not None:
            first_stats = out.stats
    mean_objective_loss = sum(objective_losses) / max(1, len(objective_losses))
    mean_lm_loss = sum(lm_losses) / max(1, len(lm_losses))
    metrics: dict[str, float] = {
        "loss": mean_objective_loss,
        "lm_loss": mean_lm_loss,
        "ppl": math.exp(min(mean_lm_loss, 20.0)),
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
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

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
                "device": str(device),
                "dtype": args.dtype,
            },
            indent=2,
        )
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

        for micro in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

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
                loss = out.loss / args.grad_accum

            loss.backward()
            accum_loss += float(loss.detach().cpu())
            running_tokens += int((batch["labels"] != -100).sum().item())
            if out.stats is not None and (collect_micro_stats or stats is None):
                stats = out.stats

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        step_num = step + 1
        running_loss += accum_loss

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
            print(json.dumps(log, ensure_ascii=False))
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
            print(json.dumps({"step": step_num, "eval": metrics}, ensure_ascii=False))
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
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


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
    print(tokenizer.decode(out[0], skip_special_tokens=True))


def main() -> None:
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_cmd(args)
    elif args.cmd == "generate":
        generate_cmd(args)
    else:
        raise ValueError(f"unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
