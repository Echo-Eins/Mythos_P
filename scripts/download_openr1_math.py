#!/usr/bin/env python3
"""
Download open-r1/OpenR1-Math-220k for local training.

Default behavior:
    python scripts/download_openr1_math.py

This downloads the `default` subset, saves the Hugging Face DatasetDict with
`save_to_disk`, and writes a compact metadata file.  Optional JSONL export is
provided for quick inspection or non-HF dataloaders.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


DATASET_ID = "open-r1/OpenR1-Math-220k"
SUBSETS = ("default", "all", "extended")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download open-r1/OpenR1-Math-220k from Hugging Face."
    )
    parser.add_argument(
        "--subset",
        choices=SUBSETS,
        default="default",
        help="Dataset subset/config to download.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split to load, e.g. train. By default loads the DatasetDict.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "openr1_math_220k",
        help="Output directory.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also export each split to JSONL.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to datasets.load_dataset.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face token. If omitted, uses HF_TOKEN or "
            "HUGGINGFACE_HUB_TOKEN from the environment."
        ),
    )
    return parser.parse_args()


def dataset_lengths(ds: Dataset | DatasetDict) -> dict[str, int]:
    if isinstance(ds, DatasetDict):
        return {name: len(split) for name, split in ds.items()}
    return {"dataset": len(ds)}


def dataset_columns(ds: Dataset | DatasetDict) -> dict[str, list[str]]:
    if isinstance(ds, DatasetDict):
        return {name: list(split.column_names) for name, split in ds.items()}
    return {"dataset": list(ds.column_names)}


def write_jsonl(path: Path, ds: Dataset) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_jsonl(out_dir: Path, ds: Dataset | DatasetDict) -> None:
    jsonl_dir = out_dir / "jsonl"
    if isinstance(ds, DatasetDict):
        for split_name, split in ds.items():
            write_jsonl(jsonl_dir / f"{split_name}.jsonl", split)
    else:
        write_jsonl(jsonl_dir / "dataset.jsonl", ds)


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hf_token = resolve_hf_token(args.hf_token)
    load_kwargs: dict[str, Any] = {
        "path": DATASET_ID,
        "name": args.subset,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.split is not None:
        load_kwargs["split"] = args.split
    if args.cache_dir is not None:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    if hf_token is not None:
        load_kwargs["token"] = hf_token

    ds = load_dataset(**load_kwargs)

    hf_dir = args.out_dir / f"hf_{args.subset}"
    if args.split is not None:
        hf_dir = args.out_dir / f"hf_{args.subset}_{args.split}"
    ds.save_to_disk(str(hf_dir))

    if args.jsonl:
        export_jsonl(args.out_dir, ds)

    metadata = {
        "dataset_id": DATASET_ID,
        "subset": args.subset,
        "split": args.split,
        "saved_to": str(hf_dir),
        "lengths": dataset_lengths(ds),
        "columns": dataset_columns(ds),
    }
    metadata_path = args.out_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
