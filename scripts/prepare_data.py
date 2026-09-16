"""Prepare EOS-separated TinyStories token streams for language-model training."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Avoid the native hf-xet downloader on hosts where it aborts at interpreter exit.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import DownloadConfig, load_dataset

from tinygpt.dataset import TokenStreamStats, write_stories_to_binary
from tinygpt.tokenizer import load_tokenizer


DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_TOKENIZER = Path("artifacts/tokenizer/tokenizer.json")
DEFAULT_TRAIN_OUTPUT = Path("data/train.bin")
DEFAULT_VAL_OUTPUT = Path("data/val.bin")
FALLBACK_VALIDATION_SIZE = 10_000


def find_text_field(split: Any) -> str:
    """Return the canonical TinyStories text field after validating the schema."""

    columns = list(split.column_names)
    if "text" not in columns:
        raise ValueError(f"Could not find the expected text field in columns: {columns}")
    return "text"


def story_iterator(
    split: Any,
    text_field: str,
    max_samples: int | None = None,
) -> Iterator[str]:
    """Yield story text lazily, optionally stopping after a sample limit."""

    for index, example in enumerate(split):
        if max_samples is not None and index >= max_samples:
            break
        text = example[text_field]
        if not isinstance(text, str):
            raise ValueError(f"TinyStories field '{text_field}' must contain strings")
        yield text


def prepare_split(
    split_name: str,
    split: Any,
    text_field: str,
    tokenizer: Any,
    output_path: Path,
    max_samples: int | None = None,
) -> TokenStreamStats:
    """Prepare one dataset split and print its resulting stream statistics."""

    if max_samples is not None:
        print(f"Preparing a limited {split_name} split: {max_samples} samples")

    stats = write_stories_to_binary(
        story_iterator(split, text_field, max_samples),
        tokenizer,
        output_path,
    )
    file_size = output_path.stat().st_size
    print(f"Split: {split_name}")
    print(f"Stories processed: {stats.stories_processed}")
    print(f"Tokens written: {stats.tokens_written}")
    print(f"Output path: {output_path}")
    print(f"File size: {file_size} bytes")
    print(f"EOS token ID: {stats.eos_token_id}")
    return stats


def make_fallback_splits(train_split: Any) -> tuple[Any, Any]:
    """Make deterministic train/validation views when no validation exists."""

    if hasattr(train_split, "select"):
        total_examples = len(train_split)
        if total_examples <= FALLBACK_VALIDATION_SIZE:
            raise RuntimeError(
                "TinyStories training split is too small for the deterministic "
                "validation fallback"
            )
        validation_split = train_split.select(range(FALLBACK_VALIDATION_SIZE))
        remaining_indices = range(FALLBACK_VALIDATION_SIZE, total_examples)
        return train_split.select(remaining_indices), validation_split

    validation_split = train_split.take(FALLBACK_VALIDATION_SIZE)
    return train_split.skip(FALLBACK_VALIDATION_SIZE), validation_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER,
        help=f"Tokenizer JSON path (default: {DEFAULT_TOKENIZER})",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=DEFAULT_TRAIN_OUTPUT,
        help=f"Training token stream path (default: {DEFAULT_TRAIN_OUTPUT})",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=DEFAULT_VAL_OUTPUT,
        help=f"Validation token stream path (default: {DEFAULT_VAL_OUTPUT})",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Prepare at most this many training stories for debugging",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Prepare at most this many validation stories for debugging",
    )
    args = parser.parse_args()
    for option_name in ("max_train_samples", "max_val_samples"):
        value = getattr(args, option_name)
        if value is not None and value <= 0:
            parser.error(f"--{option_name.replace('_', '-')} must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    if not args.tokenizer.is_file():
        raise FileNotFoundError(
            f"Tokenizer artifact not found: {args.tokenizer}. "
            "Run scripts/train_tokenizer.py first."
        )

    tokenizer = load_tokenizer(args.tokenizer)
    dataset = load_dataset(
        DATASET_NAME,
        streaming=False,
        download_config=DownloadConfig(local_files_only=True),
    )
    available_splits = list(dataset.keys())
    if "train" not in dataset:
        raise RuntimeError(f"TinyStories has no training split: {available_splits}")

    train_split = dataset["train"]
    if "validation" in dataset:
        validation_split = dataset["validation"]
        print(f"Using TinyStories validation split. Available splits: {available_splits}")
    else:
        train_split, validation_split = make_fallback_splits(train_split)
        print(
            "TinyStories has no validation split; using the first "
            f"{FALLBACK_VALIDATION_SIZE} training stories as validation and "
            "the remaining stories for training."
        )

    train_text_field = find_text_field(train_split)
    validation_text_field = find_text_field(validation_split)
    prepare_split(
        "train",
        train_split,
        train_text_field,
        tokenizer,
        args.train_output,
        args.max_train_samples,
    )
    prepare_split(
        "validation",
        validation_split,
        validation_text_field,
        tokenizer,
        args.val_output,
        args.max_val_samples,
    )


if __name__ == "__main__":
    main()
