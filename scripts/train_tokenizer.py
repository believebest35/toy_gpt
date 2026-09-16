"""Train the TinyGPT BPE tokenizer on TinyStories."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Avoid the native hf-xet downloader on hosts where it aborts at interpreter exit.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset

from tinygpt.tokenizer import (
    DEFAULT_VOCAB_SIZE,
    SPECIAL_TOKENS,
    decode,
    encode,
    save_tokenizer,
    train_tokenizer,
)


DATASET_NAME = "roneneldan/TinyStories"
DEFAULT_OUTPUT = Path("artifacts/tokenizer/tokenizer.json")


def find_text_field(split: Any) -> str:
    """Return the story-text column after checking the loaded schema."""

    columns = list(split.column_names)
    if "text" in columns:
        return "text"
    raise ValueError(f"Could not find the expected text field in columns: {columns}")


def story_iterator(
    split: Any, text_field: str, max_samples: int | None = None
) -> Iterator[str]:
    """Yield story text lazily, optionally stopping after a sample limit."""

    for index, example in enumerate(split):
        if max_samples is not None and index >= max_samples:
            break
        text = example[text_field]
        if not isinstance(text, str):
            raise ValueError(f"TinyStories field '{text_field}' must contain strings")
        yield text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Tokenizer JSON output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
        help=f"Target vocabulary size (default: {DEFAULT_VOCAB_SIZE})",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Train on at most this many stories for debugging",
    )
    args = parser.parse_args()
    if args.vocab_size < len(SPECIAL_TOKENS):
        parser.error(f"--vocab-size must be at least {len(SPECIAL_TOKENS)}")
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    streaming = args.max_samples is not None
    dataset = load_dataset(DATASET_NAME, streaming=streaming)
    if "train" not in dataset:
        raise RuntimeError(f"TinyStories has no training split: {list(dataset.keys())}")

    training_split = dataset["train"]
    text_field = find_text_field(training_split)
    if args.max_samples is not None:
        print(f"Using a limited TinyStories subset: {args.max_samples} samples")

    tokenizer = train_tokenizer(
        story_iterator(training_split, text_field, args.max_samples),
        vocab_size=args.vocab_size,
    )
    vocabulary_size = tokenizer.get_vocab_size()
    if args.max_samples is None and vocabulary_size != args.vocab_size:
        raise RuntimeError(
            f"Full-data tokenizer vocabulary has size {vocabulary_size}; "
            f"expected {args.vocab_size}"
        )
    if args.max_samples is not None and vocabulary_size != args.vocab_size:
        print(
            "Limited-sample vocabulary size: "
            f"{vocabulary_size} (target: {args.vocab_size})"
        )

    output_path = save_tokenizer(tokenizer, args.output)
    special_token_ids = {
        token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS
    }
    missing_tokens = [
        token for token, token_id in special_token_ids.items() if token_id is None
    ]
    if missing_tokens:
        raise RuntimeError(f"Missing special tokens after training: {missing_tokens}")

    example_text = "Once upon a time there was a little dog."
    encoded = encode(tokenizer, example_text)
    decoded = decode(tokenizer, encoded)

    print(f"Tokenizer output: {output_path}")
    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Special-token IDs: {special_token_ids}")
    print(f"Example text: {example_text}")
    print(f"Encoded token IDs: {encoded}")
    print(f"Decoded example: {decoded}")


if __name__ == "__main__":
    main()
