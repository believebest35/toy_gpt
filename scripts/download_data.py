"""Verify access to the public TinyStories dataset."""

from __future__ import annotations

import os
from typing import Any

# Avoid the native hf-xet downloader on hosts where it aborts at interpreter exit.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset, load_dataset_builder


DATASET_NAME = "roneneldan/TinyStories"


def find_text_field(split: Any) -> str:
    """Return the story-text column after checking the loaded schema."""

    columns = list(split.column_names)
    if "text" in columns:
        return "text"

    first_example = next(iter(split), None)
    if first_example is None:
        raise ValueError("TinyStories training split is empty")
    for column in columns:
        if isinstance(first_example.get(column), str):
            return column
    raise ValueError(f"Could not find a text field in TinyStories columns: {columns}")


def main() -> None:
    builder = load_dataset_builder(DATASET_NAME)
    split_info = builder.info.splits
    dataset = load_dataset(DATASET_NAME, streaming=True)
    available_splits = list(dataset.keys())
    if "train" not in dataset or "train" not in split_info:
        raise RuntimeError(
            f"TinyStories dataset does not contain a training split: {available_splits}"
        )

    training_split = dataset["train"]
    text_field = find_text_field(training_split)
    first_example = next(iter(training_split), None)
    if first_example is None:
        raise RuntimeError("TinyStories training split is empty")
    example = first_example[text_field]
    short_example = " ".join(example.split())[:240]

    print(f"Dataset: {DATASET_NAME}")
    print(f"Available splits: {available_splits}")
    print(f"Training examples: {split_info['train'].num_examples}")
    print(f"Text field: {text_field}")
    print(f"Example: {short_example}")


if __name__ == "__main__":
    main()
