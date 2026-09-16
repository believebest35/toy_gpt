"""Token-stream writing and memory-mapped language-model datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from .tokenizer import EOS_TOKEN, encode


UINT16_MAX = np.iinfo(np.uint16).max
WRITE_BUFFER_SIZE = 8192


@dataclass(frozen=True)
class TokenStreamStats:
    """Small summary returned after writing one token stream."""

    stories_processed: int
    tokens_written: int
    eos_token_id: int


def _validate_token_id(token_id: object) -> int:
    if isinstance(token_id, bool) or not isinstance(token_id, Integral):
        raise ValueError(f"token ID must be an integer, got {token_id!r}")

    value = int(token_id)
    if not 0 <= value <= UINT16_MAX:
        raise ValueError(
            f"token ID {value} is outside the uint16 range "
            f"[0, {UINT16_MAX}]"
        )
    return value


def write_stories_to_binary(
    stories: Iterable[str],
    tokenizer: Tokenizer,
    output_path: str | Path,
) -> TokenStreamStats:
    """Encode stories, append EOS, and write a raw uint16 token stream."""

    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_token_id is None:
        raise ValueError(f"tokenizer does not define the {EOS_TOKEN!r} token")
    eos_token_id = _validate_token_id(eos_token_id)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stories_processed = 0
    tokens_written = 0
    buffer: list[int] = []

    with path.open("wb") as output_file:
        for story in stories:
            if not isinstance(story, str):
                raise ValueError(f"story text must be a string, got {story!r}")

            story_ids = [_validate_token_id(token_id) for token_id in encode(tokenizer, story)]
            story_ids.append(eos_token_id)
            buffer.extend(story_ids)
            stories_processed += 1
            tokens_written += len(story_ids)

            if len(buffer) >= WRITE_BUFFER_SIZE:
                np.asarray(buffer, dtype=np.uint16).tofile(output_file)
                buffer.clear()

        if buffer:
            np.asarray(buffer, dtype=np.uint16).tofile(output_file)

    return TokenStreamStats(
        stories_processed=stories_processed,
        tokens_written=tokens_written,
        eos_token_id=eos_token_id,
    )


class GPTDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Expose overlapping next-token windows from a raw uint16 token stream."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        if isinstance(seq_len, bool) or not isinstance(seq_len, Integral):
            raise ValueError("seq_len must be an integer")
        seq_len = int(seq_len)
        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0")

        token_path = Path(path)
        if not token_path.is_file():
            raise FileNotFoundError(f"token stream not found: {token_path}")

        file_size = token_path.stat().st_size
        item_size = np.dtype(np.uint16).itemsize
        if file_size % item_size != 0:
            raise ValueError(
                f"token stream size must be a multiple of {item_size} bytes: "
                f"{token_path} has {file_size} bytes"
            )

        token_count = file_size // item_size
        if token_count <= seq_len:
            raise ValueError(
                f"token stream must contain more than seq_len tokens: "
                f"{token_count} <= {seq_len}"
            )

        self.path = token_path
        self.seq_len = seq_len
        self.tokens = np.memmap(
            token_path,
            dtype=np.uint16,
            mode="r",
            shape=(token_count,),
        )
        self._length = token_count - seq_len

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if not 0 <= index < self._length:
            raise IndexError(f"dataset index out of range: {index}")

        window = np.asarray(
            self.tokens[index : index + self.seq_len + 1],
            dtype=np.int64,
        ).copy()
        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:])
        return x, y


__all__ = ["GPTDataset", "TokenStreamStats", "write_stories_to_binary"]
