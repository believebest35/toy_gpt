from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import tinygpt.dataset as dataset_module
from tinygpt.dataset import GPTDataset, write_stories_to_binary
from tinygpt.tokenizer import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    encode,
    train_tokenizer,
)


def write_tokens(path: Path, tokens: list[int]) -> None:
    np.asarray(tokens, dtype=np.uint16).tofile(path)


def test_binary_uint16_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    expected = [1, 2, 3, 4, 5]
    write_tokens(path, expected)

    tokens = np.memmap(path, dtype=np.uint16, mode="r")

    assert tokens.dtype == np.uint16
    assert tokens.tolist() == expected


def test_gpt_dataset_length(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, list(range(10)))

    dataset = GPTDataset(path, seq_len=4)

    assert len(dataset) == 6
    assert isinstance(dataset.tokens, np.memmap)


def test_gpt_dataset_sample_shape_and_dtype(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, list(range(10)))
    dataset = GPTDataset(path, seq_len=4)

    x, y = dataset[0]

    assert x.shape == (4,)
    assert y.shape == (4,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_gpt_dataset_shift_relationship(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, list(range(10)))
    dataset = GPTDataset(path, seq_len=4)

    for index in range(len(dataset)):
        x, y = dataset[index]
        assert torch.equal(x[1:], y[:-1])


def test_gpt_dataset_exact_first_and_second_samples(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, [0, 1, 2, 3, 4, 5])
    dataset = GPTDataset(path, seq_len=3)

    first_x, first_y = dataset[0]
    second_x, second_y = dataset[1]

    assert first_x.tolist() == [0, 1, 2]
    assert first_y.tolist() == [1, 2, 3]
    assert second_x.tolist() == [1, 2, 3]
    assert second_y.tolist() == [2, 3, 4]


def test_gpt_dataset_rejects_short_token_stream(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, [0, 1, 2, 3])

    with pytest.raises(ValueError, match="more than seq_len"):
        GPTDataset(path, seq_len=4)


@pytest.mark.parametrize("seq_len", [0, -1])
def test_gpt_dataset_rejects_invalid_seq_len(tmp_path: Path, seq_len: int) -> None:
    path = tmp_path / "tokens.bin"
    write_tokens(path, [0, 1, 2, 3])

    with pytest.raises(ValueError, match="seq_len"):
        GPTDataset(path, seq_len=seq_len)


def test_gpt_dataset_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="token stream not found"):
        GPTDataset(tmp_path / "missing.bin", seq_len=2)


def test_write_stories_appends_exactly_one_eos(tmp_path: Path) -> None:
    stories = ["hello world", "little dog"]
    tokenizer = train_tokenizer(iter(stories), vocab_size=128)
    output_path = tmp_path / "stories.bin"

    stats = write_stories_to_binary(stories, tokenizer, output_path)
    tokens = np.memmap(output_path, dtype=np.uint16, mode="r").tolist()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    pad_id = tokenizer.token_to_id(PAD_TOKEN)
    expected = []
    for story in stories:
        expected.extend(encode(tokenizer, story))
        expected.append(eos_id)

    assert stats.stories_processed == 2
    assert stats.tokens_written == len(expected)
    assert stats.eos_token_id == eos_id
    assert tokens == expected
    assert tokens.count(eos_id) == 2
    assert bos_id not in tokens
    assert pad_id not in tokens


def test_write_stories_rejects_token_ids_outside_uint16(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stories = ["hello"]
    tokenizer = train_tokenizer(iter(stories), vocab_size=64)
    monkeypatch.setattr(dataset_module, "encode", lambda _tokenizer, _text: [65536])

    with pytest.raises(ValueError, match="uint16 range"):
        write_stories_to_binary(stories, tokenizer, tmp_path / "stories.bin")
