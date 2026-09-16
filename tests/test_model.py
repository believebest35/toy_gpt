from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinygpt import load_config
from tinygpt.model import GPTEmbedding


ROOT = Path(__file__).parents[1]
DEBUG_CONFIG = load_config(ROOT / "configs" / "debug.yaml").model


def test_embedding_output_shape() -> None:
    embedding = GPTEmbedding(DEBUG_CONFIG)
    token_ids = torch.randint(0, DEBUG_CONFIG.vocab_size, (2, 16))

    output = embedding(token_ids)

    assert output.shape == (2, 16, 64)


def test_embedding_output_is_floating_point() -> None:
    embedding = GPTEmbedding(DEBUG_CONFIG)
    token_ids = torch.randint(0, DEBUG_CONFIG.vocab_size, (2, 16))

    assert embedding(token_ids).is_floating_point()


def test_embedding_accepts_maximum_sequence_length() -> None:
    embedding = GPTEmbedding(DEBUG_CONFIG)
    token_ids = torch.zeros((1, DEBUG_CONFIG.max_seq_len), dtype=torch.long)

    output = embedding(token_ids)

    assert output.shape == (1, DEBUG_CONFIG.max_seq_len, DEBUG_CONFIG.n_embd)


def test_embedding_rejects_sequence_longer_than_maximum() -> None:
    embedding = GPTEmbedding(DEBUG_CONFIG)
    token_ids = torch.zeros((1, DEBUG_CONFIG.max_seq_len + 1), dtype=torch.long)

    with pytest.raises(ValueError, match="max_seq_len"):
        embedding(token_ids)


def test_embedding_rejects_non_rank_two_input() -> None:
    embedding = GPTEmbedding(DEBUG_CONFIG)
    token_ids = torch.zeros((2, 4, 1), dtype=torch.long)

    with pytest.raises(ValueError, match=r"shape \[B, T\]"):
        embedding(token_ids)


def test_same_token_at_different_positions_has_position_information() -> None:
    torch.manual_seed(0)
    embedding = GPTEmbedding(DEBUG_CONFIG).eval()
    token_ids = torch.tensor([[5, 5]], dtype=torch.long)

    output = embedding(token_ids)

    assert not torch.allclose(output[:, 0, :], output[:, 1, :])
