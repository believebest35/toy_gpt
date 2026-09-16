from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tinygpt import load_config
from tinygpt.attention import CausalSelfAttention


ROOT = Path(__file__).parents[1]
DEBUG_CONFIG = load_config(ROOT / "configs" / "debug.yaml").model


def make_attention() -> CausalSelfAttention:
    return CausalSelfAttention(DEBUG_CONFIG).eval()


def test_attention_output_shape() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = attention(hidden_states)

    assert output.shape == (2, 16, 64)


def test_attention_derives_head_dimension() -> None:
    attention = make_attention()

    assert attention.n_head == 2
    assert attention.head_dim == 32


def test_attention_registers_lower_triangular_mask() -> None:
    attention = make_attention()
    expected = torch.tril(
        torch.ones(DEBUG_CONFIG.max_seq_len, DEBUG_CONFIG.max_seq_len, dtype=torch.bool)
    )

    assert torch.equal(attention.causal_mask, expected)
    assert not attention.causal_mask.requires_grad


def test_future_tokens_do_not_change_earlier_outputs() -> None:
    torch.manual_seed(0)
    attention = make_attention()
    first = torch.randn(2, 8, DEBUG_CONFIG.n_embd)
    second = first.clone()
    second[:, 4:, :] = torch.randn(2, 4, DEBUG_CONFIG.n_embd)

    with torch.no_grad():
        first_output = attention(first)
        second_output = attention(second)

    torch.testing.assert_close(
        first_output[:, :4, :],
        second_output[:, :4, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_single_token_attention_is_finite() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 1, DEBUG_CONFIG.n_embd)

    output = attention(hidden_states)

    assert output.shape == (2, 1, DEBUG_CONFIG.n_embd)
    assert torch.isfinite(output).all()


def test_attention_accepts_maximum_sequence_length() -> None:
    attention = make_attention()
    hidden_states = torch.randn(1, DEBUG_CONFIG.max_seq_len, DEBUG_CONFIG.n_embd)

    output = attention(hidden_states)

    assert output.shape == hidden_states.shape


def test_attention_rejects_sequence_longer_than_maximum() -> None:
    attention = make_attention()
    hidden_states = torch.randn(
        1,
        DEBUG_CONFIG.max_seq_len + 1,
        DEBUG_CONFIG.n_embd,
    )

    with pytest.raises(ValueError, match="max_seq_len"):
        attention(hidden_states)


def test_attention_rejects_non_rank_three_input() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, DEBUG_CONFIG.n_embd)

    with pytest.raises(ValueError, match=r"shape \[B, T, C\]"):
        attention(hidden_states)


def test_attention_output_is_finite_for_random_input() -> None:
    attention = make_attention()
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = attention(hidden_states)

    assert torch.isfinite(output).all()
