from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from tinygpt import load_config
from tinygpt.model import GPTEmbedding, GPTMLP, TransformerBlock


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


def test_mlp_output_shape() -> None:
    mlp = GPTMLP(DEBUG_CONFIG)
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = mlp(hidden_states)

    assert output.shape == hidden_states.shape


def test_mlp_dimensions_and_activation() -> None:
    mlp = GPTMLP(DEBUG_CONFIG)

    assert mlp.fc1.in_features == 64
    assert mlp.fc1.out_features == 256
    assert mlp.fc2.in_features == 256
    assert mlp.fc2.out_features == 64
    assert isinstance(mlp.gelu, nn.GELU)


def test_mlp_output_is_finite() -> None:
    mlp = GPTMLP(DEBUG_CONFIG)
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = mlp(hidden_states)

    assert torch.isfinite(output).all()


def test_mlp_gradients_reach_input_and_parameters() -> None:
    torch.manual_seed(0)
    mlp = GPTMLP(DEBUG_CONFIG)
    hidden_states = torch.randn(
        2,
        16,
        DEBUG_CONFIG.n_embd,
        requires_grad=True,
    )

    mlp(hidden_states).sum().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert all(parameter.grad is not None for parameter in mlp.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in mlp.parameters())


def test_transformer_block_contains_expected_components() -> None:
    block = TransformerBlock(DEBUG_CONFIG)

    assert isinstance(block.ln1, nn.LayerNorm)
    assert isinstance(block.attention, nn.Module)
    assert isinstance(block.ln2, nn.LayerNorm)
    assert isinstance(block.mlp, GPTMLP)


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = block(hidden_states)

    assert output.shape == hidden_states.shape


def test_transformer_block_output_is_finite() -> None:
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = block(hidden_states)

    assert torch.isfinite(output).all()


def test_transformer_block_residuals_preserve_input_with_zero_submodules() -> None:
    class ZeroModule(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    block = TransformerBlock(DEBUG_CONFIG)
    block.attention = ZeroModule()
    block.mlp = ZeroModule()
    hidden_states = torch.randn(2, 16, DEBUG_CONFIG.n_embd)

    output = block(hidden_states)

    torch.testing.assert_close(output, hidden_states)


def test_transformer_block_preserves_causal_behavior() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(DEBUG_CONFIG).eval()
    first = torch.randn(2, 8, DEBUG_CONFIG.n_embd)
    second = first.clone()
    second[:, 4:, :] = torch.randn(2, 4, DEBUG_CONFIG.n_embd)

    with torch.no_grad():
        first_output = block(first)
        second_output = block(second)

    torch.testing.assert_close(
        first_output[:, :4, :],
        second_output[:, :4, :],
        rtol=1e-5,
        atol=1e-6,
    )


def test_transformer_block_gradients_reach_all_component_groups() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(
        2,
        8,
        DEBUG_CONFIG.n_embd,
        requires_grad=True,
    )

    block(hidden_states).square().mean().backward()

    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    for module in (block.attention, block.mlp, block.ln1, block.ln2):
        parameters = list(module.parameters())
        assert parameters
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_transformer_block_handles_single_token() -> None:
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(2, 1, DEBUG_CONFIG.n_embd)

    output = block(hidden_states)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()


def test_transformer_block_accepts_maximum_sequence_length() -> None:
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(
        1,
        DEBUG_CONFIG.max_seq_len,
        DEBUG_CONFIG.n_embd,
    )

    output = block(hidden_states)

    assert output.shape == hidden_states.shape


def test_transformer_block_rejects_sequence_longer_than_maximum() -> None:
    block = TransformerBlock(DEBUG_CONFIG)
    hidden_states = torch.randn(
        1,
        DEBUG_CONFIG.max_seq_len + 1,
        DEBUG_CONFIG.n_embd,
    )

    with pytest.raises(ValueError, match="max_seq_len"):
        block(hidden_states)


def test_transformer_block_eval_is_deterministic() -> None:
    torch.manual_seed(0)
    block = TransformerBlock(DEBUG_CONFIG).eval()
    hidden_states = torch.randn(2, 8, DEBUG_CONFIG.n_embd)

    with torch.no_grad():
        first_output = block(hidden_states)
        second_output = block(hidden_states)

    torch.testing.assert_close(first_output, second_output)
