from __future__ import annotations

import pytest
import torch

from tinygpt.config import ModelConfig
from tinygpt.generation import generate
from tinygpt.model import GPT


GENERATION_CONFIG = ModelConfig(
    vocab_size=32,
    max_seq_len=8,
    n_layer=1,
    n_head=1,
    n_embd=16,
    mlp_ratio=2,
    dropout=0.0,
    bias=True,
)


def test_generate_preserves_prompt_and_restores_training_mode() -> None:
    torch.manual_seed(0)
    model = GPT(GENERATION_CONFIG).train()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = generate(model, prompt, max_new_tokens=4)

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], prompt)
    assert model.training


def test_generate_respects_context_window_without_cache() -> None:
    torch.manual_seed(0)
    model = GPT(GENERATION_CONFIG).eval()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = generate(model, prompt, max_new_tokens=10)

    assert output.shape == (1, 13)
    assert torch.equal(output[:, :3], prompt)
    assert torch.isfinite(output.float()).all()


def test_generate_rejects_invalid_arguments() -> None:
    model = GPT(GENERATION_CONFIG)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="temperature"):
        generate(model, prompt, max_new_tokens=1, temperature=0.0)
