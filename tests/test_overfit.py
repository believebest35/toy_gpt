from __future__ import annotations

import torch
from torch.nn.utils import clip_grad_norm_

from tinygpt.config import ModelConfig
from tinygpt.model import GPT


def test_tiny_gpt_reduces_loss_on_a_fixed_batch() -> None:
    config = ModelConfig(
        vocab_size=16,
        max_seq_len=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        mlp_ratio=2,
        dropout=0.0,
        bias=True,
    )
    torch.manual_seed(0)
    model = GPT(config)
    token_ids = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]],
        dtype=torch.long,
    )
    targets = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 3, 4, 5, 6, 7, 0]],
        dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    model.train()
    with torch.no_grad():
        _, initial_loss = model(token_ids, targets)

    for _ in range(25):
        optimizer.zero_grad()
        _, loss = model(token_ids, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    with torch.no_grad():
        _, final_loss = model(token_ids, targets)

    assert torch.isfinite(initial_loss)
    assert torch.isfinite(final_loss)
    assert final_loss.item() < initial_loss.item()
