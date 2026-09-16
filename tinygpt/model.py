"""Embedding components for the educational TinyGPT model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ModelConfig


class GPTEmbedding(nn.Module):
    """Add learned token and absolute position embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.max_seq_len = config.max_seq_len
        self.n_embd = config.n_embd
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [B, T]")
        if token_ids.dtype != torch.long:
            raise ValueError("token_ids must have dtype torch.long")

        sequence_length = token_ids.size(1)
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        token_embeddings = self.token_embedding(token_ids)
        positions = torch.arange(sequence_length, device=token_ids.device)
        position_embeddings = self.position_embedding(positions)
        return self.dropout(token_embeddings + position_embeddings)


__all__ = ["GPTEmbedding"]
