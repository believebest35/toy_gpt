"""Embedding components for the educational TinyGPT model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
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


class GPTMLP(nn.Module):
    """Apply the GPT-style GELU feed-forward transformation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        hidden_dim = config.mlp_ratio * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """A single Pre-LayerNorm GPT Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = GPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


__all__ = ["GPTEmbedding", "GPTMLP", "TransformerBlock"]
