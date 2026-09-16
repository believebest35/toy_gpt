"""Explicit causal multi-head self-attention for TinyGPT."""

from __future__ import annotations

import math

import torch
from torch import nn
from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Compute scaled dot-product self-attention with a causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.max_seq_len = config.max_seq_len
        self.qkv_proj = nn.Linear(
            config.n_embd,
            3 * config.n_embd,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(
            config.n_embd,
            config.n_embd,
            bias=config.bias,
        )

        causal_mask = torch.tril(
            torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, C]")

        batch_size, sequence_length, embedding_size = x.shape
        if embedding_size != self.n_embd:
            raise ValueError(
                f"input embedding dimension {embedding_size} does not match "
                f"n_embd {self.n_embd}"
            )
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(batch_size, sequence_length, self.n_head, self.head_dim)
        k = k.reshape(batch_size, sequence_length, self.n_head, self.head_dim)
        v = v.reshape(batch_size, sequence_length, self.n_head, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)
        mask = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~mask, float("-inf"))

        attention_probs = torch.softmax(scores, dim=-1)
        attention_output = attention_probs @ v
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.reshape(
            batch_size,
            sequence_length,
            self.n_embd,
        )
        return self.out_proj(attention_output)


__all__ = ["CausalSelfAttention"]
