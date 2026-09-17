"""Simple autoregressive text generation without KV caching."""

from __future__ import annotations

import math

import torch

from .model import GPT


@torch.no_grad()
def generate(
    model: GPT,
    token_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    do_sample: bool = False,
    temperature: float = 1.0,
    eos_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Append tokens using repeated full forward passes.

    The model is evaluated on the most recent context window at every step.
    This deliberately does not cache keys or values from earlier steps.
    """

    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [B, T]")
    if token_ids.dtype != torch.long:
        raise ValueError("token_ids must have dtype torch.long")
    if token_ids.size(1) == 0:
        raise ValueError("token_ids must contain at least one prompt token")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise ValueError("max_new_tokens must be an integer")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than 0")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id must be inside the model vocabulary")

    generated = token_ids.clone()
    if max_new_tokens == 0:
        return generated

    was_training = model.training
    model.eval()
    finished = torch.zeros(
        generated.size(0),
        dtype=torch.bool,
        device=generated.device,
    )
    if eos_token_id is not None:
        finished = generated[:, -1].eq(eos_token_id)

    try:
        for _ in range(max_new_tokens):
            if eos_token_id is not None and finished.all():
                break

            context = generated[:, -model.config.max_seq_len :]
            logits, _ = model(context)
            next_token_logits = logits[:, -1, :] / temperature

            if do_sample:
                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=generator,
                )
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            if eos_token_id is not None:
                eos_tokens = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    eos_tokens,
                    next_token,
                )
                finished = finished | next_token.squeeze(-1).eq(eos_token_id)

            generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(was_training)

    return generated


__all__ = ["generate"]
