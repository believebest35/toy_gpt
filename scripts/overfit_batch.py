"""Overfit one fixed batch as a TinyGPT training sanity check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt import load_config
from tinygpt.dataset import GPTDataset
from tinygpt.model import GPT


DEFAULT_CONFIG_PATH = ROOT / "configs" / "debug.yaml"
DEFAULT_DATA_PATH = ROOT / "data" / "train.bin"
DEFAULT_STEPS = 300
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PRINT_EVERY = 20


def build_fixed_batch(
    dataset: GPTDataset,
    batch_size: int,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Build one deterministic batch and fit IDs to the model vocabulary."""

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be greater than 0")
    if len(dataset) < batch_size:
        raise ValueError("dataset does not contain enough samples for the batch")

    samples = [dataset[index] for index in range(batch_size)]
    token_ids = torch.stack([sample[0] for sample in samples])
    targets = torch.stack([sample[1] for sample in samples])

    source_vocab_is_larger = max(
        int(token_ids.max().item()),
        int(targets.max().item()),
    ) >= vocab_size
    if source_vocab_is_larger:
        token_ids = token_ids.remainder(vocab_size)
        targets = targets.remainder(vocab_size)

    return token_ids, targets, source_vocab_is_larger


def run_overfit(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_path: Path = DEFAULT_DATA_PATH,
    steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    print_every: int = DEFAULT_PRINT_EVERY,
) -> tuple[float, float]:
    """Train a GPT on one fixed batch and return initial and final loss."""

    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")
    if print_every <= 0:
        raise ValueError("print_every must be greater than 0")

    config = load_config(config_path)
    dataset = GPTDataset(data_path, seq_len=config.data.seq_len)
    token_ids, targets, source_vocab_is_larger = build_fixed_batch(
        dataset,
        batch_size=config.training.batch_size,
        vocab_size=config.model.vocab_size,
    )

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT(config.model).to(device)
    token_ids = token_ids.to(device)
    targets = targets.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    with torch.no_grad():
        _, initial_loss_tensor = model(token_ids, targets)
    initial_loss = float(initial_loss_tensor.item())

    print(f"device: {device}")
    print(f"batch shape: {tuple(token_ids.shape)}")
    if source_vocab_is_larger:
        print(
            "note: mapped source token IDs modulo the debug vocabulary "
            f"size ({config.model.vocab_size})"
        )
    print(f"initial loss: {initial_loss:.6f}")

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        _, loss = model(token_ids, targets)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % print_every == 0 or step == steps:
            print(f"step {step}/{steps} loss: {loss.item():.6f}")

    with torch.no_grad():
        _, final_loss_tensor = model(token_ids, targets)
    final_loss = float(final_loss_tensor.item())
    print(f"final loss: {final_loss:.6f}")
    return initial_loss, final_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"optimization steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=f"AdamW learning rate (default: {DEFAULT_LEARNING_RATE})",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=DEFAULT_PRINT_EVERY,
        help=f"loss reporting interval (default: {DEFAULT_PRINT_EVERY})",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be greater than 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if args.print_every <= 0:
        parser.error("--print-every must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    run_overfit(
        steps=args.steps,
        learning_rate=args.learning_rate,
        print_every=args.print_every,
    )


if __name__ == "__main__":
    main()
