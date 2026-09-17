"""Train TinyGPT on random GPTDataset batches and report validation loss."""

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


DEFAULT_CONFIG_PATH = ROOT / "configs" / "tiny.yaml"
DEFAULT_TRAIN_DATA_PATH = ROOT / "data" / "train.bin"
DEFAULT_VAL_DATA_PATH = ROOT / "data" / "val.bin"
DEFAULT_STEPS = 3000
DEFAULT_EVAL_EVERY = 100
DEFAULT_EVAL_STEPS = 20
DEFAULT_SEED = 0


def batch_from_indices(
    dataset: GPTDataset,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize one batch from explicit dataset indices."""

    samples = [dataset[index] for index in indices]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.stack([sample[1] for sample in samples]),
    )


def random_batch(
    dataset: GPTDataset,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a fresh random batch without materializing all dataset indices."""

    indices = torch.randint(
        low=0,
        high=len(dataset),
        size=(batch_size,),
        generator=generator,
    ).tolist()
    return batch_from_indices(dataset, indices)


@torch.no_grad()
def validation_loss(
    model: GPT,
    dataset: GPTDataset,
    batch_size: int,
    max_batches: int,
    device: torch.device,
) -> float:
    """Evaluate on a deterministic prefix of the validation stream."""

    was_training = model.training
    model.eval()
    try:
        losses: list[torch.Tensor] = []
        for batch_index in range(max_batches):
            start = batch_index * batch_size
            end = start + batch_size
            if end > len(dataset):
                break
            token_ids, targets = batch_from_indices(
                dataset,
                list(range(start, end)),
            )
            _, loss = model(token_ids.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("GPT returned no loss for validation targets")
            losses.append(loss.detach())

        if not losses:
            raise ValueError("validation dataset does not contain a complete batch")
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def run_training(
    config_path: Path = DEFAULT_CONFIG_PATH,
    train_data_path: Path = DEFAULT_TRAIN_DATA_PATH,
    val_data_path: Path = DEFAULT_VAL_DATA_PATH,
    steps: int = DEFAULT_STEPS,
    eval_every: int = DEFAULT_EVAL_EVERY,
    eval_steps: int = DEFAULT_EVAL_STEPS,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
) -> list[tuple[int, float, float]]:
    """Run small real training and return step/train-loss/val-loss records."""

    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if eval_every <= 0:
        raise ValueError("eval_every must be greater than 0")
    if eval_steps <= 0:
        raise ValueError("eval_steps must be greater than 0")

    config = load_config(config_path)
    train_dataset = GPTDataset(train_data_path, seq_len=config.data.seq_len)
    val_dataset = GPTDataset(val_data_path, seq_len=config.data.seq_len)
    batch_size = config.training.batch_size
    device = resolve_device(device_name)

    torch.manual_seed(seed)
    train_generator = torch.Generator().manual_seed(seed + 1)
    model = GPT(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )

    print(f"device: {device}", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"train data: {train_data_path}", flush=True)
    print(f"validation data: {val_data_path}", flush=True)
    print(f"batch size: {batch_size}", flush=True)
    print(f"sequence length: {config.data.seq_len}", flush=True)
    print(f"training steps: {steps}", flush=True)
    print(f"validation batches: {eval_steps}", flush=True)

    model.train()
    initial_train_ids, initial_train_targets = random_batch(
        train_dataset,
        batch_size,
        train_generator,
    )
    with torch.no_grad():
        _, initial_train_loss_tensor = model(
            initial_train_ids.to(device),
            initial_train_targets.to(device),
        )
    initial_val_loss = validation_loss(
        model,
        val_dataset,
        batch_size,
        eval_steps,
        device,
    )
    initial_train_loss = initial_train_loss_tensor.item()
    print(
        f"step 0: train_loss={initial_train_loss:.6f} "
        f"val_loss={initial_val_loss:.6f}",
        flush=True,
    )

    records = [(0, initial_train_loss, initial_val_loss)]
    interval_loss_total = 0.0
    interval_step_count = 0

    for step in range(1, steps + 1):
        token_ids, targets = random_batch(
            train_dataset,
            batch_size,
            train_generator,
        )
        token_ids = token_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        _, loss = model(token_ids, targets)
        if loss is None:
            raise RuntimeError("GPT returned no loss for training targets")
        loss.backward()
        clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        interval_loss_total += loss.item()
        interval_step_count += 1
        if step % eval_every == 0 or step == steps:
            mean_train_loss = interval_loss_total / interval_step_count
            val_loss = validation_loss(
                model,
                val_dataset,
                batch_size,
                eval_steps,
                device,
            )
            records.append((step, mean_train_loss, val_loss))
            print(
                f"step {step}: train_loss={mean_train_loss:.6f} "
                f"val_loss={val_loss:.6f}",
                flush=True,
            )
            interval_loss_total = 0.0
            interval_step_count = 0

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be greater than 0")
    if args.eval_every <= 0:
        parser.error("--eval-every must be greater than 0")
    if args.eval_steps <= 0:
        parser.error("--eval-steps must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    run_training(
        steps=args.steps,
        eval_every=args.eval_every,
        eval_steps=args.eval_steps,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
