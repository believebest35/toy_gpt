"""Train TinyGPT with validation, checkpoints, resume, and generation."""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tinygpt import load_config
from tinygpt.config import Config
from tinygpt.dataset import GPTDataset
from tinygpt.generation import generate
from tinygpt.model import GPT
from tinygpt.tokenizer import EOS_TOKEN, encode, load_tokenizer


DEFAULT_CONFIG_PATH = ROOT / "configs" / "tiny.yaml"
DEFAULT_TRAIN_DATA_PATH = ROOT / "data" / "train.bin"
DEFAULT_VAL_DATA_PATH = ROOT / "data" / "val.bin"
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_STEPS = 1000
DEFAULT_EVAL_INTERVAL = 500
DEFAULT_EVAL_BATCHES = 20
DEFAULT_SAVE_INTERVAL = 500
DEFAULT_GENERATION_INTERVAL = 500
DEFAULT_MAX_NEW_TOKENS = 40
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_SEED = 1337
DEFAULT_VALIDATION_SEED = 20260917


def set_seed(seed: int) -> None:
    """Seed Python, PyTorch CPU, and CUDA generators when available."""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def sample_validation_batches(
    dataset_size: int,
    batch_size: int,
    max_batches: int,
    seed: int,
) -> list[list[int]]:
    """Sample fixed, non-contiguous validation batches from the full dataset."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be greater than 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if max_batches <= 0:
        raise ValueError("max_batches must be greater than 0")

    batch_count = min(max_batches, dataset_size // batch_size)
    if batch_count == 0:
        raise ValueError("validation dataset does not contain a complete batch")

    generator = torch.Generator().manual_seed(seed)
    total_indices = batch_count * batch_size
    indices = torch.randperm(dataset_size, generator=generator)[:total_indices]
    return [
        indices[start : start + batch_size].tolist()
        for start in range(0, total_indices, batch_size)
    ]


@torch.no_grad()
def evaluate(
    model: GPT,
    dataset: GPTDataset,
    validation_batches: list[list[int]],
    device: torch.device,
) -> float:
    """Evaluate average loss and restore the model's previous train/eval mode."""

    was_training = model.training
    model.eval()
    try:
        losses: list[torch.Tensor] = []
        for indices in validation_batches:
            token_ids, targets = batch_from_indices(dataset, indices)
            _, loss = model(token_ids.to(device), targets.to(device))
            if loss is None:
                raise RuntimeError("GPT returned no loss for validation targets")
            losses.append(loss.detach())

        if not losses:
            raise ValueError("validation batches must not be empty")
        return torch.stack(losses).mean().item()
    finally:
        model.train(was_training)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(device_name)


def generation_schedule(steps: int, interval: int) -> tuple[int, ...]:
    """Return step zero and every requested generation interval."""

    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if interval <= 0:
        raise ValueError("generation interval must be greater than 0")
    return tuple(range(0, steps + 1, interval))


def capture_rng_state() -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state().cpu(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [rng_state.cpu() for rng_state in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: object) -> None:
    if not isinstance(state, dict):
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    torch_state = state.get("torch")
    if torch.is_tensor(torch_state):
        torch.set_rng_state(torch_state.cpu())

    cuda_states = state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, (list, tuple)):
        torch.cuda.set_rng_state_all(
            [rng_state.cpu() for rng_state in cuda_states if torch.is_tensor(rng_state)]
        )


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move tensor-valued AdamW state after loading on another device."""

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: Config,
    config_path: Path,
    seed: int,
    validation_seed: int,
    train_generator: torch.Generator,
    runtime_config: dict[str, object],
) -> Path:
    """Save a numbered checkpoint and a portable latest checkpoint."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config.model),
        "training_config": asdict(config.training),
        "data_config": asdict(config.data),
        "config_path": str(config_path),
        "seed": seed,
        "validation_seed": validation_seed,
        "train_generator_state": train_generator.get_state(),
        "rng_state": capture_rng_state(),
        "runtime_config": runtime_config,
    }
    checkpoint_path = checkpoint_dir / f"step_{step:06d}.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, checkpoint_dir / "latest.pt")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    device: torch.device,
    config: Config,
) -> int:
    """Restore model, optimizer, step, and random states from a checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a mapping")

    if checkpoint.get("model_config") != asdict(config.model):
        raise ValueError("checkpoint model configuration does not match current config")
    if checkpoint.get("data_config") != asdict(config.data):
        raise ValueError("checkpoint data configuration does not match current config")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)

    train_generator_state = checkpoint.get("train_generator_state")
    if torch.is_tensor(train_generator_state):
        train_generator.set_state(train_generator_state.cpu())
    restore_rng_state(checkpoint.get("rng_state"))

    step = checkpoint.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("checkpoint step must be a non-negative integer")
    return step


def print_generation(
    model: GPT,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    eos_token_id: int | None,
    step: int,
    strategy_name: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> None:
    generator = None
    if do_sample:
        generator = torch.Generator(device=prompt_ids.device.type).manual_seed(seed)
    generated_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=eos_token_id,
        generator=generator,
    )
    generated_text = tokenizer.decode(
        generated_ids[0].tolist(),
        skip_special_tokens=True,
    )
    print(
        f"\n===== generation at step {step} ({strategy_name}) =====",
        flush=True,
    )
    print(generated_text, flush=True)


def print_checkpoint_generations(
    model: GPT,
    tokenizer: object,
    prompt_ids: torch.Tensor,
    eos_token_id: int | None,
    step: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> None:
    """Compare multiple decoding strategies at one unchanged checkpoint."""

    strategies = (
        ("greedy", False, 1.0, None),
        (f"temperature={temperature}", True, temperature, None),
        (f"top_k={top_k}", True, 1.0, top_k),
    )
    for strategy_index, (
        name,
        do_sample,
        strategy_temperature,
        strategy_top_k,
    ) in enumerate(strategies):
        print_generation(
            model,
            tokenizer,
            prompt_ids,
            eos_token_id,
            step,
            name,
            max_new_tokens,
            do_sample=do_sample,
            temperature=strategy_temperature,
            top_k=strategy_top_k,
            seed=seed + step + strategy_index,
        )


def run_demo(
    config_path: Path = DEFAULT_CONFIG_PATH,
    train_data_path: Path = DEFAULT_TRAIN_DATA_PATH,
    val_data_path: Path = DEFAULT_VAL_DATA_PATH,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    resume: Path | None = None,
    prompt: str = DEFAULT_PROMPT,
    steps: int = DEFAULT_STEPS,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    eval_batches: int = DEFAULT_EVAL_BATCHES,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    generation_interval: int = DEFAULT_GENERATION_INTERVAL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    seed: int = DEFAULT_SEED,
    validation_seed: int = DEFAULT_VALIDATION_SEED,
    device_name: str = "auto",
) -> None:
    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be greater than 0")
    if eval_batches <= 0:
        raise ValueError("eval_batches must be greater than 0")
    if save_interval <= 0:
        raise ValueError("save_interval must be greater than 0")
    generation_steps = generation_schedule(steps, generation_interval)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")

    config = load_config(config_path)
    if top_k <= 0 or top_k > config.model.vocab_size:
        raise ValueError("top_k must be inside the model vocabulary")

    tokenizer = load_tokenizer(tokenizer_path)
    train_dataset = GPTDataset(train_data_path, seq_len=config.data.seq_len)
    val_dataset = GPTDataset(val_data_path, seq_len=config.data.seq_len)
    validation_batches = sample_validation_batches(
        len(val_dataset),
        config.training.batch_size,
        eval_batches,
        validation_seed,
    )
    device = resolve_device(device_name)
    prompt_ids = torch.tensor(
        [encode(tokenizer, prompt)],
        dtype=torch.long,
        device=device,
    )
    if prompt_ids.size(1) == 0:
        raise ValueError("prompt must encode to at least one token")
    if prompt_ids.size(1) > config.model.max_seq_len:
        raise ValueError("prompt is longer than the model context window")

    eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
    set_seed(seed)
    train_generator = torch.Generator().manual_seed(seed + 1)
    model = GPT(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.beta1, config.training.beta2),
        weight_decay=config.training.weight_decay,
    )

    start_step = 0
    if resume is not None:
        start_step = load_checkpoint(
            resume,
            model,
            optimizer,
            train_generator,
            device,
            config,
        )
        if start_step > steps:
            raise ValueError(
                f"checkpoint is at step {start_step}, beyond requested final step {steps}"
            )

    runtime_config = {
        "eval_interval": eval_interval,
        "eval_batches": eval_batches,
        "save_interval": save_interval,
        "generation_interval": generation_interval,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "device": str(device),
    }

    print(f"device: {device}", flush=True)
    print(f"config: {config_path}", flush=True)
    print(f"train data: {train_data_path}", flush=True)
    print(f"validation data: {val_data_path}", flush=True)
    print(f"batch size: {config.training.batch_size}", flush=True)
    print(f"sequence length: {config.data.seq_len}", flush=True)
    print(f"final training step: {steps}", flush=True)
    print(f"starting global step: {start_step}", flush=True)
    print(f"evaluation: every {eval_interval} steps, {eval_batches} batches", flush=True)
    print(f"validation sampling seed: {validation_seed}", flush=True)
    print(f"checkpoint directory: {checkpoint_dir}", flush=True)
    if resume is not None:
        print(f"resumed from: {resume}", flush=True)

    initial_val_loss = evaluate(model, val_dataset, validation_batches, device)
    print(
        f"step {start_step} | train loss: n/a | val loss: {initial_val_loss:.6f} "
        f"| lr: {optimizer.param_groups[0]['lr']:.6g}",
        flush=True,
    )

    if start_step in generation_steps:
        print_checkpoint_generations(
            model,
            tokenizer,
            prompt_ids,
            eos_token_id,
            step=start_step,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        )

    model.train()
    interval_loss_total = 0.0
    interval_step_count = 0
    interval_token_count = 0
    interval_started_at = time.perf_counter()

    for step in range(start_step + 1, steps + 1):
        token_ids, targets = random_batch(
            train_dataset,
            config.training.batch_size,
            train_generator,
        )
        token_ids = token_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        _, loss = model(token_ids, targets)
        if loss is None:
            raise RuntimeError("GPT returned no loss for training targets")
        loss.backward()
        optimizer.step()

        interval_loss_total += loss.item()
        interval_step_count += 1
        interval_token_count += token_ids.numel()

        should_evaluate = step % eval_interval == 0 or step == steps
        if should_evaluate:
            elapsed = max(time.perf_counter() - interval_started_at, 1e-9)
            mean_train_loss = interval_loss_total / interval_step_count
            val_loss = evaluate(model, val_dataset, validation_batches, device)
            tokens_per_second = interval_token_count / elapsed
            print(
                f"step {step} | train loss: {mean_train_loss:.6f} "
                f"| val loss: {val_loss:.6f} "
                f"| lr: {optimizer.param_groups[0]['lr']:.6g} "
                f"| tokens/sec: {tokens_per_second:.1f}",
                flush=True,
            )
            interval_loss_total = 0.0
            interval_step_count = 0
            interval_token_count = 0
            interval_started_at = time.perf_counter()

        if step in generation_steps:
            print_checkpoint_generations(
                model,
                tokenizer,
                prompt_ids,
                eos_token_id,
                step=step,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            )

        if step % save_interval == 0 or step == steps:
            saved_path = save_checkpoint(
                checkpoint_dir,
                step,
                model,
                optimizer,
                config,
                config_path,
                seed,
                validation_seed,
                train_generator,
                runtime_config,
            )
            print(f"checkpoint saved: {saved_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA_PATH)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA_PATH)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--eval-batches", type=int, default=DEFAULT_EVAL_BATCHES)
    parser.add_argument("--save-interval", type=int, default=DEFAULT_SAVE_INTERVAL)
    parser.add_argument(
        "--generation-interval",
        type=int,
        default=DEFAULT_GENERATION_INTERVAL,
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--validation-seed",
        type=int,
        default=DEFAULT_VALIDATION_SEED,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be greater than 0")
    if args.eval_interval <= 0:
        parser.error("--eval-interval must be greater than 0")
    if args.eval_batches <= 0:
        parser.error("--eval-batches must be greater than 0")
    if args.save_interval <= 0:
        parser.error("--save-interval must be greater than 0")
    if args.generation_interval <= 0:
        parser.error("--generation-interval must be greater than 0")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative")
    if args.temperature <= 0.0:
        parser.error("--temperature must be greater than 0")
    if args.top_k <= 0:
        parser.error("--top-k must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    run_demo(
        config_path=args.config,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        tokenizer_path=args.tokenizer,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        prompt=args.prompt,
        steps=args.steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        save_interval=args.save_interval,
        generation_interval=args.generation_interval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        validation_seed=args.validation_seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
