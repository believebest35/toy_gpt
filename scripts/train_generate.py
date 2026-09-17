"""Train TinyGPT and generate text at selected training steps."""

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
from tinygpt.generation import generate
from tinygpt.model import GPT
from tinygpt.tokenizer import EOS_TOKEN, encode, load_tokenizer


DEFAULT_CONFIG_PATH = ROOT / "configs" / "tiny.yaml"
DEFAULT_TRAIN_DATA_PATH = ROOT / "data" / "train.bin"
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"
DEFAULT_PROMPT = "Once upon a time"
DEFAULT_STEPS = 1000
DEFAULT_GENERATION_INTERVAL = 500
DEFAULT_MAX_NEW_TOKENS = 40
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_SEED = 0


def batch_from_indices(
    dataset: GPTDataset,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
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
    indices = torch.randint(
        low=0,
        high=len(dataset),
        size=(batch_size,),
        generator=generator,
    ).tolist()
    return batch_from_indices(dataset, indices)


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
    for strategy_index, (name, do_sample, strategy_temperature, strategy_top_k) in enumerate(
        strategies
    ):
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
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    prompt: str = DEFAULT_PROMPT,
    steps: int = DEFAULT_STEPS,
    generation_interval: int = DEFAULT_GENERATION_INTERVAL,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    seed: int = DEFAULT_SEED,
    device_name: str = "auto",
) -> None:
    if steps <= 0:
        raise ValueError("steps must be greater than 0")
    generation_steps = generation_schedule(steps, generation_interval)
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be greater than 0")

    config = load_config(config_path)
    if top_k <= 0 or top_k > config.model.vocab_size:
        raise ValueError("top_k must be inside the model vocabulary")
    tokenizer = load_tokenizer(tokenizer_path)
    dataset = GPTDataset(train_data_path, seq_len=config.data.seq_len)
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
    print(f"prompt: {prompt!r}", flush=True)
    print(f"prompt token IDs: {prompt_ids[0].tolist()}", flush=True)
    print(f"training steps: {steps}", flush=True)
    print(f"generation steps: {generation_steps}", flush=True)
    print_checkpoint_generations(
        model,
        tokenizer,
        prompt_ids,
        eos_token_id,
        step=0,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )

    model.train()
    for step in range(1, steps + 1):
        token_ids, targets = random_batch(
            dataset,
            config.training.batch_size,
            train_generator,
        )
        optimizer.zero_grad()
        _, loss = model(token_ids.to(device), targets.to(device))
        if loss is None:
            raise RuntimeError("GPT returned no loss for training targets")
        loss.backward()
        clip_grad_norm_(model.parameters(), config.training.grad_clip)
        optimizer.step()

        if step in generation_steps:
            print(f"training loss at step {step}: {loss.item():.6f}", flush=True)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
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
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be greater than 0")
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
        prompt=args.prompt,
        steps=args.steps,
        generation_interval=args.generation_interval,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
