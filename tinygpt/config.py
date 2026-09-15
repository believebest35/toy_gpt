"""Typed configuration loading and validation for TinyGPT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Configuration for the future GPT-style model."""

    vocab_size: int
    max_seq_len: int
    n_layer: int
    n_head: int
    n_embd: int
    mlp_ratio: int
    dropout: float
    bias: bool


@dataclass
class TrainingConfig:
    """Configuration for future model training."""

    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    min_learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    grad_clip: float
    warmup_steps: int


@dataclass
class DataConfig:
    """Configuration for future language-model data preparation."""

    seq_len: int


@dataclass
class Config:
    """Complete TinyGPT configuration."""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig

    def validate(self) -> None:
        """Raise ``ValueError`` when a configuration is obviously invalid."""

        model = self.model
        for field_name in (
            "vocab_size",
            "max_seq_len",
            "n_layer",
            "n_head",
            "n_embd",
            "mlp_ratio",
        ):
            _require_int(getattr(model, field_name), f"model.{field_name}")

        _require_real(model.dropout, "model.dropout")
        if not isinstance(model.bias, bool):
            raise ValueError("model.bias must be a boolean")

        if model.vocab_size <= 0:
            raise ValueError("model.vocab_size must be greater than 0")
        if model.max_seq_len <= 0:
            raise ValueError("model.max_seq_len must be greater than 0")
        if model.n_layer <= 0:
            raise ValueError("model.n_layer must be greater than 0")
        if model.n_head <= 0:
            raise ValueError("model.n_head must be greater than 0")
        if model.n_embd <= 0:
            raise ValueError("model.n_embd must be greater than 0")
        if model.n_embd % model.n_head != 0:
            raise ValueError("model.n_embd must be divisible by model.n_head")
        if model.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio must be greater than 0")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("model.dropout must satisfy 0.0 <= dropout < 1.0")

        training = self.training
        for field_name in ("batch_size", "grad_accum_steps", "warmup_steps"):
            _require_int(getattr(training, field_name), f"training.{field_name}")
        for field_name in (
            "learning_rate",
            "min_learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "grad_clip",
        ):
            _require_real(getattr(training, field_name), f"training.{field_name}")

        if training.batch_size <= 0:
            raise ValueError("training.batch_size must be greater than 0")
        if training.grad_accum_steps <= 0:
            raise ValueError("training.grad_accum_steps must be greater than 0")
        if training.learning_rate <= 0:
            raise ValueError("training.learning_rate must be greater than 0")
        if training.min_learning_rate < 0:
            raise ValueError("training.min_learning_rate must be non-negative")
        if training.min_learning_rate > training.learning_rate:
            raise ValueError(
                "training.min_learning_rate must be less than or equal to "
                "training.learning_rate"
            )
        if training.weight_decay < 0:
            raise ValueError("training.weight_decay must be non-negative")
        if not 0 < training.beta1 < 1:
            raise ValueError("training.beta1 must satisfy 0 < beta1 < 1")
        if not 0 < training.beta2 < 1:
            raise ValueError("training.beta2 must satisfy 0 < beta2 < 1")
        if training.grad_clip <= 0:
            raise ValueError("training.grad_clip must be greater than 0")
        if training.warmup_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative")

        data = self.data
        _require_int(data.seq_len, "data.seq_len")
        if data.seq_len <= 0:
            raise ValueError("data.seq_len must be greater than 0")
        if data.seq_len > model.max_seq_len:
            raise ValueError("data.seq_len must be less than or equal to model.max_seq_len")


def _require_int(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")


def _require_real(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a number")


def _section(raw_config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = raw_config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return section


def load_config(path: str | Path) -> Config:
    """Load, construct, and validate a configuration from a YAML file."""

    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw_config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw_config, Mapping):
        raise ValueError("configuration root must be a mapping")

    try:
        config = Config(
            model=ModelConfig(**_section(raw_config, "model")),
            training=TrainingConfig(**_section(raw_config, "training")),
            data=DataConfig(**_section(raw_config, "data")),
        )
    except TypeError as exc:
        raise ValueError(f"invalid configuration fields: {exc}") from exc

    config.validate()
    return config
