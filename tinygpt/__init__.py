"""Public package interface for TinyGPT Phase 1."""

from .config import Config, DataConfig, ModelConfig, TrainingConfig, load_config

__all__ = [
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "load_config",
]
