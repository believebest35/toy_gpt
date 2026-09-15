from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tinygpt import load_config


ROOT = Path(__file__).parents[1]
CONFIG_DIR = ROOT / "configs"


def test_tiny_yaml_loads_successfully() -> None:
    config = load_config(CONFIG_DIR / "tiny.yaml")

    assert config.model.vocab_size == 8192


def test_debug_yaml_loads_successfully() -> None:
    config = load_config(CONFIG_DIR / "debug.yaml")

    assert config.model.vocab_size == 256


def test_tiny_yaml_has_expected_values() -> None:
    config = load_config(CONFIG_DIR / "tiny.yaml")

    assert config.model.vocab_size == 8192
    assert config.model.max_seq_len == 256
    assert config.model.n_layer == 8
    assert config.model.n_head == 8
    assert config.model.n_embd == 512
    assert config.data.seq_len == 256


def test_debug_yaml_has_expected_values() -> None:
    config = load_config(CONFIG_DIR / "debug.yaml")

    assert config.model.vocab_size == 256
    assert config.model.n_layer == 2
    assert config.model.n_head == 2
    assert config.model.n_embd == 64
    assert config.data.seq_len == 64


@pytest.mark.parametrize("config_name", ["tiny.yaml", "debug.yaml"])
def test_embedding_size_is_compatible_with_number_of_heads(config_name: str) -> None:
    config = load_config(CONFIG_DIR / config_name)

    assert config.model.n_embd % config.model.n_head == 0


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _debug_config_data() -> dict[str, object]:
    return yaml.safe_load((CONFIG_DIR / "debug.yaml").read_text(encoding="utf-8"))


def test_non_divisible_embedding_size_is_rejected(tmp_path: Path) -> None:
    config_data = _debug_config_data()
    config_data["model"] = {**config_data["model"], "n_embd": 63}

    with pytest.raises(ValueError, match="divisible"):
        load_config(_write_config(tmp_path, config_data))


def test_sequence_length_larger_than_model_limit_is_rejected(tmp_path: Path) -> None:
    config_data = _debug_config_data()
    config_data["data"] = {**config_data["data"], "seq_len": 65}

    with pytest.raises(ValueError, match="seq_len"):
        load_config(_write_config(tmp_path, config_data))


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "vocab_size", 0),
        ("model", "max_seq_len", 0),
        ("model", "n_layer", 0),
        ("model", "n_head", 0),
        ("model", "n_embd", 0),
        ("model", "mlp_ratio", 0),
        ("training", "batch_size", 0),
        ("training", "grad_accum_steps", 0),
        ("training", "learning_rate", 0),
        ("training", "grad_clip", 0),
        ("data", "seq_len", 0),
    ],
)
def test_non_positive_values_are_rejected(
    tmp_path: Path, section: str, field: str, value: int
) -> None:
    config_data = deepcopy(_debug_config_data())
    config_data[section] = {**config_data[section], field: value}

    with pytest.raises(ValueError):
        load_config(_write_config(tmp_path, config_data))
