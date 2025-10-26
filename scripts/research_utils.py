"""Shared helpers for HCMRL research runners."""
from __future__ import annotations

import ast
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    tomllib = None

try:  # Optional dependency
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - YAML support optional
    yaml = None


JSONDecodeError = json.JSONDecodeError


def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    """Load a configuration file (JSON, TOML, or YAML)."""
    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("tomllib is unavailable; use Python 3.11+ for TOML configs")
        with path.open("rb") as handle:
            return tomllib.load(handle)
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed; install pyyaml or use JSON/TOML configs")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    raise ValueError(f"Unsupported config format: {suffix}")


def coerce_value(value: str) -> Any:
    """Convert a string override into a Python object."""
    stripped = value.strip()

    # Try JSON first (handles numbers, booleans, lists)
    try:
        return json.loads(stripped)
    except JSONDecodeError:
        pass

    # Fall back to literal eval for Python-like inputs
    try:
        return ast.literal_eval(stripped)
    except (ValueError, SyntaxError):
        return stripped


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Apply dotted-path overrides of the form key1.key2=value."""
    result = copy.deepcopy(config)

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override (missing '='): {override}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = coerce_value(raw_value)

        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    return result


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def dump_json(data: Dict[str, Any], path: Path) -> None:
    """Write a dictionary to disk in pretty JSON format."""
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def configure_logging(verbosity: int = 0) -> None:
    """Configure root logger with sensible defaults."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
    )