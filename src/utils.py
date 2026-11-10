"""Shared utility helpers for configuration, logging, and file I/O."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

REQUIRED_CONFIG_SECTIONS = ("dataset", "preprocessing", "models", "evaluation", "output")


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    logger = logging.getLogger("house_price_prediction")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load and validate the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML content is empty, malformed, or missing required sections.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML syntax in configuration file {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file {path} must define a YAML mapping.")

    missing = [section for section in REQUIRED_CONFIG_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Configuration file {path} is missing required sections: {missing}")

    return config


def ensure_dir(directory: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], filepath: str | Path) -> Path:
    """Serialize a dictionary to a UTF-8 JSON file."""
    path = Path(filepath)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
    return path


def load_json(filepath: str | Path) -> dict[str, Any]:
    """Load a dictionary from a UTF-8 JSON file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"JSON file {path} must contain an object at the top level.")
    return data

