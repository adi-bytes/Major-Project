"""Configuration loader and schemas for PG-SBD."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        # Look relative to project root
        root = Path(__file__).resolve().parent.parent.parent
        path = root / config_path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
