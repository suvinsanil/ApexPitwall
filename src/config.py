"""Configuration and small IO helpers shared by every module."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def load_config() -> dict:
    """Return the base config with all circuit configs attached under ``circuit_info``."""
    cfg = _read_yaml(CONFIG_DIR / "base_config.yaml")
    cfg["circuit_info"] = {
        cid: _read_yaml(CONFIG_DIR / "circuits" / f"{cid}.yaml") for cid in cfg["circuits"]
    }
    return cfg


def path(key: str, *parts: str) -> Path:
    """Absolute path for a directory key in the config (e.g. ``path('processed_dir', 'x.csv')``)."""
    p = PROJECT_ROOT / load_config()[key]
    return p.joinpath(*parts) if parts else p


def ensure_dir(p: Path) -> Path:
    p = Path(p)
    (p if p.suffix == "" else p.parent).mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: dict, p: Path) -> None:
    ensure_dir(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)


def read_json(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def chdir_root() -> None:
    """Scripts are run from anywhere; make relative paths resolve from the project root."""
    os.chdir(PROJECT_ROOT)

