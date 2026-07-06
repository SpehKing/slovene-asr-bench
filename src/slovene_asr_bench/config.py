"""Config loading. Configs are plain YAML passed to Modal functions as dicts
so the local environment needs nothing beyond pyyaml."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "seed", "data"):
        if key not in cfg:
            raise ValueError(f"config {path} missing required key {key!r}")
    hours = cfg["data"]["hours"]
    for split in ("train", "dev", "test"):
        if split not in hours:
            raise ValueError(f"config {path} missing data.hours.{split}")
    return cfg


def required_pool_hours(cfg: dict) -> float:
    """Hours of usable utterances needed before splitting (targets + 20% slack)."""
    h = cfg["data"]["hours"]
    return 1.2 * (h["train"] + h["dev"] + h["test"])


def to_plain_json(obj):
    """Round-trip through JSON so Modal return values contain only built-in
    types — the local venv has no numpy to unpickle np.float64 and friends."""
    import json

    return json.loads(
        json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    )
