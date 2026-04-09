"""Config loading via OmegaConf."""
from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path) -> DictConfig:
    cfg = OmegaConf.load(str(path))
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected dict-like config, got {type(cfg)}")
    return cfg
