"""Dataset loader registry."""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .wm811k import WM811KDataset
from .deeppcb import DeepPCBDataset
from .neudet import NEUDETDataset
from .mvtec import MVTecADDataset

_REGISTRY: dict[str, type[Dataset]] = {
    "wm811k": WM811KDataset,
    "deeppcb": DeepPCBDataset,
    "neudet": NEUDETDataset,
    "mvtec": MVTecADDataset,
}


def build_dataset(cfg: Any) -> Dataset:
    """Build a dataset from a config node with `name`, `root`, and kwargs."""
    name = cfg.name.lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**{k: v for k, v in cfg.items() if k != "name"})


__all__ = ["build_dataset", "WM811KDataset", "DeepPCBDataset", "NEUDETDataset", "MVTecADDataset"]
