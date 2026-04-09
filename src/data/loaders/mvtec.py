"""MVTec AD loader.

Expected layout::

    root/<category>/train/good/*.png
    root/<category>/test/<defect>/*.png

By default we load the *defect* test images of a given category for generative
training (since "good" is the in-distribution baseline).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from ._base import IMG_EXTS, default_transform


class MVTecADDataset(Dataset):
    def __init__(
        self,
        root: str,
        category: str = "bottle",
        image_size: int = 256,
        split: str = "test",
        include_good: bool = False,
        channels: int = 3,
    ) -> None:
        self.root = Path(root) / category / split
        self.image_size = image_size
        self.channels = channels
        self.transform = default_transform(image_size, channels)

        samples: list[tuple[Path, int]] = []
        classes: list[str] = []
        if self.root.exists():
            for sub in sorted(self.root.iterdir()):
                if not sub.is_dir():
                    continue
                if sub.name == "good" and not include_good:
                    continue
                idx = len(classes)
                classes.append(sub.name)
                for p in sub.rglob("*"):
                    if p.suffix.lower() in IMG_EXTS:
                        samples.append((p, idx))
        self.samples = samples
        self.classes = classes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path)
        return {"image": self.transform(img), "label": torch.tensor(label, dtype=torch.long)}
