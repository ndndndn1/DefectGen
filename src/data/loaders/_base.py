"""Shared base for image-folder style defect datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def default_transform(image_size: int, channels: int) -> Callable[[Image.Image], Tensor]:
    mode = "L" if channels == 1 else "RGB"

    def _t(img: Image.Image) -> Tensor:
        img = img.convert(mode)
        return T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5] * channels, [0.5] * channels),  # -> [-1, 1]
        ])(img)

    return _t


class ImageFolderDefectDataset(Dataset):
    """Generic per-class folder layout: ``root/<defect_type>/*.png``.

    Provides class-conditional labels via the sorted directory listing.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 64,
        channels: int = 3,
        defect_types: Optional[list[str]] = None,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.channels = channels
        self.transform = transform or default_transform(image_size, channels)

        if not self.root.exists():
            # Empty placeholder; allows shape tests on CI without the data.
            self.samples: list[tuple[Path, int]] = []
            self.classes: list[str] = defect_types or []
            return

        classes = defect_types or sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.classes = classes
        cls_to_idx = {c: i for i, c in enumerate(classes)}
        samples: list[tuple[Path, int]] = []
        for c in classes:
            for p in (self.root / c).rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    samples.append((p, cls_to_idx[c]))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        path, label = self.samples[idx]
        img = Image.open(path)
        return {"image": self.transform(img), "label": torch.tensor(label, dtype=torch.long)}
