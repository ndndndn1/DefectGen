"""WM-811K wafer map defect loader.

Expected layout::

    root/
        Center/        *.png
        Donut/         *.png
        Edge-Loc/      *.png
        ...
"""
from __future__ import annotations

from ._base import ImageFolderDefectDataset

WM811K_CLASSES = [
    "Center", "Donut", "Edge-Loc", "Edge-Ring",
    "Loc", "Random", "Scratch", "Near-full", "none",
]


class WM811KDataset(ImageFolderDefectDataset):
    def __init__(self, root: str, image_size: int = 64, **kwargs) -> None:
        super().__init__(
            root=root,
            image_size=image_size,
            channels=1,
            defect_types=kwargs.pop("defect_types", WM811K_CLASSES),
            **kwargs,
        )
