"""NEU-DET steel surface defect loader.

Six classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale,
scratches.
"""
from __future__ import annotations

from ._base import ImageFolderDefectDataset

NEUDET_CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


class NEUDETDataset(ImageFolderDefectDataset):
    def __init__(self, root: str, image_size: int = 200, **kwargs) -> None:
        super().__init__(
            root=root,
            image_size=image_size,
            channels=3,
            defect_types=kwargs.pop("defect_types", NEUDET_CLASSES),
            **kwargs,
        )
