"""DeepPCB defect loader.

Expected layout: ``root/<defect_type>/*.jpg`` where defect types include
missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper.
"""
from __future__ import annotations

from ._base import ImageFolderDefectDataset

DEEPPCB_CLASSES = [
    "missing_hole", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]


class DeepPCBDataset(ImageFolderDefectDataset):
    def __init__(self, root: str, image_size: int = 256, **kwargs) -> None:
        super().__init__(
            root=root,
            image_size=image_size,
            channels=3,
            defect_types=kwargs.pop("defect_types", DEEPPCB_CLASSES),
            **kwargs,
        )
