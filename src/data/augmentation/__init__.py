"""Defect-aware augmentation primitives."""
from .copy_paste import CopyPaste
from .elastic import ElasticDeformation
from .cutout import Cutout

__all__ = ["CopyPaste", "ElasticDeformation", "Cutout"]
