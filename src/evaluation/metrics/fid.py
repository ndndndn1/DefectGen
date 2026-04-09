"""FID via torchvision Inception V3 features (Heusel et al., NeurIPS 2017)."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseMetric


def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    # Symmetric PSD matrix square root via eigendecomposition.
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


class FID(BaseMetric):
    """Frechet Inception Distance.

    Args:
        device: torch device for the feature extractor.
        feature_dim: dimensionality of pooled features (2048 for InceptionV3).
    """

    name = "fid"

    def __init__(self, device: Optional[torch.device] = None, feature_dim: int = 2048) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self._build_extractor()
        self.real_feats: list[Tensor] = []
        self.fake_feats: list[Tensor] = []

    def _build_extractor(self) -> None:
        from torchvision.models import inception_v3, Inception_V3_Weights

        net = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        net.fc = nn.Identity()
        self.extractor = net.eval().to(self.device)

    @torch.no_grad()
    def _features(self, x: Tensor) -> Tensor:
        # x in [-1, 1]; convert to [0, 1] then resize to 299 and 3-channel.
        x = (x + 1) / 2
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        # ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std
        return self.extractor(x.to(self.device))

    def update(self, real: Tensor, fake: Tensor) -> None:
        self.real_feats.append(self._features(real).cpu())
        self.fake_feats.append(self._features(fake).cpu())

    def compute(self) -> float:
        real = torch.cat(self.real_feats, dim=0).numpy()
        fake = torch.cat(self.fake_feats, dim=0).numpy()
        mu_r, mu_f = real.mean(0), fake.mean(0)
        sig_r = np.cov(real, rowvar=False)
        sig_f = np.cov(fake, rowvar=False)
        diff = mu_r - mu_f
        covmean = _matrix_sqrt(sig_r @ sig_f)
        fid = float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))
        return fid

    def reset(self) -> None:
        self.real_feats.clear()
        self.fake_feats.clear()
