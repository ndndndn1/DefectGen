"""Base class for evaluation metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor


class BaseMetric(ABC):
    """Stateful metric: ``update`` then ``compute``."""

    name: str = "metric"

    @abstractmethod
    def update(self, real: Tensor, fake: Tensor) -> None: ...

    @abstractmethod
    def compute(self) -> float: ...

    def reset(self) -> None:  # pragma: no cover - default no-op
        pass
