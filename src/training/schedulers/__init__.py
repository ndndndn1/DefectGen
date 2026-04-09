"""LR / noise schedules. Currently re-exports the linear beta schedule."""
from ...models.diffusion.ddpm import linear_beta_schedule

__all__ = ["linear_beta_schedule"]
