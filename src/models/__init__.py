from .base import BaseGenerator
from .diffusion.ddpm import DDPMGenerator
from .flow_matching.ot_cfm import FlowMatchingGenerator

__all__ = ["BaseGenerator", "DDPMGenerator", "FlowMatchingGenerator"]


def build_model(cfg) -> BaseGenerator:
    """Factory: build a generator from an OmegaConf config node.

    Args:
        cfg: config with `name` in {"ddpm", "flow_matching"} and model kwargs.
    """
    name = cfg.name.lower()
    if name == "ddpm":
        return DDPMGenerator(**cfg.params)
    if name in ("flow_matching", "ot_cfm", "cfm"):
        return FlowMatchingGenerator(**cfg.params)
    raise ValueError(f"Unknown model: {name}")
