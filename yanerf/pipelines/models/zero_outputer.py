import warnings
from typing import Any, NamedTuple, Optional

import torch

from yanerf.utils.logging import get_logger

from .builder import MODELS

logger = get_logger(__name__)


@MODELS.register_module()
class ZeroOutputer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        warnings.warn("Should not use ZeroOutputer, Debug only.")

    def forward(
        self,
        # ray_bundle: RayBundle,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: torch.Tensor,
        # xys: torch.Tensor,
        # fun_viewpool=None,
        # cameras=None,
        # camera: Optional[CamerasBase] = None,
        global_codes: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        B, *spatial_shape, _ = origins.shape
        *_, n_pts_per_ray = lengths.shape
        raw_densities = origins.new_zeros(B, *spatial_shape, n_pts_per_ray, 1)
        rays_colors = origins.new_zeros(B, *spatial_shape, n_pts_per_ray, 3)
        return dict(rays_densities=raw_densities, rays_features=rays_colors, aux={})
