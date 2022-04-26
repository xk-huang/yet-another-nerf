from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, NamedTuple, Optional

import torch


class EvaluationMode(str, Enum):
    TRAINING: str = "training"
    EVALUATION: str = "evaluation"


@dataclass
class RendererOutput:
    """
    A structure for storing the output of a renderer.

    Args:
        features: rendered features (usually RGB colors), (B, ..., C) tensor.
        depth: rendered ray-termination depth map, in NDC coordinates, (B, ..., 1) tensor.
        mask: rendered object mask, values in [0, 1], (B, ..., 1) tensor.
        prev_stage: for multi-pass renderers (e.g. in NeRF),
            a reference to the output of the previous stage.
        normals: surface normals, for renderers that estimate them; (B, ..., 3) tensor.
        points: ray-termination points in the world coordinates, (B, ..., 3) tensor.
        aux: dict for implementation-specific renderer outputs.
    """

    features: torch.Tensor
    depths: torch.Tensor
    masks: torch.Tensor
    prev_stage: Optional[RendererOutput] = None
    normals: Optional[torch.Tensor] = None
    points: Optional[torch.Tensor] = None  # TODO: redundant with depths
    aux: Dict[str, Any] = field(default_factory=lambda: {})


class RayBundle(NamedTuple):
    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


class RayPointRefiner(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        random_sampling: bool,
        add_input_samples: bool = True,
    ) -> None:
        super().__init__()
        self.n_pts_per_ray: int = n_pts_per_ray
        self.random_sampling: bool = random_sampling
        self.add_input_samples: bool = add_input_samples

    def forward(self, origins, directions, lengths, xys, ray_weights):
        z_vals = lengths
        with torch.no_grad():
            z_vals_mid = torch.lerp(z_vals[..., 1:], z_vals[..., :-1], 0.5)
            z_samples = sample_pdf(
                z_vals_mid.view(-1, z_vals_mid.shape[-1]),
                ray_weights.view(-1, ray_weights.shape[-1])[..., 1:-1],
                self.n_pts_per_ray,
                det=not self.random_sampling,
            ).view(*z_vals.shape[:-1], self.n_pts_per_ray)

        if self.add_input_samples:
            # Add the new samples to the input ones.
            z_vals = torch.cat((z_vals, z_samples), dim=-1)
        else:
            z_vals = z_samples
        # Resort by depth.
        z_vals, _ = torch.sort(z_vals, dim=-1)

        return RayBundle(
            origins=origins, directions=directions, lengths=z_vals, xys=xys
        )  # FIXME bug, shold return the z_vals


def sample_pdf(
    bins: torch.Tensor,
    weights: torch.Tensor,
    n_samples: int,
    det: bool = False,
    eps: float = 1e-5,
):
    # TODO: implement the C++ version
    return sample_pdf_python(bins, weights, n_samples, det, eps)


def sample_pdf_python(
    bins: torch.Tensor,
    weights: torch.Tensor,
    N_samples: int,
    det: bool = False,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    This is a pure python implementation of the `sample_pdf` function.
    It may be faster than sample_pdf when the number of bins is very large,
    because it behaves as O(batchsize * [n_bins + log(n_bins) * n_samples] )
    whereas sample_pdf behaves as O(batchsize * n_bins * n_samples).
    For 64 bins sample_pdf is much faster.

    Samples probability density functions defined by bin edges `bins` and
    the non-negative per-bin probabilities `weights`.

    Note: This is a direct conversion of the TensorFlow function from the original
    release [1] to PyTorch. It requires PyTorch 1.6 or greater due to the use of
    torch.searchsorted.

    Args:
        bins: Tensor of shape `(..., n_bins+1)` denoting the edges of the sampling bins.
        weights: Tensor of shape `(..., n_bins)` containing non-negative numbers
            representing the probability of sampling the corresponding bin.
        N_samples: The number of samples to draw from each set of bins.
        det: If `False`, the sampling is random. `True` yields deterministic
            uniformly-spaced sampling from the inverse cumulative density function.
        eps: A constant preventing division by zero in case empty bins are present.

    Returns:
        samples: Tensor of shape `(..., N_samples)` containing `N_samples` samples
            drawn from each probability distribution.

    Refs:
        [1] https://github.com/bmild/nerf/blob/55d8b00244d7b5178f4d003526ab6667683c9da9/run_nerf_helpers.py#L183  # noqa E501
    """

    # Get pdf
    weights = weights + eps  # prevent nans
    if weights.min() <= 0:
        raise ValueError("Negative weights provided.")
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples u of shape (..., N_samples)
    if det:
        u = torch.linspace(0.0, 1.0, N_samples, device=cdf.device, dtype=cdf.dtype)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples]).contiguous()
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=cdf.device, dtype=cdf.dtype)

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    # inds has shape (..., N_samples) identifying the bin of each sample.
    below = (inds - 1).clamp(0)
    above = inds.clamp(max=cdf.shape[-1] - 1)
    # Below and above are of shape (..., N_samples), identifying the bin
    # edges surrounding each sample.

    inds_g = torch.stack([below, above], -1).view(*below.shape[:-1], below.shape[-1] * 2)
    cdf_g = torch.gather(cdf, -1, inds_g).view(*below.shape, 2)
    bins_g = torch.gather(bins, -1, inds_g).view(*below.shape, 2)
    # cdf_g and bins_g are of shape (..., N_samples, 2) and identify
    # the cdf and the index of the two bin edges surrounding each sample.

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < eps, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    # t is of shape  (..., N_samples) and identifies how far through
    # each sample is in its bin.

    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
