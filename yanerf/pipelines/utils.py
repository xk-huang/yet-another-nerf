from enum import Enum
from typing import Any, Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch


class EvaluationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class RayBundle(NamedTuple):
    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor


class PartialFunctionWrapper(torch.nn.Module):
    def __init__(self, fn: torch.nn.Module):
        super().__init__()
        self._fn = fn
        self.bound_args: Dict[str, Any] = {}

    def bind_args(self, **bound_args):
        self.bound_args = bound_args

    def unbind_args(self):
        self.bound_args = {}

    def forward(self, *args, **kwargs):
        return self._fn(*args, **{**kwargs, **self.bound_args})


class ViewMetrics(torch.nn.Module):
    def forward(
        self,
        image_sampling_grid: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        images_pred: Optional[torch.Tensor] = None,
        depths: Optional[torch.Tensor] = None,
        depths_pred: Optional[torch.Tensor] = None,
        # alpha_masks: Optional[torch.Tensor] = None,
        # alpha_masks_pred: Optional[torch.Tensor] = None,
        # masks_crop: Optional[torch.Tensor] = None,
        # grad_theta: Optional[torch.Tensor] = None,
        # density_grid: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
        # mask_renders_by_pred: bool = False,
    ):
        """
        Calculates various differentiable metrics useful for supervising
        differentiable rendering pipelines.

        Args:
            image_sampling_grid: A tensor of shape `(B, ..., 2)` containing 2D
                    image locations at which the predictions are defined.
                    All ground truth inputs are sampled at these
                    locations in order to extract values that correspond
                    to the predictions.
            images: A tensor of shape `(B, H, W, 3)` containing ground truth
                rgb values.
            images_pred: A tensor of shape `(B, ..., 3)` containing predicted
                rgb values.
            depths: A tensor of shape `(B, Hd, Wd, 1)` containing ground truth
                depth values.
            depths_pred: A tensor of shape `(B, ..., 1)` containing predicted
                depth values.
            masks: A tensor of shape `(B, Hm, Wm, 1)` containing ground truth
                foreground masks.
            masks_pred: A tensor of shape `(B, ..., 1)` containing predicted
                foreground masks.
            grad_theta: A tensor of shape `(B, ..., 3)` containing an evaluation
                of a gradient of a signed distance function w.r.t.
                input 3D coordinates used to compute the eikonal loss.
            density_grid: A tensor of shape `(B, Hg, Wg, Dg, 1)` containing a
                `Hg x Wg x Dg` voxel grid of density values.
            keys_prefix: A common prefix for all keys in the output dictionary
                containing all metrics.
            mask_renders_by_pred: If `True`, masks rendered images by the predicted
                `masks_pred` prior to computing all rgb metrics.

        Returns:
            metrics: A dictionary `{metric_name_i: metric_value_i}` keyed by the
                names of the output metrics `metric_name_i` with their corresponding
                values `metric_value_i` represented as 0-dimensional float tensors.

                The calculated metrics are:
                    rgb_huber: A robust huber loss between `image_pred` and `image`.
                    rgb_mse: Mean squared error between `image_pred` and `image`.
                    rgb_psnr: Peak signal-to-noise ratio between `image_pred` and `image`.
                    rgb_psnr_fg: Peak signal-to-noise ratio between the foreground
                        region of `image_pred` and `image` as defined by `mask`.
                    rgb_mse_fg: Mean squared error between the foreground
                        region of `image_pred` and `image` as defined by `mask`.
                    mask_neg_iou: (1 - intersection-over-union) between `mask_pred`
                        and `mask`.
                    mask_bce: Binary cross entropy between `mask_pred` and `mask`.
                    mask_beta_prior: A loss enforcing strictly binary values
                        of `mask_pred`: `log(mask_pred) + log(1-mask_pred)`
                    depth_abs: Mean per-pixel L1 distance between
                        `depth_pred` and `depth`.
                    depth_abs_fg: Mean per-pixel L1 distance between the foreground
                        region of `depth_pred` and `depth` as defined by `mask`.
                    eikonal: Eikonal regularizer `(||grad_theta|| - 1)**2`.
                    density_tv: The Total Variation regularizer of density
                        values in `density_grid` (sum of L1 distances of values
                        of all 4-neighbouring cells).
                    depth_neg_penalty: `min(depth_pred, 0)**2` penalizing negative
                        predicted depth values.
        """

        def _sample_grid(tensor):
            if tensor is None:
                return tensor
            return sample_grid(tensor, image_sampling_grid)

        images = _sample_grid(images)
        depths = _sample_grid(depths)

        preds = {}
        if images is not None and images_pred is not None:
            preds.update(_rgb_metrics(images, images_pred))

        if depths is not None and depths_pred is not None:
            _, abs_ = eval_depth(depths_pred, depths, get_best_scale=True, mask=None, crop=0)
            preds["depth_abs"] = abs_.mean()
        if keys_prefix is not None:
            preds = {(keys_prefix + k): v for k, v in preds.items()}

        return preds


def _rgb_metrics(
    images: torch.Tensor,
    images_pred: torch.Tensor,
):
    batch_size = images.shape[0]
    images = images.view(batch_size, -1)
    images_pred = images_pred.view(batch_size, -1)
    rgb_squared = ((images_pred - images) ** 2).mean(dim=-1)
    rgb_loss = huber(rgb_squared, scaling=0.03)
    preds = {
        "rgb_huber": rgb_loss,
        "rgb_mse": rgb_squared,
        "rgb_psnr": calc_psnr(images_pred, images),
    }

    return preds


def calc_psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    base: float = 1.0,
) -> torch.Tensor:
    """
    Calculates the Peak-signal-to-noise ratio between tensors `x` and `y`.
    """
    mse = calc_mse(x, y, mask=mask)
    psnr = torch.log10(mse.clamp(1e-10)) * (-10.0) + 20.0 * np.log10(base)
    return psnr


def calc_mse(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates the mean square error between tensors `x` and `y`.
    """
    if mask is None:
        return torch.mean((x - y) ** 2, dim=-1)
    else:
        return (((x - y) ** 2) * mask).sum(dim=-1) / mask.expand_as(x).sum(dim=-1).clamp(1e-5)


def huber(dfsq: torch.Tensor, scaling: float = 0.03) -> torch.Tensor:
    """
    Calculates the huber function of the input squared error `dfsq`.
    The function smoothly transitions from a region with unit gradient
    to a hyperbolic function at `dfsq=scaling`.
    """
    loss = (safe_sqrt(1 + dfsq / (scaling * scaling), eps=1e-4) - 1) * scaling
    return loss


def safe_sqrt(A: torch.Tensor, eps: float = float(1e-4)) -> torch.Tensor:
    """
    Performs safe differentiable sqrt
    """
    return (torch.clamp(A, float(0)) + eps).sqrt()


def eval_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    crop: int = 1,
    mask: Optional[torch.Tensor] = None,
    get_best_scale: bool = True,
    mask_thr: float = 0.5,
    best_scale_clamp_thr: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the depth error between the prediction `pred` and the ground
    truth `gt`.

    Args:
        pred: A tensor of shape (N, 1, H, W) denoting the predicted depth maps.
        gt: A tensor of shape (N, 1, H, W) denoting the ground truth depth maps.
        crop: The number of pixels to crop from the border.
        mask: A mask denoting the valid regions of the gt depth.
        get_best_scale: If `True`, estimates a scaling factor of the predicted depth
            that yields the best mean squared error between `pred` and `gt`.
            This is typically enabled for cases where predicted reconstructions
            are inherently defined up to an arbitrary scaling factor.
        mask_thr: A constant used to threshold the `mask` to specify the valid
            regions.
        best_scale_clamp_thr: The threshold for clamping the divisor in best
            scale estimation.

    Returns:
        mse_depth: Mean squared error between `pred` and `gt`.
        abs_depth: Mean absolute difference between `pred` and `gt`.
    """
    # chuck out the border
    if crop > 0:
        gt = gt[:, :, crop:-crop, crop:-crop]
        pred = pred[:, :, crop:-crop, crop:-crop]

    if mask is not None:
        # mult gt by mask
        if crop > 0:
            mask = mask[:, :, crop:-crop, crop:-crop]
        gt = gt * (mask > mask_thr).float()

    dmask = (gt > 0.0).float()
    dmask_mass = torch.clamp(dmask.sum((1, 2, 3)), 1e-4)

    if get_best_scale:
        # mult preds by a scalar "scale_best"
        # 	s.t. we get best possible mse error
        scale_best = estimate_depth_scale_factor(pred, gt, dmask, best_scale_clamp_thr)
        pred = pred * scale_best[:, None, None, None]

    df = gt - pred

    mse_depth = (dmask * (df**2)).sum((1, 2, 3)) / dmask_mass
    abs_depth = (dmask * df.abs()).sum((1, 2, 3)) / dmask_mass

    return mse_depth, abs_depth


def estimate_depth_scale_factor(pred, gt, mask, clamp_thr):
    xy = pred * gt * mask
    xx = pred * pred * mask
    scale_best = xy.mean((1, 2, 3)) / torch.clamp(xx.mean((1, 2, 3)), clamp_thr)
    return scale_best


def sample_grid(tensor: torch.Tensor, image_sampling_grid: torch.Tensor) -> torch.Tensor:
    batch_size, *tensor_spatial_shape, last_dim = tensor.shape
    _, *grid_spatial_shape, _ = image_sampling_grid.shape
    flat_tensor = tensor.view(batch_size, -1, last_dim)
    flat_image_sampling_grid = image_sampling_grid.view(batch_size, -1, 2)
    flat_image_sampling_grid = (
        flat_image_sampling_grid[:, :, 0] + tensor_spatial_shape[-1] * flat_image_sampling_grid[:, :, 1]
    )
    flat_image_sampling_grid = flat_image_sampling_grid[:, :, None].expand(-1, -1, last_dim)
    flat_image_sampling_grid = flat_image_sampling_grid.long()

    sampled_tensor = torch.gather(flat_tensor, -2, flat_image_sampling_grid)
    return sampled_tensor.view(batch_size, *grid_spatial_shape, last_dim)


@torch.no_grad()
def scatter_rays_to_image(
    tensor: torch.Tensor,
    image_sampling_grid: torch.Tensor,
    image_height: int,
    image_width: int,
    bg_color: Optional[torch.Tensor] = None,
):
    batch_size, *tensor_spatial_shape, last_dim = tensor.shape
    _, *grid_spatial_shape, _ = image_sampling_grid.shape
    assert tensor_spatial_shape == grid_spatial_shape

    flat_tensor = tensor.view(batch_size, -1, last_dim)
    flat_image_sampling_grid = image_sampling_grid.view(batch_size, -1, 2)
    flat_image_sampling_grid = flat_image_sampling_grid[..., 0] + image_width * flat_image_sampling_grid[..., 1]
    flat_image_sampling_grid = flat_image_sampling_grid[..., None].expand(-1, -1, last_dim)
    flat_image_sampling_grid = flat_image_sampling_grid.long()

    output_tensor = tensor.new_zeros(batch_size, image_height, image_width, last_dim)
    if bg_color is not None and bg_color.shape[-1] == last_dim:
        output_tensor = output_tensor + bg_color
    output_tensor = output_tensor.view(batch_size, -1, last_dim)
    output_tensor.scatter_(1, flat_image_sampling_grid, flat_tensor)

    return output_tensor.view(batch_size, image_height, image_width, last_dim)
