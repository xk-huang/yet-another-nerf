from typing import Optional, Tuple

import torch

from .builder import RAY_SAMPLERS
from .utils import EvaluationMode, RayBundle, RenderSamplingMode


@RAY_SAMPLERS.register_module()
class RaySampler(torch.nn.Module):
    def __init__(
        self,
        image_width: int = 400,
        image_height: int = 400,
        scene_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        scene_extent: float = 0.0,
        sampling_mode_training: str = "mask_sample",
        sampling_mode_evaluation: str = "full_grid",
        n_pts_per_ray_training: int = 64,
        n_pts_per_ray_evaluation: int = 64,
        n_rays_per_image_sampled_from_mask: int = 1024,
        min_depth: float = 0.1,
        max_depth: float = 8.0,
        # stratified sampling vs taking points at deterministic offsets,
        stratified_point_sampling_training: bool = True,
        stratified_point_sampling_evaluation: bool = False,
    ) -> None:
        super().__init__()

        self._sampling_mode = {
            EvaluationMode.TRAINING: RenderSamplingMode(sampling_mode_training),
            EvaluationMode.EVALUATION: RenderSamplingMode(sampling_mode_evaluation),
        }

        self._raysamplers = {
            EvaluationMode.TRAINING: _RaySampler(
                image_width=image_width,
                image_height=image_height,
                n_pts_per_ray=n_pts_per_ray_training,
                min_depth=min_depth,
                max_depth=max_depth,
                n_rays_per_image=n_rays_per_image_sampled_from_mask
                if self._sampling_mode[EvaluationMode.TRAINING] == RenderSamplingMode.MASK_SAMPLE
                else None,
                unit_directions=True,
                stratified_sampling=stratified_point_sampling_training,
            ),
            EvaluationMode.EVALUATION: _RaySampler(
                image_width=image_width,
                image_height=image_height,
                n_pts_per_ray=n_pts_per_ray_evaluation,
                min_depth=min_depth,
                max_depth=max_depth,
                n_rays_per_image=n_rays_per_image_sampled_from_mask
                if self._sampling_mode[EvaluationMode.EVALUATION] == RenderSamplingMode.MASK_SAMPLE
                else None,
                unit_directions=True,
                stratified_sampling=stratified_point_sampling_evaluation,
            ),
        }

        self.register_buffer("scene_center", torch.Tensor(scene_center), persistent=False)
        self.scene_extent = scene_extent

    def forward(
        self,
        poses: torch.Tensor,
        focal_lengths: torch.Tensor,
        evaluation_mode: EvaluationMode,
        mask: Optional[torch.Tensor] = None,
    ) -> RayBundle:
        sample_mask = None
        if (
            # pyre-fixme[29]
            self._sampling_mode[evaluation_mode] == RenderSamplingMode.MASK_SAMPLE
            and mask is not None
        ):
            sample_mask = torch.nn.functional.interpolate(
                mask,
                # pyre-fixme[6]: Expected `Optional[int]` for 2nd param but got
                #  `List[int]`.
                size=[self.image_height, self.image_width],
                mode="nearest",
            )[:, 0]

        if self.scene_extent > 0.0:
            # Override the min/max depth set in initialization based on the
            # input cameras.
            min_depth, max_depth = get_min_max_depth_bounds(poses, self.scene_center, self.scene_extent)

        ray_bundle = self._raysamplers[evaluation_mode](
            poses,
            focal_lengths,
            mask=sample_mask,
            min_depth=float(min_depth[0]) if self.scene_extent > 0.0 else None,
            max_depth=float(max_depth[0]) if self.scene_extent > 0.0 else None,
        )

        return ray_bundle


class _RaySampler(torch.nn.Module):
    def __init__(
        self,
        *,
        image_width: int,
        image_height: int,
        n_pts_per_ray: int,
        min_depth: float,
        max_depth: float,
        n_rays_per_image: Optional[int] = None,
        unit_directions: bool = False,
        stratified_sampling: bool = False,
    ) -> None:
        super().__init__()
        self._image_width = image_width
        self._image_height = image_height
        self._n_pts_per_ray = n_pts_per_ray
        self._min_depth = min_depth
        self._max_depth = max_depth
        self._n_rays_per_image = n_rays_per_image
        self._unit_directions = unit_directions
        self._stratified_sampling = stratified_sampling

        _xy_grid = torch.stack(
            tuple(
                reversed(
                    torch.meshgrid(
                        torch.linspace(0, image_height - 1, image_height, dtype=torch.float32),
                        torch.linspace(0, image_width - 1, image_width, dtype=torch.float32),
                        indexing="ij",
                    )
                )
            ),
            dim=-1,
        )

        self.register_buffer("_xy_grid", _xy_grid, persistent=False)

    def forward(
        self,
        poses,
        focal_lengths,
        *,
        mask: Optional[torch.Tensor] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        n_rays_per_image: Optional[int] = None,
        n_pts_per_ray: Optional[int] = None,
        stratified_sampling: bool = False,
    ) -> RayBundle:
        batch_size = poses.shape[0]
        device = poses.device

        poses = poses[:, :3, :4]
        xy_grid = self._xy_grid.to(device).expand(batch_size, -1, -1, -1)  # type: ignore[operator]

        num_rays = n_rays_per_image or self._n_rays_per_image
        if mask is not None and num_rays is None:
            # if num rays not given, sample according to the smallest mask
            num_rays = num_rays or mask.sum(dim=(1, 2)).min().int().item()  # type: ignore[assignment]

        if num_rays is not None:
            if mask is not None:
                assert mask.shape == xy_grid.shape[:3]
                weights = mask.reshape(batch_size, -1)
            else:
                _, width, height, _ = xy_grid.shape
                weights = xy_grid.new_ones(batch_size, width * height)
            rays_idx = _safe_multinomial(weights, num_rays)[..., None].expand(-1, -1, 2)

            xy_grid = torch.gather(xy_grid.reshape(batch_size, -1, 2), 1, rays_idx)[:, :, None]

        min_depth = min_depth if min_depth is not None else self._min_depth
        max_depth = max_depth if max_depth is not None else self._max_depth
        n_pts_per_ray = n_pts_per_ray if n_pts_per_ray is not None else self._n_pts_per_ray
        stratified_sampling = stratified_sampling if stratified_sampling is not None else self._stratified_sampling

        return _xy_to_ray_bundle(
            poses,
            self._image_width,
            self._image_height,
            focal_lengths,
            xy_grid,
            min_depth,
            max_depth,
            n_pts_per_ray,
            self._unit_directions,
            stratified_sampling,
        )


def _xy_to_ray_bundle(
    poses: torch.Tensor,
    image_width: int,
    image_height: int,
    focal_lengths: torch.Tensor,
    xy_grid: torch.Tensor,
    min_depth: float,
    max_depth: float,
    n_pts_per_ray: int,
    unit_directions: bool,
    stratified_sampling: bool = False,
) -> RayBundle:
    """_summary_

    Args:
        poses  (torch.Tensor): (batch_size, 3, 4)
        image_width (int): _description_
        image_height (int): _description_
        focal_lengths (torch.Tensor): (batch_size, )
        xy_grid (torch.Tensor): (batch_size, H, W, 2)
        min_depth (float): _description_
        max_depth (float): _description_
        n_pts_per_ray (int): _description_
        unit_directions (bool): _description_
        stratified_sampling (bool, optional): _description_. Defaults to False.
    """
    batch_size = xy_grid.shape[0]
    spatial_size = xy_grid.shape[1:-1]
    n_rays_per_image = spatial_size.numel()

    rays_zs = xy_grid.new_empty((0,))
    if n_pts_per_ray > 0:
        depths = torch.linspace(
            min_depth,
            max_depth,
            n_pts_per_ray,
            dtype=xy_grid.dtype,
            device=xy_grid.device,
        )
        rays_zs = depths[None, None].expand(batch_size, n_rays_per_image, n_pts_per_ray)
        rays_zs = rays_zs.view(batch_size, *spatial_size, n_pts_per_ray)
        if stratified_sampling:
            rays_zs = _jiggle_within_stratas(rays_zs)

    origins = poses[:, :, -1].view(batch_size, *([1] * len(spatial_size)), -1).expand(batch_size, *spatial_size, -1)
    focal_lengths = focal_lengths.view(batch_size, 1, 1)
    directions = torch.stack(
        (
            (xy_grid[..., 0] - image_width * 0.5) / focal_lengths,
            (xy_grid[..., 1] - image_height * 0.5) / focal_lengths,
            xy_grid.new_ones(batch_size, *spatial_size),
        ),
        dim=-1,
    )

    return RayBundle(origins=origins, directions=directions, lengths=rays_zs, xys=xy_grid)


def _safe_multinomial(input: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Wrapper around torch.multinomial that attempts sampling without replacement
    when possible, otherwise resorts to sampling with replacement.

    Args:
        input: tensor of shape [B, n] containing non-negative values;
                rows are interpreted as unnormalized event probabilities
                in categorical distributions.
        num_samples: number of samples to take.

    Returns:
        LongTensor of shape [B, num_samples] containing
        values from {0, ..., n - 1} where the elements [i, :] of row i make
            (1) if there are num_samples or more non-zero values in input[i],
                a random subset of the indices of those values, with
                probabilities proportional to the values in input[i, :].

            (2) if not, a random sample with replacement of the indices of
                those values, with probabilities proportional to them.
                This sample might not contain all the indices of the
                non-zero values.
        Behavior undetermined if there are no non-zero values in a whole row
        or if there are negative values.
    """
    try:
        res = torch.multinomial(input, num_samples, replacement=False)
    except RuntimeError:
        # this is probably rare, so we don't mind sampling twice
        res = torch.multinomial(input, num_samples, replacement=True)
        no_repl = (input > 0.0).sum(dim=-1) >= num_samples
        res[no_repl] = torch.multinomial(input[no_repl], num_samples, replacement=False)
        return res

    # in some versions of Pytorch, zero probabilty samples can be drawn without an error
    # due to this bug: https://github.com/pytorch/pytorch/issues/50034. Handle this case:
    repl = (input > 0.0).sum(dim=-1) < num_samples
    # pyre-fixme[16]: Undefined attribute `torch.ByteTensor` has no attribute `any`.
    if repl.any():
        res[repl] = torch.multinomial(input[repl], num_samples, replacement=True)

    return res


def _jiggle_within_stratas(bin_centers: torch.Tensor) -> torch.Tensor:
    """
    Performs sampling of 1 point per bin given the bin centers.

    More specifically, it replaces each point's value `z`
    with a sample from a uniform random distribution on
    `[z - delta_−, z + delta_+]`, where `delta_−` is half of the difference
    between `z` and the previous point, and `delta_+` is half of the difference
    between the next point and `z`. For the first and last items, the
    corresponding boundary deltas are assumed zero.

    Args:
        `bin_centers`: The input points of size (..., N); the result is broadcast
            along all but the last dimension (the rows). Each row should be
            sorted in ascending order.

    Returns:
        a tensor of size (..., N) with the locations jiggled within stratas/bins.
    """
    # Get intervals between bin centers.
    mids = 0.5 * (bin_centers[..., 1:] + bin_centers[..., :-1])
    upper = torch.cat((mids, bin_centers[..., -1:]), dim=-1)
    lower = torch.cat((bin_centers[..., :1], mids), dim=-1)
    # Samples in those intervals.
    jiggled = lower + (upper - lower) * torch.rand_like(lower)
    return jiggled


def get_min_max_depth_bounds(poses, scene_center, scene_extent):
    """
    Estimate near/far depth plane as:
    near = dist(cam_center, self.scene_center) - self.scene_extent
    far  = dist(cam_center, self.scene_center) + self.scene_extent
    """
    # cam_center = cameras.get_camera_center()
    cam_center = poses[:, :, -1]  # in world coord
    center_dist = ((cam_center - (poses[:, :3, :-1]) @ scene_center) ** 2).sum(dim=-1).clamp(0.001).sqrt()
    center_dist = center_dist.clamp(scene_extent + 1e-3)
    min_depth = center_dist - scene_extent
    max_depth = center_dist + scene_extent
    return min_depth.mean().item(), max_depth.mean().item()
