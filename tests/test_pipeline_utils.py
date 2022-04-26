import torch

from yanerf.pipelines.ray_samplers.ray_sampler import _safe_multinomial
from yanerf.pipelines.ray_samplers.utils import get_xy_grid
from yanerf.pipelines.utils import sample_grid, scatter_rays_to_image


def test_grid():
    B, H, W, _ = 2, 6, 10, 1
    num_samples = 4
    # tensor = torch.(B, H, W, C)
    torch.manual_seed(0)

    xys = get_xy_grid(H, W)[None].expand(B, -1, -1, -1)
    tensor = (xys[..., 0] + xys[..., 1] * W)[..., None]
    _, width, height, _ = xys.shape
    weights = xys.new_ones(B, width * height)
    rays_idx = _safe_multinomial(weights, num_samples)[..., None].expand(-1, -1, 2)

    sampled_xys = torch.gather(xys.reshape(B, -1, 2), 1, rays_idx)[:, :, None]
    sampled_tensor = sample_grid(tensor, sampled_xys)

    _ = scatter_rays_to_image(sampled_tensor, sampled_xys, H, W, torch.Tensor([0, 0, 0]))
