import pytest
import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.models.nerf_mlp import ModelOutputs
from yanerf.utils.config import Config

cfg_paths = [
    "tests/configs/pipelines/models/nerf_mlp.yml",
    "tests/configs/pipelines/models/nerf_mlp_without_global_code.yml",
]


@pytest.mark.parametrize("cfg_path", cfg_paths)
def test_on_cuda(cfg_path):
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    test_NeRFMLP(cfg_path)


@pytest.mark.parametrize("cfg_path", cfg_paths)
def test_NeRFMLP(cfg_path):
    cfg = Config.fromfile(cfg_path)
    print(cfg.pretty_text)
    model = MODELS.build(cfg.model)
    print(model)

    B = 3
    spatial = [4, 5]
    data_shape_prefix = [B, *spatial]
    num_pts_per_ray_dim = 2

    color_dim = cfg.model.color_dim
    latent_dim = cfg.model.latent_dim

    if latent_dim > 0:
        global_codes = torch.randn(B, latent_dim)
    else:
        global_codes = None

    data = dict(
        origins=torch.randn(*(data_shape_prefix + [3])).abs(),
        directions=torch.randn(*(data_shape_prefix + [3])).abs(),
        lengths=torch.randn(*(data_shape_prefix + [num_pts_per_ray_dim])).abs().sort(dim=-1)[0],
        xys=torch.randn(*(data_shape_prefix + [2])),
        global_codes=global_codes,
    )

    outs: ModelOutputs = model(**data)
    print(outs._fields)
    print([i.shape for i in outs if isinstance(i, torch.Tensor)])

    check_pair = (
        (list(outs[0].shape), [B, *spatial, num_pts_per_ray_dim, 1]),
        (list(outs[1].shape), [B, *spatial, num_pts_per_ray_dim, color_dim]),
    )
    for a, b in check_pair:
        assert a == b
