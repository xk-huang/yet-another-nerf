import os.path as osp

import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.models.nerf_mlp import ModelOutputs
from yanerf.utils.config import Config


def test_NeRFMLP():
    cfg = Config.fromfile(osp.join(osp.dirname(__file__), "configs/pipelines/models/nerf_mlp.yml"))
    model = MODELS.build(cfg.model)

    data_shape_prefix = [3, 5, 5]
    xyz_dims = 3
    num_pts_per_ray_dim = 6
    data = (
        torch.randn(*(data_shape_prefix + [xyz_dims])),
        torch.randn(*(data_shape_prefix + [xyz_dims])),
        torch.randn(*(data_shape_prefix + [num_pts_per_ray_dim])),
        None,
    )

    print(model)
    outs: ModelOutputs = model(*data)
    print(outs._fields)
    print([i.shape for i in outs])
