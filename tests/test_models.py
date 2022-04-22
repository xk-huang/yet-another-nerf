import os.path as osp

import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.models.nerf_mlp import ModelOutputs
from yanerf.utils.config import Config


def test_NeRFMLP():
    cfg = Config.fromfile(osp.join(osp.dirname(__file__), "configs/pipelines/models/nerf_mlp.yml"))
    model = MODELS.build(cfg.model)

    data_shape_prefix = [3, 5, 5]
    num_pts_per_ray_dim = 6
    data = dict(
        origins=torch.randn(*(data_shape_prefix + [3])).abs(),
        directions=torch.randn(*(data_shape_prefix + [3])).abs(),
        lengths=torch.randn(*(data_shape_prefix + [num_pts_per_ray_dim])).abs().sort(dim=-1)[0],
        xys=torch.randn(*(data_shape_prefix + [2])),
        global_codes=None,
    )

    print(model)
    outs: ModelOutputs = model(**data)
    print(outs._fields)
    print([i.shape for i in outs if isinstance(i, torch.Tensor)])
