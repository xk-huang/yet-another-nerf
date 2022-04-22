import os.path as osp

import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.renderers import RENDERERS
from yanerf.pipelines.renderers.utils import EvaluationMode, RendererOutput
from yanerf.utils.config import Config


def test_NeRFMLP():
    renderer_cfg = Config.fromfile(
        osp.join(osp.dirname(__file__), "configs/pipelines/renderers/multipass_emission_absorption_renderer.yml")
    )
    renderer = RENDERERS.build(renderer_cfg.renderer)

    model_cfg = Config.fromfile(osp.join(osp.dirname(__file__), "configs/pipelines/models/nerf_mlp.yml"))
    model = MODELS.build(model_cfg.model)

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
    print(renderer)

    renderer_output: RendererOutput = renderer(
        **data, implicit_functions=[model, model], evaluation_mode=EvaluationMode.TRAINING
    )
    print_renderer_output(renderer_output, 0)

    renderer_output: RendererOutput = renderer(
        **data, implicit_functions=[model, model], evaluation_mode=EvaluationMode.EVALUATION
    )
    print_renderer_output(renderer_output, 0)


def print_renderer_output(renderer_output: RendererOutput, level=0):
    if renderer_output is None:
        return

    string_prefix = "  " * level
    print(f"level {level}:")
    print(f"{string_prefix}{renderer_output.__dict__.keys()}")
    print(
        f"{string_prefix}\
            {[f'{k}: {v.shape}' for k, v in renderer_output.__dict__.items() if isinstance(v, torch.Tensor)]}"
    )

    print_renderer_output(renderer_output.prev_stage, level + 1)
