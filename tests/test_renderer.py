import os.path as osp

import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.renderers import RENDERERS
from yanerf.pipelines.renderers.utils import RendererOutput
from yanerf.pipelines.utils import EvaluationMode
from yanerf.utils.config import Config


def test_on_cuda():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    test_renderer()


def test_renderer():
    renderer_cfg = Config.fromfile(
        osp.join(osp.dirname(__file__), "configs/pipelines/renderers/multipass_emission_absorption_renderer.yml")
    )
    renderer = RENDERERS.build(renderer_cfg.renderer)

    model_cfg = Config.fromfile(osp.join(osp.dirname(__file__), "configs/pipelines/models/nerf_mlp.yml"))
    model = MODELS.build(model_cfg.model)

    B = 3
    spatial = [4, 5]
    data_shape_prefix = [B, *spatial]
    num_pts_per_ray_dim = 6
    latent_dim = model_cfg.model.latent_dim

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
        bg_color=torch.randn(*(data_shape_prefix + [3])).abs(),
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
