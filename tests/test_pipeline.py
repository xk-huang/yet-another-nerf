import os.path as osp
import warnings
from pathlib import Path

import imageio
import torch

from yanerf.pipelines.builder import PIPELINES
from yanerf.pipelines.models.zero_outputer import ZeroOutputer
from yanerf.pipelines.nerf_pipeline import NeRFPipeline
from yanerf.pipelines.ray_samplers.utils import get_xy_grid
from yanerf.pipelines.utils import EvaluationMode, sample_grid
from yanerf.utils.config import Config


def test_sample_grid():
    B = 10
    H, W, C = 7, 4, 12
    image = torch.randn(B, H, W, C)
    grid = get_xy_grid(H, W)[None].expand(B, -1, -1, -1)
    sampled_image = sample_grid(image, grid)
    assert torch.allclose(image, sampled_image)

    mask = torch.randint(0, 2, (H, W)).bool()
    mask = torch.stack([mask] * B)
    masked_grid = grid[mask].view(B, -1, 2)[:, :, None, :]
    masked_image = image[mask].view(B, -1, C)[:, :, None, :]
    sampled_masked_image = sample_grid(image, masked_grid)
    assert torch.allclose(masked_image, sampled_masked_image)


def test_on_cuda():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    test_pipeline()


def test_pipeline_global_codes():
    pipeline_cfg = Config.fromfile(osp.join("tests/configs/pipelines/nerf_pipeline_cfg_with_conditional_mlp.py"))
    print(pipeline_cfg.filename)
    pipeline_cfg.pipeline.renderer.blend_output = True
    pipeline_cfg.pipeline.renderer.density_noise_std_train = 0.0
    print(pipeline_cfg.pretty_text)

    pipeline = PIPELINES.build(pipeline_cfg.pipeline)

    save_dir = Path("tests/tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    B = 3
    poses = torch.randn(B, 3, 4)
    focal_lengths = torch.ones(B) * 500

    bg_image_rgb = imageio.imread("tests/data/image.png")
    bg_image_rgb = torch.Tensor(bg_image_rgb)[None, ..., :3].expand(B, -1, -1, -1)
    imageio.imwrite(save_dir / "image.png", bg_image_rgb.cpu().numpy()[0, ..., :3])
    bg_image_rgb = bg_image_rgb.float() / 255.0
    global_codes = torch.randn(B, pipeline_cfg.pipeline.model.latent_dim)
    _ = pipeline(
        poses=poses,
        focal_lengths=focal_lengths,
        bg_image_rgb=bg_image_rgb,
        evaluation_mode=EvaluationMode.TRAINING,
        global_codes=global_codes,
    )


def test_pipeline():
    pipeline_cfg = Config.fromfile(osp.join("tests/configs/pipelines/nerf_pipeline_cfg_with_zero_outputer.py"))
    print(pipeline_cfg.filename)
    pipeline_cfg.pipeline.renderer.blend_output = True
    pipeline_cfg.pipeline.renderer.density_noise_std_train = 0.0
    print(pipeline_cfg.pretty_text)

    pipeline = PIPELINES.build(pipeline_cfg.pipeline)

    save_dir = Path("tests/tmp")
    save_dir.mkdir(parents=True, exist_ok=True)

    B = 3
    poses = torch.randn(B, 3, 4)
    focal_lengths = torch.ones(B) * 500

    bg_image_rgb = imageio.imread("tests/data/image.png")
    bg_image_rgb = torch.Tensor(bg_image_rgb)[None, ..., :3].expand(B, -1, -1, -1)
    imageio.imwrite(save_dir / "image.png", bg_image_rgb.cpu().numpy()[0, ..., :3])
    bg_image_rgb = bg_image_rgb.float() / 255.0
    train_preds = pipeline(
        poses=poses, focal_lengths=focal_lengths, bg_image_rgb=bg_image_rgb, evaluation_mode=EvaluationMode.TRAINING
    )

    SAVED_KEYS = ("rendered_images", "rendered_depths", "rendered_alpha_masks")
    if SAVED_KEYS[0] in train_preds:
        for k in SAVED_KEYS:
            out = (train_preds[k][0].cpu().numpy() * 255).astype("uint8")
            imageio.imwrite(save_dir / f"{k}.train.png", out)
    with torch.no_grad():
        eval_preds = pipeline(
            poses=poses,
            focal_lengths=focal_lengths,
            bg_image_rgb=bg_image_rgb,
            evaluation_mode=EvaluationMode.EVALUATION,
        )

    SAVED_KEYS = ("rendered_images", "rendered_depths", "rendered_alpha_masks")
    if SAVED_KEYS[0] in eval_preds:

        for k in SAVED_KEYS:
            out = (eval_preds[k][0].cpu().numpy() * 255).astype("uint8")
            imageio.imwrite(save_dir / f"{k}.eval.png", out)

    check_numerical = isinstance(pipeline, NeRFPipeline) and (
        isinstance(pipeline.implicit_functions, ZeroOutputer)
        or isinstance(pipeline.implicit_functions[0], ZeroOutputer)
        or (isinstance(pipeline.implicit_functions[0]._fn, ZeroOutputer))
    )

    if check_numerical:
        for preds in (train_preds, eval_preds):
            assert torch.allclose(
                preds["rendered_images"],
                (preds["sampled_grids"] if preds.get("sampled_grids", None) is not None else 1.0) * bg_image_rgb,
            )
    else:
        warnings.warn(f"{__file__}: not check the nemerical consistency of bg color injection")

    _H, _W = 2, 4
    _bg_image_rgb = torch.randn(B, _H, _W, 3)
    train_preds = pipeline(
        poses=poses,
        focal_lengths=focal_lengths,
        bg_image_rgb=_bg_image_rgb,
        image_rgb=_bg_image_rgb,
        evaluation_mode=EvaluationMode.EVALUATION,
        image_width=_W,
        image_height=_H,
    )
    eval_preds = pipeline(
        poses=poses,
        focal_lengths=focal_lengths,
        bg_image_rgb=_bg_image_rgb,
        image_rgb=_bg_image_rgb,
        evaluation_mode=EvaluationMode.EVALUATION,
        image_width=_W,
        image_height=_H,
    )
    if check_numerical:
        for preds in (train_preds, eval_preds):
            assert torch.allclose(preds["objective"], torch.zeros(1))
            assert torch.all(preds["rendered_images"] == _bg_image_rgb)
