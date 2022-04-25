import logging
import os.path as osp
import random
from pathlib import Path
from typing import NamedTuple

import imageio
import numpy as np
import torch

from yanerf.pipelines.builder import PIPELINES
from yanerf.runners.apis import eval_one_epoch, train_one_epoch
from yanerf.runners.utils import create_loader, create_lr_scheduler
from yanerf.utils.config import Config
from yanerf.utils.logging import get_logger


class DummyDatasetWrapper(NamedTuple):
    poses: torch.Tensor
    focal_lengths: torch.Tensor
    image_rgb: torch.Tensor


class DummyDataset:
    data_wrapper = DummyDatasetWrapper

    def __init__(self, *args) -> None:
        self.data_loader_ls = args

    def __len__(self):
        return len(self.data_loader_ls[0])

    def __getitem__(self, index):
        return tuple(data_loader_ls[index] for data_loader_ls in self.data_loader_ls)


def test_on_cuda():
    torch.set_default_tensor_type("torch.FloatTensor")
    test_runner_simple(True)


def test_runner_simple(use_cuda=False):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    H, W = 2, 2
    pipeline_cfg = Config.fromfile(osp.join("tests/configs/pipelines/nerf_pipeline_cfg_with_mlp.py"))
    pipeline_cfg.pipeline.ray_sampler.image_height = H
    pipeline_cfg.pipeline.ray_sampler.image_width = W
    print(pipeline_cfg.filename)
    print(pipeline_cfg.pretty_text)
    pipeline = PIPELINES.build(pipeline_cfg.pipeline)

    runner_cfg = Config.fromfile("tests/configs/runner/runner.yml")
    print(runner_cfg.pretty_text)
    log_level = logging.DEBUG if runner_cfg.runner.get("debug", None) else logging.INFO
    logger = get_logger(
        __name__, log_file=osp.join(runner_cfg.runner.output_dir, "run.log"), log_level=log_level, file_mode="a"
    )

    output_dir = Path("tests/tmp/simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_cfg.output_dir = output_dir

    B = 3
    poses = torch.cat([torch.eye(3), torch.FloatTensor([[0], [0], [-1]])], dim=-1)
    poses = poses[None].expand(B, 3, 4)
    focal_lengths = torch.ones(B)

    image_rgb = (torch.randn(H, W, 3).abs() * 255).to(dtype=torch.uint8).numpy()
    image_rgb = torch.Tensor(image_rgb)[None, ..., :3].expand(B, -1, -1, -1)
    imageio.imwrite(output_dir / "image.png", image_rgb.cpu().numpy()[0, ..., :3])
    image_rgb = image_rgb.float() / 255.0

    data = [poses, focal_lengths, image_rgb]
    dataloader = create_loader(DummyDataset(*data), None, runner_cfg.runner.batch_size_train, 0, True, None, False)

    optimizer = torch.optim.Adam(pipeline.parameters(), lr=runner_cfg.runner.init_lr)
    scheduler = create_lr_scheduler(optimizer, runner_cfg.runner)
    if use_cuda:
        pipeline = pipeline.cuda()
    device = torch.device("cuda" if use_cuda else "cpu")
    for epoch in range(runner_cfg.runner.num_epochs):
        preds, _ = train_one_epoch(
            runner_cfg.runner,
            epoch,
            pipeline,
            dataloader,
            optimizer,
            scheduler,
            device,
        )
    SAVED_KEYS = ("rendered_images", "rendered_depths", "rendered_alpha_masks", "sampled_grids")
    if SAVED_KEYS[0] in preds:
        for k in SAVED_KEYS:
            out = (preds[k][0].detach().cpu().numpy() * 255).astype("uint8")
            imageio.imwrite(output_dir / f"{k}.train.runner.png", out)
    dataloader = create_loader(DummyDataset(*data), None, runner_cfg.runner.batch_size_eval, 0, False, None, False)
    preds, _ = eval_one_epoch(runner_cfg.runner, runner_cfg.runner.num_epochs - 1, pipeline, dataloader, device)
    logger.debug(f"RenderOuputs keys: {preds.keys()}")
    assert torch.all(preds["objective"] < 0.01)
