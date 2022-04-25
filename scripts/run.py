import argparse
import datetime
import json
import logging
import os.path as osp
import random
import warnings
from ensurepip import version
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import imageio
import numpy as np
import torch
import torch.distributed as dist
from imageio import imwrite
from torch.utils.data import Subset

from yanerf.dataset.builder import DATASETS
from yanerf.pipelines.builder import PIPELINES
from yanerf.runners import utils
from yanerf.runners.apis import eval_one_epoch, inference, train_one_epoch
from yanerf.runners.utils import (
    create_loader,
    create_lr_scheduler,
    create_sampler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    pause_to_debug,
)
from yanerf.utils.config import Config, DictAction
from yanerf.utils.logging import get_logger, print_log
from yanerf.utils.timer import Timer

VIS_PREFIXES = ["rendered", "image_rgb"]


def to_img(tensor_img: torch.Tensor) -> np.ndarray:
    return torch.clamp(tensor_img * 255, 0, 255).cpu().numpy().astype(np.uint8)


def get_version(path: Path):
    versions = path.parent.glob(f"{path.stem}_version_*")
    return len(list(versions))


def main(args, config):

    init_distributed_mode(args)

    # Output Directory
    if args.output_dir is not None:
        config.runner.output_dir = args.output_dir

    output_dir = Path(config.runner.output_dir)

    if output_dir.exists():
        output_dir = output_dir.parent / f"{output_dir.stem}_version_{get_version(output_dir)}"
        config.runner.output_dir = str(output_dir)

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "visualization").mkdir(parents=True, exist_ok=True)
        (output_dir / "ckpts").mkdir(parents=True, exist_ok=True)
        config.dump(osp.join(output_dir, "config.yml"))

    # Logger
    log_level = logging.DEBUG if config.runner.get("debug", False) is True else logging.INFO
    logger = get_logger(
        __name__, log_file=osp.join(config.runner.output_dir, "run.log"), log_level=log_level, file_mode="a"
    )

    if is_main_process():
        logger.info("Set up Environment.")

    rank = get_rank()
    world_size = get_world_size()

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Data: Dataset
    if is_main_process():
        logger.info("Prepare Dataset.")
    datasets = [DATASETS.build(dataset_cfg) for dataset_cfg in config.datasets]

    # Prepare Debug Data
    if config.runner.debug:
        warnings.warn("In DEBUG mode, some hyperparamters have been changed.")
        config.runner.val_per_epoch = 1
        config.runner.save_per_epoch = 1
        config.runner.batch_size_list = [3 * world_size, 3 * world_size, 3 * world_size]

        for index in (0, 1, 2):
            subset_dataset = Subset(datasets[index], list(range(config.runner.batch_size_list[index] + 1)))
            subset_dataset.data_wrapper = datasets[index].data_wrapper
            datasets[index] = subset_dataset

        config.runner.batch_size_list = [3] * 3
        config.runner.num_epochs = 2
    else:
        if is_main_process():
            logger.info("Val dataset is to large, we should re-arrange the dataset...")  # FIXME
        subset_dataset = Subset(datasets[1], list(range(0, len(datasets[1]), len(datasets[1]) // 8)))
        subset_dataset.data_wrapper = datasets[1].data_wrapper
        datasets[1] = subset_dataset

    # Data: Sampler
    samplers = [
        create_sampler(
            dataset, shuffle=True if dataset_cfg.split == "train" else False, world_size=world_size, rank=rank
        )
        for dataset, dataset_cfg in zip(datasets, config.datasets)
    ]

    # Data: DataLoader
    dataloaders = [
        create_loader(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True if dataset_cfg.split == "train" else False,
            collate_fn=None,
            pin_memory=True,
        )
        for dataset, sampler, batch_size, num_workers, dataset_cfg in zip(
            datasets, samplers, config.runner.batch_size_list, config.runner.num_workers_list, config.datasets
        )
    ]

    for i, dataloader in enumerate(dataloaders):
        if is_main_process():
            logger.info(f"Data: Length of no.{i} dataset: {len(dataloader.dataset)}, dataloader: {len(dataloader)}")
        assert len(dataloader) > 0, f"The no.{i} dataloader is empty"

    # Model
    if is_main_process():
        logger.info("Prepare Model")
    device = torch.device(args.device)
    model = PIPELINES.build(config.pipeline)
    model = model.to(device)

    if is_dist_avail_and_initialized() and config.runner.linear_scale:
        for lr in (config.runner.init_lr, config.runner.min_lr):
            if is_main_process():
                logger.info(f"Linear scale lr: from {lr} to {lr * world_size}")
        config.runner.init_lr = config.runner.init_lr * world_size
        config.runner.min_lr = config.runner.min_lr * world_size

    optimizer = torch.optim.Adam(model.parameters(), lr=config.runner.init_lr, weight_decay=config.runner.weight_decay)
    scheduler = create_lr_scheduler(optimizer, config.runner)

    # Model: Load Checkpoint
    start_epoch = 0
    if args.checkpoint:
        if is_main_process():
            logger.info("Load Checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        if is_main_process():
            logger.info(f"Resume checkpoint from {args.checkpoint}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    if args.debug:
        pause_to_debug()

    # Training
    _timer = Timer()
    if not args.test_only:
        if is_main_process():
            logger.info("Start Training.")

        train(config, logger, *dataloaders[:2], device, model, optimizer, scheduler, start_epoch, model_without_ddp)

        total_time_str = str(datetime.timedelta(seconds=int(_timer.since_last_check())))
        if is_main_process():
            logger.info(f"Training time {total_time_str}")

    # Testing
    if is_main_process():
        logger.info("Start Testing.")

    test(config, dataloaders[2], device, model)

    total_time_str = str(datetime.timedelta(seconds=int(_timer.since_last_check())))
    if is_main_process():
        logger.info(f"Testing time {total_time_str}")


def test(config, dataloader, device, model):
    test_preds, test_stats = eval_one_epoch(config.runner, -1, model, dataloader, device)
    log_stats = {
        **{f"test_{k}": v for k, v in test_stats.items()},
    }
    if is_main_process():
        with open(osp.join(config.runner.output_dir, "test_stats.json"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        vis_batch_img(config.runner.output_dir, -1, "test", test_preds)


def train(
    config,
    logger,
    train_dataloader,
    val_dataloader,
    device,
    model,
    optimizer,
    scheduler,
    start_epoch,
    model_without_ddp,
):
    _timer = Timer()
    for epoch in range(start_epoch, config.runner.num_epochs):
        _timer.since_last_check()

        train_preds, train_stats = train_one_epoch(
            config.runner, epoch, model, train_dataloader, optimizer, scheduler, device
        )

        total_time_str = str(datetime.timedelta(seconds=int(_timer.since_last_check())))
        if is_main_process():
            logger.info(f"Training One Epoch time {total_time_str}")

        _log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        if is_main_process():
            with open(osp.join(config.runner.output_dir, "train_stats.json"), "a") as f:
                f.write(json.dumps(_log_stats) + "\n")

        if (epoch + 1) % config.runner.val_per_epoch == 0:
            if is_main_process():
                logger.info(f"Start val at epoch {epoch}")

            _timer.since_last_check()

            val_preds, val_stats = eval_one_epoch(config.runner, epoch, model, val_dataloader, device)

            total_time_str = str(datetime.timedelta(seconds=int(_timer.since_last_check())))
            if is_main_process():
                logger.info(f"Validating One Epoch time {total_time_str}")

            if is_main_process():
                _log_stats = {
                    "epoch": epoch,
                    **{f"val_{k}": v for k, v in val_stats.items()},
                }
                with open(osp.join(config.runner.output_dir, "val_stats.json"), "a") as f:
                    f.write(json.dumps(_log_stats) + "\n")

                for run_type, preds in zip(("train", "val"), (train_preds, val_preds)):
                    vis_batch_img(config.runner.output_dir, epoch, run_type, preds)

        if is_main_process() and (epoch + 1) % config.runner.save_per_epoch == 0:
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            torch.save(save_obj, osp.join(config.runner.output_dir, "ckpts", "ckpts_%04d.pth" % epoch))

        if is_dist_avail_and_initialized():
            dist.barrier()


def vis_batch_img(output_dir, epoch, run_type, preds):
    for k, v in preds.items():
        if any(k.startswith(prefix) for prefix in VIS_PREFIXES):
            vis_dir = _get_vis_dir(output_dir, run_type, k)
            for batch_id, _v in enumerate(v):
                imwrite(
                    vis_dir / f"{run_type}_{k}_{epoch}_{batch_id}.png",
                    to_img(_v),
                )


@lru_cache()
def _get_vis_dir(output_dir, run_type, k):
    vis_dir = Path(output_dir) / "visualization" / f"{run_type}_{k}"
    vis_dir.mkdir(exist_ok=True, parents=True)
    return vis_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--gpu", default=None, help="No need to specify, `init_distributed_mode` takes care of it.")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    args = parser.parse_args()
    if args.device == "cpu":
        args.distributed = False

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    if args.debug:
        cfg.runner.debug = args.debug

    main(args, cfg)
