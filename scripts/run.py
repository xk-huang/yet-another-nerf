import argparse
import datetime
import json
import logging
import os.path as osp
import random
from enum import Enum
from math import ceil, floor
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from yanerf.dataset.builder import DATASETS
from yanerf.pipelines.builder import PIPELINES
from yanerf.runners.apis import eval_one_epoch, train_one_epoch
from yanerf.runners.utils import (
    RunType,
    collate_only_array,
    create_loader,
    create_lr_scheduler,
    create_param_groups,
    create_sampler,
    get_rank,
    get_world_size,
    init_distributed_mode,
    is_dist_avail_and_initialized,
    is_main_process,
    pause_to_debug,
)
from yanerf.utils.config import Config, DictAction
from yanerf.utils.logging import get_logger
from yanerf.utils.timer import Timer

MONITOR_METRIC_NAME = "loss_rgb_psnr"


class MonitorMetricType(Enum):
    HIGH = "high"
    LOW = "low"


def setup_output_dir_for_training(output_dir):
    output_dir = Path(output_dir)

    if output_dir.stem.startswith("version_"):
        output_dir = output_dir.parent
    output_dir = output_dir / f"version_{get_version(output_dir)}"

    if is_dist_avail_and_initialized():
        dist.barrier()

    return output_dir


def main(args, config):
    # Initialize Distributed Environment
    init_distributed_mode(args)
    rank = get_rank()
    world_size = get_world_size()

    if not hasattr(config.runner, "seed"):
        config.runner.seed = 42
    if args.seed is not None:
        config.runner.seed = args.seed

    seed = config.runner.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True

    # Output Directory
    if args.output_dir is not None:
        config.runner.output_dir = args.output_dir

    output_dir = Path(config.runner.output_dir)
    if not args.test_only:
        # Distributed data (directory) access, must wait for sync.
        output_dir = setup_output_dir_for_training(output_dir)
        config.runner.output_dir = str(output_dir)

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        config.dump(osp.join(output_dir, "config.yml"))

        (output_dir / "visualization").mkdir(parents=True, exist_ok=True)
        if not args.test_only:
            (output_dir / "ckpts").mkdir(parents=True, exist_ok=True)

    # Logger
    log_level = logging.DEBUG if config.runner.get("debug", False) is True else logging.INFO
    logger = get_logger(
        __name__, log_file=osp.join(config.runner.output_dir, "run.log"), log_level=log_level, file_mode="a"
    )
    logger.info(f"Output Directory: {output_dir}")

    logger.info("Set up Environment.")
    logger.info(f"World Size: {world_size}")

    # Data: Dataset
    logger.info("Prepare Dataset.")
    datasets = [DATASETS.build(dataset_cfg) for dataset_cfg in config.datasets]

    # Prepare Debug Data
    if config.runner.debug:
        setup_debug_env(config.runner, datasets, logger)
    else:
        logger.warning("Val dataset might be too large, we should re-arrange the dataset...")

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
            collate_fn=collate_only_array,
            pin_memory=True,
        )
        for dataset, sampler, batch_size, num_workers, dataset_cfg in zip(
            datasets, samplers, config.runner.batch_size_list, config.runner.num_workers_list, config.datasets
        )
    ]

    for i, dataloader in enumerate(dataloaders):
        logger.info(f"Data: Length of dataset No.{i}: {len(dataloader.dataset)}, dataloader: {len(dataloader)}")
        if len(dataloader) == 0:
            raise ValueError(f"The dataloader No.{i} is empty at rank {rank}")

    # Change iter-based runner to epoch-based runner according to the dataloaders
    setup_iter_based_runner(config.runner, dataloaders[0], logger)

    # Model
    logger.info("Prepare Model")
    device = torch.device(args.device)
    model = PIPELINES.build(config.pipeline)
    model = model.to(device)

    if is_dist_avail_and_initialized() and config.runner.linear_scale:
        for lr in (config.runner.init_lr, config.runner.min_lr):
            logger.info(f"Linear scale lr: from {lr} to {lr * world_size}")
        config.runner.init_lr = config.runner.init_lr * world_size
        config.runner.min_lr = config.runner.min_lr * world_size

    param_groups = create_param_groups(model, config.runner, logger)
    optimizer = torch.optim.Adam(param_groups, lr=config.runner.init_lr, weight_decay=config.runner.weight_decay)
    scheduler = create_lr_scheduler(optimizer, config.runner)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Model: Load Checkpoint
    start_epoch = 0
    if args.checkpoint:
        logger.info("Load Checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

        logger.info(f"Resume checkpoint from: {args.checkpoint}")

    if args.debug:
        pause_to_debug(config)

    # Create Hooks
    def create_hooks(runner_config):

        from yanerf.runners.hooks import HOOKS

        if not hasattr(runner_config, "hooks"):
            hooks = []
        elif isinstance(runner_config.hooks, Dict):
            hooks = [HOOKS.build(runner_config.hooks)]
        elif isinstance(runner_config.hooks, Sequence):
            hooks = [HOOKS.build(hook) for hook in config.runner.hooks]

        logger.info(f"Hooks: {[type(hook).__name__ for hook in hooks]}")
        return hooks

    config.runner.hooks = create_hooks(config.runner)

    # Training
    if not args.test_only:
        train(
            config.runner,
            logger,
            *dataloaders[:2],
            device,
            model,
            optimizer,
            scheduler,
            start_epoch,
            model_without_ddp,
            MONITOR_METRIC_NAME,
            MonitorMetricType.HIGH,
        )

        # Finish training, try to load the best checkpoint
        if config.runner.eval_last_epoch_model is False:
            best_model_checkpoint = output_dir / "ckpts" / f"ckpts_{-1:04d}.pth"
            if best_model_checkpoint.exists():
                logger.info("Load best checkpoint")

                checkpoint = torch.load(best_model_checkpoint, map_location="cpu")
                model_without_ddp.load_state_dict(checkpoint["model"])

                logger.info(f"Best checkpoint is found: {best_model_checkpoint}.")
            else:
                logger.info(f"Best checkpoint is not found. Use the model from the last epoch.")
        else:
            logger.info("eval last epoch model")
    # Testing
    test(config.runner, dataloaders[2], device, model, logger)

    # Wait for sync, not to interfere the next job
    if is_dist_avail_and_initialized():
        dist.barrier()


def get_version(path: Path):
    versions = path.glob("version_*")
    return len(list(versions))


def setup_iter_based_runner(runner_config, dataloader: DataLoader, logger):
    iters_per_epoch = len(dataloader) * get_world_size() * dataloader.batch_size

    runner_config.num_iters_on_one_gpu = runner_config.num_iters
    runner_config.num_epochs = ceil(runner_config.num_iters / iters_per_epoch)
    runner_config.num_iters = runner_config.num_epochs * len(dataloader)

    runner_config.val_per_epoch = max(1, floor(runner_config.val_per_iter / iters_per_epoch))
    runner_config.save_per_epoch = max(1, floor(runner_config.save_per_iter / iters_per_epoch))

    logger.info("Modify iter-based runner to epoch-based runner according to the dataloaders.")
    logger.info("After modification:")

    _pair_keys = (
        ("val_per_iter", "val_per_epoch"),
        ("save_per_iter", "save_per_epoch"),
        ("num_iters_on_one_gpu", "num_iters"),
    )
    for (old_k, new_k) in _pair_keys:
        logger.info(f"\t{old_k}: {getattr(runner_config, old_k)} -> {new_k}: {getattr(runner_config, new_k)}")
    logger.info(f"\tnum_epochs: null -> {runner_config.num_epochs}")

    for key in runner_config.keys():
        if key != "num_iters" and "iters" in key:
            x_iters = runner_config[key]
            runner_config[key] = ceil(x_iters * (runner_config.num_iters / runner_config.num_iters_on_one_gpu))
            logger.info(f"\t{key}: {x_iters} -> {runner_config[key]}")

    return None


def setup_debug_env(runner_config, datasets, logger):
    logger.warning("In DEBUG mode, some hyperparamters have been changed.")

    runner_config.val_per_epoch = 1
    runner_config.save_per_epoch = 1

    for index in (0, 1, 2):
        subset_dataset = Subset(datasets[index], list(range(runner_config.batch_size_list[index] + 1)))
        subset_dataset.data_wrapper = datasets[index].data_wrapper
        datasets[index] = subset_dataset

    runner_config.num_iters = 1
    runner_config.print_per_iter = 1
    runner_config.save_per_iter = 1
    runner_config.val_per_iter = 1
    runner_config.num_workers_list = [0 for _ in runner_config.num_workers_list]


def test(runner_config, dataloader, device, model, logger):
    timer = Timer()
    logger.info("Start Testing.")

    test_stats = eval_one_epoch(RunType.TEST, runner_config, -1, model, dataloader, device)
    log_stats = {
        **{f"test_{k}": v for k, v in test_stats.items()},
    }

    if is_dist_avail_and_initialized():
        dist.barrier()

    total_time_str = str(datetime.timedelta(seconds=int(timer.since_last_check())))
    logger.info(f"Testing time: {total_time_str}")

    if is_main_process():
        with open(osp.join(runner_config.output_dir, "test_stats.json"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")


def get_compare_func(monitor_metric_type: MonitorMetricType):
    def func(base, new):
        if monitor_metric_type == MonitorMetricType.HIGH:
            return base < new
        elif monitor_metric_type == MonitorMetricType.LOW:
            return base > new
        else:
            raise ValueError(f"Invalid MonitorMetricType: {monitor_metric_type}.")

    return func


def train(
    runner_config,
    logger,
    train_dataloader,
    val_dataloader,
    device,
    model,
    optimizer,
    scheduler,
    start_epoch,
    model_without_ddp,
    monitor_metric_name,
    monitor_metric_type: MonitorMetricType,
):

    logger.info("Start Training.")
    logger.info(f"Epoch range: {start_epoch} -> {runner_config.num_epochs}")

    # Setup Monitor
    if monitor_metric_type == MonitorMetricType.HIGH:
        best_metric = -1e10
    elif monitor_metric_type == MonitorMetricType.LOW:
        best_metric = 1e10
    else:
        raise ValueError(f"Invalid MonitorMetricType: {monitor_metric_type}.")

    compare_metric = get_compare_func(monitor_metric_type)

    # Start Training
    timer = Timer()
    for epoch in range(start_epoch, runner_config.num_epochs):

        train_stats = train_one_epoch(
            RunType.TRAIN, runner_config, epoch, model, train_dataloader, optimizer, scheduler, device
        )

        log_stats = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_stats.items()},
        }
        if is_main_process():
            with open(osp.join(runner_config.output_dir, "train_stats.json"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Validationg and Save Model
        if (epoch + 1) % runner_config.val_per_epoch == 0:
            logger.info(f"Start val at epoch: {epoch}")

            timer.since_last_check()

            val_stats = eval_one_epoch(RunType.VAL, runner_config, epoch, model, val_dataloader, device)

            total_time_str = str(datetime.timedelta(seconds=int(timer.since_last_check())))

            logger.info(f"Validating One Epoch time: {total_time_str}")

            if is_main_process():
                log_stats = {
                    "epoch": epoch,
                    **{f"val_{k}": v for k, v in val_stats.items()},
                }
                with open(osp.join(runner_config.output_dir, "val_stats.json"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                current_metric = val_stats.get(monitor_metric_name, None)
                try:
                    if compare_metric(best_metric, current_metric):
                        logger.info(f"Monitor Metric: from {best_metric} -> {current_metric}.")
                        best_metric = current_metric

                        save_model(runner_config.output_dir, optimizer, model_without_ddp, -1)
                        logger.info(f"Save Best Model to Epoch: {-1}")
                except AttributeError:
                    logger.warning(f'Moniter metric name "{monitor_metric_name}" is not found in {val_stats.keys()}')

            if is_dist_avail_and_initialized():
                dist.barrier()

        if is_main_process() and (epoch + 1) % runner_config.save_per_epoch == 0:
            save_model(runner_config.output_dir, optimizer, model_without_ddp, epoch)
            logger.info(f"Save Model at Epoch: {epoch}")

        if is_dist_avail_and_initialized():
            dist.barrier()

    if is_main_process():
        total_time_str = str(datetime.timedelta(seconds=int(timer.since_last_check())))
        logger.info(f"Training time: {total_time_str}")

        save_model(runner_config.output_dir, optimizer, model_without_ddp, runner_config.num_epochs - 1)


def save_model(output_dir, optimizer, model_without_ddp, epoch):
    save_obj = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(save_obj, osp.join(output_dir, "ckpts", f"ckpts_{epoch:04d}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # DDP configs
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--gpu", default=None, help="No need to specify, `init_distributed_mode` takes care of it.")

    # Script configs
    parser.add_argument("--config", default="./configs/pretrain.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--cfg_options",
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
