import logging
import os.path as osp
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from yanerf.pipelines.utils import EvaluationMode
from yanerf.utils.config import ConfigDict
from yanerf.utils.logging import get_logger
from yanerf.utils.timer import Timer

from .hooks import EvalDataHook, EvalOutputsHook, TrainDataHook, TrainOutputsHook
from .utils import (
    RunType,
    concat_all_gather,
    create_stats,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    vis_batch_img,
    warmup_lr_scheduler,
)

LOG_HEADER = "{}\tEpoch:\t[{}]"


def train_one_epoch(
    run_type: RunType,
    config: Dict,
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Callable[..., None],
    device: torch.device = torch.device("cuda"),
):
    logger = _get_logger(config)

    passed_iter = epoch * len(dataloader)
    header = LOG_HEADER.format(run_type.value, epoch)
    print_per_iter = config.get("print_per_iter", 100)

    model.train()
    sampler: Optional[DistributedSampler] = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)

    timer = Timer()
    data: Tuple[torch.Tensor, ...]
    for i, data in enumerate(dataloader):
        _times = {}
        data = dataloader.dataset.data_wrapper(
            *((_data.to(device, non_blocking=True) if isinstance(_data, torch.Tensor) else _data) for _data in data)
        )._asdict()

        # Run DataHook
        if hasattr(config, "hooks"):
            for hook in config.hooks:
                if isinstance(hook, TrainDataHook):
                    data = hook(data=data, iter=passed_iter, epoch=epoch, config=config)
        _times["data"] = timer.since_last_check()

        scheduler(iter=passed_iter)
        if config["warmup_steps"] > 0 and passed_iter <= config["warmup_steps"]:
            warmup_lr_scheduler(optimizer, passed_iter, config["warmup_steps"], config["warmup_lr"])

        optimizer.zero_grad()

        timer.since_last_check()
        preds = inference(
            model=model,
            data=data,
            evaluation_mode=EvaluationMode.TRAINING,
            compute_metrics=True,
        )
        # Run OutputsHook
        if hasattr(config, "hooks"):
            for hook in config.hooks:
                if isinstance(hook, TrainOutputsHook):
                    preds = hook(outputs=preds, config=config, iter=passed_iter, epoch=epoch)
        _times["inference"] = timer.since_last_check()

        try:
            loss = preds["objective"].mean()
            loss.backward()
            optimizer.step()
        except KeyError:
            raise KeyError("In train mode, but no loss (`objective`) is found.")

        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 0
        if passed_iter % print_per_iter == 0:
            lr_string = ", ".join([f"{param_group['lr']:3e}" for param_group in optimizer.param_groups])
            logger.info(f"{header}\tlr: {lr_string}.")

            stats = create_stats(preds)
            log_string = "\t".join(
                [f"iter: {passed_iter}\tsampler: [{i * batch_size}/{len(dataloader) * batch_size}]"]
                + [f"{k}: {v:.3f}" for k, v in _times.items()]
                + [f"{k}: {v:.3f}" for k, v in stats.items()]
            )
            logger.info(f"{header}: {log_string}")

        if passed_iter % config.val_per_iter == 0:
            logger.info(f"save training image to check sanity.")
            vis_batch_img(
                preds,
                run_type,
                config.output_dir,
                0,
                dataloader.batch_size,
                f"{epoch:05d}/",
            )

        passed_iter += 1
        timer.since_last_check()

    return create_stats(preds)


@torch.no_grad()
def eval_one_epoch(
    run_type: RunType,
    config: ConfigDict,
    epoch: int,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cuda"),
    save_image: bool = True,
):
    if dataloader.drop_last is True:
        raise ValueError("Imcomplete eval due to `drop_last`.")

    logger = _get_logger(config)
    print_per_iter = config.get("print_per_iter", 50)
    header = LOG_HEADER.format(run_type.value, epoch)
    rank = get_rank()
    world_size = get_world_size()

    model.eval()
    timer = Timer()
    # every time you access the `defaultdict`, it creat an empty default object
    metric_stats: defaultdict[str, Union[List, torch.Tensor]] = defaultdict(list)
    for i, data in enumerate(dataloader):
        _times = {}
        data = dataloader.dataset.data_wrapper(
            *((_data.to(device, non_blocking=True) if isinstance(_data, torch.Tensor) else _data) for _data in data)
        )._asdict()

        # Run DataHook
        if hasattr(config, "hooks"):
            for hook in config.hooks:
                if isinstance(hook, EvalDataHook):
                    data = hook(data=data, config=config, iter=i, epoch=epoch)
        _times["data"] = timer.since_last_check()

        timer.since_last_check()
        preds = inference(
            model=model,
            data=data,
            evaluation_mode=EvaluationMode.EVALUATION,
            compute_metrics=True,
        )
        # Run OutputsHook
        if hasattr(config, "hooks"):
            for hook in config.hooks:
                if isinstance(hook, EvalOutputsHook):
                    preds = hook(outputs=preds, config=config, iter=i, epoch=epoch)
        _times["inference"] = timer.since_last_check()

        for k, v in preds.items():
            if k.startswith("loss_") or k.startswith("objective"):
                metric_stats[k].append(  # type: ignore[union-attr]
                    concat_all_gather(v) if is_dist_avail_and_initialized() else v
                )

        batch_size = dataloader.batch_size if dataloader.batch_size is not None else 0
        if i % print_per_iter == 0:
            _stats = create_stats(preds)
            log_string = "\t".join(
                [f"sampler: [{i * batch_size}/{len(dataloader.dataset)}]"]
                + [f"{k}: {v:.3f}" for k, v in _times.items()]
                + [f"{k}: {v:.3f}" for k, v in _stats.items()]
            )
            logger.info(f"{header}: {log_string}")

        if save_image:
            start_idx = (i * world_size + rank) * batch_size
            end_idx = min(len(dataloader.dataset), start_idx + batch_size)
            vis_batch_img(
                preds,
                run_type,
                config.output_dir,
                start_idx,
                end_idx,
                "" if run_type == RunType.TEST else f"{epoch:05d}/",
            )
        timer.since_last_check()

    for k, v in metric_stats.items():
        metric_stats[k] = torch.mean(torch.concat(v, dim=0)[: len(dataloader.dataset)])

    preds.update(metric_stats)
    stats = create_stats(preds)
    log_string = "\t".join(
        [f"[{len(dataloader.dataset)}/{len(dataloader.dataset)}]"] + [f"{k}: {v:.3f}" for k, v in stats.items()]
    )
    logger.info(f"{header}: {log_string}")

    return stats


def _get_logger(config):
    log_level = logging.DEBUG if config.get("debug", None) else logging.INFO
    logger = get_logger(__name__, log_file=osp.join(config.output_dir, "run.log"), log_level=log_level, file_mode="a")
    return logger


def inference(
    model: torch.nn.Module,
    data: DictConfig,
    evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
    compute_metrics: bool = True,
):
    if compute_metrics is False and data.get("image_rgb", None) is not None:
        data.pop("image_rgb")

    preds: Dict = model(
        **data,
        evaluation_mode=evaluation_mode,
    )
    preds.update(data)

    return preds
