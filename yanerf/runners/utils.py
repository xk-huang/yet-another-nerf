import math
import os
from ast import Str
from enum import Enum
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from imageio import imwrite  # type: ignore[import]
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler


class RunType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def to_img(tensor_img: torch.Tensor) -> np.ndarray:
    return torch.clamp(tensor_img * 255, 0, 255).cpu().numpy().astype(np.uint8)


def vis_batch_img(
    preds,
    run_type: RunType,
    output_dir: Union[Path, Str],
    output_start_idx: int,
    output_end_idx: int,
    file_name_prefix: str = "",
    file_name_ext: str = ".png",
    render_prefixes: List[str] = ["rendered_", "image_rgb_"],
):
    file_name_template = file_name_prefix + "{:05d}" + file_name_ext
    for rendered_type, renders in preds.items():
        if any(rendered_type.startswith(prefix) for prefix in render_prefixes):
            _output_end_idx = output_start_idx + min(output_end_idx - output_start_idx, len(renders))
            vis_dir = _get_vis_dir(output_dir, run_type, rendered_type)
            for batch_idx, file_name_idx in enumerate(range(output_start_idx, _output_end_idx)):
                imwrite(vis_dir / file_name_template.format(file_name_idx), to_img(renders[batch_idx]))


@lru_cache()
def _get_vis_dir(output_dir, run_type: RunType, rendered_type):
    vis_dir = Path(output_dir) / "visualization" / run_type.value / rendered_type
    vis_dir.mkdir(exist_ok=True, parents=True)
    return vis_dir


def warmup_lr_scheduler(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def cosine_lr_scheduler(optimizer, iter, lr_decay_iter_interval, init_lr, min_lr, num_iters):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * (iter / lr_decay_iter_interval) / num_iters)) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_scheduler(optimizer, iter, lr_decay_iter_interval, init_lr, min_lr, lr_decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (lr_decay_rate ** (iter / lr_decay_iter_interval)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_lr_scheduler(optimizer, config):
    if config.lr_decay_type == "exponential":
        sched = partial(
            step_lr_scheduler,
            optimizer=optimizer,
            lr_decay_iter_interval=config["lr_decay_iter_interval"],
            init_lr=config["init_lr"],
            min_lr=config["min_lr"],
            lr_decay_rate=config["lr_decay_rate"],
        )
    elif config.lr_decay_type == "cosine":
        sched = partial(
            cosine_lr_scheduler,
            optimizer=optimizer,
            lr_decay_iter_interval=config["lr_decay_iter_interval"],
            init_lr=config["init_lr"],
            min_lr=config["min_lr"],
            num_iters=config["num_iters"],
        )
    else:
        raise ValueError

    return sched


def create_sampler(dataset: Dataset, shuffle: bool, world_size: int, rank: int):
    if is_dist_avail_and_initialized():
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
    else:
        return None


def create_loader(
    dataset: Dataset,
    sampler: Sampler,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = True,
):

    if is_train:
        shuffle = sampler is None
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}, word {args.world_size}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def create_stats(preds, prefixes=["loss_", "objective"]):
    return {k: v.mean().item() for k, v in preds.items() if any([k.startswith(prefix) for prefix in prefixes])}


def pause_to_debug(config):
    if is_main_process():
        from IPython.core.debugger import set_trace  # type: ignore[import]

        set_trace()

    if is_dist_avail_and_initialized():
        dist.barrier()
