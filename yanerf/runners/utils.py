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
    return torch.clamp(tensor_img * 255, 0, 255).detach().cpu().numpy().astype(np.uint8)


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
    if file_name_prefix.endswith("/"):
        prefix = file_name_prefix
        file_name_prefix = ""
    else:
        prefix = None
    file_name_template = file_name_prefix + "{:05d}" + file_name_ext
    for rendered_type, renders in preds.items():
        if any(rendered_type.startswith(prefix) for prefix in render_prefixes):
            if "depths" in rendered_type or "alpha_masks" in rendered_type:
                batch_size = renders.shape[0]
                num_dims = len(renders.shape)
                renders = renders / torch.max(renders.view(batch_size, -1), dim=1)[0].view(-1, *([1] * (num_dims - 1)))

            _output_end_idx = output_start_idx + min(output_end_idx - output_start_idx, len(renders))
            vis_dir = _get_vis_dir(output_dir, run_type, rendered_type, prefix)
            for batch_idx, file_name_idx in enumerate(range(output_start_idx, _output_end_idx)):

                imwrite(vis_dir / file_name_template.format(file_name_idx), to_img(renders[batch_idx]))


@lru_cache()
def _get_vis_dir(output_dir, run_type: RunType, rendered_type, prefix=None):
    vis_dir = Path(output_dir) / "visualization" / run_type.value / rendered_type
    if prefix is not None:
        vis_dir = vis_dir / prefix
    vis_dir.mkdir(exist_ok=True, parents=True)
    return vis_dir


def warmup_lr_scheduler(optimizer, step, max_step, warmup_lr):
    """Warmup the learning rate"""
    for param_group in optimizer.param_groups:
        init_lr = param_group["init_lr"]
        lr = min(init_lr, warmup_lr + (init_lr - warmup_lr) * step / max_step)
        param_group["lr"] = lr


def cosine_lr_scheduler(optimizer, iter, lr_decay_iters, min_lr, num_iters):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        lr = (param_group["init_lr"] - min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (iter / lr_decay_iters) / num_iters)
        ) + min_lr
        param_group["lr"] = lr


def step_lr_scheduler(optimizer, iter, lr_decay_iters, min_lr, lr_decay_rate):
    """Decay the learning rate"""
    for param_group in optimizer.param_groups:
        lr = max(min_lr, param_group["init_lr"] * (lr_decay_rate ** (iter / lr_decay_iters)))
        param_group["lr"] = lr


def create_lr_scheduler(optimizer, config):
    if config.lr_decay_type == "exponential":
        sched = partial(
            step_lr_scheduler,
            optimizer=optimizer,
            lr_decay_iters=config["lr_decay_iters"],
            min_lr=config["min_lr"],
            lr_decay_rate=config["lr_decay_rate"],
        )
    elif config.lr_decay_type == "cosine":
        sched = partial(
            cosine_lr_scheduler,
            optimizer=optimizer,
            lr_decay_iters=config["lr_decay_iters"],
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


def create_param_groups(model, runner_config, logger):
    if not hasattr(runner_config, "lr_param_groups"):
        logger.info(f"all params: {len(tuple(model.parameters()))}, lr: {runner_config.init_lr}")
        return [{"params": model.parameters(), "init_lr": runner_config.init_lr}]

    logger.warning("filtered params: this step may be bottleneck due to large amount of parameters.")

    def filter_name(prefixes, name):
        for i, prefix in enumerate(prefixes):
            if name.startswith(prefix):
                return i
        else:
            return -1

    prefixes = [param_group_config.prefix for param_group_config in runner_config.lr_param_groups]
    prefixes.append("")
    params_list = [[] for _ in range(len(prefixes))]

    num_params = 0
    for name, param in model.named_parameters():
        params_list_idx = filter_name(prefixes, name)
        params_list[params_list_idx].append(param)
        num_params += 1

    init_lr = runner_config.init_lr
    init_lrs = [param_group_config.base * init_lr for param_group_config in runner_config.lr_param_groups]
    init_lrs.append(init_lr)
    assert len(init_lrs) == len(params_list)

    logger.info("filtered params:")
    param_group_ls = []
    for prefix, params, lr in zip(prefixes, params_list, init_lrs):
        logger.info(f"\tprefix: {prefix},\tparams: {len(params)},\tlr: {lr}")
        param_group_ls.append({"params": iter(params), "lr": lr, "init_lr": lr})

    logger.info(f"all params: {num_params}")
    return param_group_ls


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


def mse2psnr(mse, base=1.0):
    return np.log10(max(1e-10, mse)) * (-10.0) + 20.0 * np.log10(base)


def create_stats(preds, prefixes=["loss_", "objective"]):
    stats = {}
    for k, v in preds.items():
        if any([k.startswith(prefix) for prefix in prefixes]):
            stats[k] = v.mean().item()

            if "mse" in k:
                psnr_name = "psnr".join(k.split("mse"))
                stats[psnr_name] = mse2psnr(stats[k])
    return stats


def pause_to_debug(config):
    if is_main_process():
        from IPython.core.debugger import set_trace  # type: ignore[import]

        set_trace()

    if is_dist_avail_and_initialized():
        dist.barrier()


import collections

from torch._six import string_classes
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern


def collate_only_array(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return collate_only_array([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return np.mean(batch)
    elif isinstance(elem, int):
        return batch[0]
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_only_array([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_only_array(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [collate_only_array(samples) for samples in transposed]
    elif elem is None:
        return None
    raise TypeError(default_collate_err_msg_format.format(elem_type))
