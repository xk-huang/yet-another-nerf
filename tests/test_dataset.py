import torch

from yanerf.dataset.blender_dataset import BlenderDataset
from yanerf.dataset.builder import DATASETS
from yanerf.runners.utils import create_loader, create_sampler, get_rank, get_world_size
from yanerf.utils.config import Config
from yanerf.utils.logging import get_logger

import os

os.makedirs("tests/tmp/", exist_ok=True)
logger = get_logger(__name__, log_file="tests/tmp/log.log")


def test_dataset():
    cfg = Config.fromfile("tests/configs/data/dataset.yml")
    cfg.dataset["debug"] = True
    print(cfg.pretty_text)
    try:
        dataset: BlenderDataset = DATASETS.build(cfg.dataset)
    except FileNotFoundError:
        return
    sampler = create_sampler(dataset, shuffle=True, rank=get_rank(), world_size=get_world_size())

    batch_size = 3
    num_workers = 2
    dataloader = create_loader(
        dataset=dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, is_train=True, collate_fn=None
    )

    blob = next(iter(dataloader))
    for data in blob:
        assert data.dtype == torch.float32
        logger.info(f"{data.shape}, {data.dtype}, {data.device}")
