# flake8: noqa
import os.path as osp
import pathlib

from yanerf.utils.config import Config
from yanerf.utils.registry import Registry

TRAINER = Registry("Trainer")


@TRAINER.register_module()
class MyTrainer:
    def __init__(self, lr: int, epochs: int) -> None:
        self.lr = lr
        self.epochs = epochs


def test_builder():
    cfg = Config.fromfile(osp.join(osp.dirname(__file__), "configs/test_utils_config.yml"))
    trainer = TRAINER.build(cfg.trainer)
    print(f"\nMY CFG: {cfg.pretty_text}")
    print(trainer.__dict__)
