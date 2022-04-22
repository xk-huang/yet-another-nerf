from enum import Enum
from typing import NamedTuple
import torch


class EvaluationMode(Enum):
    TRAINING = "training"
    EVALUATION = "evaluation"


class RenderSamplingMode(Enum):
    MASK_SAMPLE = "mask_sample"
    FULL_GRID = "full_grid"


class RayBundle(NamedTuple):
    origins: torch.Tensor
    directions: torch.Tensor
    lengths: torch.Tensor
    xys: torch.Tensor
