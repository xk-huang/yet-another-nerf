from enum import Enum
from typing import NamedTuple
import torch
from yanerf.pipelines.utils import RayBundle, EvaluationMode


class RenderSamplingMode(Enum):
    MASK_SAMPLE = "mask_sample"
    FULL_GRID = "full_grid"


def get_xy_grid(image_height, image_width):
    return torch.stack(
        tuple(
            reversed(
                torch.meshgrid(
                    torch.linspace(0, image_height - 1, image_height, dtype=torch.float32),
                    torch.linspace(0, image_width - 1, image_width, dtype=torch.float32),
                    indexing="ij",
                )
            )
        ),
        dim=-1,
    )
