import json
from pathlib import Path
from typing import Callable, NamedTuple, Tuple

import cv2  # type: ignore[import]
import numpy as np
import torch
from torch.utils.data import Dataset

from yanerf.utils.logging import get_logger

from .builder import DATASETS
from .utils import load_image

logger = get_logger(__name__)


class BlenderDatasetWrapper(NamedTuple):
    poses: torch.Tensor
    focal_lengths: torch.Tensor
    image_rgb: torch.Tensor


@DATASETS.register_module()
class BlenderDataset(Dataset):
    data_wrapper: Callable = BlenderDatasetWrapper

    def __init__(self, base_dir, split, scale_down=1, debug=False):
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}.")

        self.base_dir = Path(base_dir)
        self.split = split
        with open(self.base_dir / f"transforms_{split}.json", "r") as fp:
            meta = json.load(fp)
        self.frames = meta["frames"]
        camera_angle_x = float(meta["camera_angle_x"])

        img_path = self.base_dir / f"{self.frames[0]['file_path']}.png"
        img = load_image(img_path)
        H, W = img.shape[:2]
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        if debug:
            scale_down = 32
            logger.info(f"[DEBUG] scale_down from {H}x{W} to {H//scale_down}x{W//scale_down}")

        if scale_down < 0 or not isinstance(scale_down, (float, int)):
            raise TypeError(f"Invalid type scale_down: {type(scale_down)}.")
        self.H = H // scale_down
        self.W = W // scale_down
        self.focal = focal / scale_down
        self.scale_down = scale_down

        calib_mat = np.eye(4).astype(np.float32)
        calib_mat[1, 1] = calib_mat[2, 2] = -1.0
        self.calib_mat = calib_mat

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path = self.frames[index]["file_path"]

        pose = np.array(self.frames[index]["transform_matrix"]).astype(np.float32)
        # FIX: for both camera and world coordiantes,
        # we use right-hand system, but the z-axis for the camera space is pointed inward the screen;
        # that of the world space is outward (for NeRF synthetics dataset)
        pose = pose @ self.calib_mat

        normalized_img = load_image(self.base_dir / f"{file_path}.png")
        if self.scale_down != 1:
            normalized_img = cv2.resize(normalized_img, dsize=(self.H, self.W), interpolation=cv2.INTER_LINEAR)

        return torch.from_numpy(pose), torch.FloatTensor([self.focal]), torch.from_numpy(normalized_img)

    def __len__(self):
        return len(self.frames)
