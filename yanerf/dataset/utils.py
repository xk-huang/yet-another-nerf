from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image  # type: ignore[import]


def load_image(path: Union[str, Path]) -> np.ndarray:
    with Image.open(path) as pil_im:
        im = np.array(pil_im.convert("RGB"))
    return im.astype(np.float32) / 255.0
