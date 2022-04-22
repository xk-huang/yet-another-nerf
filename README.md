# yet-another-nerf

yanerf

## Installation

Environment installation:

```shell
conda env create -f envrionment.yml
pre-commit install
pip install -e .
```

Pip packages:

```text
torch torchvision pytorch3d addict yapf pytest
```

## Data Preparation

Dataset: <https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>

blender_files.zip: <https://drive.google.com/file/d/1RjwxZCUoPlUgEWIUiuCmMmG0AhuV8A2Q/view?usp=sharing>

nerf_example_data.zip: <https://drive.google.com/file/d/1xzockqgkO-H3RCGfkZvIZNjOnk3l7AcT/view?usp=sharing>

nerf_llff_data.zip: <https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing>

nerf_real_360.zip: <https://drive.google.com/file/d/1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB/view>

nerf_synthetic.zip: <https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=sharing>

## The Code Structure

### Stucture of Mine

1. pipelines/
    1. networks/
        ray_bundle to points: (origins, directions, lengths)
    2. renderer/
        ray_point_finer, sample_pdf
        background_deltas / backgroud_opacity = 1e10, and use alpha mask to blend bg_color
        use a dataclass to wrap the outputs fromprevious stage, and recursively call the render function
    3. raysampler/
2. data/
3. runner/

### Strcture of nerf.pl

1. models
    1. renderer
    2. networks
2. data module
3. trainer
    1. train / eval
    2. losses
    3. metrics
    4. opt
    5. (utils) optimizer / scheduler
    6. (utils) ckpt io
    7. (utils) visualization

### Entry of implicitron

`projects/implicitron_trainer/expertiments.py`
Use logger from logging

Global args:

- exp_dir
- dataset_args / dataloader_args (both are non_leaf)

Running Pipeline

- Build exp_dir
- Get dataset & dataloader (function)
- Build model (`init_model`)
  - Take responsibility for resume so also return the training stats & optimizer_state
  - Then move to devices
- Build optimizer & scheduler from former optimizer_state
- Training loops
  - seed all
  - Record lr from lr scheduler
  - train&val `trainvalidate`
  - test  `run_eval`
  - save checkpoint
- `test_when_finish` flag for final test

Outputs:

- Checkpoints
- Stats
- Visualizations
