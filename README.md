# yet-another-nerf

Yet another NeRF, with extensibility and scalability. Implemented in PyTorch.

## Installation

Environment installation:

```shell
conda env create -f envrionment.yml
pre-commit install
pip install -e .
```

Or use pip to install packages:

```shell
pip install torch torchvision addict yapf pytest
```

Run tests:

```shell
pytest .
```

## Data Preparation

Download and extract the zip file to `data/`.
<details>
    <summary>Dataset Links</summary>
Dataset: <https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>

blender_files.zip: <https://drive.google.com/file/d/1RjwxZCUoPlUgEWIUiuCmMmG0AhuV8A2Q/view?usp=sharing>

nerf_example_data.zip: <https://drive.google.com/file/d/1xzockqgkO-H3RCGfkZvIZNjOnk3l7AcT/view?usp=sharing>

nerf_llff_data.zip: <https://drive.google.com/file/d/16VnMcF1KJYxN9QId6TClMsZRahHNMW5g/view?usp=sharing>

nerf_real_360.zip: <https://drive.google.com/file/d/1jzggQ7IPaJJTKx9yLASWHrX8dXHnG5eB/view>

nerf_synthetic.zip: <https://drive.google.com/file/d/18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG/view?usp=sharing>

</details>

## Usage

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc-per-node=8 scripts/run.py --config $config [--output-dir $OUTPUT-DIR] [--checkpoint $CHECKPOINT-PATH] [--device $DEVICE ("cuda" or "cpu")] [--test-only] [--render-only] [--debug] {--cfg-options "xxx=yyy"}
```

## The Code Structure

<details>
  <summary> Structure of the Codebase (click to expand) </summary>

### Structure

1. pipelines/
    pipeline
    the shapes of gt_rgb & bg_rgb should both be `(B, H, W, 3)` (to be compatible with the chunkify function, and used in `renderer`)
    [TODO]: `global_codes` is coupled with through the pipeline (include pipeline, renderer, and network), but this variable is only used in network)
    loss computing: to be compatible with distributed evaluation: per-sample losses are returned, with a `torch.mean` calling in the `runner.apis`.
    1. networks/
        - ray_bundle to points: (origins, directions, lengths)
        - input dim check.
        - The networks are hard to initialized, need stochastic sampling to break the bad initialization: `pipeline.ray_sampler.stratified_point_sampling_training` (main) & `pipeline.renderer.density_noise_std_train`
    2. renderer/
        - ray_point_finer, sample_pdf
        - background_deltas / background_opacity = 1e10, and use alpha mask to blend bg_color
        - use a `dataclass` to wrap the outputs from previous stage, and recursively call the render function
        - [FIXME]: the default `bg_color` is 0.0
        - `density_noise_std`, in original paper?
        `- blend_output=False`, the foreground mask is 1, but the also use the predicted background mask
    3. ray_sampler/
        - Right-hand coordinates: x-axis points to right, y-axis points to down, z-axis points to inward
        - camera: cam2world
        - tensor shape: `(batch_size, *spatial, -1)`, `spatial` is `[height, width]` or `[n_rays_per_image, 1]`
        - `directions` are not normalized
        - The shape `poses` could both be `(..., 4, 4)` or `(..., 3, 4)`
        - Supports custom `min/max_depth` & `image_width, image_height`, `xy_grid` from `image_width, image_height` leverages `functools.lru_cache`
2. dataset/
    the shapes of gt_rgb & bg_rgb should both be `(B, H, W, 3)` (to be compatible with the chunkify function)
    the range of images should be normalized to `[0, 1]` to compatible with the sigmoid activation.
    define a `dataset_bundle: NamedTuple` in the `Dataset`; in `runner.apis` wraps the data accordingly.
3. runner/
    Multiprocess loading is on CPU.

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

</details>

## Citation and Acknowledgement

Kudos to the authors for their amazing results:

```
@misc{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    eprint={2003.08934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

Also heavily refer to the following repositories:

- nerf, nerf-pytorch (yenchenlin), nerf-pytorch (krrish94), nerf_pl, MMCV, MMDetection, Pytorch3D

<!-- However, if you find this implementation or pre-trained models helpful, please consider to cite:

```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
} -->
