import collections
import dataclasses
import math
import warnings
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import torch

from yanerf.pipelines.models import MODELS
from yanerf.pipelines.ray_samplers import RAY_SAMPLERS
from yanerf.pipelines.ray_samplers.utils import RayBundle, RenderSamplingMode
from yanerf.pipelines.renderers import RENDERERS
from yanerf.pipelines.renderers.utils import RendererOutput
from yanerf.pipelines.utils import EvaluationMode
from yanerf.utils.config import ConfigDict
from yanerf.utils.logging import get_logger

from .builder import PIPELINES
from .utils import PartialFunctionWrapper, ViewMetrics, sample_grid, scatter_rays_to_image

logger = get_logger(__name__)


@PIPELINES.register_module()
class NeRFPipeline(torch.nn.Module):
    def __init__(
        self,
        ray_sampler: ConfigDict,
        model: Union[ConfigDict, Sequence[ConfigDict]],
        renderer: ConfigDict,
        chunk_size_grid: int,
        num_passes: int,
        loss_weights: Dict[str, float] = {
            "loss_rgb_mse": 1.0,
            "loss_prev_stage_rgb_mse": 1.0,
        },
        output_rasterized_mc: bool = False,
    ) -> None:
        super().__init__()

        self.ray_sampler = RAY_SAMPLERS.build(ray_sampler)
        self.render_image_height = ray_sampler.image_height
        self.render_image_width = ray_sampler.image_width
        self.sampling_mode_training: RenderSamplingMode = RenderSamplingMode.MASK_SAMPLE
        self.sampling_mode_evaluation: RenderSamplingMode = RenderSamplingMode.FULL_GRID

        # construct models with `self._construct_models()`
        if isinstance(model, Sequence) and (len(model) != num_passes):
            logger.info(f"Rewrite `num_pass` from {num_passes} to {len(model)}.")
            num_passes = len(model)
        self.num_passes = num_passes
        self.implicit_functions = self._construct_models(model)

        self.renderer = RENDERERS.build(renderer)
        bg_color = renderer.bg_color
        if not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color)
        self.register_buffer("bg_color", bg_color, persistent=False)

        self.chunk_size_grid = chunk_size_grid
        self.output_rasterized_mc = output_rasterized_mc

        # loss weights: Dict[str, float]
        self.loss_weights = loss_weights
        self.log_loss_weights()

        self.view_metrics = ViewMetrics()

        # log_vars: List[str]

    def _construct_models(self, model_cfg) -> torch.nn.ModuleList:
        if not isinstance(model_cfg, Sequence):
            model_cfg = [model_cfg] * self.num_passes
        implicit_functions_list = [PartialFunctionWrapper(MODELS.build(_model_cfg)) for _model_cfg in model_cfg]
        return torch.nn.ModuleList(implicit_functions_list)

    def forward(
        self,
        *,
        poses: torch.Tensor,
        focal_lengths: torch.Tensor,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        min_depth: Optional[float] = None,
        max_depth: Optional[float] = None,
        global_codes: Optional[torch.Tensor] = None,
        # fg_probability: Optional[torch.Tensor],
        mask_crop: Optional[torch.Tensor] = None,
        # sequence_name: Optional[List[str]],
        bg_image_rgb: Optional[torch.Tensor] = None,
        image_rgb: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
    ):
        """
        Args:
            image_rgb: A tensor of shape `(B, 3, H, W)` containing a batch of rgb images;
                the first `min(B, n_train_target_views)` images are considered targets and
                are used to supervise the renders; the rest corresponding to the source
                viewpoints from which features will be extracted.
            camera: An instance of CamerasBase containing a batch of `B` cameras corresponding
                to the viewpoints of target images, from which the rays will be sampled,
                and source images, which will be used for intersecting with target rays.
            fg_probability: A tensor of shape `(B, 1, H, W)` containing a batch of
                foreground masks.
            mask_crop: A binary tensor of shape `(B, 1, H, W)` deonting valid
                regions in the input images (i.e. regions that do not correspond
                to, e.g., zero-padding). When the `RaySampler`'s sampling mode is set to
                "mask_sample", rays  will be sampled in the non zero regions.
            depth_map: A tensor of shape `(B, 1, H, W)` containing a batch of depth maps.
            sequence_name: A list of `B` strings corresponding to the sequence names
                from which images `image_rgb` were extracted. They are used to match
                target frames with relevant source frames.
            evaluation_mode: one of EvaluationMode.TRAINING or
                EvaluationMode.EVALUATION which determines the settings used for
                rendering.

        Returns:
            preds: A dictionary containing all outputs of the forward pass including the
                rendered images, depths, masks, losses and other metrics.
        """
        # Determine the used ray sampling mode.
        sampling_mode = RenderSamplingMode(
            self.sampling_mode_training if evaluation_mode == EvaluationMode.TRAINING else self.sampling_mode_evaluation
        )

        ray_bundle: RayBundle = self.ray_sampler(
            poses,
            focal_lengths,
            evaluation_mode=evaluation_mode,
            # mask=mask_crop if mas k_crop is not None and sampling_mode == RenderSamplingMode.MASK_SAMPLE else None,
            mask=mask_crop if mask_crop is not None and sampling_mode == RenderSamplingMode.MASK_SAMPLE else None,
            image_height=image_height,
            image_width=image_width,
            min_depth=min_depth,
            max_depth=max_depth,
        )

        xys = ray_bundle._asdict()["xys"]
        bg_color: Optional[torch.Tensor]
        if bg_image_rgb is not None:
            bg_color = sample_grid(bg_image_rgb, xys)
        else:
            bg_color = None

        for func in self.implicit_functions:
            func.bind_args(global_codes=global_codes)

        rendered: RendererOutput = self._render(
            *ray_bundle,
            bg_color=bg_color,
            sampling_mode=sampling_mode,
            implicit_functions=self.implicit_functions,
            # kwargs
            global_codes=global_codes,
            evaluation_mode=evaluation_mode,
        )

        for func in self.implicit_functions:
            func.unbind_args()

        preds = self._get_view_metrics(raymarched=rendered, xys=xys, image_rgb=image_rgb, depth_map=depth_map)

        # [TODO] Visualize the monte-carlo pixel renders by splatting onto an image grid.
        if sampling_mode == RenderSamplingMode.MASK_SAMPLE:

            if self.output_rasterized_mc:
                (
                    preds["rendered_images"],
                    preds["rendered_depths"],
                    preds["rendered_alpha_masks"],
                ) = self._rasterize_mc_samples(
                    xys,
                    bg_image_rgb if bg_image_rgb is not None else self.bg_color,  # type: ignore[arg-type]
                    image_height,
                    image_width,
                    rendered.features,
                    rendered.depths,
                    rendered.alpha_masks,
                )
        elif sampling_mode == RenderSamplingMode.FULL_GRID:
            preds["rendered_images"] = rendered.features
            preds["rendered_depths"] = rendered.depths
            preds["rendered_alpha_masks"] = rendered.alpha_masks
        else:
            raise ValueError(f"Invalid RenderSamplingMode: {sampling_mode}.")
        # Loss
        objective = self._get_objective(preds)
        if objective is not None:
            preds["objective"] = objective

        return preds

    def _render(self, origins, directions, lengths, xys, *, bg_color, sampling_mode, **kwargs):
        if sampling_mode == RenderSamplingMode.FULL_GRID and self.chunk_size_grid > 0:
            return _apply_chunked(
                self.renderer,
                _chunk_generator(
                    self.chunk_size_grid,
                    origins,
                    directions,
                    lengths,
                    xys,
                    bg_color,
                    **kwargs,
                ),
                lambda chunks: _tensor_collator(chunks, lengths.shape[:-1]),
            )

        else:
            return self.renderer(
                origins=origins, directions=directions, lengths=lengths, xys=xys, bg_color=bg_color, **kwargs
            )

    def log_loss_weights(self) -> None:
        """
        Print a table of the loss weights.
        """
        loss_weights_message = (
            "-------\nloss_weights:\n"
            + "\n".join(f"{k:40s}: {w:1.2e}" for k, w in self.loss_weights.items())
            + "\n-------"
        )
        logger.info(loss_weights_message)

    def _get_view_metrics(
        self,
        raymarched: RendererOutput,
        xys: torch.Tensor,
        image_rgb: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        keys_prefix: str = "loss_",
    ):
        metrics = self.view_metrics(
            image_sampling_grid=xys,
            images_pred=raymarched.features,
            images=image_rgb,
            depths_pred=raymarched.depths,
            depths=depth_map,
            keys_prefix=keys_prefix,
        )

        prev_stage_raymarched = raymarched.prev_stage
        while prev_stage_raymarched is not None:
            metrics.update(
                self.view_metrics(
                    image_sampling_grid=xys,
                    images_pred=raymarched.features,
                    images=image_rgb,
                    depths_pred=raymarched.depths,
                    depths=depth_map,
                    keys_prefix=(keys_prefix + "prev_stage_"),
                )
            )
            prev_stage_raymarched = prev_stage_raymarched.prev_stage

        return metrics

    def _get_objective(self, preds) -> Optional[torch.Tensor]:
        """
        A helper function to compute the overall loss as the dot product
        of individual loss functions with the corresponding weights.
        """
        losses_weighted = [preds[k] * float(w) for k, w in self.loss_weights.items() if (k in preds and w != 0.0)]
        if len(losses_weighted) == 0:
            warnings.warn("No main objective found.")
            return None
        loss = sum(losses_weighted)
        assert torch.is_tensor(loss)
        return loss  # type: ignore[return-value]

    def _rasterize_mc_samples(
        self,
        xys: torch.Tensor,
        bg_color: Optional[torch.Tensor],
        image_height: Optional[int],
        image_width: Optional[int],
        *args: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        if image_height is None or image_width is None:
            image_height = self.render_image_height
            image_width = self.render_image_width

        def _scatter_rays_to_image(features):
            return scatter_rays_to_image(features, xys, image_height, image_width, bg_color)

        return tuple(_scatter_rays_to_image(_tensor) for _tensor in args)


def _apply_chunked(func, chunk_generator, tensor_collator):
    processed_chunks = [func(*chunk_args, **chunk_kwargs) for chunk_args, chunk_kwargs in chunk_generator]

    return cat_dataclass(processed_chunks, tensor_collator)


def _chunk_generator(
    chunk_size: int,
    origins: torch.Tensor,
    directions: torch.Tensor,
    lengths: torch.Tensor,
    xys: torch.Tensor,
    bg_color: Optional[torch.Tensor] = None,
    # global_codes: Optional[torch.Tensor] = None,
    # tqdm_trigger_threshold: int,
    *args,
    **kwargs,
):
    batch_size, *spatial_dim, n_pts_per_ray = lengths.shape
    # if n_pts_per_ray > 0 and chunk_size % n_pts_per_ray != 0:
    #     raise ValueError(f"chunk_size_grid ({chunk_size}) should be divisible " f"by n_pts_per_ray ({n_pts_per_ray})")

    n_rays = math.prod(spatial_dim)
    # special handling for raytracing-based methods
    n_chunks = -(-n_rays * max(n_pts_per_ray, 1) // chunk_size)
    chunk_size_in_rays = -(-n_rays // n_chunks)

    iter = range(0, n_rays, chunk_size_in_rays)
    # if len(iter) >= tqdm_trigger_threshold:
    #     iter = tqdm.tqdm(iter)

    origins_last_dim = origins.shape[-1]
    directions_last_dim = directions.shape[-1]
    lengths_last_dim = lengths.shape[-1]
    xys_last_dim = xys.shape[-1]
    # xys_last_dim = xys.shape[-1]
    if bg_color is not None:
        bg_color_last_dim = bg_color.shape[-1]
    for start_idx in iter:
        end_idx = min(start_idx + chunk_size_in_rays, n_rays)
        _origins = origins.reshape(batch_size, -1, origins_last_dim)[:, start_idx:end_idx]
        _directions = directions.reshape(batch_size, -1, directions_last_dim)[:, start_idx:end_idx]
        _lengths = lengths.reshape(batch_size, -1, lengths_last_dim)[:, start_idx:end_idx]
        _xys = xys.reshape(batch_size, -1, xys_last_dim)[:, start_idx:end_idx]
        _bg_color: Optional[torch.Tensor]
        if bg_color is not None:
            _bg_color = bg_color.view(batch_size, -1, bg_color_last_dim)[:, start_idx:end_idx]
        else:
            _bg_color = None

        yield [_origins, _directions, _lengths, _xys, _bg_color, *args], kwargs


def _tensor_collator(batch, new_dims) -> torch.Tensor:
    """
    Helper function to reshape the batch to the desired shape
    Args:
        batch (_type_): _description_
        new_dims (_type_): shapes before chunking, `(B, H, W, -1)` or `(B, n_rays_per_image, 1, -1)`

    Returns:
        torch.Tensor: _description_
    """
    return torch.cat(batch, dim=1).reshape(*new_dims, -1)


def cat_dataclass(batch, tensor_collator: Callable):
    """
    Concatenate all fields of a list of dataclasses `batch` to a single
    dataclass object using `tensor_collator`.

    Args:
        batch: Input list of dataclasses.

    Returns:
        concatenated_batch: All elements of `batch` concatenated to a single
            dataclass object.
        tensor_collator: The function used to concatenate tensor fields.
    """
    elem = batch[0]
    collated: Dict[str, Any] = {}

    for f in dataclasses.fields(elem):
        elem_f = getattr(elem, f.name)
        if elem_f is None:
            collated[f.name] = None
        elif torch.is_tensor(elem_f):
            collated[f.name] = tensor_collator([getattr(e, f.name) for e in batch])
        elif dataclasses.is_dataclass(elem_f):
            collated[f.name] = cat_dataclass([getattr(e, f.name) for e in batch], tensor_collator)
        elif isinstance(elem_f, collections.abc.Mapping):
            collated[f.name] = {
                k: tensor_collator([getattr(e, f.name)[k] for e in batch]) if elem_f[k] is not None else None
                for k in elem_f
            }
        else:
            raise ValueError("Unsupported field type for concatenation")

    return type(elem)(**collated)
