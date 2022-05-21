from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from yanerf.pipelines.utils import EvaluationMode, RayBundle

from .builder import RENDERERS
from .utils import RayPointRefiner, RendererOutput


@RENDERERS.register_module()
class MultipassEmissionAbsorpsionRenderer(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray_fine_training: int = 64,
        n_pts_per_ray_fine_evaluation: int = 64,
        stratified_sampling_coarse_training: bool = True,
        stratified_sampling_coarse_evaluation: bool = False,
        append_coarse_samples_to_fine: bool = True,
        bg_color: Tuple[float, ...] = (0.0,),
        density_noise_std_train: float = 0.0,
        capping_function: str = "exponential",  # exponential | cap1
        weight_function: str = "product",  # product | minimum
        background_opacity: float = 1e10,
        blend_output: bool = False,
        background_density_bias: float = 0.0,
        hard_background: bool = False,
    ) -> None:
        super().__init__()
        self.density_noise_std_train = density_noise_std_train
        self._refiners = {
            EvaluationMode.TRAINING: RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray_fine_training,
                random_sampling=stratified_sampling_coarse_training,
                add_input_samples=append_coarse_samples_to_fine,
            ),
            EvaluationMode.EVALUATION: RayPointRefiner(
                n_pts_per_ray=n_pts_per_ray_fine_evaluation,
                random_sampling=stratified_sampling_coarse_evaluation,
                add_input_samples=append_coarse_samples_to_fine,
            ),
        }

        self._raymarcher: Callable = EmissionAbsorptionRaymarcher(
            surface_thickness=1,
            bg_color=bg_color,
            capping_function=capping_function,
            weight_function=weight_function,
            background_opacity=background_opacity,
            blend_output=blend_output,
            hard_background=hard_background,
            background_density_bias=background_density_bias,
        )

    def forward(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: torch.Tensor,
        xys: torch.Tensor,
        bg_color: Optional[torch.Tensor],
        *,
        # global_codes: torch.Tensor,
        implicit_functions: List[Callable],
        evaluation_mode: EvaluationMode = EvaluationMode.EVALUATION,
        **kwargs,
    ):
        if not implicit_functions:
            raise ValueError("EA renderer expects implicit functions")

        return self._run_raymarcher(
            origins,
            directions,
            lengths,
            xys,
            # global_codes,
            bg_color,
            implicit_functions,
            None,
            evaluation_mode,
            **kwargs,
        )

    def _run_raymarcher(
        self,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: torch.Tensor,
        xys: torch.Tensor,
        # global_codes: torch.Tensor,
        bg_color: Optional[torch.Tensor],
        implicit_functions: List[Callable[..., Any]],
        prev_stage: Optional[RendererOutput],
        evaluation_mode: EvaluationMode,
        **kwargs,
    ):
        density_noise_std = self.density_noise_std_train if evaluation_mode == EvaluationMode.TRAINING else 0.0

        features, depths, alpha_masks, weights, aux = self._raymarcher(
            **implicit_functions[0](origins, directions, lengths, **kwargs),
            ray_lengths=lengths,
            ray_directions=directions,
            density_noise_std=density_noise_std,
            bg_color=bg_color,
        )
        aux["weights"] = weights

        output = RendererOutput(
            features=features, depths=depths, alpha_masks=alpha_masks, aux=aux, prev_stage=prev_stage
        )

        if len(implicit_functions) > 1:  # type: ignore[arg-type]
            ray_bundle: RayBundle = self._refiners[evaluation_mode](origins, directions, lengths, xys, weights)
            output = self._run_raymarcher(
                *ray_bundle, bg_color, implicit_functions[1:], output, evaluation_mode, **kwargs
            )
        return output


class EmissionAbsorptionRaymarcher(torch.nn.Module):
    def __init__(
        self,
        surface_thickness: int = 1,
        bg_color: Union[Tuple[float, ...], torch.Tensor] = (0.0,),
        capping_function: str = "exponential",  # exponential | cap1
        weight_function: str = "product",  # product | minimum
        background_opacity: float = 1e10,
        density_relu: bool = True,
        blend_output: bool = True,
        background_density_bias: float = 0.0,
        hard_background: bool = False,
    ) -> None:
        super().__init__()
        self.surface_thickness = surface_thickness
        self.density_relu = density_relu
        self.background_opacity = background_opacity
        self.blend_output = blend_output
        self.background_density_bias = background_density_bias
        if not isinstance(bg_color, torch.Tensor):
            bg_color = torch.tensor(bg_color)
        self.register_buffer("_bg_color", bg_color, persistent=False)
        self.hard_background = hard_background

        self._capping_function: Callable[[torch.Tensor], torch.Tensor] = {
            "exponential": lambda x: 1.0 - torch.exp(-x),
            "cap1": lambda x: x.clamp(max=1.0),
        }[capping_function]

        self._weight_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = {
            "product": lambda curr, acc: curr * acc,
            "minimum": lambda curr, acc: torch.minimum(curr, acc),
        }[weight_function]

    def forward(
        self,
        rays_densities: torch.Tensor,
        rays_features: torch.Tensor,
        aux: Dict[str, Any],
        ray_lengths: torch.Tensor,
        ray_directions: torch.Tensor,
        density_noise_std: float = 0.0,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Args:
            rays_densities: Per-ray density values represented with a tensor
                of shape `(..., n_points_per_ray, 1)`.
            rays_features: Per-ray feature values represented with a tensor
                of shape `(..., n_points_per_ray, feature_dim)`.
            aux: a dictionary with extra information.
            ray_lengths: Per-ray depth values represented with a tensor
                of shape `(..., n_points_per_ray,)`.
            density_noise_std: the magnitude of the noise added to densities.

        Returns:
            features: A tensor of shape `(..., feature_dim)` containing
                the rendered features for each ray.
            depth: A tensor of shape `(..., 1)` containing estimated depth.
            opacities: A tensor of shape `(..., 1)` containing rendered opacsities.
            weights: A tensor of shape `(..., n_points_per_ray)` containing
                the ray-specific non-negative opacity weights. In general, they
                don't sum to 1 but do not overcome it, i.e.
                `(weights.sum(dim=-1) <= 1.0).all()` holds.
        """
        _check_raymarcher_inputs(
            rays_densities,
            rays_features,
            ray_lengths,
            z_can_be_none=True,
            features_can_be_none=False,
            density_1d=True,
        )

        deltas = torch.cat(
            (
                ray_lengths[..., 1:] - ray_lengths[..., :-1],
                self.background_opacity * torch.ones_like(ray_lengths[..., :1]),
            ),
            dim=-1,
        )
        deltas = deltas * ray_directions[..., None, :].norm(p=2, dim=-1)  # (B, *spatial, 1, 3)

        rays_densities = rays_densities[..., 0]
        if density_noise_std > 0.0:
            rays_densities = rays_densities + torch.randn_like(rays_densities) * density_noise_std
        if self.density_relu:
            rays_densities = torch.relu(rays_densities) + self.background_density_bias

        weighted_densities = deltas * rays_densities
        capped_densities = self._capping_function(weighted_densities)

        rays_opacities = self._capping_function(torch.cumsum(weighted_densities, dim=-1))
        opacities: torch.Tensor = rays_opacities[..., -1:]
        absorption_shifted = (1.0 - rays_opacities).roll(self.surface_thickness, dims=-1)
        absorption_shifted[..., : self.surface_thickness] = 1.0

        weights = self._weight_function(capped_densities, absorption_shifted)
        depths = (weights * ray_lengths)[..., None].sum(dim=-2)

        if bg_color is None:
            bg_color = self._bg_color  # type: ignore[assignment]
            bg_color = bg_color.view(*([1] * len(rays_features.shape[:-2])), -1).expand(*rays_features.shape[:-2], -1)
        else:
            bg_color = bg_color

        if not self.hard_background:
            features: torch.Tensor = (weights[..., None] * rays_features).sum(dim=-2)
            alpha: Union[int, torch.Tensor] = opacities if self.blend_output else 1
            bg_color: torch.Tensor
            if bg_color.shape[-1] not in [1, features.shape[-1]]:
                raise ValueError(
                    f"Wrong number of background color channels: _bg_color {bg_color.shape} vs. features {features.shape}."
                )
            features = alpha * features + (1 - opacities) * bg_color
        else:
            rays_features = torch.cat([rays_features[..., :-1, :], bg_color[..., None, :]], dim=-2)
            features: torch.Tensor = (weights[..., None] * rays_features).sum(dim=-2)

        return features, depths, opacities, weights, aux


def _check_raymarcher_inputs(
    rays_densities: torch.Tensor,
    rays_features: Optional[torch.Tensor],
    rays_z: Optional[torch.Tensor],
    features_can_be_none: bool = False,
    z_can_be_none: bool = False,
    density_1d: bool = True,
) -> None:
    """
    Checks the validity of the inputs to raymarching algorithms.
    """
    if not torch.is_tensor(rays_densities):
        raise ValueError("rays_densities has to be an instance of torch.Tensor.")

    if not z_can_be_none and not torch.is_tensor(rays_z):
        raise ValueError("rays_z has to be an instance of torch.Tensor.")

    if not features_can_be_none and not torch.is_tensor(rays_features):
        raise ValueError("rays_features has to be an instance of torch.Tensor.")

    if rays_densities.ndim < 1:
        raise ValueError("rays_densities have to have at least one dimension.")

    if density_1d and rays_densities.shape[-1] != 1:
        raise ValueError("The size of the last dimension of rays_densities has to be one.")

    rays_shape = rays_densities.shape[:-1]

    # pyre-fixme[16]: `Optional` has no attribute `shape`.
    if not z_can_be_none and rays_z.shape != rays_shape:  # type: ignore[union-attr]
        raise ValueError("rays_z have to be of the same shape as rays_densities.")

    if not features_can_be_none and rays_features.shape[:-1] != rays_shape:  # type: ignore[union-attr]
        raise ValueError(
            "The first to previous to last dimensions of rays_features"
            " have to be the same as all dimensions of rays_densities."
        )
