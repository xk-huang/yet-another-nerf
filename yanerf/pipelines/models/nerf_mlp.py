from typing import Any, List, NamedTuple, Optional

import torch

from yanerf.pipelines.models.utils import HarmonicEmbedding
from yanerf.utils.logging import get_logger

from .builder import MODELS
from .utils import LinearWithRepeat, create_embeddings_for_implicit_function, ray_bundle_to_ray_points

logger = get_logger(__name__)


class ModelOutputs(NamedTuple):
    raw_densities: torch.Tensor
    rays_colors: torch.Tensor
    aux: Any


@MODELS.register_module()
class NeRFMLP(torch.nn.Module):
    def __init__(
        self,
        n_layers: int = 8,
        input_skips: List[int] = [5],
        n_harmonic_functions_xyz: int = 10,
        harmonic_functions_xyz_append_intput: bool = True,
        n_hidden_neurons_xyz: int = 256,
        n_harmonic_functions_dir: int = 4,
        harmonic_functions_dir_append_intput: bool = True,
        n_hidden_neurons_dir: int = 128,
        latent_dim: int = 0,
        input_xyz: bool = True,
        input_dir: bool = True,
        # xyz_ray_dir_in_camera_coords: bool = False,
        color_dim: int = 3,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_skips = input_skips
        self.n_harmonic_functions_xyz: int = n_harmonic_functions_xyz
        self.harmonic_functions_xyz_append_intput: bool = harmonic_functions_xyz_append_intput
        self.n_hidden_neurons_xyz = n_hidden_neurons_xyz
        self.n_harmonic_functions_dir: int = n_harmonic_functions_dir
        self.harmonic_functions_dir_append_intput: bool = harmonic_functions_dir_append_intput
        self.n_hidden_neurons_dir: int = n_hidden_neurons_dir
        self.latent_dim: int = latent_dim
        self.input_xyz: bool = input_xyz
        self.input_dir: bool = input_dir
        self.color_dim: int = color_dim

        self.harmonic_embedding_xyz = HarmonicEmbedding(
            self.n_harmonic_functions_xyz, append_input=self.harmonic_functions_xyz_append_intput
        )
        self.harmonic_embedding_dir = HarmonicEmbedding(
            self.n_harmonic_functions_dir, append_input=self.harmonic_functions_dir_append_intput
        )
        if not self.input_xyz and self.latent_dim <= 0:
            raise ValueError("The latent dimension has to be > 0 if xyz is not input!")

        embedding_dim_dir = self.harmonic_embedding_dir.get_output_dim()

        self.xyz_encoder = self._construct_xyz_encoder(input_dim=self.get_xyz_embedding_dim())

        self.intermediate_linear = torch.nn.Linear(self.n_hidden_neurons_xyz, self.n_hidden_neurons_xyz)
        _xavier_init(self.intermediate_linear)

        self.density_layer = torch.nn.Linear(self.n_hidden_neurons_xyz, 1)
        _xavier_init(self.density_layer)

        # Zero the bias of the density layer to avoid
        # a completely transparent initialization.
        self.density_layer.bias.data[:] = 0.0  # fixme: Sometimes this is not enough
        self.color_layer = torch.nn.Sequential(
            LinearWithRepeat(self.n_hidden_neurons_xyz + embedding_dim_dir, self.n_hidden_neurons_dir)
            if input_dir
            else torch.nn.Linear(self.n_hidden_neurons_xyz, self.n_hidden_neurons_dir),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.n_hidden_neurons_dir, self.color_dim),
            torch.nn.Sigmoid(),
        )

    def get_xyz_embedding_dim(self):
        return self.harmonic_embedding_xyz.get_output_dim() * int(self.input_xyz) + self.latent_dim

    def _construct_xyz_encoder(self, input_dim: int):
        return MLPWithInputSkips(
            n_layers=self.n_layers,
            input_dim=input_dim,
            output_dim=self.n_hidden_neurons_xyz,
            skip_dim=input_dim,
            input_skips=self.input_skips,
        )

    def _get_colors(self, features: torch.Tensor, rays_directions: torch.Tensor):
        """
        This function takes per-point `features` predicted by `self.xyz_encoder`
        and evaluates the color model in order to attach to each
        point a 3D vector of its RGB color.
        """
        if self.input_dir:
            # Normalize the ray_directions to unit l2 norm.
            rays_directions_normed = torch.nn.functional.normalize(rays_directions, dim=-1)
            # Obtain the harmonic embedding of the normalized ray directions.
            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            rays_embedding = self.harmonic_embedding_dir(rays_directions_normed)

            # pyre-fixme[29]: `Union[torch.Tensor, torch.nn.Module]` is not a function.
            color = self.color_layer((self.intermediate_linear(features), rays_embedding))
        else:
            color = self.color_layer(self.intermediate_linear(features))

        return color

    def forward(
        self,
        # ray_bundle: RayBundle,
        origins: torch.Tensor,
        directions: torch.Tensor,
        lengths: torch.Tensor,
        # xys: torch.Tensor,
        # fun_viewpool=None,
        # cameras=None,
        # camera: Optional[CamerasBase] = None,
        global_codes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ModelOutputs:
        """
        The forward function accepts the parametrizations of
        3D points sampled along projection rays. The forward
        pass is responsible for attaching a 3D vector
        and a 1D scalar representing the point's
        RGB color and opacity respectively.

        Args:
            ray_bundle: A RayBundle object containing the following variables:
                origins: A tensor of shape `(minibatch, ..., 3)` denoting the
                    origins of the sampling rays in world coords.
                directions: A tensor of shape `(minibatch, ..., 3)`
                    containing the direction vectors of sampling rays in world coords.
                lengths: A tensor of shape `(minibatch, ..., num_points_per_ray)`
                    containing the lengths at which the rays are sampled.
            fun_viewpool: an optional callback with the signature
                    fun_fiewpool(points) -> pooled_features
                where points is a [N_TGT x N x 3] tensor of world coords,
                and pooled_features is a [N_TGT x ... x N_SRC x latent_dim] tensor
                of the features pooled from the context images.

        Returns:
            rays_densities: A tensor of shape `(minibatch, ..., num_points_per_ray, 1)`
                denoting the opacitiy of each ray point.
            rays_colors: A tensor of shape `(minibatch, ..., num_points_per_ray, 3)`
                denoting the color of each ray point.
        """
        # rays_points_world.shape = [minibatch x ... x pts_per_ray x 3]
        rays_points_world = ray_bundle_to_ray_points(origins, directions, lengths)

        # [TODO] Check the input range
        if not self._check_input(global_codes):
            raise ValueError("The shape of global codes is imcompible with the input dim of the network.")

        embeds = create_embeddings_for_implicit_function(
            xyz_world=rays_points_world,
            xyz_embedding_function=self.harmonic_embedding_xyz if self.input_xyz else None,
            global_codes=global_codes,
        )
        # embeds.shape = [minibatch x ... x pts_per_ray x 3]
        features = self.xyz_encoder(embeds)
        raw_densities = self.density_layer(features)

        rays_colors = self._get_colors(features, directions)

        return ModelOutputs(raw_densities, rays_colors, {})

    def _check_input(self, global_codes):
        if global_codes is None:
            return self.latent_dim == 0
        else:
            return global_codes.shape[-1] == self.latent_dim


class MLPWithInputSkips(torch.nn.Module):
    """
    Implements the multi-layer perceptron architecture of the Neural Radiance Field.

    As such, `MLPWithInputSkips` is a multi layer perceptron consisting
    of a sequence of linear layers with ReLU activations.

    Additionally, for a set of predefined layers `input_skips`, the forward pass
    appends a skip tensor `z` to the output of the preceding layer.

    Note that this follows the architecture described in the Supplementary
    Material (Fig. 7) of [1].

    References:
        [1] Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik
            and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng:
            NeRF: Representing Scenes as Neural Radiance Fields for View
            Synthesis, ECCV2020
    """

    def _make_affine_layer(self, input_dim, hidden_dim):
        l1 = torch.nn.Linear(input_dim, hidden_dim * 2)
        l2 = torch.nn.Linear(hidden_dim * 2, hidden_dim * 2)
        _xavier_init(l1)
        _xavier_init(l2)
        return torch.nn.Sequential(l1, torch.nn.ReLU(True), l2)

    def _apply_affine_layer(self, layer, x, z):
        mu_log_std = layer(z)
        mu, log_std = mu_log_std.split(mu_log_std.shape[-1] // 2, dim=-1)
        std = torch.nn.functional.softplus(log_std)
        return (x - mu) * std

    def __init__(
        self,
        n_layers: int = 8,
        input_dim: int = 39,
        output_dim: int = 256,
        skip_dim: int = 39,
        hidden_dim: int = 256,
        input_skips: List[int] = [5],
        skip_affine_trans: bool = False,
        no_last_relu=False,
    ):
        """
        Args:
            n_layers: The number of linear layers of the MLP.
            input_dim: The number of channels of the input tensor.
            output_dim: The number of channels of the output.
            skip_dim: The number of channels of the tensor `z` appended when
                evaluating the skip layers.
            hidden_dim: The number of hidden units of the MLP.
            input_skips: The list of layer indices at which we append the skip
                tensor `z`.
        """
        super().__init__()
        layers = []
        skip_affine_layers = []
        for layeri in range(n_layers):
            dimin = hidden_dim if layeri > 0 else input_dim
            dimout = hidden_dim if layeri + 1 < n_layers else output_dim

            if layeri > 0 and layeri in input_skips:
                if skip_affine_trans:
                    skip_affine_layers.append(self._make_affine_layer(skip_dim, hidden_dim))
                else:
                    dimin = hidden_dim + skip_dim

            linear = torch.nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(
                torch.nn.Sequential(linear, torch.nn.ReLU(True))
                if not no_last_relu or layeri + 1 < n_layers
                else linear
            )
        self.mlp = torch.nn.ModuleList(layers)
        if skip_affine_trans:
            self.skip_affines = torch.nn.ModuleList(skip_affine_layers)
        self._input_skips = set(input_skips)
        self._skip_affine_trans = skip_affine_trans

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None):
        """
        Args:
            x: The input tensor of shape `(..., input_dim)`.
            z: The input skip tensor of shape `(..., skip_dim)` which is appended
                to layers whose indices are specified by `input_skips`.
        Returns:
            y: The output tensor of shape `(..., output_dim)`.
        """
        y = x
        if z is None:
            # if the skip tensor is None, we use `x` instead.
            z = x
        skipi = 0
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                if self._skip_affine_trans:
                    y = self._apply_affine_layer(self.skip_affines[skipi], y, z)
                else:
                    y = torch.cat((y, z), dim=-1)
                skipi += 1
            y = layer(y)
        return y


def _xavier_init(linear) -> None:
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)
