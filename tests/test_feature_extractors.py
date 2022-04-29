from typing import Optional, Tuple
from yanerf.pipelines.feature_extractors.audionet import AudioNet
import torch
import pytest


class Hook:
    def __init__(self) -> None:
        self.input_tensor_tuple: Optional[Tuple[torch.Tensor]] = None
        self.output_tensor: Optional[torch.Tensor] = None
        self.module_info: Optional[str] = None

    def forward_hook(self, module: torch.nn.Module, input_tuple, output):
        self.module_info = module._get_name()
        self.input_tensor_tuple = input_tuple
        self.output_tensor = output
        return output


@pytest.mark.parametrize("pivot_position,select_value", zip(["front", "middle", "end"], [0, 4, 7]))
def test_audio_net_no_smooth(pivot_position, select_value):
    audionet_params = dict(
        feature_dims=64, feature_dims_for_attention=32, num_successive_features=8, pivot_position=pivot_position
    )

    audio_net = AudioNet(**audionet_params)

    hook = Hook()
    _ = audio_net.audio_net.register_forward_hook(hook.forward_hook)

    deepspeech_features = torch.arange(8, dtype=torch.float32)[None, :, None, None].expand(2, 8, 29, 16)
    data = dict(deepspeech_features=deepspeech_features, use_smooth=False)
    _ = audio_net(**data)

    assert torch.allclose(hook.input_tensor_tuple[0], torch.ones([1]) * select_value)


def test_audio_net_smooth(pivot_position="middle"):
    audionet_params = dict(
        feature_dims=64, feature_dims_for_attention=32, num_successive_features=8, pivot_position=pivot_position
    )

    audio_net = AudioNet(**audionet_params)

    hook = Hook()
    _ = audio_net.audio_smooth_net.attentionNet.register_forward_hook(hook.forward_hook)

    deepspeech_features = torch.arange(8, dtype=torch.float32)[None, :, None, None].expand(2, 8, 29, 16).contiguous()

    data = dict(deepspeech_features=deepspeech_features, use_smooth=True)
    _ = audio_net(**data)

    assert torch.allclose(hook.output_tensor.sum(-1), torch.ones(1))
