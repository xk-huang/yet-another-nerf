from email.mime import audio
from multiprocessing.sharedctypes import Value
from turtle import forward
from .builder import FEATURE_EXTRACTORS
import torch
from torch import nn


@FEATURE_EXTRACTORS.register_module()
class AudioNet(nn.Module):
    pivot_position_to_pivot_idx_func = {
        "middle": lambda x: x // 2,
        "end": lambda _: -1,
        "front": lambda _: 0,
    }

    def __init__(
        self, feature_dims, feature_dims_for_attention, num_successive_features, pivot_position: str = "middle"
    ) -> None:
        super().__init__()

        if pivot_position not in self.pivot_position_to_pivot_idx_func:
            raise ValueError(f"Invalid pivot position type: {pivot_position}")

        self.pivot_idx = self.pivot_position_to_pivot_idx_func[pivot_position](num_successive_features)
        self.num_successive_features = num_successive_features
        self.feature_dims_for_attention = feature_dims_for_attention
        self.feature_dims = feature_dims

        self.audio_net = _AudioNet(feature_dims)
        self.audio_smooth_net = AudioSmoothNet(feature_dims_for_attention, num_successive_features)

    def forward(self, deepspeech_features: torch.Tensor, use_smooth: bool = True):
        if not use_smooth:
            deepspeech_features = deepspeech_features[..., self.pivot_idx, :, :]  # (B, N, C, W)
            return self.audio_net(deepspeech_features).unsqueeze(-2)  # (B, C)

        batch_size, num_seq, *last_dims = deepspeech_features.shape

        deepspeech_features = deepspeech_features.view(batch_size * num_seq, *last_dims)
        audio_features = self.audio_net(deepspeech_features)
        audio_features = audio_features.view(batch_size, num_seq, -1)

        return self.audio_smooth_net(audio_features)  # (B, C)


class AudioSmoothNet(nn.Module):
    def __init__(self, feature_dims_for_attention=32, num_successive_features=8):
        super().__init__()
        self.num_successive_features = num_successive_features
        self.feature_dims_for_attention = feature_dims_for_attention
        self.attentionConvNet = nn.Sequential(  # (..., feature_dims, num_successive_features)
            nn.Conv1d(feature_dims_for_attention, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
        )  # (..., 1, num_successive_features)
        self.attentionNet = nn.Sequential(  # (..., num_successive_features)
            nn.Linear(in_features=self.num_successive_features, out_features=self.num_successive_features, bias=True),
            nn.Softmax(dim=1),
        )  # (..., num_successive_features)

    def forward(self, input_tensors):
        """_summary_

        Args:
            x (torch.Tensor): (..., num_successive_features, feature_dims)

        Returns:
            torch.Tensor: (..., feature_dims)
        """
        spatial_shape = input_tensors.shape[:-2]
        if input_tensors.shape[-2] != self.num_successive_features:
            raise ValueError(f"Invalid tensor shape for audio net: {input_tensors.shape}")

        # (..., feature_dims_for_attention, num_successive_features)
        attention_scores = input_tensors[..., : self.feature_dims_for_attention].permute(
            *list(range(len(spatial_shape))), -1, -2
        )
        # (..., num_successive_features)
        attention_scores = self.attentionConvNet(attention_scores).view(*spatial_shape, self.num_successive_features)
        # (..., num_successive_features, 1)
        attention_scores = self.attentionNet(attention_scores).view(*spatial_shape, self.num_successive_features, 1)
        return torch.sum(attention_scores * input_tensors, dim=-2)


class _AudioNet(nn.Module):
    def __init__(self, feature_dims=64):
        super().__init__()
        self.feature_dims = feature_dims
        self.encoder_conv = nn.Sequential(  # ... x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2, padding=1, bias=True),  # ... x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # ... x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # ... x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # ... x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, feature_dims),
        )

    def forward(self, x):
        """_summary_

        Args:
            x (torch.Tensor): (..., deepspeech_feature_size=29, window_size=16)

        Returns:
            torch.Tensor: (..., feature_dims)
        """
        if list(x.shape[-2:]) != [29, 16]:
            raise ValueError(f"Invalid deepspeech feature shape: {x.shape}")

        x = self.encoder_conv(x).mean(-1)
        x = self.encoder_fc1(x)
        return x
