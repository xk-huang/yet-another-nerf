from .builder import FEATURE_EXTRACTORS
import torch


@FEATURE_EXTRACTORS.register_module()
class IdentityMapper(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, **kwargs):
        return kwargs
