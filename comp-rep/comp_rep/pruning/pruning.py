"""
Modules to find subnetworks via model pruning
"""

from typing import Any, Literal

import torch.nn as nn

from comp_rep.pruning.masked_linear import ContinuousMaskLinear, SampledMaskLinear


class MaskedModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict,
    ):
        self.model = model
        self.init_model(maskedlayer_kwargs)
        self.masker: Any
        if pruning_method == "continuous":
            self.masker = ContinuousMaskLinear
        elif pruning_method == "sampled":
            self.masker = SampledMaskLinear
        else:
            raise Exception("Invalid pruning strategy method provided")

    def freeze_initial_model(self):
        for p in self.model.parameters():
            p.requires_grad_ = False

    def init_model(self, maskedlayer_kwargs: dict):
        self.freeze_initial_model()

        for m in self.model.modules():
            if isinstance(m, nn.Linear()):
                m = self.masker(m, **maskedlayer_kwargs)

    def forward(self, x):
        return self.model(x)
