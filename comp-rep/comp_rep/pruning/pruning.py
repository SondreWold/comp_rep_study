"""
Modules to find subnetworks via model pruning
"""

import torch.nn as nn

from comp_rep.pruning.masked_linear import ContinuousMaskLinear


class MaskedModel(nn.Module):
    def __init__(self, model: nn.Module, maskedlayer_kwargs: dict):
        self.model = model
        self.init_model(maskedlayer_kwargs)

    def freeze_initial_model(self):
        for p in self.model.parameters():
            p.requires_grad_ = False

    def init_model(self, maskedlayer_kwargs: dict):
        self.freeze_initial_model()

        for m in self.model.modules():
            if isinstance(m, nn.Linear()):
                m = ContinuousMaskLinear(m, **maskedlayer_kwargs)

    def forward(self, x):
        return self.model(x)
