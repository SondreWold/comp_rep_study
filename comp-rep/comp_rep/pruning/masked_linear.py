"""
Masked linear layers for model pruning.
"""

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedLinear(nn.Module, abc.ABC):
    """
    An abstract base class for a linear layer with a customizable mask.
    """

    def __init__(self, weight: Tensor, bias: Tensor | None = None):
        """
        Initializes the MaskedLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        """
        super(MaskedLinear, self).__init__()

        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

        self.s_matrix = self.init_s_matrix()

    @abc.abstractmethod
    def init_s_matrix(self) -> Tensor:
        """
        Initializes and returns the variable introduced to compute the binary mask matrix.

        Returns:
            Tensor: The additional variable.
        """
        pass

    @abc.abstractmethod
    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        """
        Computes and returns the mask to be applied to the weights.

        Returns:
            Tensor: The mask tensor.
        """
        pass

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        masked_weight = self.weight * self.compute_mask(self.s_matrix)
        return F.linear(x, masked_weight, self.bias)

    def compute_l1_norm(self) -> Tensor:
        """
        Computes and returns the L1 norm of the weights.

        Returns:
            Tensor: The L1 norm of the weights.
        """
        return torch.norm(self.weight, p=1)

    def extra_repr(self) -> str:
        return f"in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}"


class ContinuousMaskLinear(MaskedLinear):
    def __init__(
        self,
        weights: Tensor,
        bias: Tensor | None = None,
        mask_initial_value: float = 0.0,
        ticket: bool = False,
        temp: float = 1,
        temp_step_increase: float = 1.0,
    ):
        super(MaskedLinear, self).__init__(weights, bias)
        self.out_features, self.in_features = weights.shape
        self.mask_initial_value = mask_initial_value
        self.ticket = ticket  # For evaluation mode, use the actual heaviside function
        self.temp = temp
        self.temp_step_increase = temp_step_increase
        self.local_step = 1

    def init_s_matrix(self) -> Tensor:
        s_matrix = nn.Parameter(
            nn.init.constant_(
                torch.Tensor(self.in_features, self.out_features),
                self.mask_initial_value,
            )
        )
        return s_matrix

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        self.local_step += 1
        temperature_update = 1 + (self.temp_step_increase * self.local_step)
        if self.ticket:
            mask = (s_matrix > 0).float()
        else:
            mask = F.sigmoid(temperature_update * self.s_matrix)
        return mask


class SampledMaskLinear(MaskedLinear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

    def init_s_matrix(self) -> Tensor:
        return super().init_s_matrix()

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        return super().compute_mask(s_matrix)
