"""
Masked layer norm layers for model pruning.
"""

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MaskedLayerNorm(nn.Module, abc.ABC):
    """
    An abstract base class for a layer norm with a customizable mask.
    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        weight: Tensor,
        bias: Optional[Tensor],
        eps: float = 1e-5,
    ):
        super(MaskedLayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.eps = eps

    @abc.abstractmethod
    def init_s_matrix(self) -> Tensor:
        pass

    @abc.abstractmethod
    def compute_mask(self, s_matrix) -> Tensor:
        pass

    def compute_l1_norm(self, s_matrix: Tensor):
        """
        Computes the L1 norm of the s_matrix

        args:
            s_matrix (Tensor): The s_matrix

        Returns:
            Tensor: The L1 norm
        """
        return torch.norm(self.compute_mask(self.s_matrix), p=1)

    def compute_remaining_weights(self) -> float:
        """
        Computes and returns the percentage of remaining weights

        Returns:
            float: The percentage of remaining weights
        """
        below_zero = float((self.compute_mask(self.s_matrix) <= 0).sum())
        original = self.s_matrix.numel()
        return below_zero / original


class ContinuousMaskLayerNorm(MaskedLayerNorm):
    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        weight: Tensor,
        bias: Optional[Tensor],
        eps: float = 1e-5,
        mask_initial_value: float = 0.0,
        temperature_increase: float = 1.0,
        ticket: bool = False,
    ):
        super(ContinuousMaskLayerNorm, self).__init__(
            normalized_shape, weight, bias, eps
        )
        self.mask_initial_value = mask_initial_value
        self.temp = 1.0
        self.temperature_increase = temperature_increase
        self.ticket = ticket
        self.s_matrix = self.init_s_matrix()

    def init_s_matrix(self) -> Tensor:
        """
        Initializes the s_matrix with constant values.

        Returns:
            Tensor: The s_matrix.
        """
        s_matrix = nn.Parameter(
            nn.init.constant_(
                torch.Tensor(self.normalized_shape),
                self.mask_initial_value,
            )
        )
        return s_matrix

    def update_temperature(self):
        """
        Updates the temperature.
        """
        self.temp = self.temp * self.temperature_increase

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        """
        Compute the mask.

        Args:
            s_matrix (Tensor): The current s_matrix.

        Returns:
            Tensor: The computed mask.
        """
        if self.ticket:
            weight_mask = (s_matrix > 0).float()
        else:
            weight_mask = F.sigmoid(self.temp * self.s_matrix)
        return weight_mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass of the layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying layer normalization with a masked weight.
        """
        weight_mask = self.compute_mask(self.s_matrix)
        masked_weight = self.weight * weight_mask
        return F.layer_norm(x, self.normalized_shape, masked_weight)


if __name__ == "__main__":
    batch_size = 18
    in_features = 3

    # create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_features)

    # layer norm
    layer_norm = nn.LayerNorm(in_features)
    print(f"Layer norm: \n{layer_norm}")

    cont_mask_layernorm = ContinuousMaskLayerNorm(
        layer_norm.normalized_shape, layer_norm.weight, layer_norm.bias, layer_norm.eps
    )
    output_tensor_cont = cont_mask_layernorm(input_tensor)
    output_tensor = layer_norm(input_tensor)

    print(f"Continuous layer norm output: \n{output_tensor_cont}")
    print(
        f"L1 norm: \n{cont_mask_layernorm.compute_l1_norm(cont_mask_layernorm.s_matrix)}"
    )  # should be 0
    print(f"Normal layer norm output: \n{output_tensor}")
