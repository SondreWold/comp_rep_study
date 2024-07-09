"""
Masked layer norm layers for model pruning.
"""

import abc
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.batch_ops import batch_bias_add, batch_const_mul


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

    @abc.abstractmethod
    def extra_repr(self) -> str:
        """
        The module representation string.
        """
        pass


class SampledMaskLayerNorm(MaskedLayerNorm):
    """
    A masked LayerNorm module based on sampling. Masks are binarized to only keep or remove individual weights.
    This is achieved using a Gumbel-Sigmoid with a straight-through estimator.
    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        weight: Tensor,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
        tau: float = 1.0,
        num_masks: int = 1,
    ):
        """
        Initializes the SampledMaskLinear layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
            tau (float): The tau parameter for the s_i computation. Default: 1.0.
            num_masks (int): The number of mask samples. Default: 1.
        """
        super(SampledMaskLayerNorm, self).__init__(normalized_shape, weight, bias, eps)
        self.tau = tau
        self.num_masks = num_masks
        self.logits = nn.Parameter(
            torch.ones_like(weight) * torch.log(torch.tensor(9.0))
        )
        self.s_matrix = self.init_s_matrix()

    def sample_s_matrix(self, eps: float = 1e-10) -> Tensor:
        """
        Samples the s_matrix using the specified formula:
        s_i = sigma((l_i - log(log(U_1) / log(U_2))) / tau) with U_1, U_2 ~ U(0, 1).

        Returns:
            Tensor: The initialized s_matrix tensor.
        """
        min_sampled_value = torch.ones_like(self.weight) * eps
        U1 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))
        U2 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))

        log_ratio = torch.log(torch.log(U1) / torch.log(U2))
        s_matrix = torch.sigmoid((self.logits - log_ratio) / self.tau)

        return s_matrix

    def init_s_matrix(self) -> Tensor:
        """
        Initializes multiple s_matrices.

        Returns:
            Tensor: Stacked s_matrix tensors (stacked in the 0th dimension).
        """
        s_matrices = [self.sample_s_matrix(eps=1e-10) for _ in range(self.num_masks)]
        return torch.stack(s_matrices, dim=0)

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        """
        Computes and returns the mask to be applied to the weights using the straight-through estimator.

        Args:
            s_matrix (Tensor): The additional variable used to compute the mask.

        Returns:
            Tensor: The mask tensor.
        """
        with torch.no_grad():
            b_i = (s_matrix > 0.5).type_as(s_matrix)
        b_i = (b_i - s_matrix).detach() + s_matrix

        return b_i

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the layer norm to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        frac = batch_const_mul((x - mu) / (std + self.eps), self.weight)

        if self.bias is not None:
            return batch_bias_add(frac, self.bias)

        return frac

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " "s_matrix={s_matrix.shape}".format(
            **self.__dict__
        )


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
            normalized_shape,
            weight,
            bias,
            eps,
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
        return F.layer_norm(
            x, self.normalized_shape, masked_weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " "s_matrix={s_matrix.shape}".format(
            **self.__dict__
        )


if __name__ == "__main__":
    batch_size = 18
    in_features = 3

    # create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_features)

    # ContinuousMaskLayerNorm
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

    # SampledMaskLayerNorm
    new_layer_norm = nn.LayerNorm(in_features)
    print(f"Layer norm: \n{new_layer_norm}")

    sampled_mask_layernorm = SampledMaskLayerNorm(
        new_layer_norm.normalized_shape,
        new_layer_norm.weight,
        new_layer_norm.bias,
        new_layer_norm.eps,
    )
    print(f"Sampled layer norm: \n{sampled_mask_layernorm}")

    output_tensor_sampled = sampled_mask_layernorm(input_tensor)
    output_tensor = new_layer_norm(input_tensor)

    print(f"Normal layer norm output: \n{output_tensor.shape}")
    print(f"Sampled layer norm output: \n{output_tensor_sampled.shape}")
    print(
        f"L1 norm: \n{sampled_mask_layernorm.compute_l1_norm(sampled_mask_layernorm.s_matrix)}"
    )  # should be 0
