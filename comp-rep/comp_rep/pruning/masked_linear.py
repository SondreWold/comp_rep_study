"""
Masked linear layers for model pruning.
"""

import abc
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.batch_ops import batch_bias_add, batch_matmul


class MaskedLinear(nn.Module, abc.ABC):
    """
    An abstract base class for a linear layer with a customizable mask.
    """

    def __init__(self, weight: Tensor, bias: Optional[Tensor] = None):
        """
        Initializes the MaskedLinear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        """
        super(MaskedLinear, self).__init__()
        self.out_features, self.in_features = weight.shape
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

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

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        pass

    def compute_l1_norm(self, s_matrix: Tensor) -> Tensor:
        """
        Computes and returns the L1 norm of the weights.

        Returns:
            Tensor: The L1 norm of the weights.
        """
        return torch.norm(self.compute_mask(s_matrix), p=1)

    def compute_remaining_weights(self) -> float:
        """
        Computes and returns the percentage of remaining weights

        Returns:
            Tensor: The percentage of remaining weights
        """
        below_zero = float((self.compute_mask(self.s_matrix) <= 0).sum())
        original = self.s_matrix.numel()
        return 1 - below_zero / original

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class SampledMaskLinear(MaskedLinear):
    """
    A masked linear layer based on sampling. Masks are binarized to only keep or remove individual weights.
    This is achieved using a Gumbel-Sigmoid with a straight-through estimator.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
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
        super(SampledMaskLinear, self).__init__(weight, bias)
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
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        masks = self.compute_mask(self.s_matrix)
        masked_weight = self.weight.unsqueeze(0) * masks
        output = batch_matmul(
            x, masked_weight.transpose(-1, -2)
        )  # transpose for batch multiplication

        if self.bias is not None:
            output = batch_bias_add(output, self.bias)

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, s_matrix={self.s_matrix.shape}"


class ContinuousMaskLinear(MaskedLinear):
    """
    A masked linear layer based on continuous sparsification.
    Masks are binarized to only keep or remove individual weights.
    """

    def __init__(
        self,
        weights: Tensor,
        bias: Optional[Tensor] = None,
        mask_initial_value: float = 0.0,
        temperature_increase: float = 1.0,
        ticket: bool = False,
    ):
        super(ContinuousMaskLinear, self).__init__(weights, bias)
        self.mask_initial_value = mask_initial_value
        self.temp = 1.0  # Always starts at 1 and increases to the max_temp
        self.temperature_increase = temperature_increase
        self.ticket = ticket  # For evaluation mode, use the actual heaviside function
        self.s_matrix = self.init_s_matrix()

    def update_temperature(self) -> None:
        """
        Updates the temperature.
        """
        self.temp = self.temp * self.temperature_increase

    def init_s_matrix(self) -> Tensor:
        """
        Initializes the constant s_matrix.

        Returns:
            Tensor: The initialized s_matrix tensor.
        """
        s_matrix = nn.Parameter(
            nn.init.constant_(
                torch.Tensor(self.out_features, self.in_features),
                self.mask_initial_value,
            )
        )
        return s_matrix

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        """
        Computes and returns the mask to be applied to the weights using the heaviside function or sigmoid.

        Args:
            s_matrix (Tensor): The additional variable used to compute the mask.

        Returns:
            Tensor: The mask tensor.
        """
        if self.ticket:
            mask = (s_matrix > 0).float()
        else:
            mask = F.sigmoid(self.temp * self.s_matrix)
        return mask

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        masks = self.compute_mask(self.s_matrix)
        masked_weight = self.weight * masks
        return F.linear(x, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, s_matrix={self.s_matrix.shape}"


if __name__ == "__main__":
    batch_size = 18
    num_mask = 2
    tau = 1.0
    in_features = 3
    out_features = 5

    # create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_features)

    # layers
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    print(f"Linear layer: \n{linear_layer}")

    sampled_mask_linear = SampledMaskLinear(
        linear_layer.weight, linear_layer.bias, tau=tau, num_masks=num_mask
    )

    cont_mask_linear = ContinuousMaskLinear(linear_layer.weight, linear_layer.bias)
    print(isinstance(cont_mask_linear, MaskedLinear))
    print(f"Sampled masked layer: \n{sampled_mask_linear}")
    print(f"Continuous  masked layer: \n{cont_mask_linear}")

    output_tensor_sample = sampled_mask_linear(input_tensor)
    output_tensor_cont = cont_mask_linear(input_tensor)
    print(f"Sampled out tensor: \n{output_tensor_sample}")
    print(f"Continuous out tensor: \n{output_tensor_cont}")
