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
        self.out_features, self.in_features = weight.shape
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
        temp: float = 1.0,
        temp_step_increase: float = 1.0,
    ):
        super(MaskedLinear, self).__init__(weights, bias)
        self.mask_initial_value = mask_initial_value
        self.ticket = ticket  # For evaluation mode, use the actual heaviside function
        self.temp = temp
        self.temp_step_increase = temp_step_increase

    def init_s_matrix(self) -> Tensor:
        s_matrix = nn.Parameter(
            nn.init.constant_(
                torch.Tensor(self.in_features, self.out_features),
                self.mask_initial_value,
            )
        )
        return s_matrix

    def compute_mask(self, s_matrix: Tensor) -> Tensor:
        if self.ticket:
            mask = (s_matrix > 0).float()
        else:
            mask = F.sigmoid(self.temp * self.s_matrix)
            self.temp += self.temp_step_increase
        return mask


class SampledMaskLinear(MaskedLinear):
    """
    A masked linear layer based on sampling. Masks are binarized to only keep or remove individual weights.
    This is achieved using a Gumbel-Sigmoid with a straight-through estimator.
    """

    def __init__(self, weight: Tensor, bias: Tensor | None = None, tau: float = 1.0):
        """
        Initializes the SampledMaskLinear layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
            tau (float): The tau parameter for the s_i computation. Default: 1.0.
        """
        self.tau = tau
        self.logits = nn.Parameter(torch.ones_like(weight)) * 0.9
        super(SampledMaskLinear, self).__init__(weight, bias)

    def init_s_matrix(self, eps: float = 1e-10) -> Tensor:
        """
        Initializes the s_matrix using the specified formula:
        s_i = sigma((l_i - log(log(U_1) / log(U_2))) / tau) with U_1, U_2 ~ U(0, 1).

        Returns:
            Tensor: The initialized s_matrix tensor.
        """
        min_sampled_value = torch.ones_like(self.weight) * eps
        U1 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))
        U2 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))

        log_ratio = torch.log(torch.log(U1) / torch.log(U2))
        s_i = torch.sigmoid((self.logits - log_ratio) / self.tau)

        return s_i

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
