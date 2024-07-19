"""
Tests for subnetwork set operations
"""

import numpy as np
import pytest
import torch
from torch import nn

from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear
from comp_rep.pruning.subnetwork_metrics import (
    intersection_over_minimum,
    intersection_over_union,
)


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, norm_shape):
        super(Transformer, self).__init__()
        linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))
        norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

        self.linear_layer = ContinuousMaskLinear(
            weight=linear_weight, bias=None, ticket=True
        )
        self.norm_layer = ContinuousMaskLayerNorm(
            normalized_shape=norm_shape,
            weight=norm_layer_weights,
            bias=None,
            ticket=True,
        )

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.norm_layer(x)
        return x


@pytest.fixture
def modelA():
    input_dim = 4
    output_dim = 2
    norm_shape = 2
    return Transformer(input_dim, output_dim, norm_shape)


@pytest.fixture
def modelB():
    input_dim = 4
    output_dim = 2
    norm_shape = 2
    return Transformer(input_dim, output_dim, norm_shape)


def test_intersection_over_union(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # Manual IoU
    linear_intersect = (model_a_linear_b_matrix & model_b_linear_b_matrix).sum()
    norm_intersect = (model_a_layernorm_b_matrix & model_b_layernorm_b_matrix).sum()

    linear_union = (model_a_linear_b_matrix | model_b_linear_b_matrix).sum()
    norm_union = (model_a_layernorm_b_matrix | model_b_layernorm_b_matrix).sum()

    manuel_result = (linear_intersect + norm_intersect) / (linear_union + norm_union)

    # test intersection_over_union
    result = intersection_over_union(modelA, modelB)
    assert manuel_result == result


def test_intersection_over_minimum(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # Manual IoU
    linear_intersect = (model_a_linear_b_matrix & model_b_linear_b_matrix).sum()
    norm_intersect = (model_a_layernorm_b_matrix & model_b_layernorm_b_matrix).sum()

    minimum = min(
        (model_a_linear_b_matrix.sum() + model_a_layernorm_b_matrix.sum()),
        (model_b_linear_b_matrix.sum() + model_b_layernorm_b_matrix.sum()),
    )

    manuel_result = (linear_intersect + norm_intersect) / minimum

    # test intersection_over_union
    result = intersection_over_minimum(modelA, modelB)
    assert manuel_result == result
