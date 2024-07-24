"""
Tests for subnetwork set operations
"""

import pytest
import torch
from torch import nn

from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear
from comp_rep.pruning.subnetwork_set_operations import (
    complement_model,
    complement_model_,
    difference_model,
    difference_model_,
    intersection_model,
    intersection_model_,
    union_model,
    union_model_,
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
    input_dim = 10
    output_dim = 5
    norm_shape = 5
    return Transformer(input_dim, output_dim, norm_shape)


@pytest.fixture
def modelB():
    input_dim = 10
    output_dim = 5
    norm_shape = 5
    return Transformer(input_dim, output_dim, norm_shape)


def test_complement_(modelA):
    linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = linear_b_matrix
    modelA.norm_layer.b_matrix = layernorm_b_matrix

    # test complement
    complement_model_(modelA)

    # the target
    linear_target = 1 - linear_b_matrix
    layernorm_target = 1 - layernorm_b_matrix

    assert (modelA.linear_layer.b_matrix == linear_target).all()
    assert (modelA.norm_layer.b_matrix == layernorm_target).all()


def test_complement(modelA):
    linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = linear_b_matrix
    modelA.norm_layer.b_matrix = layernorm_b_matrix

    # test complement
    new_model = complement_model(modelA)

    # old model should remain same
    assert (modelA.linear_layer.b_matrix == linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == layernorm_b_matrix).all()

    # the target
    linear_target = 1 - linear_b_matrix
    layernorm_target = 1 - layernorm_b_matrix

    # new model should be inverted
    assert (new_model.linear_layer.b_matrix == linear_target).all()
    assert (new_model.norm_layer.b_matrix == layernorm_target).all()


def test_intersection_(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    target_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0],
        ]
    )

    # test in-place intersection
    intersection_model_(modelA, modelB)

    # modelA
    assert (modelA.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == target_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()


def test_intersection(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    target_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0],
        ]
    )

    # test intersection
    new_model = intersection_model(modelA, modelB)

    # modelA - should remain same
    assert (modelA.linear_layer.b_matrix == model_a_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == model_a_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()

    # new_model
    assert (new_model.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (new_model.norm_layer.b_matrix == target_layernorm_b_matrix).all()


def test_union_(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    target_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 1],
        ]
    )

    # test in-place union
    union_model_(modelA, modelB)

    # modelA
    assert (modelA.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == target_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()


def test_union(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    target_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 1],
        ]
    )

    # test intersection
    new_model = union_model(modelA, modelB)

    # modelA - should remain same
    assert (modelA.linear_layer.b_matrix == model_a_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == model_a_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()

    # new_model
    assert (new_model.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (new_model.norm_layer.b_matrix == target_layernorm_b_matrix).all()


def test_differene_(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    complement_linear_b_matrix = 1 - model_b_linear_b_matrix
    complement_layernorm_b_matrix = 1 - model_b_layernorm_b_matrix

    target_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 0, 0],
        ]
    )

    # test in-place union
    difference_model_(modelA, modelB)

    # modelA
    assert (modelA.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == target_layernorm_b_matrix).all()

    # modelB - b-matrix should be complement
    assert (modelB.linear_layer.b_matrix == complement_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == complement_layernorm_b_matrix).all()


def test_difference(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [1, 1, 0, 1, 0],
        ]
    )
    modelA.linear_layer.b_matrix = model_a_linear_b_matrix
    modelA.norm_layer.b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0, 0, 1, 1],
        ]
    )
    modelB.linear_layer.b_matrix = model_b_linear_b_matrix
    modelB.norm_layer.b_matrix = model_b_layernorm_b_matrix

    # target
    target_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    target_layernorm_b_matrix = torch.tensor(
        [
            [0, 1, 0, 0, 0],
        ]
    )

    # test intersection
    new_model = difference_model(modelA, modelB)

    # modelA - should remain same
    assert (modelA.linear_layer.b_matrix == model_a_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == model_a_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()

    # new_model
    assert (new_model.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (new_model.norm_layer.b_matrix == target_layernorm_b_matrix).all()
