"""
Tests for subnetwork set  fingerained operations
"""

import pytest
import torch
from torch import nn

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear
from comp_rep.pruning.pruning import Pruner
from comp_rep.pruning.subnetwork_set_operations import (
    difference_by_layer_and_module,
    intersection_by_layer_and_module,
    union_by_layer_and_module,
)


@pytest.fixture
def modelA():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 2
    hidden_dim = 64
    dropout = 0.1
    model = Transformer(
        input_vocabulary_size, output_vocabulary_size, layers, hidden_dim, dropout
    )
    pruner = Pruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    change_b_matrix_(model)
    return model


@pytest.fixture
def modelB():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 2
    hidden_dim = 64
    dropout = 0.1
    model = Transformer(
        input_vocabulary_size, output_vocabulary_size, layers, hidden_dim, dropout
    )
    pruner = Pruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    change_b_matrix_(model, True)
    return model


@pytest.fixture
def modelC():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 2
    hidden_dim = 64
    dropout = 0.1
    model = Transformer(
        input_vocabulary_size, output_vocabulary_size, layers, hidden_dim, dropout
    )
    pruner = Pruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    for module in model.modules():
        if isinstance(module, MaskedLayer):
            b_matrix = torch.tensor(
                [
                    [1] * hidden_dim if i == 0 else [0] * hidden_dim
                    for i in range(0, hidden_dim)
                ]
            )
            module.b_matrix = b_matrix
    return model


def change_b_matrix_(model: Transformer, invert: bool = False):
    for module in model.modules():
        if isinstance(module, MaskedLayer):
            b_matrix = torch.triu(torch.ones(model.hidden_size, model.hidden_size))
            if invert:
                b_matrix = 1 - b_matrix
            module.b_matrix = b_matrix


def test_union_by_layer_and_module(modelA, modelB):
    expected_result = torch.ones(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = union_by_layer_and_module(
        modelA, modelB, [1], [ContinuousMaskLayerNorm]
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_intersection_by_layer_and_module(modelA, modelB):
    expected_result = torch.zeros(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = intersection_by_layer_and_module(
        modelA, modelB, [1], [ContinuousMaskLayerNorm]
    )

    # Assert that the intersection works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the intersection is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_difference_by_layer_and_module(modelA, modelC):
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    full_triu_matrix = torch.triu(
        torch.ones(modelA.hidden_size, modelA.hidden_size)
    ).tolist()
    full_triu_matrix[0] = [0] * modelA.hidden_size
    expected_result = torch.tensor(full_triu_matrix)

    new_model = difference_by_layer_and_module(
        modelA, modelC, [1], [ContinuousMaskLayerNorm]
    )

    # Assert that the difference works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the difference is not applied in another layer
    assert torch.all(new_model.encoder.layers[0].norm_1.b_matrix == unchanged_matrix)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )
