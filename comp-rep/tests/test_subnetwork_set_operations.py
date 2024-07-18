"""
Tests for subnetwork set operations
"""

from pathlib import Path
from typing import Callable

import pytest
import torch
from torch import nn

from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm, MaskedLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear, MaskedLinear
from comp_rep.pruning.subnetwork_set_operations import (
    complement,
    complement_,
    difference_,
    intersection,
    intersection_,
    union_,
)
from comp_rep.utils import (
    ValidateTaskOptions,
    create_transformer_from_checkpoint,
    load_model,
    load_tokenizer,
    setup_logging,
)


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, norm_shape):
        super(Transformer, self).__init__()
        linear_weights = nn.Parameter(torch.randn(output_dim, input_dim))
        norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

        self.linear_layer = ContinuousMaskLinear(
            weights=linear_weights, bias=None, ticket=True
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
    complement_(modelA)

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
    new_model = complement(modelA)

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
    intersection_(modelA, modelB)

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
    new_model = intersection(modelA, modelB)

    # modelA - should remain same
    assert (modelA.linear_layer.b_matrix == model_a_linear_b_matrix).all()
    assert (modelA.norm_layer.b_matrix == model_a_layernorm_b_matrix).all()

    # modelB - should remain same
    assert (modelB.linear_layer.b_matrix == model_b_linear_b_matrix).all()
    assert (modelB.norm_layer.b_matrix == model_b_layernorm_b_matrix).all()

    # new_model
    assert (new_model.linear_layer.b_matrix == target_linear_b_matrix).all()
    assert (new_model.norm_layer.b_matrix == target_layernorm_b_matrix).all()


"""
@pytest.mark.skip()
def test_union_or_intersection(function: Callable, comparator: Callable):
    mask_name_A = "append"
    model_path_A = SAVE_PATH / mask_name_A / "pruned_model.ckpt"
    base_model_A = create_transformer_from_checkpoint(model_path_A)
    base_model_A = load_model(model_path_A, True, base_model_A, "continuous")
    mask_name_B = "remove_second"
    model_path_B = SAVE_PATH / mask_name_B / "pruned_model.ckpt"
    base_model_B = create_transformer_from_checkpoint(model_path_B)
    base_model_B = load_model(model_path_B, True, base_model_B, "continuous")

    old_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            old_masks_A.append(m.b_matrix)

    masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            masks_B.append(m.b_matrix)

    function(base_model_A, base_model_B)
    new_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, MaskedLinear) or isinstance(m, MaskedLayerNorm):
            new_masks_A.append(m.b_matrix)

    new_masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, MaskedLinear) or isinstance(m, MaskedLayerNorm):
            new_masks_B.append(m.b_matrix)

    # Union should not change subnetwork_B in any way
    for ob, nb in zip(masks_B, new_masks_B):
        assert torch.all(
            torch.eq(ob, nb)
        ), "Subnetwork B was changed, this is unexpected..."

    for i, (old_A, mask_B, new_A) in enumerate(zip(old_masks_A, masks_B, new_masks_A)):
        renew = comparator(old_A, mask_B).float()
        return torch.all(torch.eq(renew, new_A))


@pytest.mark.skip()
def test_union():
    assert test_union_or_intersection(union_, torch.logical_or)


@pytest.mark.skip()
def test_intersection():
    assert test_union_or_intersection(intersection_, torch.logical_and)


def test_difference():
    mask_name_A = "copy"
    model_path_A = SAVE_PATH / mask_name_A / "pruned_model.ckpt"
    base_model_A = create_transformer_from_checkpoint(model_path_A)
    base_model_A = load_model(model_path_A, True, base_model_A, "continuous")
    mask_name_B = "reverse"
    model_path_B = SAVE_PATH / mask_name_B / "pruned_model.ckpt"
    base_model_B = create_transformer_from_checkpoint(model_path_B)
    base_model_B = load_model(model_path_B, True, base_model_B, "continuous")

    old_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            old_masks_A.append(m.b_matrix)

    old_masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            old_masks_B.append(m.b_matrix)

    difference_(base_model_A, base_model_B)
    new_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            new_masks_A.append(m.b_matrix)

    for old_A, old_B, new_A in zip(old_masks_A, old_masks_B, new_masks_A):
        for a_row, b_row, result_row in zip(old_A, old_B, new_A):
            if a_row.dim() == 0:
                if a_row.bool().item() is True and b_row.bool().item() is False:
                    assert result_row.bool().item() is True
                continue
            for a_v, b_v, r_v in zip(a_row, b_row, result_row):
                if a_v.bool().item() is True and b_v.bool().item() is False:
                    assert r_v.bool().item() is True
"""
