from pathlib import Path
from typing import Callable

import pytest
import torch

from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm, MaskedLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear, MaskedLinear
from comp_rep.pruning.subnetwork_set_operations import (
    complement,
    difference,
    intersection,
    union,
)
from comp_rep.utils import (
    ValidateTaskOptions,
    create_transformer_from_checkpoint,
    load_model,
    load_tokenizer,
    setup_logging,
)

SAVE_PATH = Path("../base_models_trained")


def test_complement():
    mask_name = "copy"
    model_path = SAVE_PATH / mask_name / "pruned_model.ckpt"
    base_model = create_transformer_from_checkpoint(model_path)
    base_model = load_model(model_path, True, base_model, "continuous")

    # Need to do this loop in the test to get the "old masks"
    old_masks = []
    for m in base_model.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            old_masks.append(m.b_matrix)

    complement(base_model)
    new_masks = []
    for m in base_model.modules():
        if isinstance(m, ContinuousMaskLinear) or isinstance(
            m, ContinuousMaskLayerNorm
        ):
            new_masks.append(m.b_matrix)

    for ob, nb in zip(old_masks, new_masks):
        assert not torch.all(
            torch.eq(ob, nb)
        ), "Subnetwork was not changed, this is unexpected..."

    for old, new in zip(old_masks, new_masks):
        renew = ~old.bool()
        assert torch.all(torch.eq(renew, new.bool()))


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


def test_union():
    assert test_union_or_intersection(union, torch.logical_or)


def test_intersection():
    assert test_union_or_intersection(intersection, torch.logical_and)


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

    difference(base_model_A, base_model_B)
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
