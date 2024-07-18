from pathlib import Path
from typing import Callable

import pytest
import torch

from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_set_operations import (
    complement,
    difference,
    intersection,
    union,
)
from comp_rep.utils import create_transformer_from_checkpoint, load_model

SAVE_PATH = Path("../base_models_trained")


def test_complement():
    mask_name = "copy"
    model_path = SAVE_PATH / mask_name / "pruned_model.ckpt"
    base_model = create_transformer_from_checkpoint(model_path)
    base_model = load_model(model_path, True, base_model, "continuous")

    # Need to do this loop in the test to get the "old masks"
    old_masks = []
    for m in base_model.modules():
        if isinstance(m, MaskedLayer):
            m.ticket = True
            m.compute_mask()
            old_masks.append(m.b_matrix)

    complement(base_model)
    new_masks = []
    for m in base_model.modules():
        if isinstance(m, MaskedLayer):
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
    base_model_A = load_model(model_path_A, True, base_model_A)
    mask_name_B = "remove_second"
    model_path_B = SAVE_PATH / mask_name_B / "pruned_model.ckpt"
    base_model_B = create_transformer_from_checkpoint(model_path_B)
    base_model_B = load_model(model_path_B, True, base_model_B)

    old_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, MaskedLayer):
            m.ticket = True
            m.compute_mask()
            old_masks_A.append(m.b_matrix)

    masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, MaskedLayer):
            m.ticket = True
            m.compute_mask()
            masks_B.append(m.b_matrix)

    function(base_model_A, base_model_B)
    new_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, MaskedLayer):
            new_masks_A.append(m.b_matrix)

    new_masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, MaskedLayer):
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
    mask_name_A = "append"
    model_path_A = SAVE_PATH / mask_name_A / "pruned_model.ckpt"
    base_model_A = create_transformer_from_checkpoint(model_path_A)
    base_model_A = load_model(model_path_A, True, base_model_A, "continuous")
    mask_name_B = "remove_first"
    model_path_B = SAVE_PATH / mask_name_B / "pruned_model.ckpt"
    base_model_B = create_transformer_from_checkpoint(model_path_B)
    base_model_B = load_model(model_path_B, True, base_model_B, "continuous")

    old_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, MaskedLayer):
            m.ticket = True
            m.compute_mask()
            old_masks_A.append(m.b_matrix)

    old_masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, MaskedLayer):
            m.ticket = True
            m.compute_mask()
            old_masks_B.append(m.b_matrix)

    difference(base_model_A, base_model_B)
    new_masks_A = []
    for m in base_model_A.modules():
        if isinstance(m, MaskedLayer):
            new_masks_A.append(m.b_matrix)

    new_masks_B = []
    for m in base_model_B.modules():
        if isinstance(m, MaskedLayer):
            new_masks_B.append(m.b_matrix)

    """
    for ob, nb in zip(old_masks_B, new_masks_B):
        assert not torch.all(torch.eq(ob, nb)), "Subnetwork B was not changed, this is unexpected..."
    """
    for old_A, old_B, new_A, new_B in zip(
        old_masks_A, old_masks_B, new_masks_A, new_masks_B
    ):
        for old_A_row, old_B_row, new_A_row, new_B_row in zip(
            old_A, old_B, new_A, new_B
        ):
            for old_A_col, old_B_col, new_A_col, new_B_col in zip(
                old_A_row, old_B_row, new_A_row, new_B_row
            ):
                if old_A_col.bool() and not old_B_col.bool():
                    assert new_A_col.bool()
