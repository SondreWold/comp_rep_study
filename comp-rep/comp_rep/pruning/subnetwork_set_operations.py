"""
Subnetwork set operations
"""

import copy
from typing import Callable

import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer


def complement_(subnetwork: MaskedLayer):
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (MaskedLayer): The subnetwork to invert the masks for.
    """
    assert subnetwork.ticket is True
    setattr(subnetwork, "b_matrix", ~subnetwork.b_matrix.bool())


def complement(subnetwork: MaskedLayer) -> MaskedLayer:
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (MaskedLayer): The subnetwork to invert the masks for.
    """
    assert subnetwork.ticket is True
    setattr(subnetwork, "b_matrix", ~subnetwork.b_matrix.bool())
    return subnetwork


def binary_function_(
    subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer, operator: Callable
):
    """
    Replaces the binary mask of subnetwork_A with the union of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """

    assert subnetwork_A.ticket is True
    assert subnetwork_B.ticket is True
    intermediate_result = operator(subnetwork_A.b_matrix, subnetwork_B.b_matrix)
    setattr(subnetwork_A, "b_matrix", intermediate_result)


def binary_function(
    subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer, operator: Callable
) -> MaskedLayer:
    """
    Replaces the binary mask of subnetwork_A with the union of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    assert subnetwork_A.ticket is True
    assert subnetwork_B.ticket is True
    subnetwork_A.b_matrix = operator(subnetwork_A.b_matrix, subnetwork_B.b_matrix)
    return subnetwork_A


def intersection_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_and)


def intersection(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer) -> MaskedLayer:
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    return binary_function(subnetwork_A, subnetwork_B, torch.logical_and)


def union_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_or)


def union(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer) -> MaskedLayer:
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    return binary_function(subnetwork_A, subnetwork_B, torch.logical_or)


def difference_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    complement_(subnetwork_B)
    intersection_(subnetwork_A, subnetwork_B)


def difference(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer) -> MaskedLayer:
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    modified_B = complement(subnetwork_B)
    return intersection(subnetwork_A, modified_B)


def union_model(subnetwork_A: Transformer, subnetwork_B: Transformer) -> Transformer:
    subnetwork_A = copy.deepcopy(subnetwork_A)
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            sub_A = union(sub_A, sub_B)
    return subnetwork_A


def intersection_model(
    subnetwork_A: Transformer, subnetwork_B: Transformer
) -> Transformer:
    subnetwork_A = copy.deepcopy(subnetwork_A)
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            sub_A = intersection(sub_A, sub_B)
    return subnetwork_A


def difference_model(
    subnetwork_A: Transformer, subnetwork_B: Transformer
) -> Transformer:
    subnetwork_A = copy.deepcopy(subnetwork_A)
    subnetwork_B = copy.deepcopy(subnetwork_B)
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            sub_A = difference(sub_A, sub_B)
    return subnetwork_A


def complement_model(subnetwork: Transformer) -> Transformer:
    subnetwork = copy.deepcopy(subnetwork)
    for sub in subnetwork.modules():
        if isinstance(sub, MaskedLayer):
            sub = complement(sub)
    return subnetwork


def union_model_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            union_(sub_A, sub_B)


def intersection_model_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            intersection_(sub_A, sub_B)


def difference_model_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    for sub_A, sub_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            difference_(sub_A, sub_B)


def complement_model_(subnetwork: Transformer):
    for sub in subnetwork.modules():
        if isinstance(sub, MaskedLayer):
            complement_(sub)
