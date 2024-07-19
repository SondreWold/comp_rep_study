"""
Subnetwork set operations
"""

import copy
from typing import Callable

import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer


def complement_(subnetwork: Transformer):
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (Transformer): The subnetwork to invert the masks for.
    """
    for m in subnetwork.modules():
        if isinstance(m, MaskedLayer):
            assert m.ticket is True
            setattr(m, "b_matrix", ~m.b_matrix.bool())


def complement(subnetwork: Transformer) -> Transformer:
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (Transformer): The subnetwork to invert the masks for.
    """
    subnetwork_copy = copy.deepcopy(subnetwork)
    for m in subnetwork_copy.modules():
        if isinstance(m, MaskedLayer):
            assert m.ticket is True
            setattr(m, "b_matrix", ~m.b_matrix.bool())

    return subnetwork_copy


def binary_function_(
    subnetwork_A: Transformer, subnetwork_B: Transformer, operator: Callable
):
    """
    Replaces the binary mask of subnetwork_A with the union of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """

    for m_A, m_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(m_A, MaskedLayer):
            assert m_A.ticket is True

        if isinstance(m_B, MaskedLayer):
            assert m_B.ticket is True

        if isinstance(m_A, MaskedLayer) and isinstance(m_B, MaskedLayer):
            intermediate_result = operator(m_A.b_matrix, m_B.b_matrix)
            setattr(m_A, "b_matrix", intermediate_result)


def binary_function(
    subnetwork_A: Transformer, subnetwork_B: Transformer, operator: Callable
) -> Transformer:
    """
    Replaces the binary mask of subnetwork_A with the union of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    new_model = copy.deepcopy(subnetwork_A)

    for m_A, m_B in zip(new_model.modules(), subnetwork_B.modules()):
        if isinstance(m_A, MaskedLayer):
            assert m_A.ticket is True

        if isinstance(m_B, MaskedLayer):
            assert m_B.ticket is True

        if isinstance(m_A, MaskedLayer) and isinstance(m_B, MaskedLayer):
            intermediate_result = operator(m_A.b_matrix, m_B.b_matrix)
            setattr(m_A, "b_matrix", intermediate_result)

    return new_model


def intersection_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_and)


def intersection(subnetwork_A: Transformer, subnetwork_B: Transformer) -> Transformer:
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    return binary_function(subnetwork_A, subnetwork_B, torch.logical_and)


def union_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_or)


def union(subnetwork_A: Transformer, subnetwork_B: Transformer) -> Transformer:
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    return binary_function(subnetwork_A, subnetwork_B, torch.logical_or)


def difference_(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    complement_(subnetwork_B)
    intersection_(subnetwork_A, subnetwork_B)


def difference(subnetwork_A: Transformer, subnetwork_B: Transformer) -> Transformer:
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    modified_B = complement(subnetwork_B)
    return intersection(subnetwork_A, modified_B)
