import copy
from typing import Callable

import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer


def complement(subnetwork: Transformer):
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (Transformer): The subnetwork to invert the masks for.
    """
    for m in subnetwork.modules():
        if isinstance(m, MaskedLayer):
            assert m.ticket is True
            m.compute_mask()  # In the continuous case, we need to know that the mask is already binary
            setattr(m, "b_matrix", (~m.b_matrix.bool()).float())


def complement_copy(subnetwork: Transformer) -> Transformer:
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (Transformer): The subnetwork to invert the masks for.
    """
    subnetwork_copy = copy.deepcopy(subnetwork)
    for m in subnetwork_copy.modules():
        if isinstance(m, MaskedLayer):
            assert m.ticket is True
            m.compute_mask()  # In the continuous case, we need to know that the mask is already binary
            setattr(m, "b_matrix", (~m.b_matrix.bool()).float())

    return subnetwork_copy


def in_place_binary_function(
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
            m_A.compute_mask()  # In the continuous case, we need to know that the mask is already binary

        if isinstance(m_B, MaskedLayer):
            assert m_B.ticket is True
            m_B.compute_mask()
            setattr(m_A, "b_matrix", operator(m_A.b_matrix, m_B.b_matrix))


def intersection(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B.

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    in_place_binary_function(subnetwork_A, subnetwork_B, torch.logical_and)


def union(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    in_place_binary_function(subnetwork_A, subnetwork_B, torch.logical_or)


def difference(subnetwork_A: Transformer, subnetwork_B: Transformer):
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (Transformer): The first subnetwork.
        subnetwork_B (Transformer): The second subnetwork.
    """
    subnetwork_B = complement_copy(subnetwork_B)
    intersection(subnetwork_A, subnetwork_B)
