"""
Subnetwork metrics
"""

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_set_operations import intersection_model, union_model


def intersection_over_union(
    subnetwork_A: Transformer, subnetwork_B: Transformer
) -> float:
    intersection_result = intersection_model(subnetwork_A, subnetwork_B)
    union_result = union_model(subnetwork_A, subnetwork_B)
    intersection_sum = 0.0
    union_sum = 0.0
    for m_A, m_B in zip(intersection_result.modules(), union_result.modules()):
        if isinstance(m_A, MaskedLayer) and isinstance(m_B, MaskedLayer):
            intersection_sum += m_A.b_matrix.sum().item()
            union_sum += m_B.b_matrix.sum().item()
    return intersection_sum / union_sum


def intersection_over_minimum(
    subnetwork_A: Transformer, subnetwork_B: Transformer
) -> float:
    intersection_result = intersection_model(subnetwork_A, subnetwork_B)
    intersection_sum = 0
    A_masks = 0
    B_masks = 0
    for m_A, m_B in zip(subnetwork_A.modules(), subnetwork_B.modules()):
        if isinstance(m_A, MaskedLayer) and isinstance(m_B, MaskedLayer):
            A_masks += m_A.b_matrix.sum().item()
            B_masks += m_B.b_matrix.sum().item()
    for m_I in intersection_result.modules():
        if isinstance(m_I, MaskedLayer) and isinstance(m_I, MaskedLayer):
            intersection_sum += m_I.b_matrix.sum().item()
    return intersection_sum / min(A_masks, B_masks)
