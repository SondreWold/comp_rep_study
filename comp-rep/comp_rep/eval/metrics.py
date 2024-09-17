"""
Evaluation metrics
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def jensen_shannon_divergence_from_logits(
    p_logits: Tensor, q_probs: Tensor, eps: float = 1e-10
) -> float:
    """
    Computes the normalized Jensen-Shannon Divergence (JSD) between two probability distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.

    Returns:
        float: The Jensen-Shannon Divergence between the two probability distributions.
    """
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = q_probs

    # Normalize probabilities to ensure they sum to 1
    p_probs = p_probs / p_probs.sum(dim=-1, keepdim=True)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

    p_probs = p_probs + eps
    q_probs = q_probs + eps

    # compute average distribution M
    m = 0.5 * (p_probs + q_probs)

    # Compute KL divergences
    kl_pm = torch.sum(p_probs * (torch.log(p_probs) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_probs * (torch.log(q_probs) - torch.log(m)), dim=-1)

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)  # Shape: [batch_size, seq_len]

    # Average over sequence and batch
    if len(jsd.shape) == 2:
        jsd = jsd.mean(dim=-1)
    jsd_batch_mean = jsd.mean(dim=-1)
    jsd_normalized = jsd_batch_mean / math.log(2)

    return jsd_normalized.item()


def jsd_faithfulness(p_logits: Tensor, q_probs: Tensor, eps: float = 1e-10) -> float:
    """
    Computes the faithfulness of a probability distribution based on the Jensen-Shannon Divergence (JSD) between two distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.

    Returns:
        float: The faithfulness of the probability distribution, calculated as 1 minus the JSD between the two distributions.
    """
    return 1.0 - jensen_shannon_divergence_from_logits(p_logits, q_probs, eps=eps)
