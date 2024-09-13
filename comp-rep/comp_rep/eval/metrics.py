"""
Evaluation metrics
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def jensen_shannon_divergence_from_logits(
    p_logits: Tensor, q_probs: Tensor, eps: float = 1e-10
) -> float:
    """
    Computes the Jensen-Shannon Divergence (JSD) between two probability distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.

    Returns:
        float: The Jensen-Shannon Divergence between the two probability distributions.
    """
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")

    p_probs = F.softmax(p_logits, dim=-1)

    # compute average distribution M
    m = 0.5 * (p_probs + q_probs)

    # compute KL divergence between
    kl_pm = kl_loss(torch.log(m + eps), p_probs)
    kl_qm = kl_loss(torch.log(m + eps), q_probs)

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)

    return jsd.item()


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
