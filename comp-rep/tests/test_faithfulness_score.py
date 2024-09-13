"""
Unit test for JSD metric
"""

import pytest
import torch
import torch.nn.functional as F

from comp_rep.eval.metrics import jensen_shannon_divergence_from_logits


def test_same_distributions():
    """
    Test that the JSD between identical distributions is approximately zero.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_probs = F.softmax(p_logits, dim=-1)
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert abs(jsd - 0.0) < 1e-6, "JSD should be zero for identical distributions."


def test_different_distributions():
    """
    Test that the JSD between different distributions is positive.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_probs = torch.tensor([[0.1, 0.1, 0.8]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd > 0.0, "JSD should be positive for different distributions."


def test_symmetry():
    """
    Test that the JSD is symmetric: JSD(P || Q) == JSD(Q || P).
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_logits = torch.tensor([[2.0, 1.0, 0.5]])
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    jsd_pq = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    jsd_qp = jensen_shannon_divergence_from_logits(q_logits, p_probs)

    assert abs(jsd_pq - jsd_qp) < 1e-6, "JSD should be symmetric between P and Q."


def test_uniform_distributions():
    """
    Test that the JSD is zero when both distributions are uniform.
    """
    p_logits = torch.log(torch.tensor([[1.0, 1.0, 1.0]]))
    q_probs = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert abs(jsd - 0.0) < 1e-6, "JSD should be zero for uniform distributions."


def test_normalization():
    """
    Test that the function handles unnormalized logits.
    """
    p_logits = torch.tensor([[10.0, 0.0, -10.0]])
    q_probs = torch.tensor([[0.7, 0.2, 0.1]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd > 0.0, "JSD should handle unnormalized logits and return positive value."


def test_batch_inputs():
    """
    Test that the function correctly handles batch inputs.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
    q_probs = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert isinstance(jsd, float), "JSD should return a scalar value for batch inputs."


def test_invalid_inputs():
    """
    Test that the function raises an error for inputs of mismatched sizes.
    """
    p_logits = torch.tensor([[1.0, 2.0]])
    q_probs = torch.tensor([[0.5, 0.5, 0.0]])
    with pytest.raises(RuntimeError):
        jensen_shannon_divergence_from_logits(p_logits, q_probs)


def test_large_values():
    """
    Test the function with large logits and probabilities to check numerical stability.
    """
    p_logits = torch.tensor([[1000.0, 2000.0, 3000.0]])
    q_probs = torch.tensor([[0.0, 0.0, 1.0]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd >= 0.0, "JSD should handle large values without numerical issues."
