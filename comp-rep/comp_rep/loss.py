from typing import Callable

import torch
import torch.nn as nn


def get_logits_loss(
    model: nn.Module,
    batch: tuple,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple:
    source_ids, source_mask, target_ids, target_mask, source_str, target_str = batch
    # Left shift the targets so that the last token predicts the EOS
    logits = model(
        source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
    )  # [batch size, max seq len, vocab]
    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1)
    loss = criterion(logits.transpose(-2, -1), target_ids[:, 1:])
    return logits, loss


def get_regularized_logits_loss(
    model: nn.Module,
    mask_lambda: float,
    batch: tuple,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple:
    source_ids, source_mask, target_ids, target_mask, source_str, target_str = batch
    # Left shift the targets so that the last token predicts the EOS
    logits = model(
        source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
    )  # [batch size, max seq len, vocab]
    norms = model.compute_l1_norm()
    mask_loss = mask_lambda * norms
    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1)
    ce = criterion(logits.transpose(-2, -1), target_ids[:, 1:])
    loss = ce + mask_loss
    return logits, ce, mask_loss, loss
