import lightning as L
import torch.nn.functional as F


def get_logits_loss(
    model: L.LightningModule,
    batch: tuple,
) -> tuple:
    source_ids, source_mask, target_ids, target_mask, source_str, target_str = batch
    logits = model(batch)

    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1)
    loss = F.cross_entropy(logits.transpose(-2, -1), target_ids[:, 1:])
    return logits, loss


def get_regularized_logits_loss(
    model: L.LightningModule,
    mask_lambda: float,
    batch: tuple,
) -> tuple:
    logits, cross_entropy_loss = get_logits_loss(model, batch)
    norms = model.model.compute_l1_norm()
    mask_loss = mask_lambda * norms
    loss = cross_entropy_loss + mask_loss
    return logits, cross_entropy_loss, mask_loss, loss
