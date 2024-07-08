import lightning as L
import torch.nn.functional as F


def get_logits_loss(
    pl_model: L.LightningModule,
    batch: tuple,
) -> tuple:
    """
    Calculates the logits loss for a given model and batch.

    Args:
        model (L.LightningModule): The model to calculate the logits loss for.
        batch (tuple): The batch of data to calculate the logits loss on.

    Returns:
        tuple: A tuple containing the logits and the calculated loss.
    """
    _, _, target_ids, _, _, _ = batch
    logits = pl_model(batch)

    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1)
    loss = F.cross_entropy(logits.transpose(-2, -1), target_ids[:, 1:])
    return logits, loss


def get_regularized_logits_loss(
    pl_model: L.LightningModule,
    mask_lambda: float,
    batch: tuple,
) -> tuple:
    """
    Calculates the regularized logits loss based on the input model, mask lambda, and batch.

    Parameters:
        model (L.LightningModule): The LightningModule model.
        mask_lambda (float): The lambda value for masking.
        batch (tuple): The input batch tuple.

    Returns:
        tuple: A tuple containing logits, cross entropy loss, mask loss, and total loss.
    """
    logits, cross_entropy_loss = get_logits_loss(pl_model, batch)
    norms = pl_model.model.compute_l1_norm()
    mask_loss = mask_lambda * norms
    loss = cross_entropy_loss + mask_loss
    return logits, cross_entropy_loss, mask_loss, loss
