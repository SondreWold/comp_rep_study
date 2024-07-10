"""
Pytorch Lightning module for the Pruned Transformer model.
"""

import argparse
from typing import Any, Literal

import lightning as L
import torch.optim as optim
from torch import nn

import wandb
from comp_rep.loss import get_regularized_logits_loss
from comp_rep.pruning.pruning import Pruner


class LitPrunedModel(L.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.args = args
        self.pruning_method = pruning_method

        self.model = model
        self.pruner = Pruner(self.model, pruning_method, maskedlayer_kwargs)

    def forward(self, batch: tuple):
        """
        Forward pass of the model.

        Args:
            batch (tuple): The input batch.

        Returns:
            torch.Tensor: The logits of the model.
        """
        source_ids, source_mask, target_ids, target_mask, _, _ = batch
        # Left shift the targets so that the last token predicts the EOS
        logits = self.model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]
        return logits

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch: tuple, batch_idx: int):
        """
        A single training step in the LightningModule.

        Args:
            train_batch (tuple): The batch of training data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the training step.
        """
        _, _, mask_loss, loss = get_regularized_logits_loss(
            self, self.args.mask_lambda, train_batch
        )

        self.log_dict(
            {"train_loss": loss, "cross_entropy_loss": loss, "l1_norm_loss": mask_loss},
            on_step=True,
            logger=True,
            batch_size=self.args.train_batch_size,
        )
        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        """
        Calculates the validation loss for a given batch of data.

        Args:
            val_batch (tuple): The batch of data to calculate the validation loss on.
            batch_idx (int): The index of the batch.

        Returns:
            float: The calculated validation loss.
        """
        _, _, _, loss = get_regularized_logits_loss(
            self, self.args.mask_lambda, val_batch
        )
        self.log(
            "val_loss",
            loss,
            batch_size=self.args.val_batch_size,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_start(self):
        """
        Callback function that is executed at the start of each training epoch.
        If the pruning method is set to "continuous", the pruner's ticket is deactivated.

        Returns:
            None
        """
        if self.pruning_method == "continuous":
            self.pruner.deactivate_ticket()

    def on_validation_epoch_start(self):
        """
        Callback function that is executed at the start of each validation epoch.
        If the pruning method is set to "continuous", the pruner's ticket is activated.

        Returns:
            None
        """
        if self.pruning_method == "continuous":
            self.pruner.activate_ticket()

    def on_train_epoch_end(self):
        """
        Updates hyperparameters at the end of a training epoch and logs the average remaining weights.
        """
        if self.pruning_method == "continuous":
            self.pruner.update_hyperparameters()

        remaining_weights = self.pruner.get_remaining_weights()
        wandb.log(remaining_weights)  # need to use the wandb log here
