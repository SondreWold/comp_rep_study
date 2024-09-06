"""
Pytorch Lightning module for the Pruned Transformer model.
"""

import argparse
from typing import Any, Dict

import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
from comp_rep.loss import get_regularized_logits_loss
from comp_rep.models.model import Transformer
from comp_rep.pruning.pruner import Pruner
from comp_rep.utils import pruner_from_config


class LitPrunedModel(L.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace,
        pruner: Pruner,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pruner"])
        self.args = args
        self.pruner = pruner

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
        logits = self.pruner.model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]
        return logits

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
        """
        optimizer = optim.AdamW(self.pruner.model.parameters(), lr=self.args.lr)
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.T_max,
            eta_min=self.args.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def training_step(self, train_batch: tuple, batch_idx: int):
        """
        A single training step in the LightningModule.

        Args:
            train_batch (tuple): The batch of training data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the training step.
        """
        self.pruner.compute_and_update_masks()
        _, cross_entropy_loss, mask_loss, loss = get_regularized_logits_loss(
            self, self.args.mask_lambda, train_batch
        )

        self.log_dict(
            {
                "train_loss": loss,
                "cross_entropy_loss": cross_entropy_loss,
                "l1_norm_loss": mask_loss,
            },
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
        self.pruner.compute_and_update_masks()
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
        self.pruner.deactivate_ticket()

    def on_validation_epoch_start(self):
        """
        Callback function that is executed at the start of each validation epoch.
        If the pruning method is set to "continuous", the pruner's ticket is activated.

        Returns:
            None
        """
        self.pruner.activate_ticket()

    def on_train_epoch_end(self):
        """
        Updates hyperparameters at the end of a training epoch and logs the average remaining mask elements.
        """
        self.pruner.update_hyperparameters()

        remaining_mask_elements = self.pruner.get_remaining_mask()
        wandb.log(remaining_mask_elements)  # need to use the wandb log here

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Saves the state dictionary, configuration, and type of the pruner model to the checkpoint dictionary.

        Args:
            checkpoint (Dict[str, Any]): The dictionary to save the pruner state, configuration, and type.

        Returns:
            None
        """
        checkpoint["pruner_state_dict"] = self.pruner.model.state_dict()
        checkpoint["pruner_config"] = self.pruner.get_config()
        checkpoint["pruning_type"] = self.pruner.__class__.__name__

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Loads the pruner model from a given checkpoint.

        Args:
            checkpoint (Dict[str, Any]): A dictionary containing the pruner type, configuration, model hyperparameters, and state dictionary.

        Returns:
            None
        """
        pruning_type = checkpoint["pruning_type"]
        pruner_config = checkpoint["pruner_config"]
        model_hparams = checkpoint["pruner_config"]["model_hparams"]

        print(pruner_config)
        print(model_hparams)
        model = Transformer(**model_hparams)
        self.pruner = pruner_from_config(
            pruning_type=pruning_type, config=pruner_config, model=model
        )
        self.pruner.model.load_state_dict(checkpoint["pruner_state_dict"])
