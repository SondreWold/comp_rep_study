import argparse
from typing import Any

import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from loss import get_regularized_logits_loss
from model import Transformer
from pruning.pruning import MaskedModel


class LitTransformer(L.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = Transformer(
            input_vocabulary_size=self.args.input_vocabulary_size,
            output_vocabulary_size=self.args.output_vocabulary_size,
            num_transformer_layers=args.layers,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )

    def init_mask_model(self, pruning_method: str, maskedlayer_kwargs: dict[str, Any]):
        self.pruning_method = pruning_method
        if self.pruning_method == "continuous":
            temperature_increase = self.args.max_temp ** (1.0 / self.args.epochs)
            maskedlayer_kwargs["temperature_increase"] = temperature_increase
        self.model = MaskedModel(self.model, pruning_method, maskedlayer_kwargs)

    def forward(self, batch):
        source_ids, source_mask, target_ids, target_mask, source_str, target_str = batch
        # Left shift the targets so that the last token predicts the EOS
        logits = self.model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]
        return logits

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        logits, ce, mask_loss, loss = get_regularized_logits_loss(
            self.model, self.args.mask_lambda, train_batch, F.cross_entropy
        )

        self.log(
            "train_loss",
            loss,
            batch_size=self.args.train_batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def on_train_epoch_start(self):
        if self.pruning_method == "continuous":
            self.model.deactivate_ticket()

    def on_validation_epoch_start(self):
        if self.pruning_method == "continuous":
            self.model.activate_ticket()

    def on_train_epoch_end(self):
        if self.pruning_method == "continuous":
            self.model.update_hyperparameters()
        avg_remaining_weights = self.model.get_remaining_weights()
        self.log("avg_remaining_weights", avg_remaining_weights)

    def validation_step(self, val_batch, batch_idx):
        logits, ce, mask_loss, loss = get_regularized_logits_loss(
            self.model, self.args.mask_lambda, val_batch, F.cross_entropy
        )
        self.log(
            "val_loss",
            loss,
            batch_size=self.args.val_batch_size,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
