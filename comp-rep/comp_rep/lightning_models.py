import argparse

import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from loss import get_logits_loss
from model import Transformer


class LitTransformer(L.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.model = Transformer(
            input_vocabulary_size=self.args.input_vocabulary_size,
            output_vocabulary_size=self.args.output_vocabulary_size,
            num_transformer_layers=args.layers,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )

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
        logits, loss = get_logits_loss(self.model, train_batch, F.cross_entropy)
        self.log(
            "train_loss",
            loss,
            batch_size=self.args.train_batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        logits, loss = get_logits_loss(self.model, val_batch, F.cross_entropy)
        self.log(
            "val_loss",
            loss,
            batch_size=self.args.val_batch_size,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
