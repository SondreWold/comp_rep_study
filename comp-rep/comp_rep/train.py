import argparse
import logging
from pathlib import Path
from typing import Callable

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataset import CollateFunctor, SequenceDataset
from evaluator import GreedySearch, evaluate_generation
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from model import Transformer
from torch.utils.data import DataLoader
from utils import (
    ValidatePredictionPath,
    ValidateSavePath,
    create_tokenizer_dict,
    save_tokenizer,
    set_seed,
    validate_args,
)

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--train_data_path", type=Path)
    parser.add_argument("--val_data_path", type=Path)
    parser.add_argument(
        "--save_path",
        action=ValidateSavePath,
        type=Path,
        help="Path to save trained model at",
    )
    parser.add_argument(
        "--predictions_path",
        action=ValidatePredictionPath,
        type=Path,
        help="Path to save predictions at",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1860)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


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


def main(args: argparse.Namespace):
    validate_args(args)
    set_seed(args.seed)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning training loop with the config: \n{config_string}")
    train_dataset = SequenceDataset(args.train_data_path)
    train_tokenizer = create_tokenizer_dict(
        train_dataset.input_language, train_dataset.output_language
    )
    input_vocabulary_size = len(train_tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(train_tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size
    val_dataset = SequenceDataset(args.val_data_path, tokenizer=train_tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )
    model = LitTransformer(args)
    wandb_logger = WandbLogger(
        entity="pmmon-Ludwig MaximilianUniversity of Munich",
        project="compositional_representations",
        config=config,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_path,
        filename="model",
        save_top_k=1,
        mode="min",
    )
    trainer = L.Trainer(
        callbacks=[checkpoint_callback], max_epochs=args.epochs, logger=wandb_logger
    )
    trainer.fit(model, train_loader, val_loader)
    save_tokenizer(args.save_path, train_tokenizer)

    if args.eval:
        searcher = GreedySearch(model.model, val_dataset.output_language)
        accuracy = evaluate_generation(
            model.model, searcher, val_loader, args.predictions_path
        )
        logging.info(f"Final accuracy was: {accuracy}")
        wandb.log({"final_accuracy": accuracy})  # type: ignore[attr-defined]
    wandb.finish()  # type: ignore[attr-defined]


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main(parse_args())
