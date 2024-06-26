import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dataset import CollateFunctor, SequenceDataset
from model import Transformer
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import create_tokenizer_dict

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--train_data_path", type=Path)
    parser.add_argument("--val_data_path", type=Path)
    parser.add_argument("--save_path", type=Path)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def get_logits_loss(model: nn.Module, batch: tuple, criterion: _Loss) -> tuple:
    source_ids, source_mask, target_ids, target_mask, source_str, target_str = batch
    source_ids = source_ids.to(DEVICE)
    source_mask = source_mask.to(DEVICE)
    target_ids = target_ids.to(DEVICE)
    target_mask = target_mask.to(DEVICE)
    # Left shift the targets so that the last token predicts the EOS
    logits = model(
        source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
    )  # [batch size, max seq len, vocab]
    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1)
    loss = criterion(logits.transpose(-2, -1), target_ids[:, 1:])
    return logits, loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: _Loss,
    scheduler: LRScheduler,
) -> float:
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        logits, loss = get_logits_loss(model, batch, criterion)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


@torch.no_grad()
def val(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: _Loss,
) -> float:
    model.eval()
    val_loss = 0.0
    for batch in tqdm(val_loader):
        logits, loss = get_logits_loss(model, batch, criterion)
        val_loss += loss.item()
    return val_loss / len(val_loader)


def main(args: argparse.Namespace):
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning training loop with the config: \n{config_string}")
    train_dataset = SequenceDataset(args.train_data_path)
    train_tokenizer = create_tokenizer_dict(
        train_dataset.input_language, train_dataset.output_language
    )
    input_vocabulary_size = len(train_tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(train_tokenizer["output_language"]["index2word"])
    val_dataset = SequenceDataset(args.val_data_path, tokenizer=train_tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=False,
    )
    model = Transformer(
        input_vocabulary_size=input_vocabulary_size,
        output_vocabulary_size=output_vocabulary_size,
        num_transformer_layers=args.layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, None)
        val_loss = val(model, val_loader, criterion)
        logging.info(
            f"Epoch {epoch}. Train loss: {round(train_loss, 2)}, Val loss: {round(val_loss, 2)}"
        )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main(parse_args())
