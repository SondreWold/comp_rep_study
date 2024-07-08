import argparse
import logging
from pathlib import Path
from typing import Any

import lightning as L
from dataset import CollateFunctor, SequenceDataset
from evaluator import GreedySearch, evaluate_generation
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from masked_lightning_models import LitTransformer
from torch.utils.data import DataLoader
from utils import (
    ValidatePredictionPath,
    ValidateSavePath,
    load_tokenizer,
    save_tokenizer,
    set_seed,
    validate_args,
)

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Training script")
    parser.add_argument("--train_mask_path", type=Path)
    parser.add_argument("--val_mask_path", type=Path)
    parser.add_argument("--pretrained_model_path", type=Path)
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument(
        "--save_path",
        action=ValidateSavePath,
        type=Path,
        help="Path to save the trained masked model at",
    )
    parser.add_argument(
        "--predictions_path",
        action=ValidatePredictionPath,
        type=Path,
        help="Path to save predictions at",
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--pruning_method", type=str, default="continuous")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--mask_lambda", type=float, default=1e-7)
    parser.add_argument("--max_temp", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1860)
    parser.add_argument("--eval", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace):
    validate_args(args)
    set_seed(args.seed)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning mask_training loop with the config: \n{config_string}")
    train_tokenizer = load_tokenizer(args.tokenizer_path)
    train_dataset = SequenceDataset(args.train_mask_path, tokenizer=train_tokenizer)
    input_vocabulary_size = len(train_tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(train_tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size
    val_dataset = SequenceDataset(args.val_mask_path, tokenizer=train_tokenizer)
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
    model = LitTransformer.load_from_checkpoint(args.pretrained_model_path, args=args)
    pruning_methods_kwargs: dict[str, Any] = {}
    model.init_mask_model(args.pruning_method, pruning_methods_kwargs)
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
    if args.save_path:
        save_tokenizer(args.save_path, train_tokenizer)

    if args.eval:
        searcher = GreedySearch(model.model.model, val_dataset.output_language)
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
