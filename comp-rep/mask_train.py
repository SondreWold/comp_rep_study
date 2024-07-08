import argparse
import logging
from pathlib import Path
from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.utils import (
    ValidatePredictionPath,
    ValidateSavePath,
    load_tokenizer,
    save_tokenizer,
    set_seed,
    setup_logging,
    validate_args,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Model Pruning for Subnetwork Identification.")

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
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


def main() -> None:
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning pruning training loop with the config: \n{config_string}")

    wandb_logger = WandbLogger(
        entity="pmmon-Ludwig MaximilianUniversity of Munich",
        project="compositional_representations",
        config=config,
    )

    # load data
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

    # load model
    transformer_model = LitTransformer.load_from_checkpoint(
        args.pretrained_model_path, args=args
    )

    # init pruner
    pruning_methods_kwargs: dict[str, Any] = {}
    if args.pruning_method == "continuous":
        pruning_methods_kwargs["temperature_increase"] = args.max_temp ** (
            1.0 / args.epochs
        )
    elif args.pruning_method == "sampled":
        pass

    pl_pruned_model = LitPrunedModel(
        args=args,
        model=transformer_model.model,
        pruning_method=args.pruning_method,
        maskedlayer_kwargs=pruning_methods_kwargs,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_path,
        filename="model",
        save_top_k=1,
        mode="min",
    )

    # train pruner
    trainer = L.Trainer(
        callbacks=[checkpoint_callback], max_epochs=args.epochs, logger=wandb_logger
    )
    trainer.fit(pl_pruned_model, train_loader, val_loader)
    if args.save_path:
        save_tokenizer(args.save_path, train_tokenizer)

    # evaluate model
    if args.eval:
        searcher = GreedySearch(pl_pruned_model.model, val_dataset.output_language)
        accuracy = evaluate_generation(
            pl_pruned_model.model,
            searcher,
            val_loader,
            args.predictions_path,
            device=DEVICE,
        )
        logging.info(f"Final accuracy was: {accuracy}")
        wandb.log({"final_accuracy": accuracy})  # type: ignore[attr-defined]
    wandb.finish()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
