"""
Model pruning script.
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from comp_rep.callbacks.eval_callbacks import TestGenerationCallback
from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.utils import (
    ValidateSavePath,
    ValidateWandbPath,
    init_pruner_by_name,
    load_tokenizer,
    save_tokenizer,
    set_seed,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data"
SWEEP_DIR = CURR_FILE_DIR / "sweeps"
RESULT_DIR = CURR_FILE_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Model Pruning for Subnetwork Identification.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=1860, help="Random seed.")
    parser.add_argument(
        "--base_model_name", type=str, default="pcfgs_base", help="Name of base model."
    )
    parser.add_argument(
        "--acc_freq",
        type=int,
        default=20,
        help="Frequency of epochs with which generation accuracy is evaluated.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model in addition to training.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Whether to perform a hyperparameter sweep.",
    )

    # Path configs
    parser.add_argument(
        "--save_path",
        action=ValidateSavePath,
        type=Path,
        help="Path to save trained model at.",
    )
    parser.add_argument(
        "--wandb_path",
        action=ValidateWandbPath,
        type=Path,
        help="Path to save wandb metadata.",
    )

    # Pruning configs
    parser.add_argument(
        "--subtask",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=["weights", "activations"],
        default="activations",
        help="Pruning type (either 'weights' or 'activations').",
    )
    parser.add_argument(
        "--pruning_method",
        type=str,
        choices=["continuous", "sampled"],
        default="continuous",
        help="Pruning method.",
    )
    parser.add_argument("--num_masks", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--mask_initial_value", type=float, default=0.05)
    parser.add_argument(
        "--mask_lambda",
        type=float,
        default=1e-7,
        help="Lambda hyperparameter for continuous pruning.",
    )
    parser.add_argument(
        "--max_temp",
        type=int,
        default=200,
        help="Maximum temperature for continuous pruning.",
    )

    # Train parameter configs
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Validation batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--eta_min", type=float, default=0.0, help="Minimum learning rate."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Size of hidden dimension."
    )
    parser.add_argument("--layers", type=int, default=6, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout parameter.")
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.0,
        help="Value for gradient clipping. If 0 no gradient clipping is applied.",
    )
    parser.add_argument(
        "--gradient_clip_alg",
        type=str,
        choices=["norm", "value"],
        default="norm",
        help="Algorithm for gradient clipping.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    setup_logging(args.verbose)

    # sweep config
    config = vars(args)
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning pruning training loop with the config: \n{config_string}")

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

    wandb_logger = WandbLogger(
        entity="pmmon-Ludwig MaximilianUniversity of Munich",
        project="circomp-mask-training",
        name=f"{args.pruning_method}_{args.subtask}_{formatted_datetime}",
        config=config,
        save_dir=args.wandb_path,
    )

    # load data
    data_dir = DATA_DIR / "function_tasks" if args.subtask != "base_tasks" else DATA_DIR
    base_model_dir = args.save_path / args.base_model_name
    pruned_model_dir = args.save_path / args.subtask

    train_tokenizer = load_tokenizer(base_model_dir)
    train_dataset = SequenceDataset(
        path=data_dir / args.subtask / "train.csv", tokenizer=train_tokenizer
    )
    input_vocabulary_size = len(train_tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(train_tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size
    val_dataset = SequenceDataset(
        path=data_dir / args.subtask / "test.csv", tokenizer=train_tokenizer
    )

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
    lit_transformer_model = LitTransformer.load_from_checkpoint(
        checkpoint_path=base_model_dir / "base_model.ckpt",
        args=args,
    )  # type: ignore

    transformer_model = lit_transformer_model.model
    transformer_model_hparams = {
        "input_vocabulary_size": transformer_model.input_vocabulary_size,
        "output_vocabulary_size": transformer_model.output_vocabulary_size,
        "num_transformer_layers": transformer_model.num_transformer_layers,
        "hidden_size": transformer_model.hidden_size,
        "dropout": transformer_model.dropout,
    }

    # init pruner
    pruning_methods_kwargs: dict[str, Any] = {}

    if args.pruning_method == "continuous":
        pruning_methods_kwargs["temperature_increase"] = args.max_temp ** (
            1.0 / args.epochs
        )
        pruning_methods_kwargs["mask_initial_value"] = args.mask_initial_value
    elif args.pruning_method == "sampled":
        pruning_methods_kwargs["tau"] = args.tau
        pruning_methods_kwargs["num_masks"] = args.num_masks
    else:
        raise ValueError("Invalid pruning strategy method provided")

    pruner_kwargs = {
        "model": transformer_model,
        "model_hparams": transformer_model_hparams,
        "pruning_method": args.pruning_method,
        "maskedlayer_kwargs": pruning_methods_kwargs,
    }
    pruner = init_pruner_by_name(
        pruning_type=args.pruning_type, pruner_kwargs=pruner_kwargs
    )
    args.T_max = args.epochs * len(train_loader)

    pl_pruned_model = LitPrunedModel(args=args, pruner=pruner)
    searcher = GreedySearch(pl_pruned_model.pruner.model, val_dataset.output_language)

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    acc_callback = TestGenerationCallback(
        frequency=args.acc_freq,
        searcher=searcher,
        test_loader=val_loader,
        device=DEVICE,
    )
    callbacks: list = [lr_monitor, acc_callback]

    if not args.sweep:
        # checkpoint saving
        model_ckpt_name = f"{args.pruning_type}_{args.pruning_method}_pruned_model.ckpt"
        model_ckpt_path = pruned_model_dir / model_ckpt_name
        if os.path.exists(model_ckpt_path):
            logging.warning(
                f"File: {model_ckpt_path} already exists. File will be overwritten."
            )
            os.remove(model_ckpt_path)

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=pruned_model_dir,
            filename=model_ckpt_name.split(".")[0],
            save_top_k=1,
            mode="min",
        )
        callbacks.append(checkpoint_callback)

    # train pruner
    trainer = L.Trainer(
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_alg,
        max_epochs=args.epochs,
        logger=wandb_logger,
    )
    trainer.fit(pl_pruned_model, train_loader, val_loader)
    save_tokenizer(pruned_model_dir, train_tokenizer)

    # evaluate model
    if args.eval:
        prediction_path = (
            RESULT_DIR / args.subtask / args.pruning_type / args.pruning_method
        )
        os.makedirs(prediction_path, exist_ok=True)

        searcher = GreedySearch(
            pl_pruned_model.pruner.model, val_dataset.output_language
        )
        accuracy = evaluate_generation(
            pl_pruned_model.pruner.model,
            searcher,
            val_loader,
            predictions_path=prediction_path,
            device=DEVICE,
        )
        logging.info(f"Final accuracy was: {accuracy}")
        wandb.log({"final_accuracy": accuracy})  # type: ignore[attr-defined]
    wandb.finish()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
