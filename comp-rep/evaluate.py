"""
Evaluate trained models and subnetworks.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.utils import load_model, load_tokenizer, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data"
RESULT_DIR = CURR_FILE_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Evaluation script")

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--model_path", type=Path, help="Path to model checkpoint.")
    parser.add_argument(
        "--base_model_name", type=str, default="pcfgs_base", help="Name of base model."
    )
    parser.add_argument(
        "--pruning_task",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="base_tasks",
        choices=POSSIBLE_TASKS,
        help="Task to evaluate model on.",
    )
    parser.add_argument(
        "--is_masked", action="store_true", help="Whether the model is pruned."
    )
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--pruning_method", type=str, default="continuous", help="Pruning method."
    )

    return parser.parse_args()


def load_eval_data(path: Path, tokenizer: dict) -> SequenceDataset:
    """
    Load evaluation data from the given path using the provided tokenizer.

    Args:
        path (Path): The path to the evaluation data.
        tokenizer (dict): The tokenizer used to preprocess the data.

    Returns:
        SequenceDataset: The loaded evaluation dataset.
    """
    dataset = SequenceDataset(path, tokenizer)
    return dataset


def main() -> None:
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning evaluation with the config: \n{config_string}")

    # load model
    if args.is_masked:
        model_dir = args.save_path / args.pruning_task
        prediction_path = RESULT_DIR / args.pruning_task
        pl_pruner = LitPrunedModel.load_from_checkpoint(model_dir / "pruned_model.ckpt")
        model = pl_pruner.model

        if args.pruning_method == "continuous":
            pl_pruner.pruner.activate_ticket()
    else:
        model_dir = args.save_path / args.base_model_name
        prediction_path = RESULT_DIR / args.base_model_name
        pl_transformer = LitTransformer.load_from_checkpoint(
            model_dir / "base_model.ckpt"
        )
        model = pl_transformer.model

    tokenizer = load_tokenizer(model_dir)

    # load data
    data_path = DATA_DIR / args.eval_task / "test.csv"
    eval_dataset = SequenceDataset(data_path, tokenizer=tokenizer)
    input_vocabulary_size = len(tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )

    model = load_model(args.model_path, args.is_masked, args.pruning_method)
    # evaluate
    os.makedirs(prediction_path, exist_ok=True)

    searcher = GreedySearch(model, eval_dataset.output_language)
    accuracy = evaluate_generation(
        model, searcher, eval_loader, prediction_path, DEVICE
    )

    logging.info(f"Final accuracy was: {accuracy}")


if __name__ == "__main__":
    main()
