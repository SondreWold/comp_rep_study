"""
Evaluate trained models and subnetworks.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.utils import load_tokenizer, set_seed, setup_logging, validate_args

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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
    parser.add_argument("--eval_data_path", type=Path)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--model_path", type=Path)
    parser.add_argument("--tokenizer_path", type=Path)
    parser.add_argument("--is_masked", action="store_true")
    parser.add_argument("--pruning_method", type=str, default="continuous")
    parser.add_argument("--predictions_path", type=Path)

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
    validate_args(args)

    set_seed(args.seed)
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning evaluation with the config: \n{config_string}")

    # load tokenizer and data
    tokenizer = load_tokenizer(args.tokenizer_path)
    eval_dataset = SequenceDataset(args.eval_data_path, tokenizer=tokenizer)
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

    # load model
    if args.is_masked:
        pl_pruner = LitPrunedModel.load_from_checkpoint(args.model_path)
        model = pl_pruner.model

        if args.pruning_method == "continuous":
            pl_pruner.pruner.activate_ticket()
    else:
        pl_transformer = LitTransformer.load_from_checkpoint(args.model_path)
        model = pl_transformer.model

    # evaluate
    searcher = GreedySearch(model, eval_dataset.output_language)
    accuracy = evaluate_generation(
        model, searcher, eval_loader, args.predictions_path, DEVICE
    )

    logging.info(f"Final accuracy was: {accuracy}")


if __name__ == "__main__":
    main()
