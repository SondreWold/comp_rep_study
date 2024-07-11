"""
Evaluate masked models on the individual functions
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.utils import ValidateTaskOptions, load_tokenizer, setup_logging
from evaluate import load_model

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
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=POSSIBLE_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )
    parser.add_argument(
        "--eval_data_path", type=Path, help="Path to evaluation dataset."
    )
    parser.add_argument("--save_path", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--pruning_method", type=str, default="continuous", help="Pruning method."
    )
    parser.add_argument("--output_path", type=Path, help="Output path for evaluation.")

    return parser.parse_args()


def run_mask_evaluation(
    save_path: Path,
    eval_data_path: Path,
    output_path: Path,
    pruning_method: Literal["sampled", "continuous"],
    tasks: list[str],
) -> dict:
    result = defaultdict(list)
    for mask_name in tasks:
        logging.info(f"Evaluating model: {mask_name}")
        path = save_path / mask_name
        model_path = path / "model.ckpt"
        model = load_model(model_path, True, pruning_method)
        tokenizer = load_tokenizer(path)
        for function in tasks:
            eval_path = eval_data_path / function / "test.csv"
            logging.info(f"Evaluating function: {function}")
            local_output_path = output_path / f"mask_{mask_name}_function_{function}"
            Path(local_output_path).mkdir(parents=True, exist_ok=True)
            eval_dataset = SequenceDataset(eval_path, tokenizer=tokenizer)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=64,
                collate_fn=CollateFunctor(),
                shuffle=False,
                num_workers=7,
                persistent_workers=True,
            )
            searcher = GreedySearch(model, eval_dataset.output_language)
            accuracy = evaluate_generation(
                model, searcher, eval_loader, local_output_path, DEVICE
            )
            result[mask_name].append(accuracy)
    return result


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning function evaluation with the config: \n{config_string}")
    result = run_mask_evaluation(
        args.save_path,
        args.eval_data_path,
        args.output_path,
        args.pruning_method,
        args.eval_tasks,
    )
    logging.info(result)
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = args.output_path / "function_evaluation_results.json"
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
