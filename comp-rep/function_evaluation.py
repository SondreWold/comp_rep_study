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

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.eval.evaluator import eval_task
from comp_rep.utils import (
    ValidateTaskOptions,
    load_model,
    load_tokenizer,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data/function_tasks"
RESULT_DIR = CURR_FILE_DIR / "function_evaluations"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Evaluation script")

    # General Configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )

    # Eval Configs
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=POSSIBLE_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )

    # Mask Configs
    parser.add_argument("--save_path", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=["weights", "activations"],
        default="activations",
        help="Pruning type (either 'weights' or 'activations').",
    )
    parser.add_argument(
        "--pruning_method", type=str, default="continuous", help="Pruning method."
    )
    parser.add_argument(
        "--ablation_value",
        type=str,
        choices=["zero", "mean"],
        default="zero",
        help="Which value to ablate with",
    )
    return parser.parse_args()


def run_mask_evaluation(
    save_path: Path,
    pruning_type: Literal["weights", "activations"],
    pruning_method: Literal["sampled", "continuous"],
    ablation_value: Literal["zero", "mean"],
    tasks: list[str],
) -> dict:
    """
    Evaluates masked models on the individual functions.

    Args:
        save_path (Path): The path to the saved models.
        pruning_type (Literal["weights", "activations"]): The pruning type.
        pruning_method (Literal["sampled", "continuous"]): The pruning method.
        tasks (list[str]): A list of tasks to evaluate the model on.

    Returns:
        dict: A dictionary containing the evaluation results for each task.
    """
    result = defaultdict(list)

    for mask_name in tasks:
        logging.info(f"Evaluating model: {mask_name}")

        # load masked model
        model_dir = save_path / mask_name
        model_name = (
            f"{pruning_type}_{pruning_method}_{ablation_value}_pruned_model.ckpt"
        )
        model_path = model_dir / model_name

        model = load_model(model_path=model_path, is_masked=True)
        tokenizer = load_tokenizer(model_dir)

        # eval model
        for task_name in tasks:
            data_path = DATA_DIR / task_name / "test.csv"
            output_dir = RESULT_DIR / f"mask_{mask_name}_function_{task_name}"

            task_accuracy = eval_task(
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                eval_data_path=data_path,
                output_dir=output_dir,
            )
            result[mask_name].append(task_accuracy)

    return result


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning function evaluation with the config: \n{config_string}")

    result = run_mask_evaluation(
        save_path=args.save_path,
        pruning_type=args.pruning_type,
        pruning_method=args.pruning_method,
        ablation_value=args.ablation_value,
        tasks=args.eval_tasks,
    )
    logging.info(result)

    # save result
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = RESULT_DIR / "function_evaluation_results.json"
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
