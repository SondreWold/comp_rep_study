import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.pruning.subnetwork_mask_metrics import (
    iom_by_layer_and_module,
    iou_by_layer_and_module,
)
from comp_rep.utils import ValidateTaskOptions, load_model, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent


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
    parser.add_argument(
        "--result_dir", type=Path, help="Path to where the results are saved."
    )

    parser.add_argument(
        "--architecture_blocks",
        nargs="+",
        type=str,
        default=["encoder", "decoder"],
        help="Name of architecture blocks to consider.",
    )
    parser.add_argument(
        "--layer_idx",
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        action=ValidateTaskOptions,
        help="Layers to consider.",
    )

    # Eval Configs
    parser.add_argument(
        "--circuit_names",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )

    # Mask Configs
    parser.add_argument(
        "--mean_model_path", type=Path, help="Path to the saved mean ablated models."
    )
    parser.add_argument(
        "--zero_model_path", type=Path, help="Path to the saved zero ablated models."
    )
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=["weights", "activations"],
        default="activations",
        help="Pruning type (either 'weights' or 'activations').",
    )
    return parser.parse_args()


def bar_plot(labels: list, values: list, path: Path) -> None:
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    # Ensure values is a list of lists, each inner list containing two values
    assert all(
        len(v) == 2 for v in values
    ), "Each item in values should have two elements"

    # Sort the labels and values together
    sorted_pairs = sorted(zip(values, labels), key=lambda x: sum(x[0]), reverse=True)
    values, labels = zip(*sorted_pairs)

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    # Plotting the bars with different colors and patterns
    rects1 = ax.bar(
        x - width / 2, [v[0] for v in values], width, label="IoU", color="skyblue"
    )
    rects2 = ax.bar(
        x + width / 2,
        [v[1] for v in values],
        width,
        label="IoM",
        color="lightgreen",
        hatch="//",
    )

    # Setting y-axis limits from 0 to 100%
    ax.set_ylim(0, 100)

    # Adding labels and title
    ax.set_xlabel("Subtask", fontsize=12)
    ax.set_ylabel("Overlap (%)", fontsize=12)
    # ax.set_title("Comparison of IoU and IoM per Subtask", fontsize=14)

    # Adding gridlines for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    # Customize x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.0f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    autolabel(rects1)
    autolabel(rects2)

    # Add legend
    ax.legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    plt.show()


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(
        f"\nRunning calculation of global sparsity with the config: \n{config_string}"
    )

    overlap = {}
    for task in args.circuit_names:
        mean_model_path = (
            args.mean_model_path
            / f"{task}"
            / f"{args.pruning_type}_continuous_mean_pruned_model.ckpt"
        )
        mean_model = load_model(
            model_path=mean_model_path,
            is_masked=True,
            model=None,
            return_pl=False,
        )
        zero_model_path = (
            args.zero_model_path
            / f"{task}"
            / f"{args.pruning_type}_continuous_zero_pruned_model.ckpt"
        )
        zero_model = load_model(
            model_path=zero_model_path,
            is_masked=True,
            model=None,
            return_pl=False,
        )
        iou = (
            iou_by_layer_and_module(
                model_list=[mean_model, zero_model],
                architecture_blocks=args.architecture_blocks,
                layer_idx=args.layer_idx,
                fraction=True,
                average=False,
            )
            * 100
        )
        iom = (
            iom_by_layer_and_module(
                model_list=[mean_model, zero_model],
                architecture_blocks=args.architecture_blocks,
                layer_idx=args.layer_idx,
                fraction=True,
                average=True,
            )
            * 100
        )
        logging.info(f"Task: {task}, IoU: {iou}, IoM: {iom}")
        overlap[f"{task}"] = [iou, iom]
    labels = list(overlap.keys())
    values = list(overlap.values())
    os.makedirs(args.result_dir, exist_ok=True)
    save_path = args.result_dir / f"{args.pruning_type}_task_ablation_value_overlap.pdf"
    bar_plot(labels, values, save_path)


if __name__ == "__main__":
    main()
