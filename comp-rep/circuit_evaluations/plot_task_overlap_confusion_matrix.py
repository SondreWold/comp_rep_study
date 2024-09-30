import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from comp_rep.utils import setup_logging


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Plot script")

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
        "--result_path", type=Path, help="Path to the saved json results file"
    )
    parser.add_argument(
        "--output_path", type=Path, help="Path to where you want to save the heatmap"
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["IoU", "IoM"],
        default="IoU",
        help="Whether to plot IoU or IoM ",
    )

    parser.add_argument(
        "--ablation_value",
        type=str,
        choices=["zero", "mean"],
        default="zero",
        help="Which value to ablate with",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    if (args.ablation_value == "mean" and "zero" in str(args.result_path)) or (
        args.ablation_value == "zero" and "mean" in str(args.result_path)
    ):
        logging.info(
            "Are you sure you have provided the path to a model that matches the provided ablation value?"
        )

    # Load the JSON data
    with open(args.result_path, "r") as f:
        data = json.load(f)

    # Define the list of objects

    objects = [
        "echo",
        "copy",
        "repeat",
        "reverse",
        "swap_first_last",
        "shift",
        "append",
        "prepend",
        "remove_first",
        "remove_second",
    ]
    object_labels = [x.replace("swap_first_last", "swap") for x in objects]
    object_labels = [x.replace("remove_second", "rm_second") for x in object_labels]
    object_labels = [x.replace("remove_first", "rm_first") for x in object_labels]

    # Create an empty 10x10 matrix
    matrix = np.zeros((10, 10))

    # Fill the matrix with similarity scores
    for pair, scores in data.items():
        obj1, obj2 = pair.split("-")
        i, j = objects.index(obj1), objects.index(obj2)
        # Use IoU score for similarity
        similarity = scores[args.metric]
        matrix[i, j] = similarity
        matrix[j, i] = similarity  # Mirror the matrix

    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(matrix, 1)

    # Create the heatmap
    plt.figure(figsize=(11, 10))
    sns.set(font_scale=1.65)
    hmap = sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        xticklabels=object_labels,
        yticklabels=object_labels,
        fmt=".2f",
    )
    hmap.set_yticklabels(hmap.get_yticklabels(), fontsize=20)
    hmap.set_xticklabels(hmap.get_xticklabels(), rotation=45, fontsize=20)

    plt.tight_layout()
    output = args.output_path / f"task_overlap_{args.metric}_{args.ablation_value}.pdf"
    # plt.savefig(output, format="pdf")
    plt.show()
