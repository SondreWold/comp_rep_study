import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from comp_rep.utils import setup_logging

ARG_TYPE_TO_METRIC = {
    "acc": "accuracy",
    "kl": "kl_div_faithfulness",
    "jsd": "jsd_faithfulness",
}


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
        "--result_path",
        type=Path,
        help="Path to the saved json results file for the performance",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Path to where you want to save the confusion matricies",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["acc", "kl", "jsd", "all"],
        default="acc",
        help="Which metric to plot for",
    )

    parser.add_argument(
        "--ablation_value",
        type=str,
        choices=["zero", "mean"],
        default="mean",
        help="Which value to ablate with",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])

    # Load the JSON data
    with open(args.result_path, "r") as f:
        data = json.load(f)

    objects = list(data.keys())
    tasks = "".join([t[0] for t in objects])
    num_objects = len(objects)

    if args.metric == "all":
        plot_metrics = list(ARG_TYPE_TO_METRIC.values())
    else:
        plot_metrics = [ARG_TYPE_TO_METRIC[args.metric]]

    for metric in plot_metrics:
        plt.clf()
        matrix = np.zeros((num_objects, num_objects))

        # Fill the matrix with similarity scores
        for circuit, results in data.items():
            i = objects.index(circuit)
            for task in results.keys():
                j = objects.index(task)
                score = results[task][metric]
                matrix[i, j] = score

        # Create the heatmap
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=1.5)
        sns.heatmap(
            matrix,
            annot=True,
            cmap="Blues",
            xticklabels=objects,
            yticklabels=objects,
            fmt=".2f",
        )
        output = (
            args.output_path
            / f"{tasks}_performance_evaluation_{metric}_{args.ablation_value}.pdf"
        )
        plt.savefig(output, format="pdf", bbox_inches="tight")
        # plt.show()
