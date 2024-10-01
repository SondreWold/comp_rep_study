import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec

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

    # Load the JSON data for the base_model performance
    with open("../../results/base_model_results.json", "r") as f:
        base_results = json.load(f)

    objects = list(data.keys())
    og_objects = list(data.keys())
    tasks = "".join([t[0] for t in objects])
    num_objects = len(objects)
    row_objects = num_objects
    col_objects = num_objects
    object_labels = [x.replace("swap_first_last", "swap") for x in objects]
    object_labels = [x.replace("remove_second", "rm_second") for x in object_labels]
    object_labels = [x.replace("remove_first", "rm_first") for x in object_labels]
    x_object_labels = object_labels
    y_object_labels = object_labels
    fontsize = 19
    font_scale = 1.6

    if args.metric == "all":
        plot_metrics = list(ARG_TYPE_TO_METRIC.values())
    else:
        plot_metrics = [ARG_TYPE_TO_METRIC[args.metric]]

    for metric in plot_metrics:
        plt.clf()
        if metric == "accuracy":
            local_y_object_labels = ["base"] + object_labels
            local_objects = ["base"] + objects
            local_row_objects = row_objects + 1
            vmin = 0.0
            vmax = 1.0
        else:
            local_y_object_labels = object_labels
            local_objects = objects
            local_row_objects = row_objects

        matrix = np.zeros((local_row_objects, col_objects))
        # Fill the matrix with similarity scores
        for circuit, results in data.items():
            i = local_objects.index(circuit)
            for task in results.keys():
                j = og_objects.index(task)
                score = results[task][metric]
                matrix[i, j] = score

        if metric == "jsd_faithfulness":
            vmin = 0.0
            vmax = 1.0
        if metric == "kl_div_faithfulness":
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)

        if metric == "accuracy":
            i = local_objects.index("base")
            for task_label, acc in base_results.items():
                if task_label in objects:
                    j = og_objects.index(task_label)
                    matrix[i, j] = acc[0]
                else:
                    print(task_label)
            fig = plt.figure(figsize=(9, 7.8))
            gs = GridSpec(2, 2, width_ratios=[10, 1], height_ratios=[1, 10], figure=fig)

            # Create subplots
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0])

            sns.set(font_scale=font_scale)

            h1 = sns.heatmap(
                matrix[:1],
                ax=ax1,
                annot=True,
                cbar=False,
                cmap="Blues",
                xticklabels=[],
                yticklabels=["base"],
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
            )
            # ax1.set_aspect("auto")
            ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=(fontsize - 1))
            sns.set(font_scale=font_scale)
            h2 = sns.heatmap(
                matrix[1:],
                ax=ax2,
                annot=True,
                cmap="Blues",
                cbar=False,
                xticklabels=x_object_labels,
                yticklabels=x_object_labels,
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
            )
            # ax2.set_aspect("auto")
            ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=fontsize)
            ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=(fontsize - 1))
            cbar_ax = fig.add_subplot(gs[1, 1])  # [left, bottom, width, height]
            sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=1))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax)
        else:
            plt.figure(figsize=(9, 7))

            sns.set(font_scale=font_scale)
            hmap = sns.heatmap(
                matrix,
                annot=True,
                cmap="Blues",
                xticklabels=x_object_labels,
                yticklabels=x_object_labels,
                fmt=".2f",
                vmin=vmin,
                vmax=vmax,
            )
            hmap.set_xticklabels(hmap.get_xticklabels(), fontsize=fontsize)
            hmap.set_yticklabels(hmap.get_yticklabels(), fontsize=fontsize)

        output = (
            args.output_path
            / f"{tasks}_performance_evaluation_{metric}_{args.ablation_value}.pdf"
        )
        plt.savefig(output, format="pdf", bbox_inches="tight")
        # plt.show()
