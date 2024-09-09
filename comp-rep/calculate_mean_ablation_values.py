import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from nnsight import NNsight
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.utils import (
    create_transformer_from_checkpoint,
    load_model,
    load_tokenizer,
    set_seed,
)

CURR_FILE_PATH = Path(__file__).resolve()
DATA_DIR = CURR_FILE_PATH.parents[1] / "data" / "function_tasks"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        "Utility script for calculating mean ablation values"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the model you wish to get the mean ablation values from",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        help="Path to where you want to save the mean ablation values",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="The number of samples in the distribution you want to mean ablate from",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # LOADING TO BE REPLACED
    model_path = args.model_path / "base_model.ckpt"
    model = create_transformer_from_checkpoint(model_path)
    model = load_model(model_path, False, model)
    model.eval()
    model = NNsight(model)

    tokenizer = load_tokenizer(args.model_path)
    train_dataset = SequenceDataset(
        DATA_DIR / args.subtask / "train.csv", tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=CollateFunctor(),
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )

    distribution = defaultdict(list)
    sampled = 0
    for source_ids, source_mask, target_ids, target_mask, src_str, target_str in tqdm(
        train_loader
    ):
        for i in range(len(model.encoder.layers)):
            # The execution of the hook is only run upon exiting the context of the with statement... Hence, you have to repeat the tracer per call.
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                e_ff_output = model.encoder.layers[i].feed_forward.output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                e_sa_output = model.encoder.layers[i].self_attention.nns_output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_ff_output = model.decoder.layers[i].feed_forward.output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_ca_output = model.decoder.layers[i].cross_attention.nns_output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_sa_output = model.decoder.layers[i].self_attention.nns_output.save()

            distribution[f"model.encoder.layers[{i}].feed_forward"].append(
                torch.mean(e_ff_output).item()
            )
            distribution[f"model.encoder.layers[{i}].self_attention"].append(
                torch.mean(e_sa_output).item()
            )
            distribution[f"model.decoder.layers[{i}].self_attention"].append(
                torch.mean(d_sa_output).item()
            )
            distribution[f"model.decoder.layers[{i}].cross_attention"].append(
                torch.mean(d_ca_output).item()
            )
            distribution[f"model.decoder.layers[{i}].feed_forward"].append(
                torch.mean(d_ff_output).item()
            )

        sampled += args.batch_size
        if sampled > args.n:
            break

    for module, values in distribution.items():
        distribution.update({module: np.mean(values)})

    with open(args.save_path / f"{args.subtask}_mean_ablation_values.json", "w") as f:
        json.dump(distribution, f, indent=4)


if __name__ == "__main__":
    main()
