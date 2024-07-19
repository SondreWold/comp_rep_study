import argparse
from pathlib import Path

import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.subnetwork_set_operations import difference, intersection, union
from comp_rep.utils import (
    create_transformer_from_checkpoint,
    load_tokenizer,
    save_tokenizer,
)

args_to_operation = {
    "intersection": intersection,
    "union": union,
    "difference": difference,
}


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Transformation script")
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--output_path", type=Path, help="Path to save the transformed model"
    )
    parser.add_argument("--model_a", type=Path, help="Path to the saved model A")
    parser.add_argument("--model_b", type=Path, help="Path to the saved model B")
    parser.add_argument(
        "--operation",
        type=str,
        default="union",
        choices=["union", "intersection", "difference"],
        help="Set operation method.",
    )
    parser.add_argument(
        "--pruning_method", type=str, default="continuous", help="Set pruning method."
    )
    return parser.parse_args()


def load_and_transform(path_a: Path, path_b: Path, operation: str) -> Transformer:
    base_model_a = create_transformer_from_checkpoint(path_a)
    base_model_b = create_transformer_from_checkpoint(path_b)

    # Transform
    op = args_to_operation[operation]
    transformed_model = op(base_model_a, base_model_b)
    return transformed_model


def main() -> None:
    args = parse_args()
    model_a_path = args.model_a / "pruned_model.ckpt"
    model_b_path = args.model_b / "pruned_model.ckpt"

    # Get new model
    transformed_model = load_and_transform(model_a_path, model_b_path, args.operation)
    tokenizer = load_tokenizer(args.model_a)  # They all have the same tokenizer

    # Now we need to save together with the hyperparams of A
    checkpoint = torch.load(model_a_path, map_location=torch.device("cpu"))
    checkpoint["state_dict"] = transformed_model.state_dict()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_path / "pruned_model.ckpt")
    save_tokenizer(args.output_path, tokenizer)


if __name__ == "__main__":
    main()
