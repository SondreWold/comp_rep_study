import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.pruning.subnetwork_set_operations import difference, intersection, union
from comp_rep.utils import (
    create_transformer_from_checkpoint,
    load_model,
    load_tokenizer,
)

args_to_operation = {
    "intersection": intersection,
    "union": union,
    "difference": difference,
}


CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data/function_tasks"
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
    parser.add_argument("--model_c", type=Path, help="Path to the saved model C")
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


def transform(modelA, modelB, operation):
    op = args_to_operation[operation]
    return op(modelA, modelB)


def run_eval(model, tokenizer, functions: list):
    res = []
    for function in functions:
        eval_path = DATA_DIR / function / "test.csv"
        eval_dataset = SequenceDataset(eval_path, tokenizer=tokenizer)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=8,
            collate_fn=CollateFunctor(),
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )
        searcher = GreedySearch(model, eval_dataset.output_language)
        accuracy = evaluate_generation(model, searcher, eval_loader, Path(""), DEVICE)
        res.append(accuracy)
    return res


def main() -> None:
    args = parse_args()
    # Example: a: copy_echo, b: copy, c:echo
    model_a_name = str(args.model_a).split("/")[-1]
    model_b_name = str(args.model_b).split("/")[-1]
    model_c_name = str(args.model_c).split("/")[-1]
    model_a_path = args.model_a / f"{args.pruning_method}_pruned_model.ckpt"
    model_b_path = args.model_b / f"{args.pruning_method}_pruned_model.ckpt"
    model_c_path = args.model_c / f"{args.pruning_method}_pruned_model.ckpt"
    model_a = create_transformer_from_checkpoint(model_a_path)
    model_a = load_model(model_a_path, True, model_a)
    model_b = create_transformer_from_checkpoint(model_b_path)
    model_b = load_model(model_b_path, True, model_b)
    model_c = create_transformer_from_checkpoint(model_c_path)
    model_c = load_model(model_b_path, True, model_c)
    print(model_a_name, model_b_name, model_c_name)
    tokenizer = load_tokenizer(args.model_a)  # They all have the same tokenizer

    # model: b+c
    result = {}
    model_b_union_b = transform(model_b, model_b, "union")
    result["copy_union_copy"] = run_eval(
        model_b_union_b, tokenizer, [model_b_name, model_c_name, model_a_name]
    )

    model_b_intersection_b = transform(model_b, model_b, "intersection")
    result["copy_intersection_copy"] = run_eval(
        model_b_intersection_b, tokenizer, [model_b_name, model_c_name, model_a_name]
    )

    output_path = "composite_evaluation_results.json"
    json_dict = json.dumps(result)
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
