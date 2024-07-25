import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear
from comp_rep.pruning.subnetwork_metrics import (
    iom_by_layer_and_module,
    iou_by_layer_and_module,
)
from comp_rep.pruning.subnetwork_set_operations import (
    difference_by_layer_and_module,
    union_by_layer_and_module,
)
from comp_rep.utils import (
    create_transformer_from_checkpoint,
    load_model,
    load_tokenizer,
)

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


def run_eval(model, tokenizer, functions):
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


def compare_models(model_1, model_2, layers):
    results = {}
    for current_layer_iterator in layers:
        iou_linear = iou_by_layer_and_module(
            [model_1, model_2], [current_layer_iterator], [ContinuousMaskLinear], False
        )
        iou_norm = iou_by_layer_and_module(
            [model_1, model_2],
            [current_layer_iterator],
            [ContinuousMaskLayerNorm],
            False,
        )
        fraction_linear = iou_by_layer_and_module(
            [model_1, model_2], [current_layer_iterator], [ContinuousMaskLinear], True
        )
        fraction_norm = iou_by_layer_and_module(
            [model_1, model_2],
            [current_layer_iterator],
            [ContinuousMaskLayerNorm],
            True,
        )
        iom_linear = iom_by_layer_and_module(
            [model_1, model_2], [current_layer_iterator], [ContinuousMaskLinear], False
        )
        iom_norm = iom_by_layer_and_module(
            [model_1, model_2],
            [current_layer_iterator],
            [ContinuousMaskLayerNorm],
            False,
        )
        results[current_layer_iterator] = {
            "iou_linear": iou_linear,
            "iou_norm": iou_norm,
            "iom_linear": iom_linear,
            "iom_norm": iom_norm,
            "fraction_linear": fraction_linear,
            "fraction_norm": fraction_norm,
        }
    return results


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
    model_c = load_model(model_c_path, True, model_c)
    print(model_a_name, model_b_name, model_c_name)
    # tokenizer = load_tokenizer(args.model_a)  # They all have the same tokenizer
    all_layers = [0, 1, 2, 3, 4, 5]
    all_types = [ContinuousMaskLayerNorm, ContinuousMaskLinear]

    overlap_result = {}
    # PURE MODELS
    overlap_result[f"{model_a_name}_vs_{model_b_name}"] = compare_models(
        model_a, model_b, all_layers
    )
    overlap_result[f"{model_a_name}_vs_{model_c_name}"] = compare_models(
        model_a, model_c, all_layers
    )
    overlap_result[f"{model_b_name}_vs_{model_c_name}"] = compare_models(
        model_b, model_c, all_layers
    )

    model_d = difference_by_layer_and_module(model_a, model_b, all_layers, all_types)
    model_d_name = f"{model_a_name}_diff_{model_b_name}"
    model_e = difference_by_layer_and_module(model_a, model_c, all_layers, all_types)
    model_e_name = f"{model_a_name}_diff_{model_c_name}"
    model_f = union_by_layer_and_module(model_b, model_c, all_layers, all_types)
    model_f_name = f"{model_b_name}_union_{model_c_name}"

    overlap_result[f"{model_d_name}_vs_{model_a_name}"] = compare_models(
        model_d, model_a, all_layers
    )
    overlap_result[f"{model_d_name}_vs_{model_b_name}"] = compare_models(
        model_d, model_b, all_layers
    )
    overlap_result[f"{model_d_name}_vs_{model_c_name}"] = compare_models(
        model_d, model_c, all_layers
    )

    overlap_result[f"{model_e_name}_vs_{model_a_name}"] = compare_models(
        model_e, model_a, all_layers
    )
    overlap_result[f"{model_e_name}_vs_{model_b_name}"] = compare_models(
        model_e, model_b, all_layers
    )
    overlap_result[f"{model_e_name}_vs_{model_c_name}"] = compare_models(
        model_e, model_c, all_layers
    )

    overlap_result[f"{model_f_name}_vs_{model_a_name}"] = compare_models(
        model_f, model_a, all_layers
    )
    overlap_result[f"{model_f_name}_vs_{model_b_name}"] = compare_models(
        model_f, model_b, all_layers
    )
    overlap_result[f"{model_f_name}_vs_{model_c_name}"] = compare_models(
        model_f, model_c, all_layers
    )

    overlap_result[f"{model_d_name}_vs_{model_e_name}"] = compare_models(
        model_d, model_e, all_layers
    )
    overlap_result[f"{model_d_name}_vs_{model_f_name}"] = compare_models(
        model_d, model_f, all_layers
    )
    overlap_result[f"{model_e_name}_vs_{model_f_name}"] = compare_models(
        model_e, model_f, all_layers
    )

    output_path = f"overlap_results_{model_a_name}.json"
    json_dict = json.dumps(overlap_result)
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
