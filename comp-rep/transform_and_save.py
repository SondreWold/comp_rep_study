import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm
from comp_rep.pruning.masked_linear import ContinuousMaskLinear
from comp_rep.pruning.pruning import Pruner
from comp_rep.pruning.subnetwork_metrics import (
    intersection_remaining_weights_by_layer_and_module,
    intersection_remaining_weights_models,
    iom_by_layer_and_module,
    iou_by_layer_and_module,
)
from comp_rep.pruning.subnetwork_set_operations import (
    complement_model,
    difference_by_layer_and_module,
    intersection_by_layer_and_module,
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


def create_empty_set_model(reference_model):
    input_vocabulary_size = reference_model.input_vocabulary_size
    output_vocabulary_size = reference_model.output_vocabulary_size
    layers = reference_model.layers
    hidden_dim = reference_model.hidden_size
    dropout = reference_model.dropout
    model = Transformer(
        input_vocabulary_size, output_vocabulary_size, layers, hidden_dim, dropout
    )
    pruner = Pruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    return model


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


def compare_single_models(model_1, layers):
    results = {}
    for architecture_block in ["encoder", "decoder"]:
        for current_layer_iterator in layers:
            iou_linear = iou_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                False,
            )
            iou_norm = iou_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                False,
            )
            fraction_linear = intersection_remaining_weights_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                True,
            )
            fraction_norm = intersection_remaining_weights_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                True,
            )
            iom_linear = iom_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                False,
            )
            iom_norm = iom_by_layer_and_module(
                [model_1],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                False,
            )
            results[f"{architecture_block[0].upper()}_{current_layer_iterator}"] = {
                "iou_linear": iou_linear,
                "iou_norm": iou_norm,
                "iom_linear": iom_linear,
                "iom_norm": iom_norm,
                "fraction_linear": fraction_linear,
                "fraction_norm": fraction_norm,
            }
    return results


def compare_models(model_1, model_2, layers):
    results = {}
    for architecture_block in ["encoder", "decoder"]:
        for current_layer_iterator in layers:
            iou_linear = iou_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                False,
            )
            iou_norm = iou_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                False,
            )
            fraction_linear = intersection_remaining_weights_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                True,
            )
            fraction_norm = intersection_remaining_weights_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                True,
            )
            iom_linear = iom_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                False,
            )
            iom_norm = iom_by_layer_and_module(
                [model_1, model_2],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                False,
            )
            results[f"{architecture_block[0].upper()}_{current_layer_iterator}"] = {
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
    tokenizer = load_tokenizer(args.model_a)  # They all have the same tokenizer
    all_layers = [0, 1, 2, 3, 4, 5]
    functions = [model_b_name, model_c_name, model_a_name]  # copy, echo, copy_echo
    all_arch = None
    all_types = None

    overlap_result = {}

    overlap_result[f"{model_a_name}"] = run_eval(model_a, tokenizer, functions)
    overlap_result[f"{model_b_name}"] = run_eval(model_b, tokenizer, functions)
    overlap_result[f"{model_c_name}"] = run_eval(model_c, tokenizer, functions)

    # B UNION C UNION A
    """
    a_union_b_model = union_by_layer_and_module(model_a, model_b, None, None, None) 
    res_union_c_model = union_by_layer_and_module(a_union_b_model, model_c, None, None, None) 
    overlap_result[f"a_u_b_u_c"] = run_eval(res_union_c_model, tokenizer, functions)


    b_union_c_model = union_by_layer_and_module(model_b, model_c, None, None, None) 
    overlap_result[f"{model_b_name}+{model_c_name}"] = run_eval(b_union_c_model, tokenizer, functions)

    # C UNION B
    c_union_b_model = union_by_layer_and_module(model_c, model_b, None, None, None)
    overlap_result[f"{model_c_name}+{model_b_name}"] = run_eval(c_union_b_model, tokenizer, functions)
    c_union_b_model = union_by_layer_and_module(model_c, model_b, None, None, None)
    overlap_result[f"{model_c_name}"] = run_eval(model_c, tokenizer, functions)
    
    c_union_b_model_diff_b = union_by_layer_and_module(c_union_b_model, model_b, None, None, None)
    overlap_result[f"{model_c_name}+{model_b_name}-{model_b_name}"] = run_eval(c_union_b_model_diff_b, tokenizer, functions)
    # A - B
    a_difference_b_model = difference_by_layer_and_module(model_a, model_b, None, None, [ContinuousMaskLinear])
    overlap_result[f"{model_a_name}-{model_b_name}"] = run_eval(a_difference_b_model, tokenizer, functions)

    # A - C
    a_difference_c_model = difference_by_layer_and_module(model_a, model_c, None, None, [ContinuousMaskLinear])
    overlap_result[f"{model_a_name}-{model_c_name}"] = run_eval(a_difference_c_model, tokenizer, functions)
    """

    output_path = f"single_models.json"
    json_dict = json.dumps(overlap_result)
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
