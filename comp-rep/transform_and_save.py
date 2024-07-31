import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.constants import MASK_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.model import Transformer
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
    difference_model,
    intersection_by_layer_and_module,
    intersection_model,
    union_by_layer_and_module,
    union_model,
)
from comp_rep.utils import (
    create_transformer_from_checkpoint,
    iterate_subfolders,
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


def create_e_prime(path: Path):
    """
    Takes the intersection of all the circuits located at path (folder).

    Args:
        path (Path): The folder which holds all the individual circuits

    Returns:
        Transformer: the glue that holds it all together
    """
    E_prime = None
    for idx, circuit in enumerate(iterate_subfolders(path)):
        print(f"Intersecting: {circuit}")
        model_path = circuit / "continuous_pruned_model.ckpt"
        model = create_transformer_from_checkpoint(model_path)
        model = load_model(model_path, True, model)
        if idx == 0:
            E_prime = model
        else:
            E_prime = intersection_model(E_prime, model)
    return E_prime


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
    parser.add_argument(
        "--circuit_folder", type=Path, help="Path to the saved circuits"
    )
    parser.add_argument("--model_a", type=Path, help="Path to the saved model A")
    parser.add_argument("--model_b", type=Path, help="Path to the saved model B")
    parser.add_argument("--model_c", type=Path, help="Path to the saved model C")
    parser.add_argument(
        "--base_model", type=Path, help="Path to the base model circuit"
    )
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


def get_layer_fractions(E_prime, layers):
    results = {}
    for architecture_block in ["encoder", "decoder"]:
        for current_layer_iterator in layers:
            fraction_linear = intersection_remaining_weights_by_layer_and_module(
                [E_prime],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLinear],
                True,
            )
            fraction_norm = intersection_remaining_weights_by_layer_and_module(
                [E_prime],
                [architecture_block],
                [current_layer_iterator],
                [ContinuousMaskLayerNorm],
                True,
            )
            results[f"{architecture_block[0].upper()}_{current_layer_iterator}"] = {
                "fraction_linear": fraction_linear,
                "fraction_norm": fraction_norm,
            }
    fraction_linear = intersection_remaining_weights_by_layer_and_module(
        [E_prime], ["projection"], None, [ContinuousMaskLinear], True
    )
    try:
        fraction_norm = intersection_remaining_weights_by_layer_and_module(
            [E_prime], ["projection"], None, [ContinuousMaskLayerNorm], True
        )
    except:
        fraction_norm = 0.0
    results["projection_linear"] = fraction_linear
    results["projection_norm"] = fraction_norm
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

    E_prime = create_e_prime(args.circuit_folder)

    model_m_path = args.base_model / f"{args.pruning_method}_pruned_model.ckpt"
    model_m = create_transformer_from_checkpoint(model_m_path)
    model_m = load_model(model_m_path, True, model_m)

    # T_prime
    temp_step = difference_model(model_m, model_b)
    temp_step = union_model(temp_step, model_b)
    T_prime = union_model(temp_step, E_prime)
    result = run_eval(T_prime, tokenizer, MASK_TASKS)

    overlap_result = {}
    overlap_result["T_prime"] = result
    output_path = f"t_prime.json"
    json_dict = json.dumps(overlap_result)
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
