"""
Subnetwork metrics
"""

import copy
from typing import List, Optional, Type

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_set_operations import intersection_, union_
from comp_rep.utils import get_current_layer_from_module_name


def intersection_remaining_weights(
    masked_layers: List[MaskedLayer], fraction: bool = True
) -> float:
    """
    Calculates the sum or fraction of remaining weights in the intersection of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining weights in the layer, the sum if False. Defaults to True.

    Returns:
        float: The the sum or fraction of remaining weights in the intersection of the masked layers.

    Raises:
        AssertionError: If the list of masked layers is empty.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    intersection_layer = copy.deepcopy(masked_layers[0])

    for masked_layer in masked_layers:
        intersection_(intersection_layer, masked_layer)

    return intersection_layer.compute_remaining_weights(fraction)


def union_remaining_weights(
    masked_layers: List[MaskedLayer], fraction: bool = True
) -> float:
    """
    Calculates the sum or fraction of remaining weights in the union of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining weights in the layer, the sum if False. Defaults to True.

    Returns:
        float: The the sum or fraction of remaining weights in the union of the masked layers.

    Raises:
        AssertionError: If the list of masked layers is empty.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    union_layer = copy.deepcopy(masked_layers[0])

    for masked_layer in masked_layers:
        union_(union_layer, masked_layer)

    return union_layer.compute_remaining_weights(fraction)


def intersection_over_union(
    masked_layers: List[MaskedLayer], fraction: bool = False
) -> float:
    """
    Calculate the intersection over union of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining weights in the layer, the sum if False. Defaults to False.

    Returns:
        float: The intersection over union of the masked layers.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    return intersection_remaining_weights(
        masked_layers, fraction
    ) / union_remaining_weights(masked_layers, fraction)


def intersection_over_minimum(
    masked_layers: List[MaskedLayer], fraction: bool = False
) -> float:
    """
    Calculates the intersection over minimum of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining weights in the layer, the sum if False. Defaults to False.

    Returns:
        float: The intersection over minimum of the masked layers.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"
    minimum_frac = min(
        [
            masked_layer.compute_remaining_weights(fraction)
            for masked_layer in masked_layers
        ]
    )

    return intersection_remaining_weights(masked_layers, fraction) / minimum_frac


def iou_by_layer_and_module(
    model_list: List[Transformer],
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    fraction: bool = False,
) -> float:
    """
    Calculates the Intersection over Union (IoU) for dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        fraction (bool, optional): Whether to compute the fraction of remaining weights in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoU value for the specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"

    intersection_weights = 0.0
    union_weights = 0.0
    first_model = model_list[0]

    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    intersection_weights += intersection_remaining_weights(
                        masked_layers, fraction
                    )
                    union_weights += union_remaining_weights(masked_layers, fraction)

    return intersection_weights / union_weights


def iom_by_layer_and_module(
    model_list: List[Transformer],
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    fraction: bool = False,
) -> float:
    """
    Calculates the Intersection over Minimum (IoM) for dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        fraction (bool, optional): Whether to compute the fraction of remaining weights in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoM value for the specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    intersection_weights = 0.0
    weights: List[float] = [0.0] * len(model_list)

    first_model = model_list[0]
    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    print(current_layer)
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    intersection_weights += intersection_remaining_weights(
                        masked_layers, fraction
                    )
                    weights = [
                        w_sum + masked_layer.compute_remaining_weights(fraction)
                        for w_sum, masked_layer in zip(weights, masked_layers)
                    ]

    return intersection_weights / min(weights)


def iou_models(model_list: List[Transformer], fraction: bool = False) -> float:
    """
    Calculates the Intersection over Union (IoU) for all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        fraction (bool, optional): Whether to compute the fraction of remaining weights in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoU value.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return iou_by_layer_and_module(model_list=model_list, fraction=fraction)


def iom_models(model_list: List[Transformer], fraction: bool = False) -> float:
    """
    Calculates the Intersection over Minimum (IoM) for all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        fraction (bool, optional): Whether to compute the fraction of remaining weights in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoM value.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return iom_by_layer_and_module(model_list=model_list, fraction=fraction)
