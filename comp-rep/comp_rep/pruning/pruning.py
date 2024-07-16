"""
Modules to find subnetworks via model pruning
"""

from collections import defaultdict
from typing import Any, Literal

import torch
import torch.nn as nn

from comp_rep.pruning.masked_layernorm import ContinuousMaskLayerNorm, MaskedLayerNorm
from comp_rep.pruning.masked_linear import (
    ContinuousMaskLinear,
    MaskedLinear,
    SampledMaskLinear,
)


class Pruner:
    """
    A model wrapper that applies a masking strategy for model pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ):
        self.model = model
        self.init_model_pruning(pruning_method, maskedlayer_kwargs)

    def freeze_initial_model(self) -> None:
        """
        Freezes the initial model parameters to prevent updates during training.
        """
        for p in self.model.parameters():
            p.requires_grad = False

    def init_model_pruning(
        self,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ) -> None:
        """
        Initializes the model pruning by replacing linear layers with masked layers.

        Args:
            pruning_method (Literal["continuous", "sampled"]): The pruning method to deploy.
            maskedlayer_kwargs (dict[str, Any]): Additional keyword-arguments for the masked layer.
        """
        self.freeze_initial_model()

        def replace_linear(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    if pruning_method == "continuous":
                        setattr(
                            module,
                            name,
                            ContinuousMaskLinear(
                                child.weight, child.bias, **maskedlayer_kwargs
                            ),
                        )
                    elif pruning_method == "sampled":
                        setattr(
                            module,
                            name,
                            SampledMaskLinear(
                                child.weight, child.bias, **maskedlayer_kwargs
                            ),
                        )
                    else:
                        raise ValueError("Invalid pruning strategy method provided")
                else:
                    replace_linear(child)

        def replace_layernorm(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    if pruning_method == "continuous":
                        setattr(
                            module,
                            name,
                            ContinuousMaskLayerNorm(
                                child.normalized_shape,
                                child.weight,
                                child.bias,
                                child.eps,
                                **maskedlayer_kwargs,
                            ),
                        )
                    elif pruning_method == "sampled":
                        # TODO
                        pass
                    else:
                        raise ValueError("Invalid pruning strategy method provided")
                else:
                    replace_layernorm(child)

        replace_linear(self.model)
        replace_layernorm(self.model)

    def update_hyperparameters(self):
        """
        Updates the hyperparameters of the underlying Masked modules.

        Return:
            None
        """
        for m in self.model.modules():
            if isinstance(m, ContinuousMaskLinear) or isinstance(
                m, ContinuousMaskLayerNorm
            ):
                m.update_temperature()

    def get_remaining_weights(self) -> dict:
        """
        Computes the macro average remaining weights of the masked modules.

        Returns:
            float: the macro average
        """
        global_remaining_running = []
        layer_remaining_running = defaultdict(list)
        fine_grained_remaining_weights = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MaskedLinear) or isinstance(m, MaskedLayerNorm):
                local_remainder = m.compute_remaining_weights()
                name_list = name.split(".")
                try:
                    coder = f"{name_list[0]}_layer_{name_list[2]}"
                except IndexError:
                    coder = name
                layer_remaining_running[coder].append(local_remainder)
                fine_grained_remaining_weights[name] = local_remainder
                global_remaining_running.append(local_remainder)

        global_macro_average = sum(global_remaining_running) / len(
            global_remaining_running
        )
        per_layer_remaining = {
            key: sum(x) / len(x) for key, x in layer_remaining_running.items()
        }
        section_logs = {
            "global_remaining_weights": global_macro_average,
            "pruning_layers/": per_layer_remaining,
            "pruning_finegrained/": fine_grained_remaining_weights,
        }
        return section_logs

    def compute_and_update_masks(self):
        for name, m in self.model.named_modules():
            if isinstance(m, MaskedLinear) or isinstance(m, MaskedLayerNorm):
                m.compute_mask()

    def activate_ticket(self):
        """
        Activates the ticket for evaluation mode in the Continuous Mask setting
        """
        for m in self.model.modules():
            if isinstance(m, ContinuousMaskLinear) or isinstance(
                m, ContinuousMaskLayerNorm
            ):
                m.ticket = True

    def deactivate_ticket(self):
        """
        Deactivates the ticket for training mode in the Continuous Mask setting
        """
        for m in self.model.modules():
            if isinstance(m, ContinuousMaskLinear) or isinstance(
                m, ContinuousMaskLayerNorm
            ):
                m.ticket = False

    def compute_l1_norm(self):
        """
        Gathers all the L1 Norms
        """
        norms = 0.0
        for m in self.model.modules():
            if isinstance(m, MaskedLinear) or isinstance(m, MaskedLayerNorm):
                norms += m.compute_l1_norm()
        return norms


if __name__ == "__main__":

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define masked layer arguments
    sampled_maskedlayer_kwargs = {"tau": 1.0, "num_masks": 2}
    cont_maskedlayer_kwargs = {
        "mask_initial_value": 1.0,
        "ticket": False,
    }

    # Create a simple model
    model = SimpleModel()
    print(f"Toy model: \n{model}")

    sampled_masked_model = Pruner(
        model, pruning_method="sampled", maskedlayer_kwargs=sampled_maskedlayer_kwargs
    )
    print(f"Sampled Masked model: \n{model}")

    # Create dummy input data
    input_data = torch.randn(18, 10)
    sampled_output_data = model(input_data)
    print(f"in tensor: \n{input_data.shape}")
    print(f"Sampled out tensor: \n{sampled_output_data.shape}")

    model = SimpleModel()
    print(f"Toy model: \n{model}")

    cont_masked_model = Pruner(
        model, pruning_method="continuous", maskedlayer_kwargs=cont_maskedlayer_kwargs
    )
    print(f"Continuous Masked model: \n{model}")

    cont_output_data = model(input_data)
    print(f"in tensor: \n{input_data.shape}")
    print(f"Continuous out tensor: \n{cont_output_data.shape}")
