"""
Unit tests to test saving and loading of masked layers
"""

import os

import pytest
import torch
import torch.nn as nn

from comp_rep.pruning.masked_layernorm import (
    ContinuousMaskLayerNorm,
    SampledMaskLayerNorm,
)
from comp_rep.pruning.masked_linear import ContinuousMaskLinear, SampledMaskLinear


@pytest.fixture
def continuous_linear() -> ContinuousMaskLinear:
    """
    Fixture to create a ContinuousMaskLinear layer.

    Returns:
        ContinuousMaskLinear: The ContinuousMaskLinear layer.
    """
    input_dim = 4
    output_dim = 2
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))

    return ContinuousMaskLinear(weight=linear_weight, bias=None, ticket=True)


@pytest.fixture
def continuous_layernorm() -> ContinuousMaskLayerNorm:
    """
    Fixture to create a ContinuousMaskLayerNorm layer.

    Returns:
        ContinuousMaskLayerNorm: The ContinuousMaskLayerNorm layer.
    """
    norm_shape = (2,)
    norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

    return ContinuousMaskLayerNorm(
        normalized_shape=norm_shape, weight=norm_layer_weights, bias=None, ticket=True
    )


@pytest.fixture
def sampled_linear() -> SampledMaskLinear:
    """
    Fixture to create a SampledMaskLinear layer.

    Returns:
        SampledMaskLinear: The SampledMaskLinear layer.
    """
    input_dim = 4
    output_dim = 2
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))

    return SampledMaskLinear(weight=linear_weight, bias=None, ticket=True)


@pytest.fixture
def sampled_layernorm() -> SampledMaskLayerNorm:
    """
    Fixture to create a SampledMaskLayerNorm layer.

    Returns:
        SampledMaskLayerNorm: The SampledMaskLayerNorm layer.
    """
    norm_shape = (2,)
    norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

    return SampledMaskLayerNorm(
        normalized_shape=norm_shape, weight=norm_layer_weights, bias=None, ticket=True
    )


def test_save_and_load_continuous_linear(
    continuous_linear: ContinuousMaskLinear,
) -> None:
    """
    Test saving and loading the ContinuousMaskLinear layer.

    Args:
        continuous_linear (ContinuousMaskLinear): The ContinuousMaskLinear layer.
    """
    input_dim = 4
    output_dim = 2

    x = torch.randn(1, 5, 4)
    initial_s_matrix = continuous_linear.s_matrix.clone()
    initial_output = continuous_linear(x).detach().clone()

    # modify s_matrix and b_matrix
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        continuous_linear.s_matrix.copy_(model_linear_s_matrix)
    continuous_linear.compute_mask()

    # new output
    modified_output = continuous_linear(x).detach().clone()

    # save the model state_dict
    torch.save(continuous_linear.state_dict(), "test_continuous_linear.pth")

    # create a new model instance and load the state_dict
    loaded_model = ContinuousMaskLinear(weight=linear_weight, bias=None, ticket=True)
    loaded_model.load_state_dict(torch.load("test_continuous_linear.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        continuous_linear.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert not torch.allclose(
        initial_output, output_after
    ), "The initial outputs are equal to the outputs after loading."
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_continuous_linear.pth")


def test_save_and_load_continuous_layernorm(
    continuous_layernorm: ContinuousMaskLayerNorm,
) -> None:
    """
    Test saving and loading the ContinuousMaskLayerNorm.

    Args:
        continuous_layernorm (ContinuousMaskLayerNorm): The ContinuousMaskLayerNorm layer.
    """
    norm_shape = (2,)

    x = torch.randn(1, 2)
    initial_s_matrix = continuous_layernorm.s_matrix.clone()
    initial_output = continuous_layernorm(x).detach().clone()

    # modify s_matrix and b_matrix
    layernorm_weight = nn.Parameter(torch.randn(norm_shape))
    model_layernorm_s_matrix = torch.Tensor([0.6, -0.2])

    with torch.no_grad():
        continuous_layernorm.s_matrix.copy_(model_layernorm_s_matrix)
    continuous_layernorm.compute_mask()

    # new output
    modified_output = continuous_layernorm(x).detach().clone()

    # save the model state_dict
    torch.save(continuous_layernorm.state_dict(), "test_continuous_layernorm.pth")

    # create a new model instance and load the state_dict
    loaded_model = ContinuousMaskLayerNorm(
        normalized_shape=norm_shape, weight=layernorm_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_continuous_layernorm.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        continuous_layernorm.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert not torch.allclose(
        initial_output, output_after
    ), "The initial outputs are equal to the outputs after loading."
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_continuous_layernorm.pth")


def test_save_and_load_sampled_linear(sampled_linear: SampledMaskLinear) -> None:
    """
    Test saving and loading the SampledMaskLinear layer.

    Args:
        sampled_linear (SampledMaskLinear): The SampledMaskLinear layer.
    """
    input_dim = 4
    output_dim = 2

    x = torch.randn(1, 5, 4)
    initial_s_matrix = sampled_linear.s_matrix.clone()

    # modify s_matrix and b_matrix
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        sampled_linear.s_matrix.copy_(model_linear_s_matrix)
    sampled_linear.compute_mask()

    # new output
    modified_output = sampled_linear(x).detach().clone()

    # save the model state_dict
    torch.save(sampled_linear.state_dict(), "test_sampled_linear.pth")

    # create a new model instance and load the state_dict
    loaded_model = SampledMaskLinear(weight=linear_weight, bias=None, ticket=True)
    loaded_model.load_state_dict(torch.load("test_sampled_linear.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        sampled_linear.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_sampled_linear.pth")


def test_save_and_load_sampled_layernorm(
    sampled_layernorm: SampledMaskLayerNorm,
) -> None:
    """
    Test saving and loading the SampledMaskLayerNorm.

    Args:
        sampled_layernorm (SampledMaskLayerNorm): The SampledMaskLayerNorm layer.
    """
    norm_shape = (2,)

    x = torch.randn(1, 2)
    initial_s_matrix = sampled_layernorm.s_matrix.clone()

    # modify s_matrix and b_matrix
    layernorm_weight = nn.Parameter(torch.randn(norm_shape))
    model_layernorm_s_matrix = torch.Tensor([0.6, -0.2])

    with torch.no_grad():
        sampled_layernorm.s_matrix.copy_(model_layernorm_s_matrix)
    sampled_layernorm.compute_mask()

    # new output
    modified_output = sampled_layernorm(x).detach().clone()

    # save the model state_dict
    torch.save(sampled_layernorm.state_dict(), "test_sampled_layernorm.pth")

    # create a new model instance and load the state_dict
    loaded_model = SampledMaskLayerNorm(
        normalized_shape=norm_shape, weight=layernorm_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_sampled_layernorm.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        sampled_layernorm.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_sampled_layernorm.pth")
