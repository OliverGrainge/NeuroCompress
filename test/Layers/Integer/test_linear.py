import pytest
import torch
import torch.nn as nn
from NeuroPress.QLayers.Integer.linear import LinearW8A16, LinearW4A16, LinearW2A16, LinearW8A8, LinearW4A8, LinearW2A8

@pytest.fixture
def input_tensor():
    return torch.randn(10, 20)  # Example input tensor

@pytest.fixture
def setup_linear():
    linear = nn.Linear(20, 30, bias=True)
    nn.init.uniform_(linear.weight, -1, 1)
    if linear.bias is not None:
        nn.init.uniform_(linear.bias, -1, 1)
    return linear

@pytest.mark.parametrize("cls", [LinearW8A16, LinearW4A16, LinearW2A16, LinearW8A8, LinearW4A8, LinearW2A8])
def test_forward_pass(cls, input_tensor):
    model = cls(20, 30)
    output = model(input_tensor)
    assert output.shape == (10, 30), "Output shape mismatch."

def test_init(setup_linear):
    model = LinearW8A16(20, 30)
    model.setup(setup_linear)
    assert torch.allclose(model.weight.data, setup_linear.weight.data), "Weights were not copied correctly."
    assert torch.allclose(model.bias.data, setup_linear.bias.data), "Biases were not copied correctly."

def test_multiple_quantization_levels():
    input_tensor = torch.randn(10, 20)
    models = [LinearW8A16(20, 30), LinearW4A16(20, 30), LinearW2A16(20, 30)]
    outputs = [model(input_tensor) for model in models]
    # Assert different outputs due to different quantization bit levels
    for i in range(1, len(outputs)):
        assert not torch.allclose(outputs[i], outputs[i-1]), "Quantization level did not affect the output."

def test_combined_quantization_activation():
    model = LinearW8A8(20, 30)
    input_tensor = torch.randn(10, 20)
    output = model(input_tensor)
    assert output.shape == (10, 30), "Output shape mismatch."

def test_dequantization_process():
    model = LinearW4A8(20, 30)
    input_tensor = torch.randn(10, 20)
    output = model(input_tensor)
    assert output.shape == (10, 30), "Output shape mismatch due to incorrect dequantization."

# Add more tests as necessary for other classes or methods
