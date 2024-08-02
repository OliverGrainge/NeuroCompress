import pytest
import torch
import torch.nn as nn
from NeuroPress.QLayers.Integer.conv import Conv2dW8A16, Conv2dW4A16, Conv2dW2A16, Conv2dW8A8, Conv2dW4A8, Conv2dW2A8

@pytest.fixture
def input_tensor():
    # Create an input tensor that simulates a mini-batch of 4 images, each with 3 color channels, 64x64 pixels
    return torch.randn(4, 3, 64, 64)

@pytest.fixture
def setup_conv2d():
    # Setup a Conv2D layer to copy parameters from
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
    nn.init.uniform_(conv.weight, -1, 1)
    if conv.bias is not None:
        nn.init.uniform_(conv.bias, -1, 1)
    return conv

@pytest.mark.parametrize("cls", [Conv2dW8A16, Conv2dW4A16, Conv2dW2A16, Conv2dW8A8, Conv2dW4A8, Conv2dW2A8])
def test_forward_pass(cls, input_tensor):
    model = cls(3, 16, kernel_size=3, padding=1)
    output = model(input_tensor)
    assert output.shape == (4, 16, 64, 64), "Output shape mismatch."

def test_init_and_setup(setup_conv2d):
    model = Conv2dW8A16(3, 16, kernel_size=3, padding=1)
    model.setup(setup_conv2d)
    assert torch.allclose(model.weight.data, setup_conv2d.weight.data), "Weights were not copied correctly."
    if model.bias is not None:
        assert torch.allclose(model.bias.data, setup_conv2d.bias.data), "Biases were not copied correctly."

def test_quantization_effectiveness():
    # Create input tensor and initialize the model
    input_tensor = torch.randn(4, 3, 64, 64)
    model8 = Conv2dW8A16(3, 16, kernel_size=3, padding=1)
    model4 = Conv2dW4A16(3, 16, kernel_size=3, padding=1)
    model2 = Conv2dW2A16(3, 16, kernel_size=3, padding=1)

    # Get outputs from models
    output8 = model8(input_tensor)
    output4 = model4(input_tensor)
    output2 = model2(input_tensor)

    # Check that outputs are different, confirming quantization depth has an effect
    assert not torch.allclose(output8, output4), "Quantization depth did not affect the output between 8 and 4 bits."
    assert not torch.allclose(output4, output2), "Quantization depth did not affect the output between 4 and 2 bits."
