import pytest
import torch
from NeuroPress.QLayers import (
    Conv2dW1A16,
    StochastiConv2dW1A16,
    Conv2dW1A1,
    StochastiConv2dW1A1,
)
import torch.nn.functional as F

# Constants for tests
IN_CHANNELS = 3
OUT_CHANNELS = 16
KERNEL_SIZE = 3  # Assuming a 3x3 kernel for simplicity
STRIDE = 1
PADDING = 1


@pytest.fixture
def binary_conv2d():
    return Conv2dW1A16(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING)


@pytest.fixture
def stochastic_binary_conv2d():
    return StochastiConv2dW1A16(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING)


def test_binary_conv2d_initialization(binary_conv2d):
    assert binary_conv2d.weight is not None
    assert binary_conv2d.bias is not None
    assert binary_conv2d.weight.shape == (
        OUT_CHANNELS,
        IN_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE,
    )
    assert binary_conv2d.bias.shape == (OUT_CHANNELS,)


def test_stochastic_binary_conv2d_initialization(stochastic_binary_conv2d):
    assert stochastic_binary_conv2d.weight is not None
    assert stochastic_binary_conv2d.bias is not None
    assert stochastic_binary_conv2d.weight.shape == (
        OUT_CHANNELS,
        IN_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE,
    )
    assert stochastic_binary_conv2d.bias.shape == (OUT_CHANNELS,)


def test_binary_conv2d_forward(binary_conv2d):
    x = torch.randn(1, IN_CHANNELS, 32, 32)  # A random input tensor
    output = binary_conv2d(x)
    assert output.shape == (
        1,
        OUT_CHANNELS,
        32,
        32,
    )  # Check output size, assuming no padding or stride


def test_stochastic_binary_conv2d_forward(stochastic_binary_conv2d):
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    output = stochastic_binary_conv2d(x)
    assert output.shape == (1, OUT_CHANNELS, 32, 32)


def test_sign_binarize_function_in_conv2d(binary_conv2d):
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    with torch.no_grad():
        binary_weights = binary_conv2d.binarize_function.apply(binary_conv2d.weight)
    output = F.conv2d(
        x,
        binary_weights,
        binary_conv2d.bias,
        stride=binary_conv2d.stride,
        padding=binary_conv2d.padding,
    )
    assert output.shape == (1, OUT_CHANNELS, 32, 32)
    assert all(torch.unique(binary_weights) == torch.tensor([-1.0, 1.0]))


def test_stochastic_binary_sign_function_in_conv2d(stochastic_binary_conv2d):
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    with torch.no_grad():
        stochastic_binary_weights = stochastic_binary_conv2d.binarize_function.apply(
            stochastic_binary_conv2d.weight
        )
    output = F.conv2d(
        x,
        stochastic_binary_weights,
        stochastic_binary_conv2d.bias,
        stride=stochastic_binary_conv2d.stride,
        padding=stochastic_binary_conv2d.padding,
    )
    assert output.shape == (1, OUT_CHANNELS, 32, 32)
    assert all(torch.unique(stochastic_binary_weights) == torch.tensor([-1.0, 1.0]))


@pytest.fixture
def binary_conv2d_w1a1():
    return Conv2dW1A1(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING)


@pytest.fixture
def stochastic_binary_conv2d_w1a1():
    return StochastiConv2dW1A1(IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDE, PADDING)


def test_conv2d_w1a1_initialization(binary_conv2d_w1a1):
    assert binary_conv2d_w1a1.weight is not None
    assert binary_conv2d_w1a1.bias is not None
    assert binary_conv2d_w1a1.weight.shape == (
        OUT_CHANNELS,
        IN_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE,
    )
    assert binary_conv2d_w1a1.bias.shape == (OUT_CHANNELS,)


def test_stochastic_conv2d_w1a1_initialization(stochastic_binary_conv2d_w1a1):
    assert stochastic_binary_conv2d_w1a1.weight is not None
    assert stochastic_binary_conv2d_w1a1.bias is not None
    assert stochastic_binary_conv2d_w1a1.weight.shape == (
        OUT_CHANNELS,
        IN_CHANNELS,
        KERNEL_SIZE,
        KERNEL_SIZE,
    )
    assert stochastic_binary_conv2d_w1a1.bias.shape == (OUT_CHANNELS,)


def test_conv2d_w1a1_forward(binary_conv2d_w1a1):
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    output = binary_conv2d_w1a1(x)
    assert output.shape == (1, OUT_CHANNELS, 32, 32)


def test_stochastic_conv2d_w1a1_forward(stochastic_binary_conv2d_w1a1):
    x = torch.randn(1, IN_CHANNELS, 32, 32)
    output = stochastic_binary_conv2d_w1a1(x)
    assert output.shape == (1, OUT_CHANNELS, 32, 32)
