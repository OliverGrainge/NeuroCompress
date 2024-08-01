import pytest
import torch
from NeuroPress.QLayers import (
    LinearW1A16,
    StochasticLinearW1A16,
    LinearW1A1,
    StochasticLinearW1A1,
)  # Update with the correct path to LinearW1A16
from NeuroPress.QLayers.utils import SignBinarizeFunction, StochasticBinarySignFunction

# Constants for test
IN_FEATURES = 10
OUT_FEATURES = 5


@pytest.fixture
def binary_linear():
    return LinearW1A16(IN_FEATURES, OUT_FEATURES)


@pytest.fixture
def stochastic_binary_linear():
    return StochasticLinearW1A16(IN_FEATURES, OUT_FEATURES)


def test_binary_linear_initialization(binary_linear):
    assert binary_linear.weight is not None
    assert binary_linear.bias is not None
    assert binary_linear.weight.shape == (OUT_FEATURES, IN_FEATURES)
    assert binary_linear.bias.shape == (OUT_FEATURES,)


def test_stochastic_binary_linear_initialization(stochastic_binary_linear):
    assert stochastic_binary_linear.weight is not None
    assert stochastic_binary_linear.bias is not None
    assert stochastic_binary_linear.weight.shape == (OUT_FEATURES, IN_FEATURES)
    assert stochastic_binary_linear.bias.shape == (OUT_FEATURES,)


def test_binary_linear_forward(binary_linear):
    x = torch.randn(1, IN_FEATURES)
    output = binary_linear.forward(x)
    assert output.shape == (1, OUT_FEATURES)


def test_stochastic_binary_linear_forward(stochastic_binary_linear):
    x = torch.randn(1, IN_FEATURES)
    output = stochastic_binary_linear.forward(x)
    assert output.shape == (1, OUT_FEATURES)


@pytest.mark.parametrize(
    "data", [torch.tensor([0.1, -0.2, 0.3, -0.4]), torch.tensor([1, -1, 2, -2])]
)
def test_sign_binarize_function(data):
    result = SignBinarizeFunction.apply(data)
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0])
    assert torch.equal(result, expected)


@pytest.mark.parametrize(
    "data", [torch.tensor([0.1, -0.2, 0.3, -0.4]), torch.tensor([1, -1, 2, -2])]
)
def test_stochastic_binary_sign_function(data):
    result = StochasticBinarySignFunction.apply(data)
    # Since it is stochastic, we only check for valid binary outputs
    assert all((result == 1.0) | (result == -1.0))
    assert result.shape == data.shape


@pytest.fixture
def linear_w1a1():
    return LinearW1A1(IN_FEATURES, OUT_FEATURES)


@pytest.fixture
def stochastic_linear_w1a1():
    return StochasticLinearW1A1(IN_FEATURES, OUT_FEATURES)


def test_linear_w1a1_initialization(linear_w1a1):
    assert linear_w1a1.weight is not None
    assert linear_w1a1.bias is not None
    assert linear_w1a1.weight.shape == (OUT_FEATURES, IN_FEATURES)
    assert linear_w1a1.bias.shape == (OUT_FEATURES,)


def test_stochastic_linear_w1a1_initialization(stochastic_linear_w1a1):
    assert stochastic_linear_w1a1.weight is not None
    assert stochastic_linear_w1a1.bias is not None
    assert stochastic_linear_w1a1.weight.shape == (OUT_FEATURES, IN_FEATURES)
    assert stochastic_linear_w1a1.bias.shape == (OUT_FEATURES,)


def test_linear_w1a1_forward(linear_w1a1):
    x = torch.randn(1, IN_FEATURES)
    output = linear_w1a1.forward(x)
    assert output.shape == (1, OUT_FEATURES)
    # Check if output after binarization still holds to being binary (+1, -1)
    weight_tests = SignBinarizeFunction.apply(linear_w1a1.weight)
    input_tests = SignBinarizeFunction.apply(x)
    assert torch.all((weight_tests == 1) | (weight_tests == -1))
    assert torch.all((input_tests == 1) | (input_tests == -1))


def test_stochastic_linear_w1a1_forward(stochastic_linear_w1a1):
    x = torch.randn(1, IN_FEATURES)
    output = stochastic_linear_w1a1.forward(x)
    assert output.shape == (1, OUT_FEATURES)
    # Since stochastic, check that output is valid
    stochastic_weight_tests = StochasticBinarySignFunction.apply(
        stochastic_linear_w1a1.weight
    )
    stochastic_input_tests = StochasticBinarySignFunction.apply(x)
    assert torch.all(
        (stochastic_weight_tests == 1.0) | (stochastic_weight_tests == -1.0)
    )
    assert torch.all((stochastic_input_tests == 1.0) | (stochastic_input_tests == -1.0))
