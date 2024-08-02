import torch
import pytest
from NeuroPress.QLayers.Binary.quant import SignBinarizeFunction, StochasticBinarySignFunction



def test_sign_binarize_forward():
    x = torch.tensor([-1.5, 0.0, 2.0], dtype=torch.float32)
    expected_output = torch.tensor([-1.0, -1.0, 1.0])
    output = SignBinarizeFunction.apply(x)
    assert torch.equal(
        output, expected_output
    ), "Forward pass failed for SignBinarizeFunction"


def test_sign_binarize_backward():
    x = torch.tensor([1.0, -1.0], requires_grad=True)
    y = SignBinarizeFunction.apply(x)
    y.backward(torch.tensor([1.0, 1.0]))
    expected_grads = torch.tensor([1.0, 1.0])
    assert torch.allclose(
        x.grad, expected_grads
    ), "Backward pass failed for SignBinarizeFunction"


def test_stochastic_binary_sign_forward():
    torch.manual_seed(0)  # Fix seed for reproducibility
    x = torch.tensor([0.5, -0.5], dtype=torch.float32)
    output = StochasticBinarySignFunction.apply(x)
    assert torch.sort(output.unique())[0].tolist() == [
        -1.0,
        1.0,
    ], "Forward pass failed for StochasticBinarySignFunction"


def test_stochastic_binary_sign_backward():
    x = torch.tensor([0.2, -0.2], requires_grad=True)
    y = StochasticBinarySignFunction.apply(x)
    y.backward(torch.tensor([1.0, 1.0]))
    expected_grads = torch.tensor([1.0, 1.0])
    assert torch.allclose(
        x.grad, expected_grads
    ), "Backward pass failed for StochasticBinarySignFunction"


if __name__ == "__main__":
    pytest.main()
