import pytest
import torch

from NeuroPress.QLayers.Integer.quant import (
    compute_scale,
    dequantize_per_tensor,
    forward_quantize_per_channel,
    forward_quantize_per_tensor,
    quantize_per_tensor,
)


def test_compute_scale():
    tensor = torch.randn(1, 3, 10, 10)  # Example tensor
    scale, zero_point = compute_scale(tensor, bits=8, type="signed")
    assert scale.numel() == 1  # Should be a single scale for the whole tensor
    assert zero_point.item() == 0  # For signed type

    tensor = torch.randn(1, 3, 10, 10)
    scale, zero_point = compute_scale(tensor, bits=8, type="unsigned")
    assert scale.numel() == 1
    assert zero_point.item() != 0  # Non-zero for unsigned type


def test_quantize_per_tensor():
    tensor = torch.tensor([1.5, -2.3, 0.0, 3.4])
    scale = torch.tensor([0.1])
    zero_point = torch.tensor([0])
    qtensor = quantize_per_tensor(tensor, scale, zero_point, bits=8, type="signed")
    assert qtensor.equal(torch.tensor([15, -23, 0, 34]))


def test_dequantize_per_tensor():
    tensor = torch.tensor([15, -23, 0, 34], dtype=torch.float32)
    scale = torch.tensor([0.1])
    zero_point = torch.tensor([0])
    dtensor = dequantize_per_tensor(tensor, scale, zero_point)
    assert torch.allclose(dtensor, torch.tensor([1.5, -2.3, 0.0, 3.4]))


def test_forward_quantize_per_tensor():
    tensor = torch.randn(1, 3, 10, 10)
    q_tensor, scale, zero_point = forward_quantize_per_tensor(
        tensor, bits=8, type="signed"
    )
    assert q_tensor.shape == tensor.shape


def test_forward_quantize_per_channel():
    tensor = torch.randn(5, 3, 10, 10)
    q_tensor, scale, zero_point = forward_quantize_per_channel(
        tensor, bits=8, type="signed"
    )
    assert q_tensor.shape == tensor.shape
    assert scale.shape == (5, 1, 1, 1)  # Channel-wise scaling


# Additional tests can be added as necessary, for example, to verify the gradient computation in the autograd functions.
