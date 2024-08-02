import pytest
import torch
from NeuroPress.QLayers.Integer.quant import compute_linear_scale, quantize_linear_weights, quantize_conv2d_weights, quantizelinear, quantizeconv2d

@pytest.fixture
def tensors():
    return {
        'zero': torch.zeros(10, 10),
        'one': torch.ones(10, 10),
        'random': torch.randn(10, 10)
    }

def test_compute_linear_scale(tensors):
    assert compute_linear_scale(tensors['zero']) == 0
    assert compute_linear_scale(tensors['one'], bits=8) == 1 / 127  # Default scale for a tensor of ones
    assert compute_linear_scale(tensors['random'], bits=16) > 0  # Just to check non-zero scale

def test_quantize_linear_weights(tensors):
    qtensor, scale = quantize_linear_weights(tensors['one'])
    assert torch.all(qtensor == 127)  # Expected max quantization value for default 8 bits
    assert scale == 1 / 127

def test_quantize_conv2d_weights():
    tensor = torch.randn(2, 3, 5, 5)  # Example conv2d weight (out_channels, in_channels, H, W)
    qtensor, scale = quantize_conv2d_weights(tensor)
    assert qtensor.shape == tensor.shape
    assert scale.numel() == tensor.shape[0]  # One scale per output channel

def test_QuantizeLinear_forward_backward(tensors):
    tensor = tensors['random']
    bits = 8
    qtensor, scale = quantizelinear(tensor, bits)
    assert qtensor is not None
    assert scale is not None

    # Gradient check
    tensor.requires_grad_()
    qtensor, scale = quantizelinear(tensor, bits)
    (qtensor.sum() + scale.sum()).backward()  # Trigger backward pass
    assert tensor.grad is not None

def test_QuantizeConv2d_forward_backward():
    tensor = torch.randn(2, 3, 5, 5, requires_grad=True)  # Example conv2d weight
    bits = 8
    qtensor, scale = quantizeconv2d(tensor, bits)
    assert qtensor is not None
    assert scale is not None

    # Gradient check
    (qtensor.sum() + scale.sum()).backward()
    assert tensor.grad is not None
