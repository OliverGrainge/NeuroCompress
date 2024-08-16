import pytest
import torch

from NeuroPress.QLayers.Integer.conv import (
    BaseConv2d,
    Conv2dW2A8,
    Conv2dW2A16,
    Conv2dW4A8,
    Conv2dW4A16,
    Conv2dW8A8,
    Conv2dW8A16,
    FullQuant,
    WeightOnlyQuant,
)


def test_base_conv2d_abstract():
    conv = BaseConv2d(3, 16, 3)  # Should raise error due to direct instantiation


def test_weight_only_quant_init():
    conv = WeightOnlyQuant(3, 16, 3, bits=8, type="signed")
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.kernel_size == (3, 3)  # assuming tuple
    assert conv.bits == 8
    assert conv.type == "signed"


def test_full_quant_init():
    conv = FullQuant(3, 16, 3, act_bits=8, weight_bits=8, weight_type="signed", act_type="signed")
    assert conv.in_channels == 3
    assert conv.out_channels == 16
    assert conv.act_bits == 8
    assert conv.weight_bits == 8


def test_weight_only_quant_forward():
    conv = WeightOnlyQuant(3, 16, 3, bits=8, type="signed")
    input_tensor = torch.randn(1, 3, 32, 32)
    output = conv(input_tensor)
    assert output.size() == (1, 16, 30, 30)  # Default stride=1 and padding=0


def test_full_quant_forward():
    conv = FullQuant(3, 16, 3, act_bits=8, weight_bits=8, weight_type="signed", act_type="signed")
    input_tensor = torch.randn(1, 3, 32, 32)
    output = conv(input_tensor)
    assert output.size() == (1, 16, 30, 30)  # Expected size with default settings


@pytest.mark.parametrize(
    "class_type, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, bits, type",
    [
        (Conv2dW8A16, 3, 16, 3, 1, 0, 1, 1, True, 8, "signed"),
        (Conv2dW4A16, 3, 16, 3, 1, 0, 1, 1, True, 4, "unsigned"),
        (Conv2dW2A16, 3, 16, 3, 1, 0, 1, 1, True, 2, "unsigned"),
    ],
)
def test_specific_weight_only_quant_layers(
    class_type,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    padding,
    dilation,
    groups,
    bias,
    bits,
    type,
):
    conv = class_type(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
    assert conv.bits == bits
    assert conv.type == type
    input_tensor = torch.randn(1, in_channels, 32, 32)
    output = conv(input_tensor)
    assert output.size() == (
        1,
        out_channels,
        30,
        30,
    )  # Adjust as needed for stride/padding


# Additional tests can be added as needed for more specific behaviors or error cases
