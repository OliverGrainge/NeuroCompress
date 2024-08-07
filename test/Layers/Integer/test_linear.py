import pytest
import torch

from NeuroPress.QLayers.Integer.linear import (
    BaseLinear,
    FullQuant,
    LinearW2A8,
    LinearW2A16,
    LinearW4A8,
    LinearW4A16,
    LinearW8A8,
    LinearW8A16,
    WeightOnlyQuant,
)


def test_base_linear_abstract():
    base_linear = BaseLinear(10, 5)  # Should not be instantiated directly


def test_weight_only_quant_init():
    layer = WeightOnlyQuant(10, 5, bits=8, type="signed")
    assert layer.in_features == 10
    assert layer.out_features == 5
    assert layer.bits == 8
    assert layer.type == "signed"


def test_full_quant_init():
    layer = FullQuant(
        10, 5, act_bits=8, weight_bits=8, weight_type="signed", act_type="signed"
    )
    assert layer.in_features == 10
    assert layer.out_features == 5
    assert layer.act_bits == 8
    assert layer.weight_bits == 8


def test_weight_only_quant_forward():
    in_tensor = torch.randn(1, 10)
    layer = WeightOnlyQuant(10, 5)
    output = layer(in_tensor)
    assert output.shape == (1, 5)


def test_full_quant_forward():
    in_tensor = torch.randn(1, 10)
    layer = FullQuant(
        10, 5, act_bits=8, weight_bits=8, weight_type="signed", act_type="signed"
    )
    output = layer(in_tensor)
    assert output.shape == (1, 5)


def test_specific_quant_layers():
    layer = LinearW8A16(10, 5)
    assert layer.bits == 8
    assert layer.type == "signed"

    layer = LinearW2A8(10, 5)
    assert layer.weight_bits == 2
    assert layer.act_bits == 8


@pytest.mark.parametrize(
    "class_type, in_features, out_features, bits, type",
    [
        (LinearW8A16, 10, 5, 8, "signed"),
        (LinearW4A16, 10, 5, 4, "unsigned"),
        (LinearW2A16, 10, 5, 2, "unsigned"),
    ],
)
def test_various_weight_only_quant_layers(
    class_type, in_features, out_features, bits, type
):
    layer = class_type(in_features, out_features)
    assert layer.bits == bits
    assert layer.type == type
    # Test forward function with random tensor
    input_tensor = torch.randn(1, in_features)
    output = layer(input_tensor)
    assert output.shape == (1, out_features)


# Additional tests can be added as needed
