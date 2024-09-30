import importlib
import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from NeuroPress.layers import LINEAR_LAYERS


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_layer_class_attributes(layer_class):
    """
    Test that all classes in LINEAR_LAYERS have the required methods and attributes.
    """
    # Check if the class has a 'forward' method
    assert hasattr(
        layer_class, "forward"
    ), f"{layer_class.__name__} does not have a 'forward' method"

    assert hasattr(
        layer_class, "train_forward"
    ), f"{layer_class.__name__} does not have a 'forward' method"

    assert hasattr(
        layer_class, "infer_forward"
    ), f"{layer_class.__name__} does not have a 'forward' method"

    assert hasattr(
        layer_class, "freeze_layer"
    ), f"{layer_class.__name__} does not have a 'forward' method"

    assert hasattr(
        layer_class, "unfreeze_layer"
    ), f"{layer_class.__name__} does not have a 'forward' method"

    # Check if the forward method is callable
    assert callable(
        getattr(layer_class, "forward")
    ), f"{layer_class.__name__}.forward is not callable"


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_repr(layer_class):
    layer = layer_class(4, 4, bias=True, device="cpu", dtype=torch.float32)
    assert isinstance(layer.__repr__(), str)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_train_forward_cpu(layer_class):
    layer = layer_class(4, 4, bias=True, device="cpu", dtype=torch.float32)
    x = torch.randn(1, 4).to("cpu")
    y = layer.train_forward(x)
    assert y.shape == (1, 4)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_infer_forward_cpu(layer_class):
    layer = layer_class(4, 4, bias=True, device="cpu", dtype=torch.float32)
    layer.freeze_layer()
    x = torch.randn(1, 4).to("cpu")
    y = layer.infer_forward(x)
    assert y.shape == (1, 4)


# Tests for GPU if available
@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_train_forward_gpu(layer_class):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")

    layer = layer_class(128, 128, bias=True, device="cuda", dtype=None).to("cuda")
    x = torch.randn(1, 128).to("cuda")
    y = layer.train_forward(x)
    assert y.shape == (1, 128)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_infer_forward_gpu(layer_class):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")
    x = torch.randn(1, 128).to("cuda")
    layer = layer_class(128, 128, bias=True, device="cuda", dtype=None).to("cuda")
    layer.freeze_layer()
    y = layer.infer_forward(x)
    assert y.shape == (1, 128)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_unfreeze_cpu(layer_class):
    x = torch.randn(1, 128).to("cpu")
    layer = layer_class(128, 128, bias=True, device="cuda", dtype=None).to("cpu")
    layer.eval()
    y_unfrozen = layer(x)
    layer.freeze_layer()
    y_frozen = layer(x)
    assert torch.allclose(y_frozen, y_unfrozen, 1e-3)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_unfreeze_gpu(layer_class):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")
    x = torch.randn(1, 128).to("cuda")
    layer = layer_class(128, 128, bias=True, device="cuda", dtype=None).to("cuda")
    layer.eval()
    y_unfrozen = layer(x)
    layer.freeze_layer()
    y_frozen = layer(x)
    assert torch.allclose(y_frozen, y_unfrozen, 1e-4)


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_state_dict_unfrozen(layer_class):
    layer = layer_class(128, 128, bias=True, device="cuda", dtype=None).to("cuda")
    sd = layer.state_dict()
    layer.load_state_dict(sd)
    assert True


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_state_dict_frozen(layer_class):
    layer = layer_class(128, 128, bias=True, device="cpu", dtype=None).to("cpu")
    layer.freeze_layer()
    sd = layer.state_dict()
    layer.load_state_dict(sd)
    assert True


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_backward_cpu(layer_class):
    layer = layer_class(128, 128, bias=True, device="cpu", dtype=None).to("cpu")
    x = torch.randn(1, 128)
    y = layer(x)
    target = torch.randn(1, 128)
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert True


@pytest.mark.parametrize("layer_class", LINEAR_LAYERS)
def test_backward_gpu(layer_class):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping GPU test.")
    layer = layer_class(128, 128, bias=True, device="cpu", dtype=None).to("cuda")
    x = torch.randn(1, 128).to("cuda")
    y = layer(x)
    target = torch.randn(1, 128).to("cuda")
    loss = ((y - target) ** 2).mean()
    loss.backward()
    assert True
