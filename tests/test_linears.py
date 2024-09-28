import importlib
import inspect
import os
import sys

import pytest

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

    # Check if the forward method is callable
    assert callable(
        getattr(layer_class, "forward")
    ), f"{layer_class.__name__}.forward is not callable"
