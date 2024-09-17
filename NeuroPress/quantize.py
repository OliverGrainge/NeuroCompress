import torch.nn as nn


def postquantize_layermap(model, layer_map):
    """
    Replaces layers in the model based on the provided layer_map.
    Layers that are not in layer_map remain unchanged.

    Args:
        model: The PyTorch model whose layers you want to replace.
        layer_map: A dictionary where the keys are the actual layer instances
                   in the model, and the values are the new layer types to replace them with.
    """

    def replace_module(parent, name, new_layer):
        """Replace a module within its parent module."""
        parent._modules[name] = new_layer

    # Iterate through the model's named modules
    for name, layer in model.named_modules():
        # Check if the current layer is exactly the same object as any key in the layer_map
        if layer in layer_map:
            replacement_layer_type = layer_map[layer]

            # Check if the layer has bias (if applicable)
            has_bias = layer.bias is not None if hasattr(layer, "bias") else False

            # Replace nn.Linear
            if isinstance(layer, nn.Linear):
                new_layer = replacement_layer_type(layer.in_features, layer.out_features, has_bias)

            # Replace nn.Conv2d
            elif isinstance(layer, nn.Conv2d):
                new_layer = replacement_layer_type(
                    layer.in_channels,
                    layer.out_channels,
                    layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=has_bias,
                )

            # If the new layer has a custom setup method, call it
            if hasattr(new_layer, "setup"):
                new_layer.setup(layer)

            # Handle top-level modules without a dot in the name
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module, child_name = model, name

            # Replace the layer with the new one
            replace_module(parent_module, child_name, new_layer)


def postquantize_all(model: nn.Module, qlinear: nn.Module = None, qconv: nn.Module = None):
    def replace_module(parent, name, new_layer):
        """Replace a module within its parent module."""
        parent._modules[name] = new_layer

    for name, layer in model.named_modules():
        if qlinear is not None and isinstance(layer, nn.Linear):
            has_bias = layer.bias is not None
            new_layer = qlinear(layer.in_features, layer.out_features, has_bias)
            if hasattr(new_layer, "setup"):
                new_layer.setup(layer)

            # Handle top-level modules without a dot in the name
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module, child_name = model, name

            replace_module(parent_module, child_name, new_layer)

        if qconv is not None and isinstance(layer, nn.Conv2d):
            has_bias = layer.bias is not None
            new_layer = qconv(
                layer.in_channels,
                layer.out_channels,
                layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=has_bias,
            )
            new_layer.setup(layer)

            # Handle top-level modules without a dot in the name
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = model.get_submodule(parent_name)
            else:
                parent_module, child_name = model, name

            replace_module(parent_module, child_name, new_layer)


def postquantize(model: nn.Module, qlinear: nn.Module = None, qconv: nn.Module = None, layer_map=None):
    if layer_map is not None:
        postquantize_layermap(model, layer_map)
    else:
        postquantize_all(model, qlinear=qlinear, qconv=qconv)

