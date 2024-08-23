import torch.nn as nn


def postquantize(model: nn.Module, qlinear: nn.Module = None, qconv: nn.Module = None):
    def replace_module(parent, name, new_layer):
        """Replace a module within its parent module."""
        parent._modules[name] = new_layer

    for name, layer in model.named_modules():
        if qlinear is not None and isinstance(layer, nn.Linear):
            has_bias = layer.bias is not None
            new_layer = qlinear(layer.in_features, layer.out_features, has_bias)
            new_layer.setup(layer)

            # Find the parent module and replace the layer
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
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

            # Find the parent module and replace the layer
            parent_name, child_name = name.rsplit('.', 1)
            parent_module = model.get_submodule(parent_name)
            replace_module(parent_module, child_name, new_layer)
