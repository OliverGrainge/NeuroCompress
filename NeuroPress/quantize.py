import torch.nn as nn


def postquantize(model: nn.Module, qlinear: nn.Module = None, qconv: nn.Module = None):
    for name, layer in model.named_modules():
        # quantize the linear layers
        if qlinear is not None:
            if isinstance(layer, nn.Linear):
                has_bias = True if layer.bias is not None else False
                new_layer = qlinear(layer.in_features, layer.out_features, has_bias)
                has_bias = False if layer.bias is None else True
                new_layer.setup(layer)
                setattr(model, name, new_layer)
        # quantize the conv layers
        if qconv is not None:
            if isinstance(layer, nn.Conv2d):
                has_bias = False if layer.bias is None else True
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
                setattr(model, name, new_layer)
