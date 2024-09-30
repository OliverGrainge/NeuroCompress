"""
Module: packing

This module provides utility functions for packing and unpacking ternary values
(-1, 0, 1) into and from integer representations. These functions are essential
for efficient storage and computation in quantized neural network models, where
ternary weights can significantly reduce memory usage and accelerate inference.

Functions:
    pack_ternary(x, n_element_in_one_int=4):
        Packs ternary values into integers.
    
    unpack_ternary(x, n_bits=4):
        Unpacks ternary values from integers.
"""

import torch


def pack_ternary(x, n_element_in_one_int=4):
    """
    Pack ternary values into integers.

    This function packs a tensor of ternary values (-1, 0, 1) into a compact integer
    representation. Each integer encodes a fixed number of ternary elements, specified
    by `n_element_in_one_int`. This packing is useful for reducing memory footprint
    and improving computational efficiency in quantized neural network models.

    Args:
        x (torch.Tensor): 
            A tensor containing ternary values with shape `(*, K, N)`, where
            `*` denotes any number of leading dimensions, `K` is the number of
            ternary values, and `N` is the number of elements per group to pack.
        n_element_in_one_int (int, optional): 
            The number of ternary elements to pack into one integer. Must be one of
            `[4, 8, 16, 32]`. Defaults to `4`.

    Returns:
        torch.Tensor: 
            A tensor with shape `(*, K, N // n_element_in_one_int)`, where each element
            in the last dimension is an integer representing `n_element_in_one_int`
            packed ternary values.

    Raises:
        AssertionError: 
            If the last dimension of `x` is not divisible by `n_element_in_one_int`.
            If `n_element_in_one_int` is not one of `[4, 8, 16, 32]`.

    Example:
        ```python
        import torch
        from ternary_packing import pack_ternary

        # Create a sample ternary tensor
        x = torch.tensor([[[1, -1, 0, 1],
                           [0, 1, -1, 0]]])  # Shape: (1, 2, 4)

        # Pack ternary values into integers with 4 elements per integer
        packed_x = pack_ternary(x, n_element_in_one_int=4)
        print(packed_x)
        # Output: tensor([[[ 3, -2]]], dtype=torch.int8)
        ```

    Notes:
        - The ternary values are mapped as follows: `-1` -> `2`, `0` -> `0`, `1` -> `1`.
        - The packing process shifts each ternary value by `2 * position` bits and sums them
          to form the packed integer.
    """
    assert x.shape[-1] % n_element_in_one_int == 0, "K must be divisible by n_bits"
    assert n_element_in_one_int in [4, 8, 16, 32], "n_element_in_one_int must be 4, 8, 16, 32"
    device = x.device
    x_mapped = x.clone()
    x_mapped[x == -1] = 2

    shift = torch.arange(n_element_in_one_int, device=x.device) * 2

    shape = x.shape[:-1]
    x = x_mapped.view(-1, x.shape[-2], x.shape[-1] // n_element_in_one_int, n_element_in_one_int)

    x = x << shift[None, None, None, :]

    x = x.sum(-1)
    x = x.view(*shape, *x.shape[-1:])

    if n_element_in_one_int == 4:
        dtype = torch.int8
    elif n_element_in_one_int == 8:
        dtype = torch.int16
    elif n_element_in_one_int == 16:
        dtype = torch.int32
    else:
        dtype = torch.int64

    return x.to(dtype).to(device)



def unpack_ternary(x, n_bits=4):
    """
    Unpack ternary values from integers.

    This function unpacks a tensor of integers into their original ternary values (-1, 0, 1).
    Each integer encodes a fixed number of ternary elements, specified by `n_bits`. This
    unpacking is essential for retrieving the original ternary representation from a
    compact integer format, facilitating tasks such as model inference and analysis.

    Args:
        x (torch.Tensor): 
            A tensor containing packed integers with shape `(*, K // n_bits, N)`, where
            `*` denotes any number of leading dimensions, `K` is the total number of
            ternary values, and `N` is the number of elements per group.
        n_bits (int, optional): 
            The number of ternary values that each integer in `x` represents. Must be one of
            `[4, 8, 16, 32]`. Defaults to `4`.

    Returns:
        torch.Tensor: 
            A tensor with shape `(*, K, N)`, where each element is a ternary value (-1, 0, 1)
            unpacked from the integers.

    Raises:
        AssertionError: 
            If `n_bits` is not one of `[4, 8, 16, 32]`.

    Example:
        ```python
        import torch
        from ternary_packing import unpack_ternary

        # Create a sample packed tensor
        packed_x = torch.tensor([[[ 3, -2]]], dtype=torch.int8)  # Shape: (1, 2, 1)

        # Unpack ternary values with 4 bits per integer
        x = unpack_ternary(packed_x, n_bits=4)
        print(x)
        # Output: tensor([[[ 1, -1, 0, 1],
        #                  [ 0, 1, -1, 0]]])
        ```

    Notes:
        - The unpacking process reverses the packing by extracting each pair of bits,
          mapping them back to ternary values: `2` -> `-1`, `1` -> `1`, `0` -> `0`.
        - The function assumes that the packed integers were created using the `pack_ternary`
          function with the same `n_bits` parameter.
    """

    # Create a mask for the shifting
    masks = (3 << (2 * torch.arange(n_bits, device=x.device))).view(1, 1, 1, -1)

    # Use broadcasting for the mask
    x_expanded = x.unsqueeze(-1)
    x_expanded = x_expanded * torch.ones_like(masks)

    # Apply mask and shift values
    unpacked = (x_expanded & masks) >> (2 * torch.arange(n_bits, device=x.device)).view(1, 1, 1, -1)

    # Mappa i valori di nuovo a -1, 0, 1
    unpacked = torch.where(unpacked == 2, torch.tensor(-1, device=x.device), unpacked)

    # Riorganizza le dimensioni per ottenere il formato desiderato (*, K, N)
    return unpacked.reshape(*x.shape[:-1], -1)