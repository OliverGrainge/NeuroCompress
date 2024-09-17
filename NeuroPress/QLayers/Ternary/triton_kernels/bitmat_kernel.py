
import torch
import triton
import triton.language as tl
from ..utils import custom_autotune
@custom_autotune.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=1,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=1,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=4,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=4,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=4,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=4,num_warps=4),
            triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},num_stages=4,num_warps=4),
        ],
        key=["M", "N", "K"],
        nearest_power_of_two=True,
        prune_configs_by={
            "early_config_prune": custom_autotune.kernel_config_pruner,
            "perf_model": None,
            "top_k": None,
        },
    )
@triton.jit
def _ternary_mm_kernel(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M, N, K,
        n_bits,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        # Kernel parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        ACTIVATION: tl.constexpr,
):
    #Kernel for computing the matmul C = A x B.
    #    A has shape (M, K), int8
    #    B has shape (K//n_bits, N), int8, packed
    #    C has shape (M, N),
    

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # Create pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
    a_mask = offs_am[:, None] < M
    b_ptrs = b_ptr + (
                (offs_k[:, None] // n_bits) * stride_bk + offs_bn[None, :] * stride_bn)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)

    c_dtype = tl.load(c_ptr + stride_cm).dtype # here we load the first element of c to see its dtype
    # shifter is used to extract each 2 bit of each element in the int matrix
    shifter = (offs_k % n_bits)[:, None] * 2

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, num_pid_k):
        a = (tl.load(a_ptrs, mask=a_mask, other=0.0))
        b = tl.load(b_ptrs)
        # We extract the 2 bits of each element in the int matrix
        b = (b >> shifter) & 0x3
        # We need to map back the value 2 -> -1
        b = tl.where(b == 0x2, -1, b)

        b = b.to(a.dtype)  # To be sure a.dtype == b_values.dtype

        accumulator += tl.dot(a, b, out_dtype=c_dtype)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += (BLOCK_SIZE_K // n_bits) * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def bitmat_(a, b, int_per_2_bits=4, activation="", out_dtype=torch.float16):
    
    #    a: int8 tensor (..., K)
    #    b: int8 packed tensor (K // int_per_2_bit, N)
    #    c: float16 tensor (..., N)
    #    n_bits: int, number of bits that each element in b represents
    
    # Check constraints.
    assert a.shape[-1] == b.shape[-2] * int_per_2_bits, "Incompatible dimensions"
    assert a.is_contiguous(), "A must be contiguous"
    assert b.is_contiguous(), "B must be contiguous"
    assert int_per_2_bits in [4, 8, 16, 32], "n_bits must be 4, 8, 16, 32"

    x = a.view(-1, a.shape[-1]) # flatten the tensor

    M, K = x.shape
    _, N = b.shape


    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=out_dtype).contiguous()
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # print(f"Launching kernel with M = {M}, N = {N}, K = {K}, n_bits = {n_bits}, activation = {activation}")

    _ternary_mm_kernel[grid](
        x, b, c,
        M, N, K,
        int_per_2_bits,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation
    )
    # bring c back to the original shape
    c = c.view(a.shape[:-1] + (N,))
    return c
"""



import torch
import triton
import triton.language as tl
from ..utils import custom_autotune

@custom_autotune.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    ],
    key=["M", "N", "K"],
    nearest_power_of_two=True,
    prune_configs_by={
        "early_config_prune": custom_autotune.kernel_config_pruner,
        "perf_model": None,
        "top_k": None,
    },
)
@triton.jit
def _ternary_mm_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    n_bits,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Kernel parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    #Kernel for computing the matmul C = A x B.
    #A has shape (M, K), int8
    #B has shape (K // n_bits, N), int8, packed
    #C has shape (M, N),
    
    tl.static_assert(BLOCK_SIZE_M % 16 == 0, "BLOCK_SIZE_M must be divisible by 16")

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    a_mask = offs_am[:, None] < M
    b_mask = offs_bn[None, :] < N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=OUT_DTYPE)

    for k_start in range(0, K, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        k_mask = offs_k < K

        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + (offs_k[:, None] // n_bits) * stride_bk + offs_bn[None, :] * stride_bn

        # Compute shifter and ensure it's within valid range
        shifter = (offs_k % n_bits)[:, None] * 2

        # Masks to prevent out-of-bounds memory access
        a_mask_k = a_mask & (offs_k[None, :] < K)
        b_mask_k = (offs_k[:, None] // n_bits < (K + n_bits - 1) // n_bits) & b_mask

        # Load A and B with masking
        a = tl.load(a_ptrs, mask=a_mask_k, other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=b_mask_k, other=0).to(tl.int8)

        # Extract the 2 bits of each element in the int matrix
        shifter = shifter.to(tl.int8)
        b = (b >> shifter) & 0x3

        # Map value 2 back to -1
        b = tl.where(b == 0x2, -1, b)

        # Cast to float32 for dot product
        a = a.to(tl.float32)
        b = b.to(tl.float32)

        # Perform dot product
        accumulator += tl.dot(a, b)

    # Write back the result with proper masking
    c_ptrs = c_ptr + offs_am[:, None] * stride_cm + offs_bn[None, :] * stride_cn
    c_mask = a_mask & b_mask
    tl.store(c_ptrs, accumulator, mask=c_mask)



def bitmat_(a, b, int_per_2_bits=4, activation="", out_dtype=torch.float32):
    
    #a: int8 tensor (..., K)
    #b: int8 packed tensor (K_packed, N)
    #c: float32 tensor (..., N)
    #int_per_2_bits: int, number of ternary values packed into one integer
    
    # Check constraints.
    assert a.shape[-1] == b.shape[-2] * int_per_2_bits, f"Incompatible dimensions: a.shape[-1]={a.shape[-1]}, b.shape[-2]={b.shape[-2]}, int_per_2_bits={int_per_2_bits}"
    assert a.is_contiguous(), "A must be contiguous"
    assert b.is_contiguous(), "B must be contiguous"
    assert int_per_2_bits in [4, 8, 16, 32], "int_per_2_bits must be 4, 8, 16, 32"

    x = a.view(-1, a.shape[-1])  # Flatten the tensor
    M, K = x.shape
    K_packed, N = b.shape

    # Adjusted to handle K not being a multiple of int_per_2_bits
    assert K == K_packed * int_per_2_bits, f"Incompatible dimensions: K={K}, K_packed * int_per_2_bits={K_packed * int_per_2_bits}"

    # Allocate output tensor
    c = torch.empty((M, N), device=a.device, dtype=out_dtype).contiguous()

    # Define grid size for kernel launch
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    # Launch the Triton kernel
    _ternary_mm_kernel[grid](
        x, b, c,
        M, N, K,
        int_per_2_bits,
        x.stride(0), x.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        OUT_DTYPE=tl.float32 if out_dtype == torch.float32 else tl.float16
    )

    # Reshape output to match input batch dimensions
    c = c.view(a.shape[:-1] + (N,))
    return c
"""