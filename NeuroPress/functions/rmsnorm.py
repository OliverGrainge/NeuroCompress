# Kindly borrowed from UnSloth's implementation
import torch

if torch.cuda.is_available():
    import triton
    import triton.language as tl

    MAX_FUSED_SIZE = 65536
    next_power_of_2 = triton.next_power_of_2

    def calculate_settings(n):
        BLOCK_SIZE = next_power_of_2(n)
        if BLOCK_SIZE > MAX_FUSED_SIZE:
            raise RuntimeError(
                f"Cannot launch Triton kernel since n = {n} exceeds "
                f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
            )
        num_warps = 4
        if BLOCK_SIZE >= 32768:
            num_warps = 32
        elif BLOCK_SIZE >= 8192:
            num_warps = 16
        elif BLOCK_SIZE >= 2048:
            num_warps = 8
        return BLOCK_SIZE, num_warps

    pass

    @triton.jit
    def _rms_layernorm_forward(
        Y,
        Y_row_stride,
        X,
        X_row_stride,
        W,
        W_row_stride,
        r,
        r_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fast RMS Layernorm kernel
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride

        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0)  # .to(tl.float32)

        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = tl.math.rsqrt(row_var + eps)
        tl.store(r, inv_var)
        normed = X_row * inv_var
        normed = normed.to(W_row.dtype)  # Exact copy from HF
        output = normed * W_row
        tl.store(Y + col_offsets, output, mask=mask)

    pass

    @triton.heuristics(
        {
            "GEMMA": lambda args: args["GEMMA"],
        }
    )
    @triton.jit
    def _rms_layernorm_backward(
        dY,
        dY_row_stride,
        X,
        X_row_stride,
        W,
        W_row_stride,
        r,
        r_row_stride,
        dW,
        dW_row_stride,
        n_cols,
        eps,
        GEMMA: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Fast RMS Layernorm kernel for the backward pass
        Inspiration from a Triton tutorial:
        https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
        """
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        dY += row_idx * dY_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride

        dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

        # Get saved row variance
        inv_var = tl.load(r).to(tl.float32)
        normed = X_row * inv_var

        if GEMMA:
            dY_W = dY_row * (W_row + 1.0)
        else:
            dY_W = dY_row * W_row

        rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
        output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
        tl.store(dY + col_offsets, output, mask=mask)

    pass

    @triton.jit
    def _gemma_rms_layernorm_forward(
        Y,
        Y_row_stride,
        X,
        X_row_stride,
        W,
        W_row_stride,
        r,
        r_row_stride,
        n_cols,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Copies https://github.com/google-deepmind/gemma/blob/main/gemma/layers.py#L31
        # and https://github.com/keras-team/keras-nlp/blob/v0.8.2/keras_nlp/models/gemma/rms_normalization.py#L33
        # exactly. Essentially all in float32!
        row_idx = tl.program_id(0)
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        Y += row_idx * Y_row_stride
        X += row_idx * X_row_stride
        r += row_idx * r_row_stride

        X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
        W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

        row_var = tl.sum(X_row * X_row, axis=0) / n_cols
        inv_var = 1.0 / tl.sqrt(
            row_var + eps
        )  # Must be 1/sqrt to match Deepmind's impl
        tl.store(r, inv_var)
        normed = X_row * inv_var
        output = normed * (W_row + 1.0)

        tl.store(Y + col_offsets, output, mask=mask)

    pass

    class Fast_RMS_Layernorm(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X, W, eps, gemma=False):
            shape = X.shape
            dim = shape[-1]
            X = X.view(-1, dim)
            n_rows, n_cols = X.shape
            BLOCK_SIZE, num_warps = calculate_settings(n_cols)

            Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device="cuda")
            r = torch.empty(n_rows, dtype=torch.float32, device="cuda")

            fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
            fx[(n_rows,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            ctx.eps = eps
            ctx.BLOCK_SIZE = BLOCK_SIZE
            ctx.num_warps = num_warps
            ctx.GEMMA = gemma
            ctx.save_for_backward(X, W, r)
            return Y.view(*shape)

        pass

        @staticmethod
        def backward(ctx, dY):
            shape = dY.shape
            dim = shape[-1]
            dY = dY.view(-1, dim)
            X, W, r = ctx.saved_tensors
            n_rows, n_cols = dY.shape
            dW = X

            _rms_layernorm_backward[(n_rows,)](
                dY,
                dY.stride(0),
                X,
                X.stride(0),
                W,
                W.stride(0),
                r,
                r.stride(0),
                dW,
                dW.stride(0),
                n_cols,
                ctx.eps,
                GEMMA=ctx.GEMMA,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )
            dX = dY.view(*shape)
            return dX, None, None, None

        pass

    pass

    def _gpu_rmsnorm(weight, X, eps, gemma=False):
        out = Fast_RMS_Layernorm.apply(X, weight, eps, gemma)
        return out

    pass


def _cpu_rmsnorm(weight, X, eps, gemma=False):
    """
    CPU implementation of RMS LayerNorm.

    Args:
        weight (torch.Tensor): The scaling weights (gamma in LayerNorm).
        X (torch.Tensor): The input tensor.
        eps (float): A small constant added to the variance for numerical stability.
        gemma (bool): If True, applies the GEMMA-specific scaling (W + 1.0).

    Returns:
        torch.Tensor: The normalized output.
    """
    # Compute variance of each row (mean of the squared values)
    row_var = X.pow(2).mean(dim=-1, keepdim=True)

    # Compute the inverse of the square root of the variance, adding epsilon for stability
    inv_var = torch.rsqrt(row_var + eps)

    # Normalize the input by multiplying with the inverse variance
    normed = X * inv_var

    # Apply GEMMA-specific scaling if required
    if gemma:
        normed = normed * (weight + 1.0)
    else:
        normed = normed * weight

    return normed


def rmsnorm(weight, X, eps, gemma=False):
    if X.device.type == "cuda":
        assert weight.device.type == "cuda"
        return _gpu_rmsnorm(weight, X, eps, gemma=gemma)
    elif X.device.type == "cpu":
        assert weight.device.type == "cpu"
        return _cpu_rmsnorm(weight, X, eps, gemma=gemma)
    else:
        raise Exception("Only CPU or CUDA supported")
