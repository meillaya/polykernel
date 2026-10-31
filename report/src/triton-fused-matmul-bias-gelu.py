"""
Fused MatMul + Bias + GELU (epilogue fusion)
=============================================

A matrix multiply is rarely the whole story. In a transformer feed-forward
block the GEMM is immediately followed by a bias add and a nonlinearity.
Running those as three separate kernels pays the round trip through global
memory twice: the GEMM writes its output, the bias kernel reads it back and
writes again, and the activation kernel does the same. Epilogue fusion folds
the bias and the activation into the GEMM's store, so the intermediate
results never leave on-chip registers.

This tutorial builds on :code:`03-matrix-multiplication.py`. It reuses the
same tiled, autotuned GEMM and adds two operations to the epilogue: a
per-column bias broadcast and an exact erf-based GELU, both applied while the
accumulator is still in fp32. The result matches
:code:`torch.nn.functional.gelu(a @ b + bias)` to floating point tolerance,
in a single kernel launch.

The pattern generalizes. Any elementwise epilogue (bias + SiLU, a GeGLU half,
a quantization scale and zero-point) slots into the same place between the
K-loop and the store.
"""

# %%
# Setup
# -----
#
# We need PyTorch for the reference and Triton for the kernel.

import torch

import triton
import triton.language as tl

# %%
# The autotune configuration
# --------------------------
#
# A compact config list, mirroring the larger one in tutorial 03 and trimmed
# here for readability. The autotuner re-runs the search whenever one of the
# keys (M, N, K) changes, so each problem shape gets its own best config.


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
    ]


# %%
# The kernel
# ----------
#
# The GEMM body is the grouped, tiled loop from tutorial 03. The only new
# code lives in the epilogue, marked below.


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K'])
@triton.jit
def fused_matmul_bias_gelu_kernel(
        a_ptr, b_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
):
    """Compute C = gelu(A @ B + bias).

    A has shape (M, K), B has shape (K, N), bias has shape (N,), and C has
    shape (M, N). The bias is broadcast over the M rows.
    """
    # Map this program to the output tile it owns. The grouped ordering
    # promotes L2 reuse, exactly as in tutorial 03.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate in fp32 for accuracy. The epilogue below stays in fp32 too.
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # --- Epilogue fusion (the whole point) ---------------------------------
    # bias is per output column, shape (N,). Load this tile's slice, broadcast
    # it across the M rows, add it, then apply GELU, all while acc is still
    # fp32. Nothing is written to global memory between the GEMM and the
    # activation, which is the traffic the three-kernel version would pay.
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    bias = tl.load(bias_ptr + offs_cn, mask=offs_cn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))). tl.math.erf is the
    # libdevice binding and the constant is 1 / sqrt(2).
    acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865475))

    # Cast once, at the store, to the output dtype (fp16 here, as in 03).
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


# %%
# A convenience wrapper
# ---------------------
#
# The wrapper checks the shape constraints, allocates the output, and launches
# the kernel over a 1D grid of output tiles.


def fused_matmul_bias_gelu(a, b, bias):
    assert a.shape[1] == b.shape[0], 'Incompatible dimensions'
    assert a.is_contiguous() and b.is_contiguous(), 'A and B must be contiguous'
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    fused_matmul_bias_gelu_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


# %%
# Verification
# ------------
#
# Compare against the unfused PyTorch reference. The fused kernel keeps the
# epilogue in fp32 and casts once at the store, so it matches the reference to
# floating point tolerance rather than bit for bit.

torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
bias = torch.randn((512, ), device='cuda', dtype=torch.float16)

c_fused = fused_matmul_bias_gelu(a, b, bias)
c_ref = torch.nn.functional.gelu(torch.matmul(a, b) + bias)

max_abs_err = (c_fused.float() - c_ref.float()).abs().max().item()
print(f'max absolute error vs torch reference: {max_abs_err:.3e}')
assert torch.allclose(c_fused, c_ref, atol=1e-2, rtol=1e-2), 'fused kernel diverged from reference'
print('OK: fused matmul + bias + gelu matches the unfused reference')
