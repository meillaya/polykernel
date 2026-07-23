"""PolyKernel golden reference implementations (NumPy + ml_dtypes.bfloat16).

This module is the single source of truth for kernel correctness (CUDA, HIP,
dataflow simulator). ROUNDING CONTRACT (pinned, contract C):

  1. inputs/weights rounded to bf16 (round-to-nearest-even, ml_dtypes default);
  2. all reductions / accumulation performed in fp32 (upcast before computing);
  3. each op's OUTPUT rounded back to bf16 (so chained ops see bf16 inputs);
  4. comparison is on the final bf16 output upcast to fp32 (see metrics.py).

Every public function takes bf16 numpy arrays and returns bf16. Fused variants
compose the primitive references, so a fused op is EXACTLY equal to running the
unfused primitives in sequence (the strongest consistency guarantee; real fused
kernels are compared against this within the cosine >= 0.999 tolerance).

NumPy has NO bfloat16 and NO erf; we use ml_dtypes.bfloat16 for the dtype and
the stdlib `math.erf` (C libm, correctly rounded) vectorized for exact GELU.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np

_BF16 = ml_dtypes.bfloat16
_F32 = np.float32

# Exact erf-based GELU constant: 1 / sqrt(2).
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _bf(x: np.ndarray) -> np.ndarray:
    """Round to bf16 (round-to-nearest-even, the ml_dtypes default)."""
    return np.asarray(x).astype(_BF16)


def _f32(x: np.ndarray) -> np.ndarray:
    """Upcast a bf16 tensor to fp32 for accumulation."""
    return np.asarray(x).astype(_F32)


# Vectorized exact erf (numpy ships no erf; scipy is not a dependency).
_verf = np.vectorize(math.erf, otypes=[np.float64])


def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise add with broadcasting (fp32 compute, bf16 output)."""
    return _bf(_f32(_bf(a)) + _f32(_bf(b)))


def bias(x: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add bias[N] broadcast over [..., N] (last axis)."""
    return _bf(_f32(_bf(x)) + _f32(_bf(b)))


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply [..., M, K] @ [K, N] with fp32 accumulation, bf16 output."""
    return _bf(np.matmul(_f32(_bf(a)), _f32(_bf(b))))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    xf = _f32(_bf(x))
    return _bf(0.5 * xf * (1.0 + _verf(xf * _INV_SQRT2).astype(_F32)))


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish: x / (1 + exp(-x))."""
    xf = _f32(_bf(x))
    return _bf(xf / (1.0 + np.exp(-xf)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax (subtract max), fp32 compute, bf16 output."""
    xf = _f32(_bf(x))
    shifted = xf - np.max(xf, axis=axis, keepdims=True)
    e = np.exp(shifted)
    return _bf(e / np.sum(e, axis=axis, keepdims=True))


def rmsnorm(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm over the last axis: x / sqrt(mean(x^2) + eps) [* weight]."""
    xf = _f32(_bf(x))
    rms = np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + eps)
    out = xf / rms
    if weight is not None:
        out = out * _f32(_bf(weight))
    return _bf(out)


# --- Fused variants (compose the primitives above) -------------------------


def fused_rmsnorm_matmul(
    x: np.ndarray, w: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5
) -> np.ndarray:
    """RMSNorm(x) then MatMul with w (prologue fusion)."""
    return matmul(rmsnorm(x, weight, eps), w)


def fused_matmul_bias_gelu(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GELU(MatMul(x, w) + b) (epilogue fusion; bias BEFORE gelu)."""
    return gelu(bias(matmul(x, w), b))


def fused_residual_rmsnorm(
    x: np.ndarray, residual: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5
) -> np.ndarray:
    """RMSNorm(x + residual)."""
    return rmsnorm(add(x, residual), weight, eps)


def fused_softmax_mask(x: np.ndarray, mask: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax(x + mask) — additive mask (e.g. large-negative for masked positions)."""
    return softmax(add(x, mask), axis=axis)


def qkv_projection(
    x: np.ndarray, w_qkv: np.ndarray, num_heads: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project x @ w_qkv ([..., D] @ [D, 3D]) and split last axis into Q, K, V.

    If `num_heads` is given, each of Q/K/V is reshaped to [..., num_heads, head_dim]
    (head_dim = D // num_heads); otherwise they are returned flat as [..., D].
    """
    proj = matmul(x, w_qkv)  # [..., 3D]
    q, k, v = np.split(_f32(proj), 3, axis=-1)
    if num_heads is not None:
        head_dim = q.shape[-1] // num_heads
        new_shape = q.shape[:-1] + (num_heads, head_dim)
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)
    return _bf(q), _bf(k), _bf(v)


def fused_kv_append_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Append K/V to their caches, then scaled dot-product attention.

    Layout: [B, H, S, D_h] for q/k/v/k_cache/v_cache; the head count is the
    explicit H axis (num_heads == q.shape[1]). The new k/v are concatenated onto
    the cache along the sequence axis (axis=2). Returns (attn_output,
    new_k_cache, new_v_cache), all bf16.

    The five tensor args (Q, K, V, K-cache, V-cache) plus the optional additive
    mask are the irreducible I/O of a KV-append attention; they are kept explicit
    rather than bundled, matching how the CUDA/HIP/simulator kernels pass them.
    """
    new_k = np.concatenate([_f32(_bf(k_cache)), _f32(_bf(k))], axis=2)
    new_v = np.concatenate([_f32(_bf(v_cache)), _f32(_bf(v))], axis=2)

    qf = _f32(_bf(q))
    head_dim = qf.shape[-1]
    # scores: [B, H, S_q, S_kv] = Q @ K^T / sqrt(D_h)
    scores = np.matmul(qf, np.swapaxes(new_k, -1, -2)) / math.sqrt(head_dim)
    if mask is not None:
        scores = scores + _f32(_bf(mask))
    # stable softmax over the key axis
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    e = np.exp(shifted)
    attn = e / np.sum(e, axis=-1, keepdims=True)
    out = np.matmul(attn, new_v)  # [B, H, S_q, D_h]
    return _bf(out), _bf(new_k), _bf(new_v)
