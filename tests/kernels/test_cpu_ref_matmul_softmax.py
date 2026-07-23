"""Validation bridge: CPU-reference MatMul + Softmax vs the golden (Todo 10).

Proves the CPU reference (kernels/cpu/cpu_reference.cpp) honors the golden's
math + rounding contract (tests/golden/golden.py) for the two Wave-2 ops:

    MatMul:  C = A @ B   (fp32 accumulation, bf16 RNE output; batched leading
             dims flattened into M, matching np.matmul on [...,M,K] @ [K,N])
    Softmax: stable softmax over the last axis (subtract the row max before exp,
             fp32 compute, bf16 RNE output)

compared via metrics.assert_correct (cosine >= 0.999 AND max_rel_err <= 1e-2 AND
pcc >= 0.99). The CPU reference is the LOCAL EXECUTABLE SPEC (no NVIDIA GPU
locally); the CUDA kernels in kernels/generated/{matmul,softmax}.cu share this
exact algorithm via the portable template. The non-tile-multiple matmul shape and
the odd-last-dim softmax shape exercise the CUDA kernels' bounds checks / scalar
tail (the CPU ref loops element-wise, so it handles any shape). Deterministic:
fixed seed (conftest.SEED) -> identical every run.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

import golden as G
from metrics import assert_correct, cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16

# MatMul: reduction only (fp32 dot product), so like RMSNorm the divergence source
# is sequential-vs-pairwise summation order -> a handful of 1-ULP bf16 flips at
# rounding boundaries for large K. >99% bit-exact proves the rounding contract.
_MATMUL_EXACT_FLOOR = 0.99

# Softmax: reduction (sum) AND a per-element exp; the fp32 exp libm vs numpy SIMD
# divergence plus the summation-order difference makes more elements land 1 bf16
# ULP off than a pure reduction, so the bit-exact floor is softer. The BINDING
# contract is assert_correct (cosine/rel/pcc), which softmax passes with margin.
_SOFTMAX_EXACT_FLOOR = 0.90


def _check(out: np.ndarray, ref: np.ndarray, floor: float) -> None:
    """Assert the binding golden contract + element-wise rounding fidelity."""
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    assert_correct(out, ref)  # cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99
    exact_fraction = float(np.mean(out == ref))
    assert exact_fraction > floor, (
        f"rounding fidelity too low: {exact_fraction:.6f} bit-exact "
        f"(need > {floor}); cosine={cosine(out, ref):.6f} "
        f"rel={max_rel_err(out, ref):.3e} pcc={pcc(out, ref):.6f}"
    )


# --- MatMul -----------------------------------------------------------------


def test_matmul_square(run_matmul, rand_bf16):
    # [M,N,K] = [128,128,128]: a clean tile-multiple square GEMM.
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    out = run_matmul(a, b)
    ref = G.matmul(a, b)
    _check(out, ref, _MATMUL_EXACT_FLOOR)


def test_matmul_rectangular(run_matmul, rand_bf16):
    # [M,N,K] = [256,512,256]: rectangular GEMM (M != N != K), all tile multiples.
    a = rand_bf16((256, 256))
    b = rand_bf16((256, 512))
    out = run_matmul(a, b)
    ref = G.matmul(a, b)
    _check(out, ref, _MATMUL_EXACT_FLOOR)


def test_matmul_non_tile_multiple(run_matmul, rand_bf16):
    # [M,N,K] = [130,127,65]: NONE of M/N/K is a tile multiple -> exercises the
    # CUDA kernel's bounds checks (the CPU ref handles any shape element-wise).
    a = rand_bf16((130, 65))
    b = rand_bf16((65, 127))
    out = run_matmul(a, b)
    ref = G.matmul(a, b)
    _check(out, ref, _MATMUL_EXACT_FLOOR)


def test_matmul_batched_flattens_leading_dims(run_matmul, rand_bf16):
    # Batched [2,64,128] @ [128,64] -> [2,64,64]: the driver flattens the leading
    # batch dim into M, matching golden.matmul (np.matmul on [...,M,K] @ [K,N]).
    a = rand_bf16((2, 64, 128))
    b = rand_bf16((128, 64))
    out = run_matmul(a, b)
    ref = G.matmul(a, b)
    _check(out, ref, _MATMUL_EXACT_FLOOR)


# --- Softmax ----------------------------------------------------------------


def test_softmax_tokens_vocab(run_softmax, rand_bf16):
    # [tokens, vocab] = [64, 256]: the canonical attention/LLM-head softmax shape.
    x = rand_bf16((64, 256))
    out = run_softmax(x)
    ref = G.softmax(x)
    _check(out, ref, _SOFTMAX_EXACT_FLOOR)


def test_softmax_odd_last_dim(run_softmax, rand_bf16):
    # Odd last dim (127) exercises the CUDA kernel's scalar strided tail (the CPU
    # ref handles any cols).
    x = rand_bf16((32, 127))
    out = run_softmax(x)
    ref = G.softmax(x)
    _check(out, ref, _SOFTMAX_EXACT_FLOOR)


def test_softmax_3d_flattens_leading_dims(run_softmax, rand_bf16):
    # 3-D input: the driver flattens leading dims into rows and reduces axis=-1,
    # matching golden.softmax(axis=-1).
    x = rand_bf16((2, 16, 64))
    out = run_softmax(x)
    ref = G.softmax(x)
    _check(out, ref, _SOFTMAX_EXACT_FLOOR)
