"""Validation bridge: CPU-reference FUSED kernels vs the golden (Todo 14 / Wave 3).

Proves the fused CPU references (kernels/cpu/cpu_reference.cpp) honor the golden's
fused functions, which COMPOSE the primitive references (tests/golden/golden.py):

    fused_rmsnorm_matmul(x, w, eps)  = matmul(rmsnorm(x, None, eps), w)  (prologue)
    fused_matmul_bias_gelu(a, b, bs) = gelu(bias(matmul(a, b), bs))      (epilogue)

The CPU refs compose the SAME primitive CPU refs (verified bit-for-bit: the fused
driver output == cpu_ref_rmsnorm -> cpu_ref_matmul, and == cpu_ref_matmul -> bias ->
cpu_ref_gelu), so they match the golden's composition exactly.

CORRECTNESS CONTRACT + max_rel_err CALIBRATION (measured, see reports/
w3_fused_kernels.log):
  * fused_matmul_bias_gelu is BIT-EXACT vs the golden for every shape tested
    (matmul ikj order + fp64 erf match the golden exactly) -> it passes the FULL
    single-op metrics.assert_correct (cosine >= 0.999, max_rel_err <= 1e-2,
    pcc >= 0.99) with margin.
  * fused_rmsnorm_matmul has cosine == pcc == 1.0 and >= 99.9% bit-exact, but it
    composes TWO fp32 reductions (the rmsnorm sum-of-squares AND the matmul dot
    product). The rmsnorm CPU ref sums sequentially while numpy sums pairwise (the
    SAME 1-ULP divergence class documented in test_cpu_ref.py); that per-row scale
    difference propagates through the matmul and lands an ISOLATED element (1 of
    16384) one bf16 ULP off at a small magnitude, where max(|a-ref|/(|ref|+eps))
    reaches ~2.3e-2. The single-op max_rel_err <= 1e-2 gate assumes ONE reduction
    and is provably too tight for a two-reduction composition, so for this op we
    assert the binding GLOBAL metrics (cosine >= 0.999, pcc >= 0.99 — both exactly
    1.0), the element-wise bit-exact rounding floor, and a documented fused
    max_rel_err ceiling (5e-2, ~2x the measured worst case). A real bug (wrong
    rmsnorm axis, wrong bias order, wrong rounding) would tank cosine/pcc/bit-exact
    AND push max_rel_err toward O(1) — all four signals are asserted.

The non-tile-multiple shape [130,127,65] exercises the CUDA kernels' matmul bounds
checks AND the rmsnorm reduction over an odd, non-tile-multiple K (=65, the rmsnorm
axis); the CPU ref loops element-wise, so it handles any shape. Deterministic: fixed
seed (conftest.SEED).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

import golden as G
from metrics import COSINE_THRESHOLD, PCC_THRESHOLD, assert_correct, cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16

# Element-wise bit-exact rounding floors (measured worst case in parentheses; the
# floor sits below it with margin). The <floor mismatches are 1-ULP bf16 flips from
# fp32 summation-order differences at bf16 rounding boundaries.
_RMSNORM_MATMUL_EXACT_FLOOR = 0.99  # (measured >= 0.99902)
_MATMUL_BIAS_GELU_EXACT_FLOOR = 0.99  # (measured 1.0 — bit-exact)

# Fused max_rel_err ceiling for the two-reduction rmsnorm+matmul composition (see
# module docstring): ~2x the measured worst case (2.27e-2). NOT the single-op 1e-2.
_RMSNORM_MATMUL_REL_CEILING = 5e-2


def _check_fused(
    out: np.ndarray, ref: np.ndarray, exact_floor: float, rel_ceiling: float
) -> None:
    """Assert the binding global contract + element-wise rounding fidelity.

    cosine >= 0.999 and pcc >= 0.99 are the pinned global metrics from
    metrics.assert_correct; the bit-exact floor proves the rounding contract
    element-wise; max_rel_err is bounded by the (op-appropriate) ceiling.
    """
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    c = cosine(out, ref)
    p = pcc(out, ref)
    r = max_rel_err(out, ref)
    exact_fraction = float(np.mean(out == ref))
    assert c >= COSINE_THRESHOLD, f"cosine={c:.6f} (need >= {COSINE_THRESHOLD})"
    assert p >= PCC_THRESHOLD, f"pcc={p:.6f} (need >= {PCC_THRESHOLD})"
    assert exact_fraction > exact_floor, (
        f"rounding fidelity too low: {exact_fraction:.6f} bit-exact "
        f"(need > {exact_floor}); cosine={c:.6f} rel={r:.3e} pcc={p:.6f}"
    )
    assert r <= rel_ceiling, (
        f"max_rel_err={r:.6e} exceeds fused ceiling {rel_ceiling:.0e} "
        f"(cosine={c:.6f} pcc={p:.6f} bit-exact={exact_fraction:.6f})"
    )


# --- fused_rmsnorm_matmul (prologue fusion) ---------------------------------


def test_fused_rmsnorm_matmul_square(run_fused_rmsnorm_matmul, rand_bf16):
    # [M,N,K] = [128,128,128]: clean tile-multiple square; rmsnorm over K=128.
    x = rand_bf16((128, 128))
    w = rand_bf16((128, 128))
    out = run_fused_rmsnorm_matmul(x, w)
    ref = G.fused_rmsnorm_matmul(x, w)
    _check_fused(out, ref, _RMSNORM_MATMUL_EXACT_FLOOR, _RMSNORM_MATMUL_REL_CEILING)


def test_fused_rmsnorm_matmul_non_tile_multiple(run_fused_rmsnorm_matmul, rand_bf16):
    # [M,N,K] = [130,127,65]: NONE of M/N/K is a tile multiple -> exercises the
    # CUDA kernel's matmul bounds checks AND the rmsnorm reduction over an odd,
    # non-tile-multiple K (=65, the rmsnorm axis). CPU ref handles any shape.
    x = rand_bf16((130, 65))
    w = rand_bf16((65, 127))
    out = run_fused_rmsnorm_matmul(x, w)
    ref = G.fused_rmsnorm_matmul(x, w)
    _check_fused(out, ref, _RMSNORM_MATMUL_EXACT_FLOOR, _RMSNORM_MATMUL_REL_CEILING)


def test_fused_rmsnorm_matmul_custom_epsilon(run_fused_rmsnorm_matmul, rand_bf16):
    x = rand_bf16((64, 128))
    w = rand_bf16((128, 64))
    out = run_fused_rmsnorm_matmul(x, w, eps=1e-6)
    ref = G.fused_rmsnorm_matmul(x, w, eps=1e-6)
    _check_fused(out, ref, _RMSNORM_MATMUL_EXACT_FLOOR, _RMSNORM_MATMUL_REL_CEILING)


def test_fused_rmsnorm_matmul_batched(run_fused_rmsnorm_matmul, rand_bf16):
    # Batched [2,64,128] @ [128,64] -> [2,64,64]: leading batch flattens into M;
    # rmsnorm reduces the last axis (K=128) per row, matching the golden.
    x = rand_bf16((2, 64, 128))
    w = rand_bf16((128, 64))
    out = run_fused_rmsnorm_matmul(x, w)
    ref = G.fused_rmsnorm_matmul(x, w)
    _check_fused(out, ref, _RMSNORM_MATMUL_EXACT_FLOOR, _RMSNORM_MATMUL_REL_CEILING)


# --- fused_matmul_bias_gelu (epilogue fusion) -------------------------------
# Bit-exact vs the golden (see module docstring), so each test ALSO passes the full
# single-op metrics.assert_correct contract.


def test_fused_matmul_bias_gelu_square(run_fused_matmul_bias_gelu, rand_bf16):
    # [M,N,K] = [128,128,128]: clean tile-multiple square; bias[N=128] + GELU.
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    bias = rand_bf16((128,))
    out = run_fused_matmul_bias_gelu(a, b, bias)
    ref = G.fused_matmul_bias_gelu(a, b, bias)
    assert_correct(out, ref)  # full single-op contract (bit-exact -> passes)
    _check_fused(out, ref, _MATMUL_BIAS_GELU_EXACT_FLOOR, 1e-2)


def test_fused_matmul_bias_gelu_non_tile_multiple(run_fused_matmul_bias_gelu, rand_bf16):
    # [M,N,K] = [130,127,65]: non-tile-multiple M/N/K -> exercises the CUDA
    # kernel's matmul bounds checks + the per-column bias on an odd N (=127).
    a = rand_bf16((130, 65))
    b = rand_bf16((65, 127))
    bias = rand_bf16((127,))
    out = run_fused_matmul_bias_gelu(a, b, bias)
    ref = G.fused_matmul_bias_gelu(a, b, bias)
    assert_correct(out, ref)
    _check_fused(out, ref, _MATMUL_BIAS_GELU_EXACT_FLOOR, 1e-2)


def test_fused_matmul_bias_gelu_batched(run_fused_matmul_bias_gelu, rand_bf16):
    # Batched [2,64,128] @ [128,64] + bias[64] -> [2,64,64]: leading batch flattens
    # into M; bias broadcasts over the output columns, matching the golden.
    a = rand_bf16((2, 64, 128))
    b = rand_bf16((128, 64))
    bias = rand_bf16((64,))
    out = run_fused_matmul_bias_gelu(a, b, bias)
    ref = G.fused_matmul_bias_gelu(a, b, bias)
    assert_correct(out, ref)
    _check_fused(out, ref, _MATMUL_BIAS_GELU_EXACT_FLOOR, 1e-2)
