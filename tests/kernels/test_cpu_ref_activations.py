"""Validation bridge: CPU-reference GELU + SiLU vs the golden (Todo 9 / Wave 2).

Proves the CPU reference (kernels/cpu/cpu_reference.cpp) honors the golden's
math + rounding contract (tests/golden/golden.py) for the two activations:

    GELU: y = 0.5 * x * (1 + erf(x / sqrt(2)))   (exact erf-based, NOT tanh approx)
    SiLU: y = x / (1 + exp(-x))

with bf16 in/out + fp32 compute, compared via metrics.assert_correct
(cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99). The CPU reference is
the LOCAL EXECUTABLE SPEC (no NVIDIA GPU locally); the CUDA kernels in
kernels/generated/{gelu,silu}.cu share this exact algorithm via the portable
template. Odd-total-length shapes exercise the CUDA kernel's scalar tail (the CPU
ref loops element-wise, so it handles any length). Deterministic: fixed seed
(conftest.SEED) -> identical every run.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

import golden as G
from metrics import assert_correct, cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16

# Floor on the fraction of bit-exact elements (see test_cpu_ref.py rationale):
# proves the bf16 rounding contract is honored element-wise. Activations have no
# reduction, so the only divergence source is the fp32 erf/exp libm vs numpy's
# SIMD implementation (a handful of 1-ULP differences at rounding boundaries).
_EXACT_FRACTION_FLOOR = 0.99


def _check(out: np.ndarray, ref: np.ndarray) -> None:
    """Assert the binding golden contract + element-wise rounding fidelity."""
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    assert_correct(out, ref)  # cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99
    exact_fraction = float(np.mean(out == ref))
    assert exact_fraction > _EXACT_FRACTION_FLOOR, (
        f"rounding fidelity too low: {exact_fraction:.6f} bit-exact "
        f"(need > {_EXACT_FRACTION_FLOOR}); cosine={cosine(out, ref):.6f} "
        f"rel={max_rel_err(out, ref):.3e} pcc={pcc(out, ref):.6f}"
    )


def test_gelu_small(run_gelu, rand_bf16):
    # Small input: the GELU CPU ref is bit-exact vs the golden (both evaluate erf
    # in fp64 via the same libm; no reduction => no summation-order divergence).
    x = rand_bf16((8, 16))
    out = run_gelu(x)
    ref = G.gelu(x)
    _check(out, ref)
    np.testing.assert_array_equal(out, ref)  # bit-exact for small input


def test_gelu_odd_length(run_gelu, rand_bf16):
    # Odd total element count (131*127 = 16637) exercises the CUDA kernel's
    # scalar tail (non-multiple-of-vector-width); the CPU ref handles any length.
    x = rand_bf16((131, 127))
    out = run_gelu(x)
    ref = G.gelu(x)
    _check(out, ref)


def test_gelu_canonical_shape(run_gelu, rand_bf16):
    # The canonical Wave-2 shape [2048, 4096] bf16 at scale.
    x = rand_bf16((2048, 4096))
    out = run_gelu(x)
    ref = G.gelu(x)
    _check(out, ref)


def test_silu_small(run_silu, rand_bf16):
    x = rand_bf16((8, 16))
    out = run_silu(x)
    ref = G.silu(x)
    _check(out, ref)


def test_silu_odd_length(run_silu, rand_bf16):
    # Odd total element count (131*127 = 16637) exercises the CUDA scalar tail.
    x = rand_bf16((131, 127))
    out = run_silu(x)
    ref = G.silu(x)
    _check(out, ref)


def test_silu_canonical_shape(run_silu, rand_bf16):
    x = rand_bf16((2048, 4096))
    out = run_silu(x)
    ref = G.silu(x)
    _check(out, ref)
