"""Validation bridge: CPU-reference RMSNorm vs the NumPy/ml_dtypes golden (Todo 8).

Proves the CPU reference (kernels/cpu/cpu_reference.cpp) honors the golden's
math + rounding contract (tests/golden/golden.py):

    1. bf16 inputs/weights (round-to-nearest-even);
    2. fp32 accumulation / reduction;
    3. bf16 output (RNE);
    4. comparison on the bf16 output upcast to fp32 (metrics.assert_correct:
       cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99).

The CPU reference is the LOCAL EXECUTABLE SPEC and the correctness proof since no
NVIDIA GPU is available locally; the CUDA/HIP kernels share this exact algorithm
via the portable template (kernels/template/kernel_common.h).

On bit-exactness: for small reduction lengths the CPU ref is bit-exact vs the
golden; for large `cols` a handful of elements differ by 1 bf16 ULP because NumPy
sums in fp32 with pairwise blocking while the CPU ref sums sequentially — the
fp32 difference (~1e-7 relative) only flips the bf16 round at a rounding boundary.
We therefore assert the BINDING contract (assert_correct) plus a robust
element-wise rounding-fidelity floor (>99% of elements bit-exact), NOT strict
bit-exactness. Deterministic: fixed seed (conftest.SEED) -> identical every run.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

import golden as G
from metrics import assert_correct, cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16

# Floor on the fraction of bit-exact elements: proves the bf16 rounding contract
# is honored element-wise (the <1% mismatches are 1-ULP fp32 summation-order
# differences at bf16 rounding boundaries for large reduction lengths).
_EXACT_FRACTION_FLOOR = 0.99


def _check(out: np.ndarray, ref: np.ndarray) -> None:
    """Assert the binding golden contract + element-wise rounding fidelity."""
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    assert_correct(out, ref) # cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99
    exact_fraction = float(np.mean(out == ref))
    assert exact_fraction > _EXACT_FRACTION_FLOOR, (
        f"rounding fidelity too low: {exact_fraction:.6f} bit-exact "
        f"(need > {_EXACT_FRACTION_FLOOR}); cosine={cosine(out, ref):.6f} "
        f"rel={max_rel_err(out, ref):.3e} pcc={pcc(out, ref):.6f}"
    )


def test_rmsnorm_no_weight_small(run_cpu_ref, rand_bf16):
    # Small reduction (cols=16): the CPU ref is bit-exact vs the golden here.
    x = rand_bf16((8, 16))
    out = run_cpu_ref(x, weight=None, eps=1e-5)
    ref = G.rmsnorm(x, None, 1e-5)
    _check(out, ref)
    np.testing.assert_array_equal(out, ref) # bit-exact for small cols


def test_rmsnorm_with_weight(run_cpu_ref, rand_bf16):
    x = rand_bf16((256, 512))
    w = rand_bf16((512,))
    out = run_cpu_ref(x, weight=w, eps=1e-5)
    ref = G.rmsnorm(x, w, 1e-5)
    _check(out, ref)


def test_rmsnorm_odd_cols(run_cpu_ref, rand_bf16):
    # Non-multiple-of-2 last dim (127) exercises the general-cols path (the CUDA
    # kernel's scalar tail; the CPU ref handles any cols).
    x = rand_bf16((130, 127))
    out = run_cpu_ref(x, weight=None, eps=1e-5)
    ref = G.rmsnorm(x, None, 1e-5)
    _check(out, ref)


def test_rmsnorm_3d_flattens_leading_dims(run_cpu_ref, rand_bf16):
    # 3-D input: the driver flattens leading dims into rows and reduces axis=-1,
    # matching golden.rmsnorm(axis=-1).
    x = rand_bf16((2, 64, 128))
    w = rand_bf16((128,))
    out = run_cpu_ref(x, weight=w, eps=1e-5)
    ref = G.rmsnorm(x, w, 1e-5)
    _check(out, ref)


def test_rmsnorm_custom_epsilon(run_cpu_ref, rand_bf16):
    x = rand_bf16((32, 256))
    out = run_cpu_ref(x, weight=None, eps=1e-6)
    ref = G.rmsnorm(x, None, 1e-6)
    _check(out, ref)


def test_rmsnorm_large_canonical_shape(run_cpu_ref, rand_bf16):
    # The canonical T9 RMSNorm shape [2048, 4096] bf16: exercises a large fp32
    # reduction at scale (where the pairwise-vs-sequential summation order shows).
    x = rand_bf16((2048, 4096))
    out = run_cpu_ref(x, weight=None, eps=1e-5)
    ref = G.rmsnorm(x, None, 1e-5)
    _check(out, ref)
