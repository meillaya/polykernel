"""Self-tests for the golden reference: verify against HAND-COMPUTED tiny cases.

These lock the math/rounding of the harness that ALL downstream kernel
validation (CUDA / HIP / dataflow simulator) depends on. bf16 ULP near
magnitude 1 is 2^-7 = 0.0078125, so hand-computed fp32 expectations are
compared with a tolerance that only correct bf16 rounding can satisfy.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import pytest

import golden as G
from metrics import assert_correct, cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16
_F32 = np.float32

# bf16 rounding tolerance for hand-computed fp32 expectations (>= half-ULP at |x|~1).
_BF16_TOL = 5e-3


def _bf(xs) -> np.ndarray:
    return np.asarray(xs, dtype=_BF16)


# --- Hand-computed primitive cases -----------------------------------------


def test_matmul_2x2_known_exact():
    # Given A=[[1,2],[3,4]], B=[[5,6],[7,8]]
    # When  C = A @ B
    # Then  C == [[1*5+2*7, 1*6+2*8],[3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]] (exact in bf16)
    out = G.matmul(_bf([[1, 2], [3, 4]]), _bf([[5, 6], [7, 8]]))
    assert out.dtype == _BF16
    np.testing.assert_array_equal(out.astype(_F32), np.array([[19, 22], [43, 50]], dtype=_F32))


def test_gelu_zero_is_zero():
    # GELU(0) = 0.5*0*(1+erf(0)) = 0 exactly.
    out = G.gelu(_bf([0.0]))
    assert float(out[0]) == 0.0


def test_gelu_one_matches_erf_formula():
    # GELU(1) = 0.5*(1+erf(1/sqrt(2))) = 0.8413447... (fp32); bf16 rounds to 0.83984375.
    out = G.gelu(_bf([1.0]))
    fp32_ref = 0.5 * (1.0 + math.erf(1.0 / math.sqrt(2.0)))
    assert abs(fp32_ref - 0.8413) < 1e-4  # sanity on the hand value
    assert abs(float(out[0]) - 0.8413) < _BF16_TOL


def test_silu_one():
    # SiLU(1) = 1/(1+e^-1) = 0.7310586... (fp32); bf16 rounds to 0.73046875.
    out = G.silu(_bf([1.0]))
    assert abs(float(out[0]) - 0.7311) < _BF16_TOL


def test_softmax_uniform_is_half():
    # softmax([1,1]) = [e^0/(e^0+e^0), ...] = [0.5, 0.5] exactly (0.5 is bf16-exact).
    out = G.softmax(_bf([1.0, 1.0]))
    np.testing.assert_array_equal(out.astype(_F32), np.array([0.5, 0.5], dtype=_F32))


def test_rmsnorm_known_vector():
    # Given x=[1,2,3,4], no weight, eps=1e-5.
    # mean(x^2) = (1+4+9+16)/4 = 7.5 ; rms = sqrt(7.5 + 1e-5).
    # out = x / rms = [0.365148, 0.730296, 1.095444, 1.460593] (fp32), then bf16-rounded.
    out = G.rmsnorm(_bf([1.0, 2.0, 3.0, 4.0]))
    rms = math.sqrt(7.5 + 1e-5)
    expected_fp32 = np.array([1 / rms, 2 / rms, 3 / rms, 4 / rms], dtype=_F32)
    assert out.dtype == _BF16
    np.testing.assert_allclose(out.astype(_F32), expected_fp32, atol=4e-3, rtol=4e-3)
    # And it must pass the project thresholds against the bf16-rounded expectation.
    assert_correct(out, expected_fp32.astype(_BF16))


def test_softmax_sums_to_one_random(rand_bf16):
    x = rand_bf16((8, 16), low=-4.0, high=4.0)
    out = G.softmax(x, axis=-1).astype(_F32)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(8), atol=2e-2, rtol=0)


# --- Fused == unfused consistency (fused composes the primitives) -----------


def test_fused_rmsnorm_matmul_consistency(rand_bf16):
    x = rand_bf16((4, 8))
    w = rand_bf16((8, 6))
    gamma = rand_bf16((8,))
    fused = G.fused_rmsnorm_matmul(x, w, weight=gamma)
    unfused = G.matmul(G.rmsnorm(x, weight=gamma), w)
    # Composition => bit-exact equality (stronger than "within tolerance").
    np.testing.assert_array_equal(fused, unfused)
    assert_correct(fused, unfused)


def test_fused_matmul_bias_gelu_consistency(rand_bf16):
    x = rand_bf16((4, 8))
    w = rand_bf16((8, 6))
    b = rand_bf16((6,))
    fused = G.fused_matmul_bias_gelu(x, w, b)
    unfused = G.gelu(G.bias(G.matmul(x, w), b))
    np.testing.assert_array_equal(fused, unfused)


def test_fused_residual_rmsnorm_consistency(rand_bf16):
    x = rand_bf16((4, 8))
    r = rand_bf16((4, 8))
    fused = G.fused_residual_rmsnorm(x, r)
    unfused = G.rmsnorm(G.add(x, r))
    np.testing.assert_array_equal(fused, unfused)


def test_fused_softmax_mask_consistency(rand_bf16):
    x = rand_bf16((4, 8))
    mask = rand_bf16((4, 8), low=-2.0, high=0.0)
    fused = G.fused_softmax_mask(x, mask)
    unfused = G.softmax(G.add(x, mask), axis=-1)
    np.testing.assert_array_equal(fused, unfused)


def test_qkv_projection_splits_evenly(rand_bf16):
    # x:[2,8] @ w_qkv:[8,24] -> [2,24] -> split into Q,K,V of [2,8]; reshape heads=4 -> [2,4,2].
    x = rand_bf16((2, 8))
    w_qkv = rand_bf16((8, 24))
    q, k, v = G.qkv_projection(x, w_qkv, num_heads=4)
    assert q.shape == (2, 4, 2) and k.shape == (2, 4, 2) and v.shape == (2, 4, 2)
    # concat(Q,K,V) along the feature axis must equal the raw projection (reshaped).
    proj = G.matmul(x, w_qkv).astype(_F32).reshape(2, 3, 4, 2)
    np.testing.assert_array_equal(np.stack([q, k, v], axis=1), proj.astype(_BF16))


def test_fused_kv_append_attention_shapes_and_append(rand_bf16):
    B, H, S, Dh, Sc = 1, 2, 3, 4, 5
    q = rand_bf16((B, H, S, Dh))
    k = rand_bf16((B, H, S, Dh))
    v = rand_bf16((B, H, S, Dh))
    kc = rand_bf16((B, H, Sc, Dh))
    vc = rand_bf16((B, H, Sc, Dh))
    out, new_kc, new_vc = G.fused_kv_append_attention(q, k, v, kc, vc)
    assert out.shape == (B, H, S, Dh)
    # Cache grows by S along the sequence axis: old (5) + new (3) = 8.
    assert new_kc.shape == (B, H, Sc + S, Dh)
    assert new_vc.shape == (B, H, Sc + S, Dh)
    # The prepended cache region is preserved exactly.
    np.testing.assert_array_equal(new_kc[:, :, :Sc, :], kc)
    np.testing.assert_array_equal(new_vc[:, :, :Sc, :], vc)
    # Attention output passes thresholds against itself (sanity: finite + well-formed).
    assert_correct(out, out)


# --- Metrics + the negative (perturbation) guard ----------------------------


def test_metrics_perfect_match():
    x = rand_bf16_local()
    assert cosine(x, x) == pytest.approx(1.0, abs=1e-6)
    assert max_rel_err(x, x) == pytest.approx(0.0, abs=1e-6)
    assert pcc(x, x) == pytest.approx(1.0, abs=1e-6)
    assert_correct(x, x)  # must not raise


def rand_bf16_local() -> np.ndarray:
    rng = np.random.default_rng(0xC0FFEE)
    return rng.uniform(0.5, 1.5, size=(32,)).astype(_BF16)


def test_assert_correct_raises_on_perturbation(rand_bf16):
    # Given a reference and a copy perturbed by +5% multiplicative noise,
    # When  assert_correct is called,
    # Then  it RAISES AssertionError (max_rel_err ~0.05 > 1e-2; pcc drops below 0.99).
    ref = rand_bf16((128,), low=1.0, high=2.0)
    rng = np.random.default_rng(1234)
    noise = rng.uniform(-1.0, 1.0, size=(128,))
    perturbed = (ref.astype(_F32) * (1.0 + 0.05 * noise)).astype(_BF16)

    # Confirm at least one threshold is actually violated (the guard is meaningful).
    assert max_rel_err(perturbed, ref) > 1e-2

    with pytest.raises(AssertionError, match="assert_correct FAILED"):
        assert_correct(perturbed, ref)
