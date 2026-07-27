"""PolyKernel QUANTIZATION-AWARE golden references (Todo 41 / Wave 8 elite).

A NEW golden module (the shared tests/golden/golden.py + metrics.py are NOT
modified). It is the single source of truth for the quantized-matmul correctness
gates: int8 WEIGHT-ONLY quantization and an fp8 (e4m3/e5m2) SIMULATION.

ROUNDING / QUANTIZATION CONTRACT (pinned for this task):

  int8 weight-only (per output channel = per column of W[K, N]):
    1. weights W are bf16; upcast to fp32 to compute the per-channel scale
         amax[n] = max_k |W[k, n]|;  scale[n] = amax[n] / 127   (symmetric, fp32)
       (an all-zero channel gets scale = 1 so its quantized weights stay 0);
    2. quantize  wq[k, n] = clip(round(W[k, n] / scale[n]), -127, 127)  -> int8
       (round = np.rint, round-to-nearest-even, matching the C++/GPU dequant);
    3. the kernel / CPU ref DEQUANTIZE inside the matmul:
         w_deq[k, n] = wq[k, n] * scale[n]   (fp32)
         C = bf16( fp32(bf16(A)) @ w_deq )    (fp32 accumulate, bf16 RNE out)
       Activations A stay bf16 — only the WEIGHTS are int8 (weight-only).

  fp8 simulation (weights only; NO real fp8 hardware — a simulation):
    1. weights W are bf16; round them onto the fp8 grid via ml_dtypes
         w_fp8 = W.astype(float8_e4m3fn | float8_e5m2).astype(fp32)
       which simulates storing the weights in fp8 and reading them back;
    2. C = bf16( fp32(bf16(A)) @ w_fp8 ).

THRESHOLDS (RELAXED vs the bf16 contract C, because quantization is lossy):
    cosine    >= 0.99
    max_rel_err <= 5e-2   (eps = 1e-6)
    pcc       >= 0.99
assert_quant_correct requires ALL THREE and prints every value on failure. The
metric FUNCTIONS (cosine / max_rel_err / pcc) are reused from metrics.py — only
the thresholds + the quantized references are new here.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from metrics import cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16
_F32 = np.float32

# Relaxed quantization-aware thresholds (pinned for Todo 41; see module docstring).
QUANT_COSINE_THRESHOLD = 0.99
QUANT_MAX_REL_ERR_THRESHOLD = 5e-2
QUANT_PCC_THRESHOLD = 0.99

# Symmetric int8 range (avoid the -128 asymmetry: qmax = 127).
_INT8_QMAX = 127

# fp8 simulation formats supported by the golden (ml_dtypes dtypes).
FP8_E4M3 = ml_dtypes.float8_e4m3fn
FP8_E5M2 = ml_dtypes.float8_e5m2


def _bf(x: np.ndarray) -> np.ndarray:
    """Round to bf16 (round-to-nearest-even, the ml_dtypes default)."""
    return np.asarray(x).astype(_BF16)


def _f32(x: np.ndarray) -> np.ndarray:
    """Upcast a bf16 tensor to fp32 for accumulation."""
    return np.asarray(x).astype(_F32)


# --- int8 weight-only quantization -----------------------------------------


def quantize_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric per-output-channel int8 quantization of a bf16 weight W[K, N].

    Returns (wq, scale): wq is int8 [K, N], scale is fp32 [N] (one scale per
    output channel / column). Matches the contract above; an all-zero channel
    gets scale = 1 so its quantized weights stay exactly 0 (no division by zero).
    """
    wf = _f32(_bf(w))
    amax = np.max(np.abs(wf), axis=0)  # [N]
    scale = np.where(amax > 0, amax / _INT8_QMAX, 1.0).astype(_F32)
    wq = np.rint(wf / scale[None, :])
    wq = np.clip(wq, -_INT8_QMAX, _INT8_QMAX).astype(np.int8)
    return wq, scale


def dequantize_per_channel(wq: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Dequantize int8 weights back to fp32: w_deq[k, n] = wq[k, n] * scale[n]."""
    return np.asarray(wq).astype(_F32) * np.asarray(scale).astype(_F32)[None, :]


def matmul_int8(a: np.ndarray, wq: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """int8 weight-only matmul: C = bf16( fp32(bf16(A)) @ dequant(wq, scale) ).

    A is bf16 [..., M, K]; wq is int8 [K, N]; scale is fp32 [N]. This is the
    reference the int8 GPU kernel + CPU ref are compared against. The dequant is
    per-element (wq * scale) in fp32, exactly as the kernel computes it inside the
    inner loop, so the only kernel-vs-golden difference is the fp32 K-reduction
    order (handled by the relaxed thresholds).
    """
    w_deq = dequantize_per_channel(wq, scale)
    return _bf(np.matmul(_f32(_bf(a)), w_deq))


# --- fp8 (e4m3/e5m2) simulation --------------------------------------------


def fp8_sim_dtype(fmt: str):
    """Map a format name to the ml_dtypes fp8 dtype (the simulation grid)."""
    if fmt == "e4m3":
        return FP8_E4M3
    if fmt == "e5m2":
        return FP8_E5M2
    raise ValueError(f"unknown fp8 format {fmt!r} (expected 'e4m3' or 'e5m2')")


def matmul_fp8_sim(a: np.ndarray, w: np.ndarray, fmt: str) -> np.ndarray:
    """fp8 weight SIMULATION: round bf16 weights onto the fp8 grid, then matmul.

    C = bf16( fp32(bf16(A)) @ fp8(W) ), where fp8(W) = W cast to float8_e4m3fn /
    float8_e5m2 and back to fp32 (simulated fp8 storage; NO fp8 hardware). A stays
    bf16 (weight-only). `fmt` is 'e4m3' or 'e5m2'.
    """
    dt = fp8_sim_dtype(fmt)
    w_fp8 = _f32(_bf(w)).astype(dt).astype(_F32)
    return _bf(np.matmul(_f32(_bf(a)), w_fp8))


# --- relaxed quantization-aware correctness gate ---------------------------


def assert_quant_correct(actual: np.ndarray, ref: np.ndarray) -> None:
    """Assert `actual` matches `ref` within the RELAXED quant thresholds.

    Requires cosine >= 0.99 AND max_rel_err <= 5e-2 AND pcc >= 0.99; raises
    AssertionError printing every computed value on failure. (The bf16 contract-C
    gate in metrics.assert_correct is stricter; quantization is lossy, so this
    gate is the appropriate one for int8 / fp8-sim outputs.)
    """
    c = cosine(actual, ref)
    r = max_rel_err(actual, ref)
    p = pcc(actual, ref)
    ok = (
        c >= QUANT_COSINE_THRESHOLD
        and r <= QUANT_MAX_REL_ERR_THRESHOLD
        and p >= QUANT_PCC_THRESHOLD
    )
    if not ok:
        raise AssertionError(
            "quant assert_quant_correct FAILED: "
            f"cosine={c:.6f} (need >= {QUANT_COSINE_THRESHOLD}), "
            f"max_rel_err={r:.6e} (need <= {QUANT_MAX_REL_ERR_THRESHOLD:.0e}), "
            f"pcc={p:.6f} (need >= {QUANT_PCC_THRESHOLD})"
        )
