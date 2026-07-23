"""Correctness metrics for the PolyKernel golden harness.

All metrics operate on the FLATTENED, fp32-upcast view of the inputs (the
rounding contract compares the final bf16 output upcast to fp32). Thresholds
are PINNED by the project plan (contract C):

    cosine    >= 0.999
    max_rel_err <= 1e-2   (eps = 1e-6)
    pcc       >= 0.99

`assert_correct` requires ALL THREE to hold and raises AssertionError printing
every computed value on failure.
"""

from __future__ import annotations

import numpy as np

# Pinned constants (do not change without updating the plan / downstream tasks).
COSINE_THRESHOLD = 0.999
MAX_REL_ERR_THRESHOLD = 1e-2
PCC_THRESHOLD = 0.99
REL_ERR_EPS = 1e-6


def _flat_fp32(x: np.ndarray) -> np.ndarray:
    """Flatten and upcast to fp32 (the comparison view per the rounding contract)."""
    return np.asarray(x).astype(np.float32).reshape(-1)


def cosine(a: np.ndarray, ref: np.ndarray) -> float:
    """Cosine similarity: dot(a, b) / (||a|| * ||b||) over flattened fp32 tensors."""
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        # Both-zero vectors are identical; one-zero is maximally dissimilar.
        return 1.0 if (np.linalg.norm(x) == 0.0 and np.linalg.norm(y) == 0.0) else 0.0
    return float(np.dot(x, y) / denom)


def max_rel_err(a: np.ndarray, ref: np.ndarray, eps: float = REL_ERR_EPS) -> float:
    """max(|a - ref| / (|ref| + eps)) over all elements (flattened fp32)."""
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    return float(np.max(np.abs(x - y) / (np.abs(y) + eps)))


def pcc(a: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation coefficient over flattened fp32 tensors."""
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc)) * np.sqrt(np.sum(yc * yc)))
    if denom == 0.0:
        # Zero variance: perfectly correlated if both constant-equal, else 0.
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.sum(xc * yc) / denom)


def assert_correct(actual: np.ndarray, ref: np.ndarray) -> None:
    """Assert `actual` matches `ref` within ALL pinned thresholds.

    Raises AssertionError printing cosine / max_rel_err / pcc on failure.
    """
    c = cosine(actual, ref)
    r = max_rel_err(actual, ref)
    p = pcc(actual, ref)
    ok = (c >= COSINE_THRESHOLD) and (r <= MAX_REL_ERR_THRESHOLD) and (p >= PCC_THRESHOLD)
    if not ok:
        raise AssertionError(
            "golden assert_correct FAILED: "
            f"cosine={c:.6f} (need >= {COSINE_THRESHOLD}), "
            f"max_rel_err={r:.6e} (need <= {MAX_REL_ERR_THRESHOLD:.0e}), "
            f"pcc={p:.6f} (need >= {PCC_THRESHOLD})"
        )
