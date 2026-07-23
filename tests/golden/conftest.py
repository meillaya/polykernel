"""Pytest fixtures for the PolyKernel golden harness.

Provides a DETERMINISTIC bf16 random-tensor factory backed by a FIXED seed, so
every test run produces identical tensors (the flaky-tests adversarial class is
defeated by construction).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

# Fixed seed for the whole golden suite (deterministic across runs / machines).
GOLDEN_SEED = 0xC0FFEE


@pytest.fixture
def rand_bf16():
    """Deterministic bf16 tensor factory.

    Returns a callable `make(shape, low=-1.0, high=1.0, seed=None)`:
      - draws uniform fp32 values in [low, high) and rounds to bf16;
      - with `seed=None` it draws from a shared fixed-seed generator (so a
        sequence of calls is reproducible); pass an explicit `seed` to reset.
    """
    rng = np.random.default_rng(GOLDEN_SEED)

    def _make(
        shape: tuple[int, ...],
        low: float = -1.0,
        high: float = 1.0,
        seed: int | None = None,
    ) -> np.ndarray:
        r = np.random.default_rng(seed) if seed is not None else rng
        return r.uniform(low, high, size=shape).astype(ml_dtypes.bfloat16)

    return _make
