"""Pytest config for the correctness-gated benchmarking harness tests (Todo 25).

Puts tests/golden (the NumPy golden + metrics) AND tools/polykernel-bench (the
bench CLI module) on sys.path, and provides a session fixture that builds the
hipcc bench driver once + probes the local gfx1101 GPU. If hipcc / a usable HIP
device is absent, the module is SKIPPED-local (mirrors tests/kernels/conftest.py
+ test_hip_run.py: the Todo 18 gate guards this; it would run on the first
available rental GPU instead).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "golden"
_BENCH_DIR = _PROJECT_ROOT / "tools" / "polykernel-bench"
_BENCH_PY = _BENCH_DIR / "bench.py"
_ARCH = "gfx1101"

for _d in (_GOLDEN_DIR, _BENCH_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


@pytest.fixture(scope="session")
def bench_driver() -> Path:
    """Build the bench driver once + probe the GPU; skip if hipcc/device absent."""
    if shutil.which("hipcc") is None:
        pytest.skip(
            "hipcc not found on PATH (run inside `nix develop`, rocmPackages.clr); "
            "bench-gate SKIPPED-local."
        )
    import bench  # noqa: PLC0415  (sys.path set above).

    driver = bench.build_driver(_ARCH, timeout=900)

    # GPU probe: a tiny matmul autotune must succeed (or fail specifically with a
    # no-device error => skip). Belt-and-braces on the Todo 18 gate.
    probe_cache = _BENCH_DIR.parent.parent / "build" / "polykernel-bench" / "probe.json"
    proc = subprocess.run(
        [sys.executable, str(_BENCH_PY), "--autotune", "--op", "matmul",
         "--shape", "64,64,64", "--variants", "1", "--warmup", "1", "--iters", "2",
         "--cache-out", str(probe_cache)],
        capture_output=True, text=True, timeout=300,
    )
    combined = (proc.stdout + proc.stderr).lower()
    if proc.returncode != 0 and (
        "no device" in combined or "no hip gpus" in combined or "invalid device" in combined
    ):
        pytest.skip(f"no usable HIP device ({_ARCH}): {proc.stderr.strip()[:200]}")
    if proc.returncode != 0:
        raise RuntimeError(f"bench GPU probe failed (exit {proc.returncode}):\n{proc.stderr}")
    return driver
