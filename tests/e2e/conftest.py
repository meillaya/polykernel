"""Pytest config for the PolyKernel end-to-end MLP correctness test (Todo 17).

Reuses the CPU-reference validation mechanism from tests/kernels/conftest.py —
the SAME built driver executables, the SAME file-based .npy bridge, and the SAME
fixed seed — plus the golden harness (tests/golden). pytest fixtures are
directory-scoped, so this file re-exposes that mechanism for tests/e2e by loading
the kernels conftest as a module and reusing its constants + driver helpers (no
duplicated seed, executable paths, or .npy bridge logic).

The plain functions ``make_rand_bf16`` and ``build_cpu_drivers`` are importable so
the test module's standalone ``main()`` (the non-zero-exit harness used for the
negative-test evidence) can drive the identical mechanism without pytest fixtures.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "golden"
if str(_GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GOLDEN_DIR))

# Load the kernels conftest as a plain module to reuse its mechanism (SEED, the
# built CPU-ref executable paths, _require_exe, _drive_activation) — single source
# of truth, no duplication.
_KERNELS_CONFTEST = _PROJECT_ROOT / "tests" / "kernels" / "conftest.py"
_spec = importlib.util.spec_from_file_location("kernels_conftest", _KERNELS_CONFTEST)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load kernels conftest from {_KERNELS_CONFTEST}")
_k = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_k)

SEED = _k.SEED
_BF16 = ml_dtypes.bfloat16


def make_rand_bf16(seed: int = SEED):
    """Deterministic bf16 tensor factory (same fixed seed as the kernel suite)."""
    rng = np.random.default_rng(seed)

    def _make(shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> np.ndarray:
        return rng.uniform(low, high, size=shape).astype(_BF16)

    return _make


def _save(workdir: Path, name: str, arr: np.ndarray) -> Path:
    path = workdir / name
    np.save(path, arr)
    return path


def _run(cmd: list[str], name: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed (exit {proc.returncode}):\n{proc.stderr}")


def build_cpu_drivers(workdir: Path) -> SimpleNamespace:
    """Bundle the CPU-ref drivers composing the MLP block (reuses kernels exes).

    Each callable drives the SAME built executable + .npy bridge as
    tests/kernels/conftest.py and returns a bf16 array. The residual ``add`` has
    NO CPU-ref executable (kernels/cpu ships no add kernel — it is a trivial
    elementwise op fully determined by the rounding contract), so it is computed
    inline as fp32-add + bf16-RNE-round and flagged ``ref_only`` by the harness.
    """
    rmsnorm_exe = _k._require_exe(_k.CPU_REF_EXE)
    gelu_exe = _k._require_exe(_k.CPU_REF_GELU_EXE)
    matmul_exe = _k._require_exe(_k.CPU_REF_MATMUL_EXE)
    fused_rm_exe = _k._require_exe(_k.CPU_REF_FUSED_RMSNORM_MATMUL_EXE)

    def rmsnorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        out = workdir / "rms_out.npy"
        _run([str(rmsnorm_exe), str(_save(workdir, "rms_in.npy", x)), str(out),
              "--epsilon", repr(eps)], "cpu_ref_rmsnorm")
        return np.load(out).view(_BF16)

    def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        out = workdir / "mm_c.npy"
        _run([str(matmul_exe), str(_save(workdir, "mm_a.npy", a)),
              str(_save(workdir, "mm_b.npy", b)), str(out)], "cpu_ref_matmul")
        return np.load(out).view(_BF16)

    def gelu(x: np.ndarray) -> np.ndarray:
        return _k._drive_activation(gelu_exe, x, workdir)

    def fused_rmsnorm_matmul(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        out = workdir / "frm_out.npy"
        _run([str(fused_rm_exe), str(_save(workdir, "frm_x.npy", x)),
              str(_save(workdir, "frm_w.npy", w)), str(out),
              "--epsilon", repr(eps)], "cpu_ref_fused_rmsnorm_matmul")
        return np.load(out).view(_BF16)

    def add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # ref-only: no CPU-ref kernel exists for add; fp32-add + bf16 RNE round is
        # the rounding-contract definition (identical to golden.add by construction).
        return (a.astype(np.float32) + b.astype(np.float32)).astype(_BF16)

    return SimpleNamespace(rmsnorm=rmsnorm, matmul=matmul, gelu=gelu,
                           fused_rmsnorm_matmul=fused_rmsnorm_matmul, add=add)


@pytest.fixture
def rand_bf16():
    """Deterministic bf16 tensor factory (fresh fixed-seed RNG per test)."""
    return make_rand_bf16()


@pytest.fixture
def cpu_drivers(tmp_path: Path) -> SimpleNamespace:
    """CPU-ref driver bundle bound to a per-test temp workdir."""
    return build_cpu_drivers(tmp_path)
