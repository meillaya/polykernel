"""Pytest config for the PolyKernel CPU-reference kernel tests (Todo 8 / Wave 2).

Puts tests/golden on sys.path (so `import golden` / `from metrics import ...`
resolve) and provides fixtures that locate + drive the built `cpu_ref_rmsnorm`
executable through the file-based .npy validation bridge (Pinned contract B):

    python (fixed-seed bf16 input) -> input.npy -> cpu_ref_rmsnorm -> output.npy
        -> python (np.load(...).view(ml_dtypes.bfloat16)) -> metrics.assert_correct

The .npy encoding is '<V2' (raw uint16 = bf16 bits): ml_dtypes.bfloat16 has no
native NumPy dtype, so np.save serializes it as void-2-bytes and the reader
reinterprets those bytes via `.view(ml_dtypes.bfloat16)` (exact round-trip).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "golden"
if str(_GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GOLDEN_DIR))

# The CPU-ref driver lands at build/kernels/cpu_ref_rmsnorm (no
# LLVM_RUNTIME_OUTPUT_INTDIR is set project-wide; see kernels/CMakeLists.txt).
CPU_REF_EXE = _PROJECT_ROOT / "build" / "kernels" / "cpu_ref_rmsnorm"

# Fixed seed -> identical tensors on every run (defeats the flaky-tests class).
SEED = 0xC0FFEE


@pytest.fixture(scope="session")
def cpu_ref_exe() -> Path:
    """Path to the built cpu_ref_rmsnorm driver; clear error if not built."""
    if not CPU_REF_EXE.exists():
        raise FileNotFoundError(
            f"cpu_ref_rmsnorm not found at {CPU_REF_EXE}. Build it first:\n"
            f"  nix develop -c cmake --build build --target cpu_ref_rmsnorm"
        )
    return CPU_REF_EXE


@pytest.fixture
def run_cpu_ref(cpu_ref_exe: Path, tmp_path: Path):
    """Drive the CPU-ref RMSNorm on a bf16 input; return the bf16 output.

    Returns a callable `run(x, weight=None, eps=1e-5) -> np.ndarray(bf16)`.
    """

    def _run(
        x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5
    ) -> np.ndarray:
        in_path = tmp_path / "input.npy"
        out_path = tmp_path / "output.npy"
        np.save(in_path, x)
        cmd = [str(cpu_ref_exe), str(in_path), str(out_path), "--epsilon", repr(eps)]
        if weight is not None:
            w_path = tmp_path / "weight.npy"
            np.save(w_path, weight)
            cmd += ["--weight", str(w_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cpu_ref_rmsnorm failed (exit {proc.returncode}):\n{proc.stderr}"
            )
        return np.load(out_path).view(ml_dtypes.bfloat16)

    return _run


@pytest.fixture
def rand_bf16():
    """Deterministic bf16 tensor factory (fresh fixed-seed RNG per test)."""
    rng = np.random.default_rng(SEED)

    def _make(shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> np.ndarray:
        return rng.uniform(low, high, size=shape).astype(ml_dtypes.bfloat16)

    return _make


# Activation CPU-ref drivers (Todo 9): elementwise input.npy -> output.npy.
CPU_REF_GELU_EXE = _PROJECT_ROOT / "build" / "kernels" / "cpu_ref_gelu"
CPU_REF_SILU_EXE = _PROJECT_ROOT / "build" / "kernels" / "cpu_ref_silu"


def _require_exe(path: Path) -> Path:
    """Resolve a built CPU-ref driver, with a clear build hint if absent."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found at {path}. Build it first:\n"
            f"  nix develop -c cmake --build build --target {path.name}"
        )
    return path


def _drive_activation(exe: Path, x: np.ndarray, tmp_path: Path) -> np.ndarray:
    """Drive an elementwise CPU-ref driver: bf16 input.npy -> bf16 output.npy."""
    in_path = tmp_path / "input.npy"
    out_path = tmp_path / "output.npy"
    np.save(in_path, x)
    proc = subprocess.run(
        [str(exe), str(in_path), str(out_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{exe.name} failed (exit {proc.returncode}):\n{proc.stderr}"
        )
    return np.load(out_path).view(ml_dtypes.bfloat16)


@pytest.fixture
def run_gelu(tmp_path: Path):
    """Drive the CPU-ref GELU on a bf16 input; return the bf16 output."""
    exe = _require_exe(CPU_REF_GELU_EXE)
    return lambda x: _drive_activation(exe, x, tmp_path)


@pytest.fixture
def run_silu(tmp_path: Path):
    """Drive the CPU-ref SiLU on a bf16 input; return the bf16 output."""
    exe = _require_exe(CPU_REF_SILU_EXE)
    return lambda x: _drive_activation(exe, x, tmp_path)
