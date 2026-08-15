"""CUDA runtime launch + golden correctness on the RTX 6000 Ada (Todo 8 / Wave 3).

The CUDA twin of tests/kernels/test_hip_run.py (Todo 20 / Wave 4): the SAME
generated kernels (kernels/generated/*.cu) the HIP backend compiles with
-DPOLYKERNEL_HIP are here compiled UNCHANGED by nvcc -DPOLYKERNEL_CUDA
--arch=sm_89 and ACTUALLY EXECUTED on a real NVIDIA GPU (RTX 6000 Ada, sm_89;
todo 9 runs this module on the pod - there is no local NVIDIA GPU). Proving the
portable compute logic is numerically correct when run on a real GPU validates
BOTH backends.

Mechanism (REUSED, not reinvented): mirrors tests/kernels/conftest.py exactly - a
session fixture compiles a `cuda_run` launcher with nvcc (the thin error-checked
CUDA-runtime driver lib/Runtime/cuda_run_main.cpp + the reused .npy bridge
kernels/cpu/npy_io.cpp + the generated kernels), then each test drives it through
the SAME fixed-seed (conftest.SEED) bf16 .npy file bridge and compares the GPU
output to the golden via metrics.assert_correct (contract C: cosine >= 0.999 AND
max_rel_err <= 1e-2 AND pcc >= 0.99).

BUILD GOTCHA (todo 7, documented): the nvcc invocation MUST include `-include
cstdio`. kernel_common.h's PK_CHECK uses fprintf/stderr with NO #include
<cstdio>, and - unlike hip_runtime.h, which pulls <cstdio> transitively -
cuda_runtime.h does NOT. Without the force-include the generated kernel TUs fail
host compilation with `identifier "stderr" is undefined`.

SHAPES (representative + golden-tractable; the GPU runs them fast, the NumPy
golden is the cost): per op a tile-multiple shape, an odd / non-tile-multiple
shape (bounds checks + scalar tail), and (matmul/softmax/fused) a batched/3-D
shape that flattens leading dims. Measured on gfx1101 (the HIP twin): cosine ==
pcc == 1.0 and >= 99.99% bit-exact for every binding case.

max_rel_err CALIBRATION (mirrors test_cpu_ref_fused.py + test_hip_run.py): the
single-op gate max_rel_err <= 1e-2 assumes ONE reduction summed in NumPy's order.
The tiled GEMM accumulates over K in BLOCK_K tiles (a different fp32 summation
order than NumPy), so at LARGE K an ISOLATED element can land 1 bf16 ULP off at
small magnitude where max(|a-ref|/(|ref|+eps)) exceeds 1e-2 - with cosine == pcc
== 1.0 and >= 99.99% bit-exact (NOT a bug; a real bug tanks cosine/pcc toward 0).
test_cuda_matmul_large_k_reduction_order documents this class with the same
calibrated ceiling (5e-2) the committed CPU fused test and the HIP twin use;
every other matmul shape (K <= 128) passes the FULL assert_correct.

Negative path: `cuda_run neg` launches a kernel with a too-large block (2048 >
the max threads-per-block of 1024 on sm_80/sm_89/sm_90 alike); the runtime must
CATCH + report the CUDA error (no silent corruption, no crash) - see
reports/pass2_cuda_run_neg.log (todo 7's launcher QA).

NO-GPU GOTCHA (todo 7, documented): a machine with no NVIDIA driver makes
cudaGetDeviceCount return cudaErrorInsufficientDriver ("CUDA driver version is
insufficient for CUDA runtime version"), NOT cudaErrorNoDevice. `_gpu_probe`
therefore treats ANY non-success of the `cuda_run dev` probe (the direct
cudaGetDeviceCount call, exit 0 only when a usable device exists) as no-device -
so this module is SKIPPED-local here and only runs on the pod (todo 9).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import golden as G
from metrics import (
    COSINE_THRESHOLD,
    PCC_THRESHOLD,
    assert_correct,
    cosine,
    max_rel_err,
    pcc,
)

_BF16 = ml_dtypes.bfloat16
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CUDA_RUN_EXE = _PROJECT_ROOT / "build" / "cuda_run" / "cuda_run"
_ARCH = "sm_89"

# The generated kernels under test (the closed op set; silu is not launched here).
_KERNELS = (
    "rmsnorm",
    "gelu",
    "matmul",
    "softmax",
    "fused_rmsnorm_matmul",
    "fused_matmul_bias_gelu",
)

# Element-wise bit-exact rounding floor (measured worst case >= 0.99994 on the
# HIP twin's gfx1101; the CUDA path uses the identical generated kernels).
_EXACT_FLOOR = 0.99
# Calibrated max_rel_err ceiling for the large-K tiled-GEMM reduction-order class.
_LARGE_K_REL_CEILING = 5e-2


@pytest.fixture(scope="session")
def cuda_run_exe() -> Path:
    """Compile the CUDA launcher once per session; skip if nvcc is unavailable."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip(
            "nvcc not found on PATH (run inside `nix develop`, cudaPackages_12_6); "
            "CUDA-run SKIPPED-local."
        )
    sources = [
        "lib/Runtime/cuda_run_main.cpp",
        "kernels/cpu/npy_io.cpp",
        *(f"kernels/generated/{k}.cu" for k in _KERNELS),
    ]
    # `-include cstdio` is REQUIRED (todo 7 gotcha): cuda_runtime.h does not pull
    # <cstdio>, and kernel_common.h's PK_CHECK uses fprintf/stderr - without the
    # force-include the generated kernel TUs fail with `stderr is undefined`.
    cmd = [
        nvcc,
        "-x",
        "cu",
        "-DPOLYKERNEL_CUDA",
        "-include",
        "cstdio",
        f"-arch={_ARCH}",
        "-Ikernels/template",
        "-Iinclude",
        "-Ikernels/cpu",
        *sources,
        "-o",
        str(_CUDA_RUN_EXE),
    ]
    _CUDA_RUN_EXE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvcc failed to build cuda_run (exit {proc.returncode}):\n{proc.stderr}"
        )
    return _CUDA_RUN_EXE


@pytest.fixture(scope="session", autouse=True)
def _gpu_probe(cuda_run_exe: Path) -> None:
    """Skip the module if no usable CUDA device (belt-and-braces on the T18 gate).

    `cuda_run dev` IS the direct cudaGetDeviceCount probe: it exits 0 only when
    a usable device exists (printing DEVICE <name> COMPUTE_CAPABILITY <cc>). ANY
    other exit means no usable device - a machine with no NVIDIA driver makes
    cudaGetDeviceCount return cudaErrorInsufficientDriver (NOT
    cudaErrorNoDevice), so the skip is keyed on non-success, not on a specific
    error string (todo 7's documented gotcha).
    """
    pr = subprocess.run([str(cuda_run_exe), "dev"], capture_output=True, text=True,
                        timeout=120)
    if pr.returncode != 0:
        pytest.skip(f"no usable CUDA device ({_ARCH}): {pr.stderr.strip()[:200]}")


@pytest.fixture
def cuda_drive(cuda_run_exe: Path, tmp_path: Path):
    """Drive `cuda_run <op> ...`; raise (with stderr) on a non-zero exit."""

    def _drive(args: list[str]) -> subprocess.CompletedProcess:
        proc = subprocess.run([str(cuda_run_exe), *args], capture_output=True,
                              text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"cuda_run {args[0]} failed (exit {proc.returncode}):\n{proc.stderr}"
            )
        return proc

    return _drive


def _load(path: Path) -> np.ndarray:
    return np.load(path).view(_BF16)


@pytest.fixture
def run_cuda_rmsnorm(cuda_drive, tmp_path: Path):
    def _run(x: np.ndarray, weight: np.ndarray | None = None, eps: float = 1e-5):
        inp = tmp_path / "input.npy"
        out = tmp_path / "output.npy"
        np.save(inp, x)
        cmd = ["rmsnorm", str(inp), str(out), "--epsilon", repr(eps)]
        if weight is not None:
            w = tmp_path / "weight.npy"
            np.save(w, weight)
            cmd += ["--weight", str(w)]
        cuda_drive(cmd)
        return _load(out)

    return _run


@pytest.fixture
def run_cuda_gelu(cuda_drive, tmp_path: Path):
    def _run(x: np.ndarray):
        inp = tmp_path / "input.npy"
        out = tmp_path / "output.npy"
        np.save(inp, x)
        cuda_drive(["gelu", str(inp), str(out)])
        return _load(out)

    return _run


@pytest.fixture
def run_cuda_matmul(cuda_drive, tmp_path: Path):
    def _run(a: np.ndarray, b: np.ndarray):
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        out = tmp_path / "c.npy"
        np.save(ap, a)
        np.save(bp, b)
        cuda_drive(["matmul", str(ap), str(bp), str(out)])
        return _load(out)

    return _run


@pytest.fixture
def run_cuda_softmax(cuda_drive, tmp_path: Path):
    def _run(x: np.ndarray):
        inp = tmp_path / "input.npy"
        out = tmp_path / "output.npy"
        np.save(inp, x)
        cuda_drive(["softmax", str(inp), str(out)])
        return _load(out)

    return _run


@pytest.fixture
def run_cuda_fused_rmsnorm_matmul(cuda_drive, tmp_path: Path):
    def _run(x: np.ndarray, w: np.ndarray, eps: float = 1e-5):
        xp = tmp_path / "x.npy"
        wp = tmp_path / "w.npy"
        out = tmp_path / "out.npy"
        np.save(xp, x)
        np.save(wp, w)
        cuda_drive(["fused_rmsnorm_matmul", str(xp), str(wp), str(out),
                    "--epsilon", repr(eps)])
        return _load(out)

    return _run


@pytest.fixture
def run_cuda_fused_matmul_bias_gelu(cuda_drive, tmp_path: Path):
    def _run(a: np.ndarray, b: np.ndarray, bias: np.ndarray):
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        bs = tmp_path / "bias.npy"
        out = tmp_path / "out.npy"
        np.save(ap, a)
        np.save(bp, b)
        np.save(bs, bias)
        cuda_drive(["fused_matmul_bias_gelu", str(ap), str(bp), str(bs), str(out)])
        return _load(out)

    return _run


def _check(out: np.ndarray, ref: np.ndarray, rel_ceiling: float = 1e-2,
           full_contract: bool = True) -> None:
    """Assert the binding global contract + element-wise rounding fidelity.

    cosine >= 0.999 and pcc >= 0.99 are the pinned global metrics; the bit-exact
    floor proves the bf16 rounding contract element-wise; max_rel_err is bounded
    by the (op-appropriate) ceiling. With full_contract=True the binding single-op
    metrics.assert_correct (incl. max_rel_err <= 1e-2) must also hold.
    """
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    c = cosine(out, ref)
    p = pcc(out, ref)
    r = max_rel_err(out, ref)
    exact = float(np.mean(out == ref))
    assert c >= COSINE_THRESHOLD, f"cosine={c:.6f} (need >= {COSINE_THRESHOLD})"
    assert p >= PCC_THRESHOLD, f"pcc={p:.6f} (need >= {PCC_THRESHOLD})"
    assert exact > _EXACT_FLOOR, (
        f"rounding fidelity too low: {exact:.6f} bit-exact (need > {_EXACT_FLOOR}); "
        f"cosine={c:.6f} rel={r:.3e} pcc={p:.6f}"
    )
    assert r <= rel_ceiling, (
        f"max_rel_err={r:.6e} exceeds ceiling {rel_ceiling:.0e} "
        f"(cosine={c:.6f} pcc={p:.6f} bit-exact={exact:.6f})"
    )
    if full_contract:
        assert_correct(out, ref)


# --- RMSNorm ----------------------------------------------------------------


def test_cuda_rmsnorm_with_weight(run_cuda_rmsnorm, rand_bf16):
    x = rand_bf16((256, 512))
    w = rand_bf16((512,))
    _check(run_cuda_rmsnorm(x, w), G.rmsnorm(x, w))


def test_cuda_rmsnorm_odd_cols(run_cuda_rmsnorm, rand_bf16):
    x = rand_bf16((130, 127))  # odd last dim -> scalar tail path
    _check(run_cuda_rmsnorm(x), G.rmsnorm(x, None, 1e-5))


def test_cuda_rmsnorm_3d_flattens(run_cuda_rmsnorm, rand_bf16):
    x = rand_bf16((2, 64, 128))
    w = rand_bf16((128,))
    _check(run_cuda_rmsnorm(x, w), G.rmsnorm(x, w))


# --- GELU -------------------------------------------------------------------


def test_cuda_gelu(run_cuda_gelu, rand_bf16):
    x = rand_bf16((64, 128))
    _check(run_cuda_gelu(x), G.gelu(x))


def test_cuda_gelu_odd_n(run_cuda_gelu, rand_bf16):
    x = rand_bf16((3, 127))  # odd element count -> scalar tail
    _check(run_cuda_gelu(x), G.gelu(x))


# --- MatMul -----------------------------------------------------------------


def test_cuda_matmul_square(run_cuda_matmul, rand_bf16):
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    _check(run_cuda_matmul(a, b), G.matmul(a, b))


def test_cuda_matmul_rectangular(run_cuda_matmul, rand_bf16):
    a = rand_bf16((256, 128))  # M != N, K = 128 (tile-multiple)
    b = rand_bf16((128, 256))
    _check(run_cuda_matmul(a, b), G.matmul(a, b))


def test_cuda_matmul_non_tile_multiple(run_cuda_matmul, rand_bf16):
    a = rand_bf16((130, 65))  # none of M/N/K is a tile multiple -> bounds checks
    b = rand_bf16((65, 127))
    _check(run_cuda_matmul(a, b), G.matmul(a, b))


def test_cuda_matmul_batched_flattens(run_cuda_matmul, rand_bf16):
    a = rand_bf16((2, 64, 128))  # leading batch flattens into M
    b = rand_bf16((128, 64))
    _check(run_cuda_matmul(a, b), G.matmul(a, b))


def test_cuda_matmul_large_k_reduction_order(run_cuda_matmul, rand_bf16):
    # K = 256: the tiled GEMM's K-accumulation order differs from NumPy's, so an
    # ISOLATED element lands 1 bf16 ULP off at small magnitude (measured max_rel_err
    # ~3e-2 on gfx1101) while cosine == pcc == 1.0 and >= 99.99% bit-exact. This is
    # the documented reduction-order class (see module docstring), NOT a bug, so it
    # uses the calibrated ceiling instead of the single-op max_rel_err <= 1e-2 gate.
    a = rand_bf16((256, 256))
    b = rand_bf16((256, 512))
    _check(run_cuda_matmul(a, b), G.matmul(a, b),
           rel_ceiling=_LARGE_K_REL_CEILING, full_contract=False)


# --- Softmax ----------------------------------------------------------------


def test_cuda_softmax_tokens_vocab(run_cuda_softmax, rand_bf16):
    x = rand_bf16((64, 256))
    _check(run_cuda_softmax(x), G.softmax(x))


def test_cuda_softmax_odd_last_dim(run_cuda_softmax, rand_bf16):
    x = rand_bf16((32, 127))
    _check(run_cuda_softmax(x), G.softmax(x))


def test_cuda_softmax_3d_flattens(run_cuda_softmax, rand_bf16):
    x = rand_bf16((2, 16, 64))
    _check(run_cuda_softmax(x), G.softmax(x))


# --- Fused ------------------------------------------------------------------


def test_cuda_fused_rmsnorm_matmul_square(run_cuda_fused_rmsnorm_matmul, rand_bf16):
    x = rand_bf16((128, 128))
    w = rand_bf16((128, 128))
    _check(run_cuda_fused_rmsnorm_matmul(x, w), G.fused_rmsnorm_matmul(x, w))


def test_cuda_fused_rmsnorm_matmul_non_tile(run_cuda_fused_rmsnorm_matmul, rand_bf16):
    x = rand_bf16((130, 65))  # rmsnorm over odd, non-tile-multiple K (=65)
    w = rand_bf16((65, 127))
    _check(run_cuda_fused_rmsnorm_matmul(x, w), G.fused_rmsnorm_matmul(x, w))


def test_cuda_fused_matmul_bias_gelu_square(run_cuda_fused_matmul_bias_gelu, rand_bf16):
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    bias = rand_bf16((128,))
    _check(run_cuda_fused_matmul_bias_gelu(a, b, bias),
           G.fused_matmul_bias_gelu(a, b, bias))


def test_cuda_fused_matmul_bias_gelu_non_tile(run_cuda_fused_matmul_bias_gelu, rand_bf16):
    a = rand_bf16((130, 65))
    b = rand_bf16((65, 127))
    bias = rand_bf16((127,))  # per-column bias on an odd N (=127)
    _check(run_cuda_fused_matmul_bias_gelu(a, b, bias),
           G.fused_matmul_bias_gelu(a, b, bias))


# --- Negative: an invalid launch must be CAUGHT, not silent -----------------


def test_cuda_negative_launch_caught(cuda_run_exe: Path):
    # `cuda_run neg` launches a probe kernel with a 2048-thread block (> the max
    # threads-per-block of 1024 on sm_80/sm_89/sm_90 alike - the per-block limit,
    # not the 1536 per-SM limit). The runtime must catch + report the CUDA error
    # and exit non-zero (no silent wrong result, no crash).
    proc = subprocess.run([str(cuda_run_exe), "neg"], capture_output=True, text=True,
                          timeout=120)
    assert proc.returncode == 1, (
        f"negative launch must exit 1 (caught); got {proc.returncode}\n{proc.stderr}"
    )
    assert "invalid launch CAUGHT" in proc.stderr, proc.stderr
    assert "invalid configuration argument" in proc.stderr, proc.stderr
    assert "PASS - invalid launch CAUGHT" in proc.stderr, proc.stderr
