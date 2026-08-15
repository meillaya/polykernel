"""CUDA MMA bf16 tensor-core MatMul on sm_80+ (RTX 6000 Ada target sm_89) (Todo 10 / Wave 4 pass 2).

This is WHERE the CUDA MMA matmul variant (kernels/generated/matmul_mma.cu) is
proven correct against the golden — and proven ADDITIVE: the tiled SCALAR baseline
(kernels/generated/matmul.cu, Todo 10 / Wave 2) remains the correctness reference
and is UNCHANGED. MMA is a codegen VARIANT of the closed `polykernel.matmul` op the
autotuner (Todo 15) can SELECT on NVIDIA GPUs; it is correctness-gated, so an MMA
variant that fails golden is excluded (validated=false) and the scalar baseline
stands — MMA NEVER blocks the wave. This is the CUDA twin of the RDNA3 WMMA path
(tests/kernels/test_wmma.py, Todo 22 / Wave 4): same additive + correctness-gated
contract, `path=mma` report annotation (contract H, no new op).

INSTRUCTION + FRAGMENTS (sm_80+, `mma.sync.aligned.row.col.m16n16k16.f32.bf16.bf16.f32`,
driven by nvcuda::wmma from <mma.h>; one warp computes a 16x16 f32 accumulator tile
from 16x16 bf16 A and 16x16 bf16 B). The fragment API abstracts the hardware lane
mapping — the kernel stages shared tiles and calls load_matrix_sync/mma_sync/
store_matrix_sync. Contract K (the CUDA twin):
    A: fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>  As[16][16] row-major
    B: fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>   Bs[16][16] col-major
       — B MUST be col_major (NOT row_major): the KxN B tile is staged TRANSPOSED in
       shared memory (each N-column's 16 K-values contiguous), which is what the
       col_major fragment load expects; row_major would read an NxK/B^T view and
       transpose the operand, failing golden.
    C: fragment<accumulator, 16, 16, 16, float>; mma_sync(acc, a, b, acc);
       store_matrix_sync(..., mem_row_major) -> shared Cs[16][16], rounded to bf16.

Mechanism (REUSED, not reinvented): mirrors tests/kernels/test_wmma.py exactly. A
session fixture compiles a standalone `mma_run` launcher with nvcc (the reused .npy
bridge kernels/cpu/npy_io.cpp + the MMA kernel + the scalar baseline), built with
-DPOLYKERNEL_MMA_MAIN to activate the test-only driver main inside matmul_mma.cu.
Each test drives it through the SAME fixed-seed (conftest.SEED = 0xC0FFEE) bf16 .npy
file bridge and compares the GPU output to the golden via metrics.assert_correct
(contract C: cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99), plus an
element-wise bit-exact rounding floor. The driver uses the CUDA runtime API directly
(no lib/Runtime/ dependency — that is concurrent Todo 7's territory).

SHAPES (all multiples of the 16x16x16 MMA tile — the clean, no-tail case the
autotuner selects; documented per the task): 128x128x128 square, 256x128x256
rectangular (M != N), 192x160x224 non-square / non-power-of-two multiple, and
256x256x256 (larger K). Expected on the Ada: cosine == pcc == 1.0 and >= 99.99%
bit-exact for every MMA shape (the WMMA precedent measured exactly that on gfx1101).

PATH ANNOTATION (contract H): the driver prints `[polykernel-mma] path=<mma|
mma_bad|scalar> M=.. N=.. K=..` to stderr; the happy tests assert path=mma was used.

NEGATIVE PATH (proves MMA is safely additive): `mma_bad` is the SAME kernel with a
deliberately WRONG store layout (mem_col_major instead of mem_row_major — the
believable bug of forgetting the row-major output layout), which TRANSPOSES the C
tile. It fails golden decisively (cosine/pcc ~0, max_rel_err >> 1) -> the variant
would be marked validated=false and excluded — while the scalar baseline STILL
PASSES on the identical inputs. A bad MMA variant never replaces the correct scalar
baseline. Evidence: reports/pass2_mma_neg.log.

If nvcc / a usable CUDA device is absent, the module is SKIPPED-local: the session
fixture compiles the launcher (nvcc builds sm_89 with no device needed), the
`_gpu_probe` skips the GPU-run cases cleanly (SKIPPED, not FAILED), and the real
run happens on the pod (Todo 11).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
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
_MMA_RUN_EXE = _PROJECT_ROOT / "build" / "mma_run" / "mma_run"
_ARCH = "sm_89"

# Element-wise bit-exact rounding floor (WMMA measured worst case >= 0.9999 on
# gfx1101; MMA is the same bf16 rounding contract).
_EXACT_FLOOR = 0.99
# Calibrated max_rel_err ceiling for the large-K tiled-GEMM reduction-order class
# (MIRRORS test_hip_run.py's _LARGE_K_REL_CEILING / test_wmma.py). At large K the
# MMA fp32 K-accumulation order differs from NumPy's, so an ISOLATED element can
# land 1 bf16 ULP off at small magnitude where max_rel_err nudges past 1e-2 — with
# cosine == pcc == 1.0 and >= 99.99% bit-exact (NOT a bug; a real bug tanks
# cosine/pcc toward 0). The large-K shape uses this ceiling + full_contract=False,
# exactly as the committed scalar baseline + WMMA tests do.
_LARGE_K_REL_CEILING = 5e-2


@pytest.fixture(scope="session")
def mma_run_exe() -> Path:
    """Compile the standalone MMA launcher once per session; skip if no nvcc.

    Builds matmul_mma.cu (with -DPOLYKERNEL_MMA_MAIN to activate its test-only
    driver main) together with the scalar baseline matmul.cu (for the `scalar`
    mode the negative test compares against) and the reused .npy bridge — the
    nvcc invocation mirroring test_wmma.py's hipcc invocation, plus the MMA driver
    macro. nvcc compiles sm_89 with NO device required (compile is GPU-free).

    `-include cstdio` is REQUIRED: kernel_common.h's PK_CHECK uses fprintf/stderr
    without #include <cstdio>, so a full host+device nvcc link of ANY generated
    kernel (here matmul.cu) fails otherwise — the pre-existing T40 gotcha; the flag
    injects <cstdio> into every TU at test-build time without touching the shared
    template.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip(
            "nvcc not found on PATH (run inside `nix develop`, cudaPackages_12_6); "
            "MMA SKIPPED-local."
        )
    assert nvcc is not None  # narrow for the type checker (pytest.skip is a raise)
    sources = [
        "kernels/cpu/npy_io.cpp",
        "kernels/generated/matmul.cu",
        "kernels/generated/matmul_mma.cu",
    ]
    cmd = [
        nvcc,
        f"-arch={_ARCH}",
        "-DPOLYKERNEL_CUDA",
        "-DPOLYKERNEL_MMA_MAIN",
        "-include",
        "cstdio",
        "-O2",
        "-std=c++17",
        "-Ikernels/template",
        "-Iinclude",
        "-Ikernels/cpu",
        *sources,
        "-o",
        str(_MMA_RUN_EXE),
    ]
    _MMA_RUN_EXE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"nvcc failed to build mma_run (exit {proc.returncode}):\n{proc.stderr}"
        )
    return _MMA_RUN_EXE


@pytest.fixture(scope="session", autouse=True)
def _gpu_probe(mma_run_exe: Path) -> None:
    """Skip the module if no usable CUDA device (belt-and-braces on the T18 gate)."""
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.npy"
        b = Path(td) / "b.npy"
        o = Path(td) / "o.npy"
        np.save(a, np.zeros((16, 16), dtype=_BF16))
        np.save(b, np.zeros((16, 16), dtype=_BF16))
        pr = subprocess.run([str(mma_run_exe), "mma", str(a), str(b), str(o)],
                            capture_output=True, text=True, timeout=120)
        err = pr.stderr.lower()
        if pr.returncode != 0 and (
            "no cuda" in err
            or "no device" in err
            or "invalid device" in err
            or "insufficient" in err  # e.g. "driver version is insufficient"
        ):
            pytest.skip(f"no usable CUDA device ({_ARCH}): {pr.stderr.strip()[:200]}")


@pytest.fixture
def mma_drive(mma_run_exe: Path):
    """Drive `mma_run <mode> A B C`; raise (with stderr) on a non-zero exit.

    Returns (output_bf16, stderr_text) so tests can assert on the path annotation.
    """

    def _drive(mode: str, a: np.ndarray, b: np.ndarray, tmp_path: Path):
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        cp = tmp_path / "c.npy"
        np.save(ap, a)
        np.save(bp, b)
        proc = subprocess.run([str(mma_run_exe), mode, str(ap), str(bp), str(cp)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"mma_run {mode} failed (exit {proc.returncode}):\n{proc.stderr}"
            )
        return np.load(cp).view(_BF16), proc.stderr

    return _drive


def _metrics(out: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """The contract-C metrics + bit-exact fidelity, for assertions and evidence."""
    return {
        "cosine": cosine(out, ref),
        "pcc": pcc(out, ref),
        "max_rel_err": max_rel_err(out, ref),
        "bit_exact": float(np.mean(out == ref)),
    }


def _check_mma(out: np.ndarray, ref: np.ndarray, rel_ceiling: float = 1e-2,
               full_contract: bool = True) -> dict[str, float]:
    """Assert the binding global contract + element-wise rounding fidelity.

    cosine >= 0.999 and pcc >= 0.99 are the pinned global metrics; the bit-exact
    floor proves the bf16 rounding contract element-wise; max_rel_err is bounded by
    the (shape-appropriate) ceiling. With full_contract=True the binding single-op
    metrics.assert_correct (incl. max_rel_err <= 1e-2) must also hold; the large-K
    reduction-order shape uses full_contract=False + the calibrated ceiling instead.
    """
    assert out.dtype == _BF16
    assert out.shape == ref.shape
    m = _metrics(out, ref)
    assert m["cosine"] >= COSINE_THRESHOLD, (
        f"cosine={m['cosine']:.6f} (need >= {COSINE_THRESHOLD})"
    )
    assert m["pcc"] >= PCC_THRESHOLD, f"pcc={m['pcc']:.6f} (need >= {PCC_THRESHOLD})"
    assert m["bit_exact"] > _EXACT_FLOOR, (
        f"rounding fidelity too low: {m['bit_exact']:.6f} bit-exact "
        f"(need > {_EXACT_FLOOR}); cosine={m['cosine']:.6f} rel={m['max_rel_err']:.3e}"
    )
    assert m["max_rel_err"] <= rel_ceiling, (
        f"max_rel_err={m['max_rel_err']:.6e} exceeds ceiling {rel_ceiling:.0e} "
        f"(cosine={m['cosine']:.6f} pcc={m['pcc']:.6f} bit-exact={m['bit_exact']:.6f})"
    )
    if full_contract:
        assert_correct(out, ref)  # FULL contract (max_rel_err <= 1e-2)
    return m


# --- MMA happy path: golden-validated on the Ada, path=mma noted -------------


def test_mma_matmul_square(mma_drive, rand_bf16, tmp_path):
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    out, log = mma_drive("mma", a, b, tmp_path)
    assert "path=mma" in log, f"report must note the mma path; got: {log!r}"
    m = _check_mma(out, G.matmul(a, b))
    print(f"\n[MMA 128x128x128] path=mma validated=true {m}")


def test_mma_matmul_rectangular(mma_drive, rand_bf16, tmp_path):
    a = rand_bf16((256, 128))  # M != N, K = 128 (tile-multiple)
    b = rand_bf16((128, 256))
    out, log = mma_drive("mma", a, b, tmp_path)
    assert "path=mma" in log
    m = _check_mma(out, G.matmul(a, b))
    print(f"\n[MMA 256x128x256] path=mma validated=true {m}")


def test_mma_matmul_non_square_multiple(mma_drive, rand_bf16, tmp_path):
    # M=192, K=160, N=224: all multiples of 16, non-square, non-power-of-two.
    a = rand_bf16((192, 160))
    b = rand_bf16((160, 224))
    out, log = mma_drive("mma", a, b, tmp_path)
    assert "path=mma" in log
    m = _check_mma(out, G.matmul(a, b))
    print(f"\n[MMA 192x160x224] path=mma validated=true {m}")


def test_mma_matmul_large_k_reduction_order(mma_drive, rand_bf16, tmp_path):
    # K=256: the MMA fp32 K-accumulation order differs from NumPy's, so an ISOLATED
    # element lands 1 bf16 ULP off at small magnitude while cosine == pcc == 1.0 and
    # >= 99.99% bit-exact. This is the documented large-K reduction-order class (see
    # _LARGE_K_REL_CEILING), NOT a bug, so it uses the calibrated ceiling instead of
    # the single-op max_rel_err <= 1e-2 gate — exactly mirroring test_wmma.py /
    # test_hip_run.py.
    a = rand_bf16((256, 256))
    b = rand_bf16((256, 256))
    out, log = mma_drive("mma", a, b, tmp_path)
    assert "path=mma" in log
    m = _check_mma(out, G.matmul(a, b), rel_ceiling=_LARGE_K_REL_CEILING,
                   full_contract=False)
    print(f"\n[MMA 256x256x256 large-K] path=mma validated=true {m}")


# --- Negative: a wrong-store-layout MMA variant FAILS, scalar STILL passes ----


def test_mma_negative_bad_store_excluded_scalar_stands(
    mma_drive, rand_bf16, tmp_path
):
    """An MMA variant with a deliberately WRONG store layout (mem_col_major ->
    transposed C) must FAIL golden (validated=false / excluded), while the UNCHANGED
    scalar baseline STILL PASSES on the identical inputs — proving MMA is safely
    ADDITIVE: a bad MMA variant is discarded by correctness-gating and never
    selected over the correct scalar.
    """
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    ref = G.matmul(a, b)

    # (1) The wrong-store-layout MMA variant FAILS golden, decisively.
    out_bad, log_bad = mma_drive("mma_bad", a, b, tmp_path)
    assert "path=mma_bad" in log_bad
    mb = _metrics(out_bad, ref)
    with pytest.raises(AssertionError):
        assert_correct(out_bad, ref)
    # The failure is a real correctness failure (not a flake): global metrics tank.
    assert mb["cosine"] < COSINE_THRESHOLD, (
        f"bad-store MMA should fail cosine; got {mb['cosine']:.6f}"
    )
    assert mb["pcc"] < PCC_THRESHOLD, (
        f"bad-store MMA should fail pcc; got {mb['pcc']:.6f}"
    )
    validated_bad = False  # correctness-gated: fails golden -> excluded

    # (2) The scalar baseline (kernels/generated/matmul.cu) STILL PASSES, same inputs.
    out_scalar, log_scalar = mma_drive("scalar", a, b, tmp_path)
    assert "path=scalar" in log_scalar
    assert_correct(out_scalar, ref)  # scalar baseline stands
    ms = _metrics(out_scalar, ref)
    validated_scalar = True

    print(
        f"\n[NEG] mma_bad  validated={validated_bad} {mb}\n"
        f"[NEG] scalar   validated={validated_scalar} {ms}\n"
        f"[NEG] bad MMA variant EXCLUDED (fails golden); scalar baseline STANDS "
        f"-> MMA is safely additive"
    )
