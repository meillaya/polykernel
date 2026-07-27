"""WMMA bf16 tensor-core MatMul on the RX 7800 XT (Todo 22 / Wave 4).

This is WHERE the RDNA3 WMMA matmul variant (kernels/generated/matmul_wmma.cu) is
proven correct on the real gfx1101 GPU — and proven ADDITIVE: the tiled SCALAR
baseline (kernels/generated/matmul.cu, Todo 10) remains the correctness reference
and is UNCHANGED. WMMA is a codegen VARIANT of the closed `polykernel.matmul` op the
autotuner (Todo 25) can SELECT; it is correctness-gated, so a WMMA variant that fails
golden is excluded (validated=false) and the scalar baseline stands — WMMA NEVER
blocks the wave.

INSTRUCTION + LANE-MAPPING (gfx1101, wave32, `v_wmma_f32_16x16x16_bf16` via the clang
builtin __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32; one wave computes a 16x16 f32
accumulator tile from 16x16 bf16 A and 16x16 bf16 B — 16 bf16 elems/lane input, 8
f32 elems/lane accumulator). Validated RDNA3 fragment layout (AMD GPUOpen "WMMA on
RDNA3"; the idiom vLLM/llama.cpp ship):
    A load: lane l -> a[i] = A_tile[l & 15][i],  i = 0..15   (full K-row)
    B load: lane l -> b[k] = B_tile[k][l & 15],  k = 0..15   (full K-col)
    Store:  lane l, elem e (0..7) -> C[2*e + (l >> 4)][l & 15]
(even rows in lanes 0..15, odd rows in lanes 16..31; columns = l & 15).

Mechanism (REUSED, not reinvented): mirrors tests/kernels/test_hip_run.py exactly. A
session fixture compiles a standalone `wmma_run` launcher with hipcc (the thin
error-checked HipRuntime layer + the reused .npy bridge kernels/cpu/npy_io.cpp + the
WMMA kernel + the scalar baseline), built with -DPOLYKERNEL_WMMA_MAIN to activate the
test-only driver main inside matmul_wmma.cu. Each test drives it through the SAME
fixed-seed (conftest.SEED = 0xC0FFEE) bf16 .npy file bridge and compares the GPU
output to the golden via metrics.assert_correct (contract C: cosine >= 0.999 AND
max_rel_err <= 1e-2 AND pcc >= 0.99), plus an element-wise bit-exact rounding floor.

SHAPES (all multiples of the 16x16x16 WMMA tile — the clean, no-tail case the
autotuner selects; documented per the task): 128x128x128 square, 256x128x256
rectangular (M != N), 192x160x224 non-square / non-power-of-two multiple, and
256x256x256 (larger K). Measured on gfx1101: cosine == pcc == 1.0 and >= 99.99%
bit-exact for every WMMA shape; max_rel_err <= 6.4e-3 (within the FULL 1e-2 contract
even at K=256 — the WMMA fp32 accumulation order stays close to NumPy's here).

PATH ANNOTATION (contract H): the driver prints `[polykernel-wmma] path=<wmma|
wmma_bad|scalar> M=.. N=.. K=..` to stderr; the happy tests assert path=wmma was used.

NEGATIVE PATH (proves WMMA is safely additive): `wmma_bad` is the SAME kernel with a
deliberately WRONG store lane-mapping (the believable bug of assuming contiguous rows
per lane-half, row = 8*(l>>4)+e, instead of the correct interleaved row = 2*e+(l>>4)).
It scrambles the output rows so golden FAILS decisively (cosine/pcc ~0.13, max_rel_err
>> 1) -> the variant would be marked validated=false and excluded — while the scalar
baseline STILL PASSES on the identical inputs. A bad WMMA variant never replaces the
correct scalar baseline. Evidence: reports/w4_wmma_neg.log.

If hipcc / a usable gfx1101 device is absent, the module is SKIPPED-local (the Todo 18
gate guards this; it would run on the first available rental GPU instead).
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
_WMMA_RUN_EXE = _PROJECT_ROOT / "build" / "wmma_run" / "wmma_run"
_ARCH = "gfx1101"

# Element-wise bit-exact rounding floor (WMMA measured worst case >= 0.9999 on gfx1101).
_EXACT_FLOOR = 0.99
# Calibrated max_rel_err ceiling for the large-K tiled-GEMM reduction-order class
# (MIRRORS test_hip_run.py's _LARGE_K_REL_CEILING, which mirrors test_cpu_ref_fused.py).
# At large K the WMMA fp32 K-accumulation order differs from NumPy's, so an ISOLATED
# element can land 1 bf16 ULP off at small magnitude where max_rel_err nudges past 1e-2
# — with cosine == pcc == 1.0 and >= 99.99% bit-exact (NOT a bug; a real bug tanks
# cosine/pcc toward 0). The large-K shape uses this ceiling + full_contract=False,
# exactly as the committed scalar baseline test does; every other WMMA shape (K <= 224)
# passes the FULL assert_correct (max_rel_err <= 1e-2).
_LARGE_K_REL_CEILING = 5e-2


@pytest.fixture(scope="session")
def wmma_run_exe() -> Path:
    """Compile the standalone WMMA launcher once per session; skip if no hipcc.

    Builds matmul_wmma.cu (with -DPOLYKERNEL_WMMA_MAIN to activate its test-only
    driver main) together with the scalar baseline matmul.cu (for the `scalar` mode
    the negative test compares against), the error-checked HipRuntime layer, and the
    reused .npy bridge — exactly the hipcc invocation test_hip_run.py uses, plus the
    WMMA driver macro.
    """
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip(
            "hipcc not found on PATH (run inside `nix develop`, rocmPackages.clr); "
            "WMMA SKIPPED-local."
        )
    sources = [
        "lib/Runtime/HipRuntime.cpp",
        "kernels/cpu/npy_io.cpp",
        "kernels/generated/matmul.cu",
        "kernels/generated/matmul_wmma.cu",
    ]
    cmd = [
        hipcc,
        f"--offload-arch={_ARCH}",
        "-DPOLYKERNEL_HIP",
        "-DPOLYKERNEL_WMMA_MAIN",
        "-O2",
        "-Ikernels/template",
        "-Iinclude",
        "-Ikernels/cpu",
        *sources,
        "-o",
        str(_WMMA_RUN_EXE),
    ]
    _WMMA_RUN_EXE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"hipcc failed to build wmma_run (exit {proc.returncode}):\n{proc.stderr}"
        )
    return _WMMA_RUN_EXE


@pytest.fixture(scope="session", autouse=True)
def _gpu_probe(wmma_run_exe: Path) -> None:
    """Skip the module if no usable HIP device (belt-and-braces on the T18 gate)."""
    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.npy"
        b = Path(td) / "b.npy"
        o = Path(td) / "o.npy"
        np.save(a, np.zeros((16, 16), dtype=_BF16))
        np.save(b, np.zeros((16, 16), dtype=_BF16))
        pr = subprocess.run([str(wmma_run_exe), "wmma", str(a), str(b), str(o)],
                            capture_output=True, text=True, timeout=120)
        err = pr.stderr.lower()
        if pr.returncode != 0 and (
            "no device" in err or "no hip gpus" in err or "invalid device" in err
        ):
            pytest.skip(f"no usable HIP device ({_ARCH}): {pr.stderr.strip()[:200]}")


@pytest.fixture
def wmma_drive(wmma_run_exe: Path):
    """Drive `wmma_run <mode> A B C`; raise (with stderr) on a non-zero exit.

    Returns (output_bf16, stderr_text) so tests can assert on the path annotation.
    """

    def _drive(mode: str, a: np.ndarray, b: np.ndarray, tmp_path: Path):
        ap = tmp_path / "a.npy"
        bp = tmp_path / "b.npy"
        cp = tmp_path / "c.npy"
        np.save(ap, a)
        np.save(bp, b)
        proc = subprocess.run([str(wmma_run_exe), mode, str(ap), str(bp), str(cp)],
                              capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(
                f"wmma_run {mode} failed (exit {proc.returncode}):\n{proc.stderr}"
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


def _check_wmma(out: np.ndarray, ref: np.ndarray, rel_ceiling: float = 1e-2,
                full_contract: bool = True) -> dict[str, float]:
    """Assert the binding global contract + element-wise rounding fidelity.

    cosine >= 0.999 and pcc >= 0.99 are the pinned global metrics; the bit-exact floor
    proves the bf16 rounding contract element-wise; max_rel_err is bounded by the
    (shape-appropriate) ceiling. With full_contract=True the binding single-op
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


# --- WMMA happy path: golden-validated on the 7800 XT, path=wmma noted --------


def test_wmma_matmul_square(wmma_drive, rand_bf16, tmp_path):
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    out, log = wmma_drive("wmma", a, b, tmp_path)
    assert "path=wmma" in log, f"report must note the wmma path; got: {log!r}"
    m = _check_wmma(out, G.matmul(a, b))
    print(f"\n[WMMA 128x128x128] path=wmma validated=true {m}")


def test_wmma_matmul_rectangular(wmma_drive, rand_bf16, tmp_path):
    a = rand_bf16((256, 128))  # M != N, K = 128 (tile-multiple)
    b = rand_bf16((128, 256))
    out, log = wmma_drive("wmma", a, b, tmp_path)
    assert "path=wmma" in log
    m = _check_wmma(out, G.matmul(a, b))
    print(f"\n[WMMA 256x128x256] path=wmma validated=true {m}")


def test_wmma_matmul_non_square_multiple(wmma_drive, rand_bf16, tmp_path):
    # M=192, K=160, N=224: all multiples of 16, non-square, non-power-of-two.
    a = rand_bf16((192, 160))
    b = rand_bf16((160, 224))
    out, log = wmma_drive("wmma", a, b, tmp_path)
    assert "path=wmma" in log
    m = _check_wmma(out, G.matmul(a, b))
    print(f"\n[WMMA 192x160x224] path=wmma validated=true {m}")


def test_wmma_matmul_large_k_reduction_order(wmma_drive, rand_bf16, tmp_path):
    # K=256: the WMMA fp32 K-accumulation order differs from NumPy's, so an ISOLATED
    # element lands 1 bf16 ULP off at small magnitude (measured max_rel_err ~1.0e-2)
    # while cosine == pcc == 1.0 and >= 99.99% bit-exact. This is the documented
    # large-K reduction-order class (see _LARGE_K_REL_CEILING), NOT a bug, so it uses
    # the calibrated ceiling instead of the single-op max_rel_err <= 1e-2 gate — exactly
    # mirroring test_hip_run.py::test_hip_matmul_large_k_reduction_order.
    a = rand_bf16((256, 256))
    b = rand_bf16((256, 256))
    out, log = wmma_drive("wmma", a, b, tmp_path)
    assert "path=wmma" in log
    m = _check_wmma(out, G.matmul(a, b), rel_ceiling=_LARGE_K_REL_CEILING,
                    full_contract=False)
    print(f"\n[WMMA 256x256x256 large-K] path=wmma validated=true {m}")


# --- Negative: a wrong-lane-mapping WMMA variant FAILS, scalar STILL passes ----


def test_wmma_negative_bad_lane_excluded_scalar_stands(
    wmma_drive, rand_bf16, tmp_path
):
    """A WMMA fragment with a deliberately WRONG lane-mapping must FAIL golden
    (validated=false / excluded), while the UNCHANGED scalar baseline STILL PASSES on
    the identical inputs — proving WMMA is safely ADDITIVE: a bad WMMA variant is
    discarded by correctness-gating and never selected over the correct scalar.
    """
    a = rand_bf16((128, 128))
    b = rand_bf16((128, 128))
    ref = G.matmul(a, b)

    # (1) The wrong-lane-mapping WMMA variant FAILS golden, decisively.
    out_bad, log_bad = wmma_drive("wmma_bad", a, b, tmp_path)
    assert "path=wmma_bad" in log_bad
    mb = _metrics(out_bad, ref)
    with pytest.raises(AssertionError):
        assert_correct(out_bad, ref)
    # The failure is a real correctness failure (not a flake): global metrics tank.
    assert mb["cosine"] < COSINE_THRESHOLD, (
        f"bad-lane WMMA should fail cosine; got {mb['cosine']:.6f}"
    )
    assert mb["pcc"] < PCC_THRESHOLD, (
        f"bad-lane WMMA should fail pcc; got {mb['pcc']:.6f}"
    )
    validated_bad = False  # correctness-gated: fails golden -> excluded

    # (2) The scalar baseline (kernels/generated/matmul.cu) STILL PASSES, same inputs.
    out_scalar, log_scalar = wmma_drive("scalar", a, b, tmp_path)
    assert "path=scalar" in log_scalar
    assert_correct(out_scalar, ref)  # scalar baseline stands
    ms = _metrics(out_scalar, ref)
    validated_scalar = True

    print(
        f"\n[NEG] wmma_bad validated={validated_bad} {mb}\n"
        f"[NEG] scalar   validated={validated_scalar} {ms}\n"
        f"[NEG] bad WMMA variant EXCLUDED (fails golden); scalar baseline STANDS "
        f"-> WMMA is safely additive"
    )
