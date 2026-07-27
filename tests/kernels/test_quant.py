"""Quantized inference: int8 weight-only + fp8 simulation (Todo 41 / Wave 8 elite).

WHERE the quantized matmul is proven correct against the QUANTIZATION-AWARE golden
(tests/golden/quant_golden.py — a NEW golden module; the shared golden.py/metrics.py
are untouched). Two paths:

  int8 WEIGHT-ONLY: weights stored as int8 with a per-output-channel scale[N],
  activations bf16, dequantize-and-multiply inside the kernel/CPU-ref:
      C[m,n] = bf16( sum_k fp32(A[m,k]) * (fp32(Wq[k,n]) * scale[n]) )
  Validated THREE ways: the CPU reference (kernels/cpu/matmul_quant.cpp `int8` mode)
  and the REAL GPU kernel (kernels/generated/matmul_int8.cu, run on the RX 7800 XT
  under hipcc) both match quant_golden.matmul_int8 within the RELAXED quant thresholds
  (cosine >= 0.99, max_rel_err <= 5e-2, pcc >= 0.99). The int8 kernel is a codegen
  VARIANT of the closed `polykernel.matmul` op (NO new op); `path=int8_weight_only`
  is a report annotation (contract H), exactly as `path=wmma` is for the WMMA variant.

  fp8 (e4m3/e5m2) SIMULATION: NO fp8 hardware. The weights are rounded onto the fp8
  grid (ml_dtypes.float8_e4m3fn / float8_e5m2 in the golden; a portable RNE round in
  the CPU ref `fp8sim` mode) and then matmul'd in bf16/fp32. The CPU ref matches
  quant_golden.matmul_fp8_sim within the relaxed thresholds (measured bit-exact for
  uniform [-1,1] weights). `path=fp8_sim`.

MECHANISM (REUSED, mirrors tests/kernels/test_wmma.py): session fixtures compile the
CPU ref with the host compiler and a standalone GPU launcher with hipcc (the thin
error-checked HipRuntime layer + the reused .npy bridge kernels/cpu/npy_io.cpp + the
int8 kernel, built with -DPOLYKERNEL_QUANT_MAIN to activate the test-only driver main
inside matmul_int8.cu). Everything builds into the TASK-LOCAL build_t41/ (never the
shared build/, which concurrent Wave-8 tasks hold). Each test drives through the fixed
-seed (conftest.SEED) bf16 .npy bridge and compares to the golden.

PATH ANNOTATION (contract H): the drivers print `[polykernel-quant] path=...` /
`[polykernel-quant-cpu] path=...` to stderr; the happy tests assert the path was noted.

NEGATIVE PATH (proves per-channel scale correctness is tested): a WRONG per-channel
scale makes the int8 output fail the quantized golden decisively -> the scale is not
cosmetic, it is verified. Evidence: reports/w8_quant_scale_neg.log.

If hipcc / a usable gfx1101 device is absent, the GPU tests SKIP-local (the CPU-ref
tests still run; the Todo 18 gate guards the GPU). The CUDA sm_80/sm_90 compile of
matmul_int8.cu is GPU-free (nvcc --ptx; see reports/w8_quant.log).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

import quant_golden as QG
from metrics import cosine, max_rel_err, pcc

_BF16 = ml_dtypes.bfloat16
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BUILD = _PROJECT_ROOT / "build_t41"
_CPU_EXE = _BUILD / "quant_cpu" / "matmul_quant"
_GPU_EXE = _BUILD / "quant_run" / "matmul_int8_run"
_ARCH = "gfx1101"

# int8 shapes: tile-multiple (clean) + a non-tile-multiple (kernel bounds checks).
_INT8_SHAPES = [(128, 128, 128), (256, 128, 256), (192, 160, 224), (130, 127, 65)]


# --- build fixtures (task-local build_t41/, never the shared build/) ---------


@pytest.fixture(scope="session")
def quant_cpu_exe() -> Path:
    """Compile the quantized CPU reference once per session; skip if no compiler."""
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no host C++ compiler on PATH (run inside `nix develop`)")
    cmd = [
        cxx, "-std=c++17", "-O2", "-Ikernels/cpu",
        "kernels/cpu/matmul_quant.cpp", "kernels/cpu/npy_io.cpp",
        "-o", str(_CPU_EXE), "-lm",
    ]
    _CPU_EXE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"c++ failed to build matmul_quant:\n{proc.stderr}")
    return _CPU_EXE


@pytest.fixture(scope="session")
def quant_gpu_exe() -> Path:
    """Compile the standalone int8 GPU launcher once per session; skip if no hipcc."""
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        pytest.skip("hipcc not found on PATH (run inside `nix develop`); GPU SKIPPED-local")
    cmd = [
        hipcc, f"--offload-arch={_ARCH}",
        "-DPOLYKERNEL_HIP", "-DPOLYKERNEL_QUANT_MAIN", "-O2",
        "-Ikernels/template", "-Iinclude", "-Ikernels/cpu",
        "lib/Runtime/HipRuntime.cpp", "kernels/cpu/npy_io.cpp",
        "kernels/generated/matmul_int8.cu",
        "-o", str(_GPU_EXE),
    ]
    _GPU_EXE.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          cwd=_PROJECT_ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"hipcc failed to build matmul_int8_run:\n{proc.stderr}")
    return _GPU_EXE


@pytest.fixture(scope="session")
def gpu_probe(quant_gpu_exe: Path) -> None:
    """Skip GPU tests if no usable HIP device (belt-and-braces on the T18 gate)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        a = Path(td) / "a.npy"
        wq = Path(td) / "wq.npy"
        sc = Path(td) / "s.npy"
        o = Path(td) / "o.npy"
        np.save(a, np.zeros((16, 16), dtype=_BF16))
        np.save(wq, np.zeros((16, 16), dtype=np.int8))
        np.save(sc, np.ones((16,), dtype=np.float32))
        pr = subprocess.run(
            [str(quant_gpu_exe), "int8", str(a), str(wq), str(sc), str(o)],
            capture_output=True, text=True, timeout=120,
        )
        err = pr.stderr.lower()
        if pr.returncode != 0 and (
            "no device" in err or "no hip gpus" in err or "invalid device" in err
        ):
            pytest.skip(f"no usable HIP device ({_ARCH}): {pr.stderr.strip()[:200]}")


# --- drive helpers (file-based .npy bridge, Pinned contract B) ---------------


def _drive(exe: Path, argv: list[str], inputs: dict[str, np.ndarray],
           out_name: str, tmp_path: Path) -> tuple[np.ndarray, str]:
    """Write `inputs` as .npy, run `exe argv` (substituting paths), load the bf16 out."""
    paths = {}
    for name, arr in inputs.items():
        p = tmp_path / f"{name}.npy"
        np.save(p, arr)
        paths[name] = str(p)
    out = tmp_path / out_name
    full = [str(exe)] + [arg.format(**paths, OUT=str(out)) for arg in argv]
    proc = subprocess.run(full, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise RuntimeError(f"{exe.name} failed (exit {proc.returncode}):\n{proc.stderr}")
    return np.load(out).view(_BF16), proc.stderr


def _metrics(out: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    return {
        "cosine": cosine(out, ref),
        "pcc": pcc(out, ref),
        "max_rel_err": max_rel_err(out, ref),
        "bit_exact": float(np.mean(out == ref)),
    }


# --- int8 weight-only: CPU reference vs quantization-aware golden ------------


@pytest.mark.parametrize("m,k,n", _INT8_SHAPES)
def test_int8_cpu_matches_golden(quant_cpu_exe, rand_bf16, tmp_path, m, k, n):
    a = rand_bf16((m, k))
    w = rand_bf16((k, n))
    wq, scale = QG.quantize_per_channel(w)
    ref = QG.matmul_int8(a, wq, scale)
    out, log = _drive(
        quant_cpu_exe,
        ["int8", "{a}", "{wq}", "{scale}", "{OUT}"],
        {"a": a, "wq": wq, "scale": scale}, "c.npy", tmp_path,
    )
    assert "path=int8_weight_only" in log, f"report must note the quant path: {log!r}"
    QG.assert_quant_correct(out, ref)
    print(f"\n[int8 CPU {m}x{k}x{n}] path=int8_weight_only {_metrics(out, ref)}")


# --- int8 weight-only: REAL GPU kernel (HIP, RX 7800 XT) vs golden -----------


@pytest.mark.parametrize("m,k,n", _INT8_SHAPES)
def test_int8_gpu_matches_golden(gpu_probe, quant_gpu_exe, rand_bf16, tmp_path,
                                 m, k, n):
    a = rand_bf16((m, k))
    w = rand_bf16((k, n))
    wq, scale = QG.quantize_per_channel(w)
    ref = QG.matmul_int8(a, wq, scale)
    out, log = _drive(
        quant_gpu_exe,
        ["int8", "{a}", "{wq}", "{scale}", "{OUT}"],
        {"a": a, "wq": wq, "scale": scale}, "c.npy", tmp_path,
    )
    assert "path=int8_weight_only" in log, f"report must note the quant path: {log!r}"
    QG.assert_quant_correct(out, ref)
    print(f"\n[int8 GPU {m}x{k}x{n}] path=int8_weight_only {_metrics(out, ref)}")


# --- fp8 (e4m3/e5m2) simulation: CPU reference vs golden ---------------------


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("m,k,n", [(128, 128, 128), (192, 160, 224)])
def test_fp8_sim_cpu_matches_golden(quant_cpu_exe, rand_bf16, tmp_path, fmt, m, k, n):
    a = rand_bf16((m, k))
    w = rand_bf16((k, n))
    ref = QG.matmul_fp8_sim(a, w, fmt)
    out, log = _drive(
        quant_cpu_exe,
        ["fp8sim", fmt, "{a}", "{w}", "{OUT}"],
        {"a": a, "w": w}, "c.npy", tmp_path,
    )
    assert "path=fp8_sim" in log, f"report must note the fp8-sim path: {log!r}"
    QG.assert_quant_correct(out, ref)
    print(f"\n[fp8_sim {fmt} {m}x{k}x{n}] path=fp8_sim {_metrics(out, ref)}")


# --- NEGATIVE: a wrong per-channel scale FAILS the quantized golden ----------


def test_int8_wrong_per_channel_scale_fails_golden(quant_cpu_exe, rand_bf16, tmp_path):
    """A WRONG per-channel scale must make the int8 output fail the quantized golden
    (proves the scale is verified, not cosmetic). The correct scale passes; a wrong
    per-channel scale (a 0.25x..4.0 ramp across the output channels) applies the wrong
    gain to each column -> the output diverges and assert_quant_correct fails on ALL
    THREE relaxed metrics (cosine, max_rel_err, pcc), decisively.
    """
    m, k, n = 128, 128, 128
    a = rand_bf16((m, k))
    w = rand_bf16((k, n))
    wq, scale = QG.quantize_per_channel(w)
    ref = QG.matmul_int8(a, wq, scale)

    # (1) Sanity: the CORRECT scale passes the quantized golden.
    out_ok, _ = _drive(
        quant_cpu_exe, ["int8", "{a}", "{wq}", "{scale}", "{OUT}"],
        {"a": a, "wq": wq, "scale": scale}, "c_ok.npy", tmp_path,
    )
    QG.assert_quant_correct(out_ok, ref)

    # (2) A WRONG per-channel scale (0.25x..4.0 ramp) FAILS the quantized golden.
    scale_wrong = (scale * np.linspace(0.25, 4.0, n).astype(np.float32)).astype(
        np.float32
    )
    assert not np.allclose(scale_wrong, scale), "corruption must actually differ"
    out_bad, _ = _drive(
        quant_cpu_exe, ["int8", "{a}", "{wq}", "{scale}", "{OUT}"],
        {"a": a, "wq": wq, "scale": scale_wrong}, "c_bad.npy", tmp_path,
    )
    mb = _metrics(out_bad, ref)
    with pytest.raises(AssertionError):
        QG.assert_quant_correct(out_bad, ref)
    # The failure is decisive on every relaxed metric (a wrong per-channel gain
    # scrambles the column magnitudes): cosine + pcc drop below threshold AND
    # max_rel_err blows far past 5e-2.
    assert mb["cosine"] < QG.QUANT_COSINE_THRESHOLD, (
        f"wrong scale should fail cosine; got {mb['cosine']:.6f}"
    )
    assert mb["max_rel_err"] > QG.QUANT_MAX_REL_ERR_THRESHOLD, (
        f"wrong scale should fail max_rel_err; got {mb['max_rel_err']:.3e}"
    )
    assert mb["pcc"] < QG.QUANT_PCC_THRESHOLD, (
        f"wrong scale should fail pcc; got {mb['pcc']:.6f}"
    )
    print(
        f"\n[NEG int8 wrong-scale] correct=PASS  wrong-scale validated=False {mb}\n"
        f"[NEG] wrong per-channel scale FAILS the quantized golden on cosine+rel+pcc "
        f"-> scale is verified"
    )
