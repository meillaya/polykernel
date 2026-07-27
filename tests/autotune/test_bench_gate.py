"""Correctness-gated benchmarking harness tests (Todo 25 / Wave 5).

Asserts the CORE INVARIANT end-to-end on the gfx1101 GPU: the harness selects the
FASTEST VALIDATED variant (cache entry validated:true, a measured time_ms, golden
correctness recorded), and a deliberately-broken (fast-but-wrong) variant is
EXCLUDED - validated:false, never timed, never selected - with the rejection
logged. The gate, not the timing, decides the winner.

The C++ gate + selection logic is unit-tested GPU-free by
lib/Autotune/tests/benchmark_test.cpp; this exercises the full HIP pipeline
(hipcc driver -> golden diff -> C++ gate -> HIP-event timing -> contract-H cache).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BENCH_PY = Path(__file__).resolve().parents[2] / "tools" / "polykernel-bench" / "bench.py"


def _run_bench(tmp_path: Path, *extra: str) -> tuple[subprocess.CompletedProcess, dict]:
    """Drive the polykernel-bench CLI; return (proc, parsed tuning-cache doc)."""
    cache = tmp_path / "cache.json"
    cmd = [sys.executable, str(_BENCH_PY), "--autotune", *extra, "--cache-out", str(cache)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, f"bench failed (exit {proc.returncode}):\n{proc.stderr}"
    return proc, json.loads(cache.read_text())


def _validated_medians(stdout: str) -> list[float]:
    return [
        float(line.split("median_ms=")[1].split()[0])
        for line in stdout.splitlines()
        if "VALIDATED" in line and "median_ms=" in line
    ]


def test_selects_fastest_validated(bench_driver: Path, tmp_path: Path) -> None:
    # matmul at small K passes the FULL strict contract (max_rel_err <= 1e-2), so
    # the strict gate applies and every real variant validates.
    proc, doc = _run_bench(
        tmp_path, "--op", "matmul", "--shape", "128,128,128", "--dtype", "bf16",
        "--backend", "hip", "--arch", "gfx1101", "--variants", "3",
        "--warmup", "2", "--iters", "5",
    )

    assert doc["version"] == 1
    (entry,) = doc["entries"]
    # The winning cache entry is a measured, VALIDATED contract-H record.
    assert entry["gpu"] == "gfx1101"
    assert entry["op"] == "matmul"
    assert entry["shape"] == {"M": 128, "N": 128, "K": 128, "dtype": "bf16"}
    assert entry["scored_by"] == "measure"
    assert entry["validated"] is True
    assert isinstance(entry["time_ms"], (int, float)) and entry["time_ms"] > 0
    # Golden correctness is recorded with the entry (contract C held).
    corr = entry["correctness"]
    assert corr["cosine"] >= 0.999
    assert corr["pcc"] >= 0.99
    assert corr["max_rel_err"] <= 1e-2

    # The selected best is the FASTEST validated variant (min measured median).
    medians = _validated_medians(proc.stdout)
    assert len(medians) >= 1, proc.stdout
    assert entry["time_ms"] == min(medians)


def test_broken_variant_excluded(bench_driver: Path, tmp_path: Path) -> None:
    # Inject a fast-but-wrong variant (constant-fill kernel: trivially fast,
    # numerically garbage). The gate must EXCLUDE it and select a real kernel.
    proc, doc = _run_bench(
        tmp_path, "--op", "fused_matmul_bias_gelu", "--shape", "256,256,128",
        "--dtype", "bf16", "--backend", "hip", "--arch", "gfx1101", "--variants", "3",
        "--warmup", "2", "--iters", "5", "--inject-broken",
    )
    out = proc.stdout + proc.stderr

    # The broken variant is rejected + logged: validated:false, NOT timed, never best.
    broken_lines = [ln for ln in out.splitlines() if "broken_fast" in ln]
    assert broken_lines, out
    assert any("REJECTED" in ln for ln in broken_lines), out
    assert any("validated:false" in ln for ln in broken_lines), out
    assert "correctness gate FAILED" in out
    assert "NOT timed" in out

    # A REAL validated kernel won (validated:true + a measured time), and the
    # selected best is NOT the broken variant.
    (entry,) = doc["entries"]
    assert entry["op"] == "fused_matmul_bias_gelu"
    assert entry["validated"] is True
    assert isinstance(entry["time_ms"], (int, float)) and entry["time_ms"] > 0
    best_lines = [ln for ln in out.splitlines() if "BEST (fastest validated)" in ln]
    assert best_lines, out
    assert "broken_fast" not in best_lines[0], out
