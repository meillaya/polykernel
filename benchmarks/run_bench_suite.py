#!/usr/bin/env python3
"""PolyKernel RunPod benchmark suite orchestrator (Todo 28 / Wave 5).

Drives the H100/A100 (CUDA) + MI300 (HIP gfx942) benchmark of the MLP fragment:
unfused baseline (1.00x) vs fused generated vs best autotuned (bounded Todo 24
grid). Two paths, selected by RunPod credential detection:

  * CREDENTIALS PRESENT (opt-in rental): invoke benchmarks/rent_runpod.sh, which
    provisions the pods, builds the engine + bench in-container, runs the bounded
    autotuner + bench_cuda/bench_hip, enforces the spend ceiling, and writes REAL
    unfused/fused/autotuned speedups to reports/{h100,mi300}_bench.{json,md}.

  * CREDENTIALS ABSENT (this environment): the suite is SKIPPED (NOT FAILED), no
    pod is provisioned, no money is spent. The reports are populated with
    clearly-labeled perf-model PROJECTIONS derived from the Todo 11/27 compile-
    time analyzer's roofline model (lib/Analysis/Roofline.cpp) + the Todo 17
    analytic traffic (reports/mlp_traffic.json), and reports/w5_rent_skipped.log
    records the graceful skip. Exit code 0.

PROJECTION METHODOLOGY (reproducible; recomputed on every run from the constants
below - no hardcoded speedups):

  For each MLP-fragment op with (flops, bytes):
      arithmetic_intensity AI = flops / bytes
      ridge(arch)           = peak_flop_s / peak_byte_s        (Roofline.cpp)
      bound                 = compute-bound iff AI >= ridge else memory-bound
      t_roofline            = max(flops/peak_flop_s, bytes/peak_byte_s)
      t_attained            = t_roofline / eta(bound)          (eta = attainment)
  Fragment time = sum of per-op t_attained; speedup = t_unfused / t_variant.

  Variants: unfused = [rmsnorm, matmul, gelu, matmul, add]; fused = the same with
  rmsnorm folded into fused_rmsnorm_matmul (its 32 MiB intermediate round-trip
  removed - the Todo 17 elimination); autotuned = fused with the matmuls at the
  higher vectorized/wmma attainment the bounded grid selects.

  Documented attainment assumptions (the modeling knobs; clearly labeled in the
  reports): scalar 16x16x16 tiled GEMM eta_compute=0.12; best-of-grid vectorized
  /wmma eta_compute=0.35; large regular elementwise eta_memory=0.80. The fragment
  is compute-bound (the two K=4096/11008 matmuls, AI~1215 >> ridge), so fusion's
  traffic elimination yields a modest projected speedup (~1.003-1.004x) while the
  scalar->tuned attainment jump yields the headline ~2.8x.

Usage:
    python3 benchmarks/run_bench_suite.py            # auto: SKIPPED+projection here
    python3 benchmarks/run_bench_suite.py --reports-dir reports
"""

# allow: SIZE_OK - one cohesive orchestrator: roofline projection model + JSON/MD
# report rendering + RunPod credential detection + rental dispatch + skip-log. The
# bulk is the reproducible projection math + the two clearly-labeled report
# renderers; splitting would fragment the single "produce the bench report"
# responsibility (mirrors the accepted tools/polykernel-bench/bench.py SIZE_OK).

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRAFFIC_JSON = _PROJECT_ROOT / "reports" / "mlp_traffic.json"

# --- Arch peak perf (mirror lib/Analysis/Roofline.cpp PerfFor + MI300X public ---
# --- specs). peak_tflops = dense FP16 TFLOP/s; peak_bw_gbps = HBM GB/s.        ---
ARCHS: dict[str, dict] = {
    "h100": {"arch": "sm_90", "backend": "cuda", "peak_tflops": 989.0,
             "peak_bw_gbps": 3350.0, "label": "NVIDIA H100 (SXM5, sm_90)"},
    "a100": {"arch": "sm_80", "backend": "cuda", "peak_tflops": 312.0,
             "peak_bw_gbps": 1555.0, "label": "NVIDIA A100 (80GB, sm_80)"},
    "mi300": {"arch": "gfx942", "backend": "hip", "peak_tflops": 1307.4,
              "peak_bw_gbps": 5300.0, "label": "AMD Instinct MI300X (gfx942)"},
    # sm_89 peak = the todo-5 VERIFIED scalar-bf16 figure (91.1) + datasheet BW
    # (960). This row is a PROJECTION; todo 14's MEASURED section (measured:true
    # ada6000_bench.json) overrides it on the real RTX 6000 Ada.
    "ada6000": {"arch": "sm_89", "backend": "cuda", "peak_tflops": 91.1,
                "peak_bw_gbps": 960.0, "label": "NVIDIA RTX 6000 Ada (sm_89)"},
}
# reports/{h100,mi300,ada6000}_bench: the headline NVIDIA report carries H100 +
# A100; ada6000 is its own report (PROJECTED here, MEASURED after todo 14).
REPORT_ARCHS = {"h100": ["h100", "a100"], "mi300": ["mi300"],
                "ada6000": ["ada6000"]}

# --- Documented attainment assumptions (the projection's modeling knobs).      ---
ETA_COMPUTE_BASE = 0.12   # scalar 16x16x16 tiled GEMM, no tensor cores.
ETA_COMPUTE_TUNED = 0.35  # best of bounded grid: vectorized/wmma (vector_width>=4).
ETA_MEMORY = 0.80         # large regular elementwise strides -> ~80% of HBM BW.

# --- MLP fragment (examples/mlp_block.mlir, bf16, M=2048). Bytes are the Todo  ---
# --- 17 analytic global traffic; flops the exact GEMM counts (elementwise flops ---
# --- are negligible: those ops are memory-bound, so only their bytes bind).     ---
_MM_UP_FLOPS = 2 * 2048 * 11008 * 4096   # A[2048,4096] @ B[4096,11008]
_MM_DN_FLOPS = 2 * 2048 * 4096 * 11008   # A[2048,11008] @ B[11008,4096]


@dataclass(frozen=True)
class Op:
    name: str
    flops: int
    bytes: int


# Unfused primitive chain; fused folds rmsnorm into the up-projection matmul.
MLP_UNFUSED = (
    Op("rmsnorm", 25_000_000, 33_554_432),
    Op("matmul_up", _MM_UP_FLOPS, 152_043_520),
    Op("gelu", 200_000_000, 90_177_536),
    Op("matmul_dn", _MM_DN_FLOPS, 152_043_520),
    Op("add_residual", 8_400_000, 50_331_648),
)
MLP_FUSED = (
    Op("fused_rmsnorm_matmul_up", _MM_UP_FLOPS, 152_043_520),
    Op("gelu", 200_000_000, 90_177_536),
    Op("matmul_dn", _MM_DN_FLOPS, 152_043_520),
    Op("add_residual", 8_400_000, 50_331_648),
)


def ridge_flop_per_byte(arch: dict) -> float:
    return (arch["peak_tflops"] * 1e12) / (arch["peak_bw_gbps"] * 1e9)


def op_time_s(op: Op, arch: dict, eta_compute: float) -> tuple[float, str]:
    """Roofline lower bound / attainment -> projected seconds + classification."""
    peak_flop = arch["peak_tflops"] * 1e12
    peak_bw = arch["peak_bw_gbps"] * 1e9
    ai = op.flops / op.bytes if op.bytes else 0.0
    bound = "compute-bound" if ai >= ridge_flop_per_byte(arch) else "memory-bound"
    t_roofline = max(op.flops / peak_flop, op.bytes / peak_bw)
    eta = eta_compute if bound == "compute-bound" else ETA_MEMORY
    return t_roofline / eta, bound


def fragment_time_s(ops: tuple[Op, ...], arch: dict, eta_compute: float) -> float:
    return sum(op_time_s(o, arch, eta_compute)[0] for o in ops)


def project_arch(arch_key: str) -> dict:
    """Project unfused/fused/autotuned for one arch (speedups vs unfused=1.0)."""
    arch = ARCHS[arch_key]
    t_unfused = fragment_time_s(MLP_UNFUSED, arch, ETA_COMPUTE_BASE)
    t_fused = fragment_time_s(MLP_FUSED, arch, ETA_COMPUTE_BASE)
    t_tuned = fragment_time_s(MLP_FUSED, arch, ETA_COMPUTE_TUNED)

    def per_op(ops: tuple[Op, ...], eta: float) -> list[dict]:
        out = []
        for o in ops:
            t, bound = op_time_s(o, arch, eta)
            out.append({"op": o.name, "flops": o.flops, "bytes": o.bytes,
                        "ai_flop_per_byte": round(o.flops / o.bytes, 2),
                        "bound": bound, "projected_ms": round(t * 1e3, 5)})
        return out

    return {
        "arch": arch["arch"], "backend": arch["backend"], "label": arch["label"],
        "peak_tflops": arch["peak_tflops"], "peak_bw_gbps": arch["peak_bw_gbps"],
        "ridge_flop_per_byte": round(ridge_flop_per_byte(arch), 2),
        "projected_total_ms": {
            "unfused": round(t_unfused * 1e3, 5),
            "fused": round(t_fused * 1e3, 5),
            "autotuned": round(t_tuned * 1e3, 5),
        },
        "speedup": {
            "unfused": 1.0,
            "fused": round(t_unfused / t_fused, 4),
            "autotuned": round(t_unfused / t_tuned, 4),
        },
        "per_op_unfused": per_op(MLP_UNFUSED, ETA_COMPUTE_BASE),
        "per_op_fused": per_op(MLP_FUSED, ETA_COMPUTE_BASE),
    }


def cross_check_traffic() -> dict:
    """Tie the projection's traffic to the authoritative Todo 17 figures."""
    if not _TRAFFIC_JSON.exists():
        return {"source": "embedded canonical (reports/mlp_traffic.json absent)"}
    t = json.loads(_TRAFFIC_JSON.read_text())
    return {
        "source": "reports/mlp_traffic.json (Todo 17)",
        "before_bytes": t.get("before_bytes"),
        "after_bytes": t.get("after_bytes"),
        "reduction_bytes": t.get("reduction_bytes"),
        "reduction_pct": round(t.get("reduction_pct", 0.0), 3),
        "eliminated_round_trip_bytes": t.get("eliminated_round_trip_bytes"),
    }


def build_report(report_key: str) -> dict:
    arch_keys = REPORT_ARCHS[report_key]
    return {
        "status": "PROJECTED",
        "projected": True,
        "measured": False,
        "banner": ("PROJECTED (no rental; run benchmarks/rent_runpod.sh with "
                   "RUNPOD_API_KEY set for real measured numbers)"),
        "task": "T28 (Wave 5): real benchmark on rented GPUs (OWNER-GATED: RunPod)",
        "fragment": "MLP block (examples/mlp_block.mlir): rmsnorm+matmul+gelu+matmul+add",
        "dtype": "bf16", "shape": {"M": 2048, "N_up": 11008, "K": 4096, "N_dn": 4096},
        "generated_by": "benchmarks/run_bench_suite.py (roofline + analytic traffic)",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": {
            "model": "GEMM roofline (lib/Analysis/Roofline.cpp) + Todo 17 analytic traffic",
            "equation": ("t = max(flops/peak_flop, bytes/peak_bw) / eta; "
                         "speedup = t_unfused / t_variant"),
            "attainment_assumptions": {
                "eta_compute_scalar_baseline": ETA_COMPUTE_BASE,
                "eta_compute_autotuned_vectorized_wmma": ETA_COMPUTE_TUNED,
                "eta_memory_elementwise": ETA_MEMORY,
            },
            "traffic": cross_check_traffic(),
        },
        "variants": {
            "unfused": "rmsnorm + matmul + gelu + matmul + add (separate primitives)",
            "fused": "fused_rmsnorm_matmul + gelu + matmul + add (32 MiB round-trip removed)",
            "autotuned": "fused, matmuls at the bounded-grid vectorized/wmma attainment",
        },
        "archs": {k: project_arch(k) for k in arch_keys},
        "caveats": [
            "PROJECTED, not measured: derived from the roofline model + analytic "
            "traffic, not from a rented GPU run.",
            "The fragment is compute-bound (the two matmuls, AI~1215 >> ridge), so "
            "fusion's traffic elimination projects a modest speedup; the headline "
            "gain is the scalar->autotuned attainment jump.",
            "The generated fused kernel recomputes the per-row RMS in every output "
            "tile, so the REAL fusion speedup is shape-dependent (verified on the "
            "local gfx1101: fused can be slower at small, compute-light shapes).",
            "Attainment factors are documented modeling assumptions, not measured "
            "efficiencies; real numbers require the opt-in RunPod rental.",
        ],
        "opt_in": ("RUNPOD_API_KEY=... python3 benchmarks/run_bench_suite.py   # or "
                   "./benchmarks/rent_runpod.sh --suite mlp"),
    }


def render_md(report: dict) -> str:
    head = report["archs"]
    lines = [
        f"# PolyKernel {report['fragment']} benchmark — {list(head)[0].upper()}",
        "",
        "> ## ⚠ PROJECTED — NOT MEASURED",
        f"> {report['banner']}.",
        f"> Numbers are perf-model projections (roofline + analytic traffic), not a "
        f"rented-GPU run. No pod was provisioned; no money was spent.",
        "",
        f"- **Task:** {report['task']}",
        f"- **Fragment:** `{report['fragment']}`",
        f"- **dtype / shape:** {report['dtype']}, M={report['shape']['M']}, "
        f"K={report['shape']['K']}, N_up={report['shape']['N_up']}, "
        f"N_dn={report['shape']['N_dn']}",
        f"- **Generated:** {report['generated_utc']} by `{report['generated_by']}`",
        "",
        "## Projected speedup (unfused = 1.00x baseline)",
        "",
        "| arch | backend | unfused | fused | autotuned |",
        "|---|---|---|---|---|",
    ]
    for key, a in head.items():
        sp = a["speedup"]
        lines.append(
            f"| {a['label']} | {a['backend']} ({a['arch']}) | "
            f"{sp['unfused']:.3f}x | {sp['fused']:.3f}x | {sp['autotuned']:.3f}x |")
    lines += ["", "## Projected fragment wall time (ms)", "",
              "| arch | unfused | fused | autotuned | ridge (flop/byte) |",
              "|---|---|---|---|---|"]
    for key, a in head.items():
        t = a["projected_total_ms"]
        lines.append(f"| {a['label']} | {t['unfused']:.4f} | {t['fused']:.4f} | "
                     f"{t['autotuned']:.4f} | {a['ridge_flop_per_byte']:.1f} |")
    lines += ["", "## Methodology (reproducible)", "",
              f"- **Model:** {report['method']['model']}",
              f"- **Equation:** `{report['method']['equation']}`",
              "- **Attainment assumptions (modeling knobs):** "
              f"scalar GEMM η={ETA_COMPUTE_BASE}, autotuned vectorized/wmma "
              f"η={ETA_COMPUTE_TUNED}, elementwise η={ETA_MEMORY}",
              f"- **Traffic source:** {report['method']['traffic'].get('source')}",
              ""]
    traf = report["method"]["traffic"]
    if "reduction_bytes" in traf:
        lines += [f"  - Todo 17 fusion eliminates {traf['eliminated_round_trip_bytes']:,} "
                  f"bytes ({traf['reduction_pct']}% of fragment traffic).", ""]
    # Per-op roofline for the headline arch.
    headline = next(iter(head.values()))
    lines += [f"## Per-op roofline ({headline['label']}, unfused)", "",
              "| op | flops | bytes | AI (flop/byte) | bound | projected ms |",
              "|---|---|---|---|---|---|"]
    for o in headline["per_op_unfused"]:
        lines.append(f"| {o['op']} | {o['flops']:,} | {o['bytes']:,} | "
                     f"{o['ai_flop_per_byte']:.2f} | {o['bound']} | {o['projected_ms']:.5f} |")
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in report["caveats"]]
    lines += ["", "## Opt-in real measurement", "",
              "```bash",
              "export RUNPOD_API_KEY=<your-key>   # owner-gated; ~$50-100 ceiling",
              "./benchmarks/rent_runpod.sh --suite mlp",
              "```",
              ""]
    return "\n".join(lines)


def detect_runpod_credentials() -> list[str]:
    """Return the names of present RunPod credentials (empty => SKIPPED path)."""
    present = []
    if os.environ.get("RUNPOD_API_KEY", "").strip():
        present.append("RUNPOD_API_KEY")
    if os.environ.get("RUNPOD_API_URL", "").strip():
        present.append("RUNPOD_API_URL")
    if (Path.home() / ".runpod").exists():
        present.append("~/.runpod")
    return present


def write_reports(reports_dir: Path) -> list[Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in REPORT_ARCHS:
        report = build_report(key)
        j = reports_dir / f"{key}_bench.json"
        m = reports_dir / f"{key}_bench.md"
        j.write_text(json.dumps(report, indent=2) + "\n")
        m.write_text(render_md(report))
        written += [j, m]
    return written


def write_skip_log(reports_dir: Path, missing: str, written: list[Path]) -> Path:
    log = reports_dir / "w5_rent_skipped.log"
    summary = build_report("h100")["archs"]["h100"]["speedup"]
    lines = [
        "=" * 70,
        "PolyKernel T28 (Wave 5): RunPod benchmark suite — SKIPPED (no credentials)",
        "=" * 70,
        f"UTC: {_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        f"STATUS: SKIPPED (not FAILED). Missing credential: {missing}",
        "RunPod credential detection found NONE of: RUNPOD_API_KEY, RUNPOD_API_URL, "
        "~/.runpod.",
        "",
        "ACTION TAKEN: graceful skip. NO pod provisioned, NO RunPod API create/start "
        "call, NO spend.",
        "The bench SUITE code is written (benchmarks/{bench_cuda.cpp,bench_hip.cpp,"
        "rent_runpod.sh});",
        "the rental is OPT-IN and gated behind the absent credential.",
        "",
        "PROJECTED reports written (clearly labeled PROJECTED, roofline-derived):",
    ]
    lines += [f"  - {p.relative_to(_PROJECT_ROOT)}" for p in written]
    lines += [
        "",
        "Projected MLP-fragment speedup (H100, unfused=1.00x baseline): "
        f"unfused={summary['unfused']:.3f}x  fused={summary['fused']:.3f}x  "
        f"autotuned={summary['autotuned']:.3f}x",
        "(PROJECTED — not measured; see reports/h100_bench.md + mi300_bench.md.)",
        "",
        "OPT-IN: to obtain REAL numbers, set the credential and re-run:",
        "    export RUNPOD_API_KEY=<your-key>     # ~$50-100 ceiling enforced",
        "    ./benchmarks/rent_runpod.sh --suite mlp",
        "  or: RUNPOD_API_KEY=<key> python3 benchmarks/run_bench_suite.py",
        "",
        "VERDICT: SKIPPED — graceful degradation, no crash, no surprise spend.",
    ]
    log.write_text("\n".join(lines) + "\n")
    return log


def run_rental(reports_dir: Path, creds: list[str]) -> int:
    """Credentials present: delegate to the opt-in rental orchestration script."""
    script = _PROJECT_ROOT / "benchmarks" / "rent_runpod.sh"
    print(f"[bench-suite] RunPod credentials present ({', '.join(creds)}); "
          f"delegating to the opt-in rental: {script}", file=sys.stderr)
    proc = subprocess.run(["bash", str(script), "--suite", "mlp",
                           "--reports-dir", str(reports_dir)], cwd=_PROJECT_ROOT)
    return proc.returncode


def main() -> int:
    reports_dir = _PROJECT_ROOT / "reports"
    for i, a in enumerate(sys.argv):
        if a == "--reports-dir" and i + 1 < len(sys.argv):
            reports_dir = Path(sys.argv[i + 1])

    creds = detect_runpod_credentials()
    if creds:
        return run_rental(reports_dir, creds)

    # SKIPPED path: no credentials -> projections + skip log, exit 0 (not FAILED).
    missing = "RUNPOD_API_KEY"
    written = write_reports(reports_dir)
    log = write_skip_log(reports_dir, missing, written)

    print("SKIPPED: RUNPOD_API_KEY not set — opt-in rental documented (no spend).")
    print(f"[bench-suite] RunPod credentials ABSENT (checked RUNPOD_API_KEY, "
          f"RUNPOD_API_URL, ~/.runpod).")
    print(f"[bench-suite] STATUS=SKIPPED (not FAILED); no pod provisioned, no "
          f"RunPod API call, no spend.")
    print(f"[bench-suite] PROJECTED reports written (roofline + Todo 17 traffic):")
    for p in written:
        print(f"  - {p.relative_to(_PROJECT_ROOT)}")
    print(f"[bench-suite] skip evidence: {log.relative_to(_PROJECT_ROOT)}")
    sp = build_report("h100")["archs"]["h100"]["speedup"]
    print(f"[bench-suite] H100 projected speedup (unfused=1.00x): "
          f"fused={sp['fused']:.3f}x autotuned={sp['autotuned']:.3f}x  [PROJECTED]")
    print(f"[bench-suite] OPT-IN real numbers: export RUNPOD_API_KEY=<key> && "
          f"./benchmarks/rent_runpod.sh --suite mlp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
