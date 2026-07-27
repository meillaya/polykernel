#!/usr/bin/env python3
"""PolyKernel analytic memory-traffic report (Todo 17 / Wave 3).

Runs the pipeline UNFUSED vs FUSED via polykernel-opt and reports the global
memory-traffic reduction fusion achieves by eliminating intermediate global
round-trips. Per-op traffic parsing lives in mlir_traffic.py.

Fusion removes each absorbed intermediate's global ROUND-TRIP: the value the
producer wrote to global memory and the consumer read back now stays on-chip.
Saved bytes per eliminated intermediate = 2 * numel * bytes (one write + one read
removed). The eliminated intermediates are read from the discardable attrs the
fusion passes attach (``polykernel.fused_from`` + ``polykernel.eliminated_type[_N]``,
see lib/Passes/Fuse*.cpp); ``polykernel.workspace_bytes`` from --plan-memory
(Todo 16) is read as a cross-check (0 for a fused op).

Two independent views are computed and MUST agree (the tool fails otherwise):
  * before/after = sum of per-op traffic over the unfused vs fused IR;
  * eliminated   = sum of 2 * bytes(eliminated_type) over the fused ops.

Usage:
    polykernel-report --traffic examples/mlp_block.mlir
    python tools/polykernel-report/traffic_report.py --traffic examples/mlp_block.mlir
"""

# pyright: reportImplicitRelativeImport=false
# This tool is an executable SCRIPT (the `polykernel-report` wrapper puts this
# directory on sys.path; it is never imported as a package member), so the
# top-level `from mlir_traffic import ...` is correct and a relative import would
# break the documented `python tools/polykernel-report/traffic_report.py` usage.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict

from mlir_traffic import OpRecord, parse_ops, tensor_bytes

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FusionRecord(TypedDict):
    ssa: str
    op: str
    fused_from: str
    eliminated_types: list[str]
    saved_round_trip_bytes: int
    workspace_bytes: int | None


class Report(TypedDict):
    input: str
    unfused_ops: list[OpRecord]
    fused_ops: list[OpRecord]
    fusions: list[FusionRecord]
    before_bytes: int
    after_bytes: int
    reduction_bytes: int
    reduction_pct: float
    eliminated_round_trip_bytes: int
    consistent: bool
    problems: list[str]

# The four Wave-3 fusion passes (Todo 12/13), plus tile-layout + plan-memory so
# the report can cross-check polykernel.workspace_bytes (Todo 15/16).
_FUSION_PASSES = [
    "--fuse-rmsnorm-matmul", "--fuse-matmul-bias-gelu",
    "--fuse-residual-rmsnorm", "--fuse-softmax-mask",
    "--infer-tile-layout", "--plan-memory",
]


def find_opt(explicit: str | None) -> Path:
    """Locate polykernel-opt: --opt flag, $POLYKERNEL_OPT, then the build tree."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("POLYKERNEL_OPT"):
        candidates.append(Path(os.environ["POLYKERNEL_OPT"]))
    candidates.append(Path.cwd() / "build/tools/polykernel-opt/polykernel-opt")
    candidates.append(_PROJECT_ROOT / "build/tools/polykernel-opt/polykernel-opt")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "polykernel-opt not found; build it (nix develop -c cmake --build build "
        f"--target polykernel-opt) or pass --opt. Tried: {[str(c) for c in candidates]}"
    )


def run_opt(opt: Path, mlir: Path, passes: list[str]) -> str:
    """Run polykernel-opt on `mlir` with `passes`; return the printed IR (stdout)."""
    proc = subprocess.run(
        [str(opt), str(mlir), *passes], capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        raise RuntimeError(f"polykernel-opt failed (exit {proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def build_report(opt: Path, mlir: Path) -> Report:
    """Compute the full before/after traffic report for `mlir`."""
    unfused = parse_ops(run_opt(opt, mlir, []))
    fused = parse_ops(run_opt(opt, mlir, _FUSION_PASSES))

    before = sum(o["traffic_bytes"] for o in unfused)
    after = sum(o["traffic_bytes"] for o in fused)
    reduction = before - after

    fusions: list[FusionRecord] = []
    for o in fused:
        if not o["fused_from"]:
            continue
        fusions.append({
            "ssa": o["ssa"], "op": o["op"], "fused_from": o["fused_from"],
            "eliminated_types": o["eliminated_types"],
            "saved_round_trip_bytes": sum(2 * tensor_bytes(t) for t in o["eliminated_types"]),
            "workspace_bytes": o["workspace_bytes"],
        })
    eliminated_total = sum(f["saved_round_trip_bytes"] for f in fusions)

    # Cross-checks: the two views of the reduction must agree, and a fused op must
    # carry zero plan-memory workspace (its intermediates are fused away).
    problems = []
    if reduction != eliminated_total:
        problems.append(
            f"before-after reduction ({reduction}) != sum of eliminated round-trips "
            f"({eliminated_total})"
        )
    for f in fusions:
        if f["workspace_bytes"] not in (None, 0):
            problems.append(f"{f['op']} has nonzero workspace_bytes={f['workspace_bytes']}")

    return {
        "input": str(mlir),
        "unfused_ops": unfused,
        "fused_ops": fused,
        "fusions": fusions,
        "before_bytes": before,
        "after_bytes": after,
        "reduction_bytes": reduction,
        "reduction_pct": (100.0 * reduction / before) if before else 0.0,
        "eliminated_round_trip_bytes": eliminated_total,
        "consistent": not problems,
        "problems": problems,
    }


def _mib(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MiB"


def render_markdown(r: Report) -> str:
    """Render the human-readable before/after traffic report."""
    lines = [
        "# PolyKernel MLP fusion memory-traffic report (Todo 17 / Wave 3)",
        "",
        f"Input: `{r['input']}`",
        "",
        "Traffic is ANALYTIC: per-op global bytes read + written, derived from tensor",
        "shapes + dtype. Fusion removes each eliminated intermediate's global",
        "round-trip (one write + one read).",
        "",
        "## Headline",
        "",
        f"- **Before fusion:** {r['before_bytes']:,} bytes ({_mib(r['before_bytes'])})",
        f"- **After fusion:**  {r['after_bytes']:,} bytes ({_mib(r['after_bytes'])})",
        f"- **Reduction:**     {r['reduction_bytes']:,} bytes ({_mib(r['reduction_bytes'])})"
        f"  = {r['reduction_pct']:.2f}%",
        f"- **Consistent** (before-after == eliminated round-trips): {r['consistent']}",
        "",
        "## Per-fused-op breakdown",
        "",
    ]
    if not r["fusions"]:
        lines.append("_No fusion applied to this input._")
    for f in r["fusions"]:
        lines.append(
            f"- `{f['op']}` (fused_from=`{f['fused_from']}`) eliminates "
            f"{_mib(f['saved_round_trip_bytes'])} intermediate write+read "
            f"({f['saved_round_trip_bytes']:,} bytes) from "
            f"{', '.join(f['eliminated_types'])}"
        )
    lines += ["", "## Unfused per-op traffic", "",
              "| op | result | reads | writes | traffic |", "|---|---|---|---|---|"]
    for o in r["unfused_ops"]:
        lines.append(f"| {o['op']} | {o['ssa']} | {o['reads_bytes']:,} | "
                     f"{o['writes_bytes']:,} | {o['traffic_bytes']:,} |")
    lines += ["", "## Fused per-op traffic", "",
              "| op | result | reads | writes | traffic | workspace |",
              "|---|---|---|---|---|---|"]
    for o in r["fused_ops"]:
        ws = "n/a" if o["workspace_bytes"] is None else f"{o['workspace_bytes']:,}"
        lines.append(f"| {o['op']} | {o['ssa']} | {o['reads_bytes']:,} | "
                     f"{o['writes_bytes']:,} | {o['traffic_bytes']:,} | {ws} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polykernel-report",
        description="Analytic before/after fusion memory-traffic report.",
    )
    parser.add_argument("--traffic", metavar="MLIR", required=True,
                        help="MLIR file to report global memory traffic for")
    parser.add_argument("--opt", metavar="PATH", default=None,
                        help="path to polykernel-opt (auto-discovered by default)")
    parser.add_argument("--json-out", metavar="PATH",
                        default=str(_PROJECT_ROOT / "reports/mlp_traffic.json"))
    parser.add_argument("--md-out", metavar="PATH",
                        default=str(_PROJECT_ROOT / "reports/mlp_traffic.md"))
    parser.add_argument("--no-write", action="store_true",
                        help="print the report but do not write the output files")
    args = parser.parse_args(argv)

    report = build_report(find_opt(args.opt), Path(args.traffic))
    if not report["consistent"]:
        for p in report["problems"]:
            print(f"error: {p}", file=sys.stderr)
        return 2

    if not args.no_write:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        Path(args.md_out).write_text(render_markdown(report))

    print(render_markdown(report))
    print(f"reduction: {report['reduction_bytes']:,} bytes "
          f"({report['reduction_pct']:.2f}%) — fusion eliminates "
          f"{_mib(report['eliminated_round_trip_bytes'])} intermediate write+read")
    if not args.no_write:
        print(f"wrote {args.json_out} and {args.md_out}")
    return 0 if report["reduction_bytes"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
