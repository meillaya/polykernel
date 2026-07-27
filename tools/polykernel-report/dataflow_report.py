#!/usr/bin/env python3
"""PolyKernel dataflow simulator metrics report (Todo 38 / Wave 7).

The `--backend=dataflow` report: drives the dataflow simulator's metrics model on
an MLP fragment's matmul (lowered onto the Todo 37 SUMMA schedule) and emits the
Cerebras-style dataflow metrics to JSON - grid utilization (active PEs / total),
local SRAM pressure (resident tiles + double-buffers vs 48 KB), messages sent
(total wavelets), average hop distance (1 cycle/hop; broadcast width P = P-1),
communication bottleneck (active/delayed/backpressure/idle), critical-path
cycles, and fusion traffic reduction (wavelet count unfused vs fused: the fused
MLP keeps the eliminated intermediate in local SRAM -> zero fabric wavelets).

The metrics are the simulator's analytic/cycle model - the same formulas the C++
metrics layer (lib/DataflowSim/Metrics.cpp) computes and metrics_test.cpp
validates against the cycle-accurate RunSumma simulation with hand-computed 4x4
values. The real-SDK analogs MODELLED are `cslc --out-routes` + `calculate_cycles`.

The MLP matmul is non-square (M != N) and too large to simulate cycle-accurately,
so the dataflow metrics are reported for a representative square SUMMA matmul
(default 32x32x32 on a 4x4 grid); the fusion traffic reduction uses the MLP's
ACTUAL eliminated intermediate (read from the fusion attrs polykernel-opt attaches,
cross-referenced against reports/mlp_traffic.json).

THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
hardware). CE + FMAC (no FMU/PMU); fabin_dsd/fabout_dsd; @set_color_config rx/tx.

Usage:
    polykernel-report --backend=dataflow examples/mlp_block.mlir
    python3 tools/polykernel-report/dataflow_report.py --backend=dataflow \\
        examples/mlp_block.mlir
"""

# pyright: reportImplicitRelativeImport=false
# Executable SCRIPT (the `polykernel-report` wrapper puts this directory on
# sys.path); the top-level `from mlir_traffic import ...` reuses the Todo 17 IR
# parser and is correct (a relative import would break the documented usage).

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mlir_traffic import ELEM_BYTES, parse_ops, tensor_bytes

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The simulator's SRAM-residency element width (fp32: the PeState buffers are
# std::vector<float>), the 48 KB per-PE SRAM, and the 2-byte (fp16) wavelet
# payload. Mirror Metrics::kElemBytes / Sram::kTotalBytes / Wavelet::kDataBits.
_ELEM_BYTES = 4
_SRAM_BYTES = 48 * 1024  # 49152.
_WAVELET_BYTES = 2

# The Wave-3 fusion passes (same set traffic_report.py uses) so the report reads
# the eliminated intermediate from the attrs the fusion passes attach.
_FUSION_PASSES = [
    "--fuse-rmsnorm-matmul", "--fuse-matmul-bias-gelu",
    "--fuse-residual-rmsnorm", "--fuse-softmax-mask",
]

_ZERO_FUSION = {
    "fused_op": None, "fused_from": None, "eliminated_intermediate": None,
    "eliminated_elements": 0, "eliminated_round_trip_bytes": 0,
    "eliminated_wavelets": 0,
}


#===----------------------------------------------------------------------===#
# SUMMA lowering + analytic metrics model (mirrors lib/DataflowSim/Metrics.cpp).
#===----------------------------------------------------------------------===#

def _choose_tile_k(k: int, pref: int) -> int:
    """Mirror ChooseTileK (LowerToDataflow.cpp): prefer pref, divide k exactly."""
    if pref <= 0:
        pref = 8
    t = pref if pref < k else k
    while t > 1 and (k % t) != 0:
        t -= 1
    return t


def lower_matmul_to_summa(m: int, n: int, k: int, grid_p: int, tile_k: int) -> dict:
    """Mirror LowerMatmulToSumma: square output tiles, an exact K-slab."""
    if m % grid_p or n % grid_p:
        raise SystemExit(f"error: shape {m}x{n} must tile evenly across gridP={grid_p}")
    tile_m, tile_n = m // grid_p, n // grid_p
    if tile_m != tile_n:
        raise SystemExit(f"error: SUMMA tiles are square (need M==N); got {m}x{n}")
    tk = _choose_tile_k(k, tile_k)
    return {"m": m, "n": n, "k": k, "gridP": grid_p, "tileM": tile_m,
            "tileN": tile_n, "tileK": tk, "steps": k // tk}


def _classify_bottleneck(m: dict) -> tuple[str, str]:
    """Mirror ClassifyBottleneck. The analytic model uses the critical-path cycle
    estimate, so Delayed (a simulation-observable stall, validated in C++) cannot
    arise here; SRAM over-subscription -> backpressure, idle fabric -> idle."""
    if m["sram_pressure_pct"] > 100.0:
        return "backpressure", "sram_over_subscription"
    if m["messages_sent"] == 0 and m["panel_computes"] == 0:
        return "idle", "none"
    return "active", "none"


def compute_metrics(plan: dict) -> dict:
    """Analytic dataflow metrics for one SUMMA plan (the simulator's model)."""
    total_pes = plan["gridP"] * plan["gridP"]
    panel = plan["tileM"] * plan["tileK"]
    a_wavelets = plan["gridP"] * panel * plan["steps"]  # == bWavelets (square).
    resident = (plan["tileM"] * plan["tileN"] * _ELEM_BYTES          # C tile
                + 2 * plan["tileM"] * plan["tileK"] * _ELEM_BYTES     # A ping-pong
                + 2 * plan["tileK"] * plan["tileN"] * _ELEM_BYTES)    # B ping-pong
    m = {
        "representative_shape": [plan["m"], plan["n"], plan["k"]],
        "grid_p": plan["gridP"],
        "tile_m": plan["tileM"], "tile_n": plan["tileN"],
        "tile_k": plan["tileK"], "steps": plan["steps"],
        "active_pes": total_pes, "total_pes": total_pes,
        "utilization_pct": 100.0,  # SUMMA maps the full P x P grid.
        "resident_bytes": resident, "sram_bytes": _SRAM_BYTES,
        "sram_pressure_pct": 100.0 * resident / _SRAM_BYTES,
        "a_wavelets": a_wavelets, "b_wavelets": a_wavelets,
        "messages_sent": 2 * a_wavelets,
        "avg_hop_distance": float(plan["gridP"] - 1),
        "critical_path_cycles": (plan["gridP"] - 1) + plan["steps"],
        "panel_computes": total_pes * plan["steps"],
    }
    m["bottleneck"], m["bottleneck_cause"] = _classify_bottleneck(m)
    return m


#===----------------------------------------------------------------------===#
# MLIR driving: extract the MLP's eliminated intermediate via polykernel-opt.
#===----------------------------------------------------------------------===#

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
    """Run polykernel-opt on `mlir` with `passes`; return the printed IR."""
    proc = subprocess.run([str(opt), str(mlir), *passes],
                          capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"polykernel-opt failed (exit {proc.returncode}):\n{proc.stderr}")
    return proc.stdout


def extract_fusion(opt: Path, mlir: Path) -> dict:
    """Run the fusion passes; return the first fused op's eliminated intermediate
    (a zeroed record when nothing fused)."""
    for o in parse_ops(run_opt(opt, mlir, _FUSION_PASSES)):
        if not o["fused_from"] or not o["eliminated_types"]:
            continue
        elim = o["eliminated_types"][0]
        dtype = elim.strip().rsplit("x", 1)[-1].rstrip(">")
        round_trip_bytes = 2 * tensor_bytes(elim)  # write + read fusion removes.
        return {
            "fused_op": o["op"], "fused_from": o["fused_from"],
            "eliminated_intermediate": elim,
            "eliminated_elements": tensor_bytes(elim) // ELEM_BYTES[dtype],
            "eliminated_round_trip_bytes": round_trip_bytes,
            "eliminated_wavelets": round_trip_bytes // _WAVELET_BYTES,
        }
    return dict(_ZERO_FUSION)


def build_report(opt: Path, mlir: Path, plan: dict) -> dict:
    """Compute the full dataflow metrics report for `mlir`."""
    metrics = compute_metrics(plan)
    fusion = extract_fusion(opt, mlir)

    # Fusion traffic reduction: the fused op eliminates the intermediate's fabric
    # round-trip; the matmul broadcast wavelets are invariant under fusion.
    fused_wavelets = metrics["messages_sent"]
    eliminated = fusion["eliminated_wavelets"]
    unfused_wavelets = fused_wavelets + eliminated
    fusion.update({
        "matmul_broadcast_wavelets": fused_wavelets,
        "unfused_wavelets": unfused_wavelets,
        "fused_wavelets": fused_wavelets,
        "traffic_reduction_pct": (100.0 * eliminated / unfused_wavelets
                                  if unfused_wavelets else 0.0),
        "cross_reference": {
            "report": "reports/mlp_traffic.json",
            "note": "Todo 17 global-traffic report: fusion eliminates the same "
                    "intermediate's global round-trip (reduction_pct there is the "
                    "whole-pipeline global-byte view; here it is the eliminated "
                    "intermediate's fabric wavelets vs the matmul broadcast).",
        },
    })
    return {
        "input": str(mlir),
        "backend": "dataflow",
        "simulator": "PolyKernel dataflow simulator (functional/cycle model; "
                     "NOT real CSL, NOT Cerebras hardware)",
        "sdk_analogs": ["cslc --out-routes", "calculate_cycles"],
        "dataflow_metrics": metrics,
        "fusion": fusion,
    }


def _mlp_traffic_xref() -> float | None:
    """Read the Todo 17 global reduction_pct for cross-reference (None if absent)."""
    p = _PROJECT_ROOT / "reports/mlp_traffic.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("reduction_pct")
    except (OSError, json.JSONDecodeError):
        return None


def render_text(r: dict) -> str:
    """Human-readable summary of the dataflow metrics."""
    m, f = r["dataflow_metrics"], r["fusion"]
    lines = [
        f"input:                  {r['input']}",
        f"backend:                {r['backend']}",
        f"simulator:              {r['simulator']}",
        f"representative_shape:   {m['representative_shape']} (M,N,K) on a "
        f"{m['grid_p']}x{m['grid_p']} grid",
        f"summa_tiles:            tileM={m['tile_m']} tileN={m['tile_n']} "
        f"tileK={m['tile_k']} steps={m['steps']}",
        f"grid_utilization:       {m['utilization_pct']:.2f}% "
        f"({m['active_pes']}/{m['total_pes']} PEs active)",
        f"sram_pressure:          {m['sram_pressure_pct']:.2f}% "
        f"({m['resident_bytes']} B resident vs {m['sram_bytes']} B)",
        f"messages_sent:          {m['messages_sent']} wavelets "
        f"(A={m['a_wavelets']} + B={m['b_wavelets']})",
        f"avg_hop_distance:       {m['avg_hop_distance']:.1f} hops (1 cycle/hop)",
        f"critical_path_cycles:   {m['critical_path_cycles']}",
        f"bottleneck:             {m['bottleneck']} (cause: {m['bottleneck_cause']})",
        f"fusion:                 {f['fused_op']} eliminates {f['eliminated_intermediate']}",
        f"traffic_reduction:      {f['traffic_reduction_pct']:.4f}% "
        f"({f['eliminated_wavelets']} wavelets removed; "
        f"{f['eliminated_round_trip_bytes']:,} B round-trip)",
    ]
    xref = _mlp_traffic_xref()
    if xref is not None:
        lines.append(f"xref mlp_traffic.json:  global reduction {xref:.2f}% "
                     f"(whole-pipeline global-byte view)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polykernel-report",
        description="Dataflow simulator metrics report (utilization / SRAM pressure / "
                    "messages / hops / bottleneck / critical-path / fusion reduction).",
    )
    parser.add_argument("mlir", nargs="?", default="examples/mlp_block.mlir",
                        help="MLIR file to report dataflow metrics for")
    parser.add_argument("--backend", default="dataflow", choices=["dataflow"],
                        help="report backend (dataflow simulator metrics)")
    parser.add_argument("--shape", default="32,32,32",
                        help="representative square SUMMA matmul shape M,N,K "
                             "(the MLP matmul is non-square/too large to simulate)")
    parser.add_argument("--grid", type=int, default=4, help="SUMMA grid dimension P")
    parser.add_argument("--tile-k", type=int, default=0, help="K-slab width (0 = auto)")
    parser.add_argument("--opt", default=None, help="path to polykernel-opt")
    parser.add_argument("--json-out",
                        default=str(_PROJECT_ROOT / "reports/dataflow_metrics.json"))
    parser.add_argument("--format", default="text", choices=["text", "json"])
    parser.add_argument("--no-write", action="store_true",
                        help="print the report but do not write the JSON output")
    args = parser.parse_args(argv)

    vals = [int(x) for x in args.shape.split(",")]
    if len(vals) != 3:
        raise SystemExit(f"error: --shape must be M,N,K (got '{args.shape}')")
    plan = lower_matmul_to_summa(vals[0], vals[1], vals[2], args.grid, args.tile_k)

    report = build_report(find_opt(args.opt), Path(args.mlir), plan)
    output = (json.dumps(report, indent=2) + "\n") if args.format == "json" else render_text(report)
    print(output, end="")
    if not args.no_write:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {args.json_out}", file=sys.stderr)

    m, red = report["dataflow_metrics"], report["fusion"]["traffic_reduction_pct"]
    print(f"dataflow: utilization {m['utilization_pct']:.2f}%, "
          f"traffic reduction {red:.4f}% from fusion, "
          f"bottleneck={m['bottleneck']}", file=sys.stderr)
    # QA gate: utilization > 0, fusion reduction > 0, a bottleneck classified.
    return 0 if (m["utilization_pct"] > 0 and red > 0 and m["bottleneck"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
