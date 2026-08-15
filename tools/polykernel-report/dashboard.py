#!/usr/bin/env python3
"""PolyKernel benchmark dashboard + HTML reports (Todo 34 / Wave 6).

The `--dashboard` report: AGGREGATES the already-committed per-backend reports
(NOT regenerates their data) into one benchmark report - text (Markdown) + three
STATIC, SELF-CONTAINED HTML pages:

  * CUDA  - reports/h100_bench.json (Todo 28; H100 sm_90 + A100 sm_80 PROJECTED
            speedups) + reports/kernel_report_example.json (Todo 27 contract-H
            per-kernel report for fused_rmsnorm_matmul cuda sm_90).
  * AMD   - reports/mi300_bench.json (Todo 28; MI300X gfx942 PROJECTED speedups).
  * Dataflow - reports/dataflow_metrics.json (Todo 38 simulator metrics). The
            dataflow HTML viz (reports/dataflow_report.html, Todo 39) is LINKED,
            never regenerated here (do NOT clobber it).
  * Compiler stats - passes (include/PolyKernel/Passes/Passes.td), generated
            kernels (kernels/generated/*.cu), lit tests (test/*.mlir), dialect
            ops (include/PolyKernel/IR/PolyKernelOps.td), and VALIDATED /
            FAILED-CORRECTNESS from the golden e2e harness.

The dashboard REFLECTS REALITY: validated + failed-correctness come from RUNNING
the golden harness tests/e2e/test_mlp_correctness.py (CPU-reference kernels vs
the NumPy/ml_dtypes golden, contract C). `--inject-failed-correctness` runs that
harness with its own `--corrupt fused_rmsnorm_matmul` flag so the report shows
"failed correctness: N>0" PROMINENTLY (proving failures are surfaced, not hidden).

SCOPE GUARDRAIL (enforced by assert_self_contained): every HTML page is a SINGLE
self-contained file - inline CSS + inline JS + an embedded JSON payload blob. NO
external dependencies, NO network fetch, NO framework, NO build step; each renders
fully OFFLINE. Mirrors the sibling tools/polykernel-report/dataflow_viz.py gate.

ALL NUMBERS ARE LABELED REAL-VS-PROJECTED: the H100/A100/MI300 speedups carry the
bench JSONs' status (PROJECTED - no rental; run benchmarks/rent_runpod.sh for real
measured numbers); the dataflow figures are the simulator's functional/cycle model
(NOT real CSL, NOT Cerebras hardware).

Usage:
    polykernel-report --dashboard            # via the wrapper's sibling CLI
    python3 tools/polykernel-report/dashboard.py --dashboard
    python3 tools/polykernel-report/dashboard.py --dashboard --inject-failed-correctness
"""

# allow: SIZE_OK - single-file dashboard: it aggregates three backends + compiler
# stats and renders one Markdown + three self-contained HTML pages + the CLI. The
# deliverable scope forbids splitting into helper modules, and it mirrors the
# sibling single-file dataflow_viz.py (which carries the same SIZE_OK marker).

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from html import escape as _esc
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REPORTS = _PROJECT_ROOT / "reports"

# Self-containment gate (mirrors dataflow_viz.py): tokens that would break offline
# rendering. Internal `url(#fragment)` SVG references are offline-safe and NOT
# listed. NOTE: the literal substring "cdn" is forbidden, so no page text may say
# "cdn" - phrased as "no external dependencies" instead.
_FORBIDDEN = ("http://", "https://", "cdn", "<script src", "<link ",
              "xlink", "xmlns", "@import", "url(http", "url(//")
# Structural QA targets: the embedded payload + the rendered dashboard sections.
_REQUIRED_IDS = ("dashboard-data", "correctness-banner", "model-fragment",
                 "backends", "cuda-speedup", "amd-speedup", "dataflow-metrics",
                 "compiler-stats")


#===----------------------------------------------------------------------===#
# Aggregation loaders (READ-ONLY: aggregate the committed reports, never regen).
#===----------------------------------------------------------------------===#

def _load_json(rel: str) -> dict:
    """Load a committed report JSON ({} if absent - the page degrades, not crashes)."""
    path = _PROJECT_ROOT / rel
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_model() -> dict:
    """Assemble the aggregated dashboard model from the committed reports."""
    cuda = _load_json("reports/h100_bench.json")
    amd = _load_json("reports/mi300_bench.json")
    fragment = cuda.get("fragment") or amd.get("fragment") or (
        "MLP block (examples/mlp_block.mlir): rmsnorm+matmul+gelu+matmul+add")
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fragment": fragment,
        "dtype": cuda.get("dtype", "bf16"),
        "shape": cuda.get("shape", {}),
        "cuda": cuda,
        "amd": amd,
        "dataflow": _load_json("reports/dataflow_metrics.json"),
        "kernel_report": _load_json("reports/kernel_report_example.json"),
        "traffic": _load_json("reports/mlp_traffic.json"),
    }


# Backends-enabled is structural facts about the implemented system (Waves 1-7),
# not a regenerated metric - documented inline so the dashboard states them.
_BACKENDS = [
    {"name": "CUDA", "archs": "sm_80 (A100), sm_89 (RTX 6000 Ada), sm_90 (H100)",
     "status": "compile + PTX + GPU-free compile-time analysis (no local NVIDIA "
               "GPU); H100/A100/RTX 6000 Ada speedups PROJECTED"},
    {"name": "HIP / ROCm", "archs": "gfx1101 (local RX 7800 XT), gfx942 (MI300)",
     "status": "runs locally on gfx1101 + WMMA bf16; MI300 cross-compile; MI300 "
               "speedups PROJECTED"},
    {"name": "Dataflow", "archs": "64x64 PE grid (CE + FMAC, 48 KB SRAM/PE)",
     "status": "functional/cycle SIMULATOR (NOT real CSL, NOT Cerebras hardware)"},
]


#===----------------------------------------------------------------------===#
# Compiler stats: passes / generated kernels / lit tests / dialect ops /
# validated + failed-correctness (the golden e2e harness is the source of truth).
#===----------------------------------------------------------------------===#

def collect_correctness(inject: bool, override_json: str | None) -> dict:
    """validated + failed-correctness from the golden e2e harness (contract C).

    The harness (tests/e2e/test_mlp_correctness.py) drives the built CPU-reference
    kernels vs the NumPy/ml_dtypes golden and prints "N failed / M ops". With
    `inject` it runs the harness's OWN `--corrupt fused_rmsnorm_matmul` flag, so a
    deliberately-broken kernel makes the dashboard surface "failed correctness: N>0"
    (the dashboard reflects reality; it never hides a failure)."""
    if override_json:
        data = json.loads(Path(override_json).read_text())
        return {**data, "source": f"override JSON ({override_json})",
                "unavailable": False, "table": data.get("table", "")}

    harness = _PROJECT_ROOT / "tests" / "e2e" / "test_mlp_correctness.py"
    cmd = [sys.executable, str(harness)]
    if inject:
        cmd += ["--corrupt", "fused_rmsnorm_matmul"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=300, cwd=_PROJECT_ROOT)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"validated": None, "failed": 0, "ops": 0, "unavailable": True,
                "exit_code": None, "table": "",
                "source": f"golden e2e harness failed to run: {exc}"}
    match = re.search(r"(\d+) failed / (\d+) ops", proc.stdout)
    if not match:
        return {"validated": None, "failed": 0, "ops": 0, "unavailable": True,
                "exit_code": proc.returncode, "table": proc.stdout.strip(),
                "source": "could not parse golden e2e harness output"}
    failed, ops = int(match.group(1)), int(match.group(2))
    source = "tests/e2e/test_mlp_correctness.py (CPU-ref vs golden, contract C)"
    if inject:
        source += " [--corrupt fused_rmsnorm_matmul: INJECTED failure]"
    return {"validated": ops - failed, "failed": failed, "ops": ops,
            "unavailable": False, "exit_code": proc.returncode,
            "table": proc.stdout.strip(), "source": source}


def collect_compiler_stats(correctness: dict) -> dict:
    """passes / generated kernels / lit tests / dialect ops + correctness rollup."""
    passes_td = (_PROJECT_ROOT / "include/PolyKernel/Passes/Passes.td").read_text()
    passes = re.findall(r"^\s*def\s+(\w+)\s*:\s*Pass<", passes_td, re.M)
    ops_td = (_PROJECT_ROOT / "include/PolyKernel/IR/PolyKernelOps.td").read_text()
    ops = re.findall(r"^\s*def\s+PolyKernel_(\w+?)\s*:", ops_td, re.M)
    kernels = sorted(p.name for p in (_PROJECT_ROOT / "kernels/generated").glob("*.cu"))
    lit_tests = sorted(p.name for p in (_PROJECT_ROOT / "test").glob("*.mlir"))
    return {
        "passes": passes, "pass_count": len(passes),
        "dialect_ops": ops, "dialect_op_count": len(ops),
        "generated_kernels": kernels, "generated_kernel_count": len(kernels),
        "lit_tests": lit_tests, "lit_test_count": len(lit_tests),
        "validated": correctness.get("validated"),
        "failed_correctness": correctness.get("failed", 0),
        "correctness_ops": correctness.get("ops"),
        "correctness_source": correctness.get("source"),
        "correctness_table": correctness.get("table", ""),
        "correctness_unavailable": correctness.get("unavailable", False),
    }


#===----------------------------------------------------------------------===#
# Markdown renderer (reports/benchmark_report.md).
#===----------------------------------------------------------------------===#

def _status_label(report: dict) -> str:
    """REAL-VS-PROJECTED label carried verbatim from the bench JSON status."""
    if report.get("measured"):
        return "REAL (measured on rental)"
    if report.get("projected") or report.get("status") == "PROJECTED":
        return "PROJECTED (not measured)"
    return report.get("status", "unknown")


def _arch_speedup_rows(report: dict) -> list[tuple[str, str, str, str, str]]:
    """(label, backend, unfused, fused, autotuned) rows from a bench JSON."""
    rows = []
    for arch in report.get("archs", {}).values():
        sp = arch.get("speedup", {})
        rows.append((arch.get("label", "?"), f'{arch.get("backend", "?")} '
                     f'({arch.get("arch", "?")})',
                     f'{sp.get("unfused", 0):.3f}x', f'{sp.get("fused", 0):.3f}x',
                     f'{sp.get("autotuned", 0):.3f}x'))
    return rows


def render_markdown(model: dict, stats: dict) -> str:
    failed = stats["failed_correctness"]
    verdict = "PASS" if failed == 0 else "FAIL"
    lines = [
        "# PolyKernel benchmark dashboard (Todo 34 / Wave 6)",
        "",
        f"Generated: {model['generated_utc']} by "
        "`tools/polykernel-report/dashboard.py --dashboard`.",
        "",
        "## Model fragment",
        "",
        f"- **Fragment:** `{model['fragment']}`",
        f"- **dtype / shape:** {model['dtype']}, {json.dumps(model['shape'])}",
        "",
        "## Backends enabled",
        "",
        "| backend | archs | status |",
        "|---|---|---|",
    ]
    for b in _BACKENDS:
        lines.append(f"| {b['name']} | {b['archs']} | {b['status']} |")
    lines += [
        "",
        f"## Correctness (golden contract C) — {verdict}",
        "",
        f"- **failed correctness: {failed}**",
    ]
    if stats["correctness_unavailable"]:
        lines.append("- validated: n/a (golden e2e harness unavailable; build the "
                     "CPU refs)")
    else:
        lines.append(f"- validated: {stats['validated']} / {stats['correctness_ops']} "
                     "ops pass the golden (cosine >= 0.999, max rel err <= 1e-2, "
                     "PCC >= 0.99)")
    lines.append(f"- source: {stats['correctness_source']}")
    lines += ["", "## CUDA speedups (unfused = 1.00x baseline)", "",
              f"> {_status_label(model['cuda'])}: {model['cuda'].get('banner', '')}",
              "", "| arch | backend | unfused | fused | autotuned |",
              "|---|---|---|---|---|"]
    for row in _arch_speedup_rows(model["cuda"]):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    lines += ["", "## AMD speedups (unfused = 1.00x baseline)", "",
              f"> {_status_label(model['amd'])}: {model['amd'].get('banner', '')}",
              "", "| arch | backend | unfused | fused | autotuned |",
              "|---|---|---|---|---|"]
    for row in _arch_speedup_rows(model["amd"]):
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")
    df = model["dataflow"].get("dataflow_metrics", {})
    fu = model["dataflow"].get("fusion", {})
    lines += ["", "## Dataflow simulator metrics", "",
              f"> {model['dataflow'].get('simulator', 'dataflow simulator')}", ""]
    if df:
        lines += [
            f"- grid utilization: {df.get('utilization_pct', 0):.2f}% "
            f"({df.get('active_pes', 0)}/{df.get('total_pes', 0)} PEs)",
            f"- SRAM pressure: {df.get('sram_pressure_pct', 0):.2f}%",
            f"- messages sent: {df.get('messages_sent', 0):,} wavelets",
            f"- avg hop distance: {df.get('avg_hop_distance', 0):.1f}",
            f"- critical path: {df.get('critical_path_cycles', 0)} cycles",
            f"- bottleneck: {df.get('bottleneck', '?')}",
            f"- fusion traffic reduction: {fu.get('traffic_reduction_pct', 0):.4f}%",
            "- visualizer: reports/dataflow_report.html (Todo 39; linked, not "
            "regenerated here)",
        ]
    lines += ["", "## Compiler stats", "",
              f"- passes: {stats['pass_count']} ({', '.join(stats['passes'])})",
              f"- generated kernels: {stats['generated_kernel_count']} "
              f"({', '.join(stats['generated_kernels'])})",
              f"- dialect ops: {stats['dialect_op_count']} (closed set)",
              f"- lit tests (check-polykernel): {stats['lit_test_count']}",
              f"- validated kernels: {stats['validated']}",
              f"- **failed correctness: {failed}**",
              "", "## Reports", "",
              "- reports/benchmark_report.html (this dashboard, self-contained)",
              "- reports/h100_report.html (CUDA H100/A100 detail, self-contained)",
              "- reports/mi300_report.html (AMD MI300 detail, self-contained)",
              "- reports/dataflow_report.html (dataflow viz, Todo 39)",
              "", "## Caveats", "",
              "- H100/A100/MI300 speedups are PROJECTED (roofline + analytic "
              "traffic), not measured on rented hardware.",
              "- The dataflow backend is a functional/cycle SIMULATOR (NOT real "
              "CSL, NOT Cerebras hardware).",
              "- No SOTA-performance claim: the goal is the compiler/runtime "
              "machinery, not beating vLLM.",
              "",
    ]
    return "\n".join(lines)


#===----------------------------------------------------------------------===#
# Self-contained HTML: shared CSS + primitives (mirror dataflow_viz.py theme).
#===----------------------------------------------------------------------===#

_CSS = """
:root {
  --bg: #0b1020; --panel: #131a2e; --panel2: #0f1526; --ink: #e6ecff;
  --muted: #93a1c4; --line: #26304d; --accent: #5b8cff;
  --a: #3b82f6; --b: #f97316; --good: #22c55e; --warn: #eab308; --bad: #ef4444;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 22px 28px 12px; border-bottom: 1px solid var(--line); }
header h1 { margin: 0 0 6px; font-size: 21px; letter-spacing: .2px; }
header .disclaimer { margin: 0; color: var(--muted); font-size: 12.5px; max-width: 104ch; }
header .meta { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
main { padding: 20px 28px 48px; max-width: 1180px; margin: 0 auto; }
section { margin: 24px 0; }
section > h2 { font-size: 15px; margin: 0 0 12px; padding-bottom: 6px;
  border-bottom: 1px solid var(--line); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
.card .k { color: var(--muted); font-size: 11.5px; text-transform: uppercase; letter-spacing: .5px; }
.card .v { font-size: 22px; font-weight: 650; margin-top: 4px; }
.card .s { color: var(--muted); font-size: 11.5px; margin-top: 2px; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; overflow-x: auto; }
.legend { color: var(--muted); font-size: 12.5px; margin-top: 10px; }
.legend code { color: var(--ink); }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 600; }
.pill.good { background: rgba(34,197,94,.16); color: #86efac; }
.pill.warn { background: rgba(234,179,8,.16); color: #fde047; }
.pill.bad { background: rgba(239,68,68,.16); color: #fca5a5; }
.pill.proj { background: rgba(91,140,255,.16); color: #aecbff; }
.pill.sim { background: rgba(249,115,22,.16); color: #fdba74; }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th, table.data td { padding: 6px 10px; border-bottom: 1px solid var(--panel2); text-align: left; }
table.data th { color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: .5px; }
table.data td { font-variant-numeric: tabular-nums; }
table.kv { border-collapse: collapse; width: 100%; font-size: 13px; }
table.kv td { padding: 5px 8px; border-bottom: 1px solid var(--panel2); vertical-align: top; }
table.kv td:first-child { color: var(--muted); width: 46%; }
table.kv td:last-child { font-variant-numeric: tabular-nums; }
.bar-row { display: grid; grid-template-columns: 120px 1fr 90px; align-items: center; gap: 10px; margin: 7px 0; }
.bar-row .lbl { color: var(--muted); font-size: 12.5px; }
.bar-track { background: var(--panel2); border-radius: 6px; height: 16px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; }
.bar-row .num { text-align: right; font-variant-numeric: tabular-nums; font-size: 12.5px; }
.banner { border-radius: 12px; padding: 16px 20px; margin: 18px 0; font-size: 15px; font-weight: 650; border: 1px solid; }
.banner.good { background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.5); color: #86efac; }
.banner.bad { background: rgba(239,68,68,.16); border-color: rgba(239,68,68,.7); color: #fca5a5; font-size: 19px; }
.banner .sub { display: block; font-weight: 400; font-size: 12.5px; margin-top: 6px; color: var(--muted); }
.banner.bad .sub { color: #fda4af; }
pre.table { background: var(--panel2); border: 1px solid var(--line); border-radius: 8px; padding: 12px; overflow-x: auto; font-size: 12px; color: var(--ink); }
a { color: var(--accent); }
footer { padding: 16px 28px 40px; color: var(--muted); font-size: 12px; border-top: 1px solid var(--line); }
"""


def _card(k: str, v: str, sub: str = "") -> str:
    return (f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div>'
            f'<div class="s">{sub}</div></div>')


def _pill(text: str, cls: str) -> str:
    return f'<span class="pill {cls}">{text}</span>'


def _bar(lbl: str, width_pct: float, color: str, num: str) -> str:
    return (f'<div class="bar-row"><span class="lbl">{lbl}</span>'
            f'<span class="bar-track"><span class="bar-fill" '
            f'style="width:{width_pct:.1f}%;background:{color}"></span></span>'
            f'<span class="num">{num}</span></div>')


def _kv(rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in rows)
    return f'<table class="kv">{body}</table>'


def _page(title: str, headline: str, disclaimer: str, meta: str, body: str,
          payload: dict) -> str:
    """Wrap body in a self-contained HTML page (inline CSS + embedded JSON blob)."""
    return (
        "<!DOCTYPE html>\n"
        "<!--\n"
        f"  {title} (Todo 34 / Wave 6).\n"
        "  STATIC + SELF-CONTAINED: inline CSS + inline JS + an embedded JSON payload\n"
        "  blob (script id dashboard-data, type application/json). No external\n"
        "  dependencies, no network fetch, no framework, no build step -> renders\n"
        "  fully OFFLINE. Generated by tools/polykernel-report/dashboard.py.\n"
        "  Numbers are labeled real-vs-projected; the dataflow backend is a simulator.\n"
        "-->\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<header><h1>{headline}</h1>"
        f'<p class="disclaimer">{disclaimer}</p>'
        f'<p class="meta">{meta}</p></header>\n'
        f"<main>\n{body}\n</main>\n"
        "<footer>PolyKernel &mdash; out-of-tree MLIR compiler + CUDA/HIP kernels + "
        "Cerebras-style dataflow simulator + Modal deployment. Self-contained "
        "static report (no external dependencies).</footer>\n"
        f'<script id="dashboard-data" type="application/json">'
        f"{json.dumps(payload)}</script>\n"
        "<script>\n"
        "  // Payload integrity check (offline): the embedded blob must parse.\n"
        "  (function () {\n"
        "    var el = document.getElementById('dashboard-data');\n"
        "    try { JSON.parse(el.textContent); }\n"
        "    catch (e) { console.error('dashboard-data payload invalid', e); }\n"
        "  })();\n"
        "</script>\n"
        "</body>\n</html>\n"
    )


#===----------------------------------------------------------------------===#
# Section renderers (correctness banner / speedups / dataflow / compiler stats).
#===----------------------------------------------------------------------===#

def _correctness_banner(stats: dict) -> str:
    """PROMINENT failed-correctness banner: green at 0, large+red at N>0."""
    failed = stats["failed_correctness"]
    if stats["correctness_unavailable"]:
        return ('<div class="banner bad" id="correctness-banner">'
                'failed correctness: unknown'
                '<span class="sub">golden e2e harness unavailable - build the CPU '
                'refs (cmake --build build) to validate. '
                f'{_esc(stats["correctness_source"] or "")}</span></div>')
    if failed > 0:
        return (f'<div class="banner bad" id="correctness-banner">'
                f'failed correctness: {failed} &mdash; FAIL'
                f'<span class="sub">{stats["validated"]} / '
                f'{stats["correctness_ops"]} ops validated; {failed} FAILED the '
                f'golden (contract C). The dashboard reflects reality - it does '
                f'not hide failures. source: {_esc(stats["correctness_source"] or "")}'
                f'</span></div>')
    return (f'<div class="banner good" id="correctness-banner">'
            f'failed correctness: 0'
            f'<span class="sub">all {stats["correctness_ops"]} ops validated against '
            f'the NumPy/ml_dtypes golden (cosine &gt;= 0.999, max rel err &lt;= 1e-2, '
            f'PCC &gt;= 0.99; bf16 round / fp32 accumulate). source: '
            f'{_esc(stats["correctness_source"] or "")}</span></div>')


def _speedup_panel(report: dict, anchor: str) -> str:
    """Speedup bars + table for one bench JSON (CUDA has 2 archs, AMD has 1)."""
    archs = report.get("archs", {})
    label = _status_label(report)
    pill = _pill(label, "proj" if "PROJECTED" in label else "good")
    parts = [f"<p>{pill} &nbsp; {_esc(report.get('banner', ''))}</p>"]
    for arch in archs.values():
        sp = arch.get("speedup", {})
        mx = max(sp.get("autotuned", 1.0), sp.get("fused", 1.0), 1.0)
        bars = (
            _bar("unfused", 100.0 * sp.get("unfused", 1.0) / mx, "#64748b",
                 f'{sp.get("unfused", 0):.3f}x')
            + _bar("fused", 100.0 * sp.get("fused", 1.0) / mx, "#5b8cff",
                   f'{sp.get("fused", 1.0):.3f}x')
            + _bar("autotuned", 100.0 * sp.get("autotuned", 1.0) / mx, "#22c55e",
                   f'{sp.get("autotuned", 0):.3f}x'))
        pt = arch.get("projected_total_ms", {})
        parts.append(
            f'<h3>{_esc(arch.get("label", "?"))} &mdash; '
            f'{_esc(arch.get("backend", "?"))} ({_esc(arch.get("arch", "?"))})</h3>'
            f'<div class="panel">{bars}'
            f'<p class="legend">projected fragment wall time (ms): unfused '
            f'{pt.get("unfused", 0):.4f} &middot; fused {pt.get("fused", 0):.4f} '
            f'&middot; autotuned {pt.get("autotuned", 0):.4f} &middot; ridge '
            f'{arch.get("ridge_flop_per_byte", 0):.1f} flop/byte &middot; peak '
            f'{arch.get("peak_tflops", 0):.0f} TFLOPS / '
            f'{arch.get("peak_bw_gbps", 0):.0f} GB/s</p></div>')
    return f'<section id="{anchor}"><h2>{anchor.replace("-", " ").title()}</h2>' \
           + "".join(parts) + "</section>"


def _dataflow_section(model: dict) -> str:
    df = model["dataflow"].get("dataflow_metrics", {})
    fu = model["dataflow"].get("fusion", {})
    if not df:
        return ('<section id="dataflow-metrics"><h2>Dataflow Metrics</h2>'
                '<p>dataflow_metrics.json not found.</p></section>')
    cards = "".join([
        _card("Grid utilization", f'{df.get("utilization_pct", 0):.0f}%',
              f'{df.get("active_pes", 0)}/{df.get("total_pes", 0)} PEs'),
        _card("SRAM pressure", f'{df.get("sram_pressure_pct", 0):.2f}%',
              f'{df.get("resident_bytes", 0):,} / {df.get("sram_bytes", 0):,} B'),
        _card("Messages sent", f'{df.get("messages_sent", 0):,}', 'wavelets'),
        _card("Avg hop distance", f'{df.get("avg_hop_distance", 0):.1f}', 'hops'),
        _card("Critical path", f'{df.get("critical_path_cycles", 0):,}', 'cycles'),
        _card("Fusion reduction", f'{fu.get("traffic_reduction_pct", 0):.2f}%',
              f'{fu.get("eliminated_wavelets", 0):,} wavelets removed'),
    ])
    sim = _esc(model["dataflow"].get("simulator", "dataflow simulator"))
    return (
        '<section id="dataflow-metrics"><h2>Dataflow Simulator Metrics</h2>'
        f'<p>{_pill("SIMULATOR", "sim")} &nbsp; {sim}</p>'
        f'<div class="cards">{cards}</div>'
        f'<div class="panel" style="margin-top:12px">{_kv([
            ("Bottleneck", f'{df.get("bottleneck", "?")} '
             f'(cause: {df.get("bottleneck_cause", "none")})'),
            ("Representative shape", f'{df.get("representative_shape", [])} (M,N,K) '
             f'on a {df.get("grid_p", 0)}x{df.get("grid_p", 0)} grid'),
            ("SUMMA mapping", "A broadcast east, B broadcast south, output-stationary C"),
            ("Fused op", f'{_esc(fu.get("fused_op", "?"))} eliminates '
             f'{_esc(fu.get("eliminated_intermediate", "?"))}'),
        ])}</div>'
        '<p class="legend">Visualizer: <a href="dataflow_report.html">'
        'reports/dataflow_report.html</a> (Todo 39; linked, not regenerated by the '
        'dashboard).</p></section>')


def _compiler_stats_section(stats: dict) -> str:
    failed = stats["failed_correctness"]
    failed_pill = _pill(f"failed correctness: {failed}",
                        "good" if failed == 0 else "bad")
    table = (f'<pre class="table">{_esc(stats["correctness_table"])}</pre>'
             if stats["correctness_table"] else "")
    return (
        '<section id="compiler-stats"><h2>Compiler Stats</h2>'
        f'<div class="cards">{"".join([
            _card("Passes", str(stats["pass_count"]), "MLIR transformation passes"),
            _card("Generated kernels", str(stats["generated_kernel_count"]),
                  "kernels/generated/*.cu"),
            _card("Dialect ops", str(stats["dialect_op_count"]), "closed op set"),
            _card("Lit tests", str(stats["lit_test_count"]), "check-polykernel"),
            _card("Validated", str(stats["validated"]), "ops pass the golden"),
        ])}</div>'
        f'<div class="panel" style="margin-top:12px">{_kv([
            ("Passes", ", ".join(stats["passes"])),
            ("Generated kernels", ", ".join(stats["generated_kernels"])),
            ("Lit tests (check-polykernel)", ", ".join(stats["lit_tests"])),
            ("Correctness", failed_pill),
            ("Correctness source", _esc(stats["correctness_source"] or "")),
        ])}</div>{table}</section>')


def _backends_section() -> str:
    rows = "".join(
        f'<tr><td>{_esc(b["name"])}</td><td>{_esc(b["archs"])}</td>'
        f'<td>{_esc(b["status"])}</td></tr>' for b in _BACKENDS)
    return ('<section id="backends"><h2>Backends Enabled</h2><div class="panel">'
            '<table class="data"><tr><th>backend</th><th>archs</th><th>status</th>'
            f'</tr>{rows}</table></div></section>')


def _model_fragment_section(model: dict) -> str:
    sh = model["shape"]
    shape = (f'M={sh.get("M", "?")}, K={sh.get("K", "?")}, '
             f'N_up={sh.get("N_up", "?")}, N_dn={sh.get("N_dn", "?")}')
    return ('<section id="model-fragment"><h2>Model Fragment</h2>'
            f'<div class="panel">{_kv([
                ("Fragment", f'<code>{_esc(model["fragment"])}</code>'),
                ("dtype", _esc(model["dtype"])),
                ("Shape", shape),
            ])}</div></section>')


def _kernel_report_section(report: dict) -> str:
    """Contract-H per-kernel report (Todo 27) for the fused CUDA kernel."""
    if not report:
        return ""
    occ, tr = report.get("occupancy", {}), report.get("traffic", {})
    fixes = report.get("suggested_fixes") or ["(none)"]
    return (
        '<section id="kernel-report"><h2>Per-Kernel Report (contract-H)</h2>'
        f'<div class="panel">{_kv([
            ("Kernel", f'{_esc(report.get("kernel", "?"))} &mdash; '
             f'{_esc(report.get("backend", "?"))} ({_esc(report.get("arch", "?"))})'),
            ("Registers / thread", str(report.get("registers_per_thread", "?"))),
            ("Smem / block", f'{report.get("smem_per_block_bytes", 0):,} B'),
            ("Spills", f'{report.get("spill_stores_bytes", 0)} B stores / '
             f'{report.get("spill_loads_bytes", 0)} B loads'),
            ("Occupancy", f'{occ.get("occupancy_pct", "?")}% '
             f'({occ.get("active_warps_per_sm", "?")}/{occ.get("max_warps_per_sm", "?")} '
             f'warps/SM, limiter: {occ.get("limiter", "?")})'),
            ("Traffic", f'read {tr.get("global_read_bytes", 0):,} B / write '
             f'{tr.get("global_write_bytes", 0):,} B'),
            ("Arithmetic intensity", f'{tr.get("arithmetic_intensity_flop_per_byte", 0)} '
             f'flop/byte ({tr.get("roofline", "?")})'),
            ("Bottleneck", _esc(report.get("bottleneck", "?"))),
            ("Path", _esc(report.get("path", "?"))),
            ("Suggested fixes", "; ".join(_esc(f) for f in fixes)),
        ])}</div></section>')


#===----------------------------------------------------------------------===#
# Page assembly (benchmark + per-arch detail pages).
#===----------------------------------------------------------------------===#

_DISCLAIMER = (
    "STATIC + SELF-CONTAINED: inline CSS + inline JS + an embedded JSON payload; no "
    "external dependencies, no network fetch, no framework; renders fully offline. "
    "H100/A100/MI300 speedups are PROJECTED (roofline + analytic traffic, not measured "
    "on rented hardware). The dataflow backend is a functional/cycle SIMULATOR (NOT "
    "real CSL, NOT Cerebras hardware). No SOTA-performance claim.")


def _meta(model: dict, stats: dict) -> str:
    return (f'generated: {model["generated_utc"]} &middot; fragment: '
            f'<code>{_esc(model["fragment"])}</code> &middot; failed correctness: '
            f'{stats["failed_correctness"]} &middot; by '
            'tools/polykernel-report/dashboard.py')


def render_benchmark_html(model: dict, stats: dict) -> str:
    body = "".join([
        _correctness_banner(stats),
        _model_fragment_section(model),
        _backends_section(),
        _speedup_panel(model["cuda"], "cuda-speedup"),
        _speedup_panel(model["amd"], "amd-speedup"),
        _dataflow_section(model),
        _compiler_stats_section(stats),
        _kernel_report_section(model["kernel_report"]),
        '<section id="reports"><h2>Reports</h2><div class="panel">'
        '<ul style="margin:0;padding-left:18px">'
        '<li><a href="h100_report.html">reports/h100_report.html</a> &mdash; CUDA '
        'H100/A100 detail</li>'
        '<li><a href="mi300_report.html">reports/mi300_report.html</a> &mdash; AMD '
        'MI300 detail</li>'
        '<li><a href="dataflow_report.html">reports/dataflow_report.html</a> &mdash; '
        'dataflow visualizer (Todo 39)</li>'
        '</ul></div></section>',
    ])
    payload = {"generated_utc": model["generated_utc"], "fragment": model["fragment"],
               "cuda": model["cuda"], "amd": model["amd"],
               "dataflow": model["dataflow"], "kernel_report": model["kernel_report"],
               "compiler_stats": {k: v for k, v in stats.items()
                                  if k != "correctness_table"}}
    return _page("PolyKernel Benchmark Dashboard", "PolyKernel Benchmark Dashboard",
                 _DISCLAIMER, _meta(model, stats), body, payload)


def _per_op_table(report: dict, arch: dict) -> str:
    rows = "".join(
        f'<tr><td>{_esc(op.get("op", "?"))}</td><td>{op.get("flops", 0):,}</td>'
        f'<td>{op.get("bytes", 0):,}</td><td>{op.get("ai_flop_per_byte", 0):.2f}</td>'
        f'<td>{_esc(op.get("bound", "?"))}</td><td>{op.get("projected_ms", 0):.5f}</td>'
        f'</tr>' for op in arch.get("per_op_unfused", []))
    return ('<table class="data"><tr><th>op</th><th>flops</th><th>bytes</th>'
            '<th>AI (flop/byte)</th><th>bound</th><th>projected ms</th></tr>'
            f'{rows}</table>')


def render_arch_html(model: dict, stats: dict, source_key: str, title: str,
                     headline: str) -> str:
    """A per-backend detail page (h100_report.html / mi300_report.html)."""
    report = model[source_key]
    label = _status_label(report)
    pill = _pill(label, "proj" if "PROJECTED" in label else "good")
    sections = [
        f'<section><p>{pill} &nbsp; {_esc(report.get("banner", ""))}</p></section>',
        _speedup_panel(report, f"{source_key}-speedup"),
    ]
    for arch in report.get("archs", {}).values():
        sections.append(
            f'<section><h2>Per-op roofline &mdash; {_esc(arch.get("label", "?"))} '
            f'(unfused)</h2><div class="panel">{_per_op_table(report, arch)}</div>'
            '<p class="legend">PROJECTED per-op roofline (lib/Analysis/Roofline.cpp + '
            'Todo 17 analytic traffic). t = max(flops/peak, bytes/bw) / eta.</p>'
            '</section>')
    sections.append(_kernel_report_section(model["kernel_report"])
                    if source_key == "cuda" else "")
    sections.append(
        '<section><h2>Caveats</h2><div class="panel"><ul style="margin:0;'
        'padding-left:18px">'
        '<li>PROJECTED, not measured: derived from the roofline model + analytic '
        'traffic, not a rented-GPU run.</li>'
        '<li>The fragment is compute-bound (matmuls AI~1215 &gt;&gt; ridge); the '
        'headline gain is the scalar&rarr;autotuned attainment jump.</li>'
        '<li>Opt-in real measurement: <code>RUNPOD_API_KEY=... '
        'benchmarks/rent_runpod.sh --suite mlp</code> (owner-gated).</li>'
        '</ul></div></section>')
    payload = {"generated_utc": model["generated_utc"], "source": source_key,
               "report": report,
               "failed_correctness": stats["failed_correctness"]}
    return _page(title, headline, _DISCLAIMER, _meta(model, stats),
                 "".join(sections), payload)


#===----------------------------------------------------------------------===#
# Self-containment gate (mirror dataflow_viz.py).
#===----------------------------------------------------------------------===#

def assert_self_contained(html: str, required: tuple[str, ...] = _REQUIRED_IDS) -> None:
    """QA gate: no external refs (offline-safe) + the structural sections present."""
    low = html.lower()
    hits = [t for t in _FORBIDDEN if t in low]
    if hits:
        raise SystemExit(f"error: HTML is NOT self-contained; external refs: {hits}")
    missing = [i for i in required if f'id="{i}"' not in html]
    if missing:
        raise SystemExit(f"error: HTML missing required sections: {missing}")


#===----------------------------------------------------------------------===#
# CLI.
#===----------------------------------------------------------------------===#

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polykernel-report",
        description="Benchmark dashboard: aggregate CUDA (real/projected) + AMD "
                    "(real/projected) + dataflow simulator metrics + compiler stats "
                    "(passes / generated kernels / validated / failed correctness) "
                    "into a Markdown report + three self-contained HTML pages.")
    parser.add_argument("--dashboard", action="store_true",
                        help="generate the benchmark dashboard (this tool's action)")
    parser.add_argument("--inject-failed-correctness", action="store_true",
                        help="run the golden e2e harness with --corrupt "
                             "fused_rmsnorm_matmul so the report shows "
                             "'failed correctness: N>0' prominently (negative test)")
    parser.add_argument("--correctness-json", default=None,
                        help="override the correctness source with a JSON file "
                             "({validated, failed, ops, table}; offline/no-build)")
    parser.add_argument("--reports-dir", default=str(_REPORTS),
                        help="output directory (default reports/)")
    parser.add_argument("--no-write", action="store_true",
                        help="run the gate + print the summary but write nothing")
    args = parser.parse_args(argv)

    if not args.dashboard:
        parser.error("--dashboard is required (this is the dashboard CLI)")

    model = load_model()
    correctness = collect_correctness(args.inject_failed_correctness,
                                      args.correctness_json)
    stats = collect_compiler_stats(correctness)

    benchmark_html = render_benchmark_html(model, stats)
    h100_html = render_arch_html(
        model, stats, "cuda", "PolyKernel CUDA Benchmark (H100 / A100)",
        "PolyKernel CUDA Benchmark &mdash; H100 / A100")
    mi300_html = render_arch_html(
        model, stats, "amd", "PolyKernel AMD Benchmark (MI300)",
        "PolyKernel AMD Benchmark &mdash; MI300X")
    markdown = render_markdown(model, stats)

    # Self-containment + structural QA gate (the benchmark page carries the full
    # required-id set; the per-arch pages carry the shared payload id only).
    assert_self_contained(benchmark_html)
    assert_self_contained(h100_html, required=("dashboard-data", "cuda-speedup"))
    assert_self_contained(mi300_html, required=("dashboard-data", "amd-speedup"))

    out = Path(args.reports_dir)
    if not args.no_write:
        out.mkdir(parents=True, exist_ok=True)
        (out / "benchmark_report.md").write_text(markdown)
        (out / "benchmark_report.html").write_text(benchmark_html)
        (out / "h100_report.html").write_text(h100_html)
        (out / "mi300_report.html").write_text(mi300_html)
        print(f"wrote {out / 'benchmark_report.md'}", file=sys.stderr)
        print(f"wrote {out / 'benchmark_report.html'}", file=sys.stderr)
        print(f"wrote {out / 'h100_report.html'}", file=sys.stderr)
        print(f"wrote {out / 'mi300_report.html'}", file=sys.stderr)

    failed = stats["failed_correctness"]
    print(markdown)
    if stats["correctness_unavailable"]:
        print("dashboard: failed correctness: unknown (golden e2e harness "
              "unavailable; build the CPU refs)", file=sys.stderr)
    elif failed > 0:
        print(f"dashboard: failed correctness: {failed} "
              f"({stats['validated']}/{stats['correctness_ops']} ops validated; "
              f"{failed} FAILED) - DASHBOARD REFLECTS REALITY (failures surfaced, "
              "not hidden)", file=sys.stderr)
    else:
        print(f"dashboard: failed correctness: 0 "
              f"({stats['correctness_ops']}/{stats['correctness_ops']} ops validated "
              "against golden contract C); self-contained HTML (no external "
              "dependencies)", file=sys.stderr)
    # Non-zero iff a correctness failure is present (the dashboard signals reality).
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
