#!/usr/bin/env python3
"""PolyKernel dataflow HTML visualizer (Todo 39 / Wave 7).

The `--backend=dataflow --viz` report: REUSES the Todo 38 dataflow metrics
(dataflow_report.py: lower_matmul_to_summa + build_report, the same analytic/cycle
model lib/DataflowSim/Metrics.cpp computes) and renders them into a STATIC,
SELF-CONTAINED HTML visualization (reports/dataflow_report.html) of the PE grid,
the SUMMA tile routes (A broadcast EAST / B broadcast SOUTH), the per-PE
SRAM-pressure heatmap, message traffic, fusion savings, and the critical path.

SCOPE GUARDRAIL (enforced by assert_self_contained): the HTML is a SINGLE
self-contained file - inline CSS + inline JS + an embedded #viz-data JSON blob.
NO external dependencies, NO network fetch, NO framework, NO build step; it
renders fully OFFLINE. The static SVG/HTML is pre-rendered here so the page works
with JS disabled; the inline JS adds per-PE hover tooltips + a payload check
(extended by Todo 43).

THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
hardware). CE + FMAC; fabin_dsd/fabout_dsd; @set_color_config rx/tx.

Usage:
    polykernel-report --backend=dataflow --viz examples/mlp_block.mlir
    python3 tools/polykernel-report/dataflow_viz.py --backend=dataflow --viz \\
        examples/mlp_block.mlir
"""

# allow: SIZE_OK - single-file static renderer: it pre-renders six SVG/HTML
# sections + the embedded JSON payload + the CLI. The deliverable scope forbids
# splitting into helper modules, and it mirrors the sibling dataflow_report.py.

# pyright: reportImplicitRelativeImport=false
# Executable SCRIPT (the `polykernel-report` wrapper puts this directory on
# sys.path); the top-level `import dataflow_report` reuses the Todo 38 metrics and
# is correct (a relative import would break the documented usage).

from __future__ import annotations

import argparse
import json
import math
import sys
from html import escape as _esc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataflow_report as dr  # noqa: E402  (reuse the Todo 38 metrics as-is)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = _PROJECT_ROOT / "lib" / "DataflowSim" / "viz_template.html"
_DEFAULT_OUT = _PROJECT_ROOT / "reports" / "dataflow_report.html"

# SVG grid geometry (px): square PE cells, gaps hold the route arrows, margins
# hold the west/north broadcast-injection arrows + labels.
_CELL, _GAP, _MARGIN = 60, 26, 58
_A, _B = "#3b82f6", "#f97316"  # route colours: A east (blue), B south (orange).

# Self-containment gate: tokens that would break offline rendering. Internal
# `url(#fragment)` SVG references are offline-safe and intentionally NOT listed.
_FORBIDDEN = ("http://", "https://", "cdn", "<script src", "<link ",
              "xlink", "xmlns", "@import", "url(http", "url(//")
# Structural QA targets (spec J): the embedded payload + the rendered sections.
_REQUIRED_IDS = ("viz-data", "pe-grid", "tile-routes", "sram-heatmap",
                 "message-traffic", "fusion-savings", "critical-path")


#===----------------------------------------------------------------------===#
# SVG / HTML primitives (static: pre-rendered so the page works with JS off).
#===----------------------------------------------------------------------===#

def _heat(pct: float) -> str:
    """Heatmap colour: green (0%) -> yellow (50%) -> red (>=100%), clamped."""
    t = max(0.0, min(float(pct), 100.0)) / 100.0
    green, yellow, red = (34, 197, 94), (234, 179, 8), (239, 68, 68)
    a, b, u = (green, yellow, t / 0.5) if t < 0.5 else (yellow, red, (t - 0.5) / 0.5)
    c = tuple(int(round(a[i] + (b[i] - a[i]) * u)) for i in range(3))
    return "#%02x%02x%02x" % c


def _xy(x: int, y: int) -> tuple[float, float]:
    """Top-left pixel of PE cell (x, y)."""
    return _MARGIN + x * (_CELL + _GAP), _MARGIN + y * (_CELL + _GAP)


def _grid_wh(p: int) -> int:
    return 2 * _MARGIN + p * _CELL + (p - 1) * _GAP


def _arrow(x1: float, y1: float, x2: float, y2: float, color: str, mk: str) -> str:
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="2.5" marker-end="url(#{mk})"/>')


def _cell(x: int, y: int, fill: str, ink: str, label: str, info: str = "") -> str:
    px, py = _xy(x, y)
    di = f' data-info="{info}"' if info else ""
    return (f'<rect class="pe-cell" data-pe="PE({x},{y})"{di} x="{px}" y="{py}" '
            f'width="{_CELL}" height="{_CELL}" rx="7" fill="{fill}" stroke="#26304d"/>'
            f'<text x="{px + _CELL / 2}" y="{py + _CELL / 2 + 4}" text-anchor="middle" '
            f'font-size="11" fill="{ink}">{label}</text>')


def _svg_open(w: int, h: int, label: str) -> str:
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{label}">')


def render_grid_svg(p: int) -> str:
    """PE grid + tile routes: A arrows east (blue), B arrows south (orange)."""
    w = _grid_wh(p)
    s = [_svg_open(w, w, "PE grid and SUMMA tile routes"),
         '<defs>'
         f'<marker id="ah-a" markerWidth="9" markerHeight="9" refX="7" refY="3" '
         f'orient="auto"><path d="M0,0 L7,3 L0,7 Z" fill="{_A}"/></marker>'
         f'<marker id="ah-b" markerWidth="9" markerHeight="9" refX="7" refY="3" '
         f'orient="auto"><path d="M0,0 L7,3 L0,7 Z" fill="{_B}"/></marker></defs>']
    for y in range(p):
        for x in range(p):
            s.append(_cell(x, y, "#0f1526", "#93a1c4", f"({x},{y})"))
    for y in range(p):  # A broadcast EAST: west-edge injection + inter-cell hops.
        cy = _xy(0, y)[1] + _CELL / 2
        s.append(_arrow(10, cy, _MARGIN - 8, cy, _A, "ah-a"))
        for x in range(p - 1):
            gx = _xy(x, y)[0] + _CELL
            s.append(_arrow(gx + 3, cy, gx + _GAP - 3, cy, _A, "ah-a"))
    for x in range(p):  # B broadcast SOUTH: north-edge injection + inter-cell hops.
        cx = _xy(x, 0)[0] + _CELL / 2
        s.append(_arrow(cx, 10, cx, _MARGIN - 8, _B, "ah-b"))
        for y in range(p - 1):
            gy = _xy(x, y)[1] + _CELL
            s.append(_arrow(cx, gy + 3, cx, gy + _GAP - 3, _B, "ah-b"))
    s.append(f'<text x="10" y="{_MARGIN - 14}" font-size="12" fill="#93c5fd">'
             f'A &#8594; EAST</text>')
    s.append(f'<text x="{_MARGIN + _CELL + _GAP}" y="22" font-size="12" '
             f'fill="#fdba74">B &#8595; SOUTH</text></svg>')
    return "".join(s)


def render_heatmap_svg(p: int, per_pe: list[float], m: dict) -> str:
    """Per-PE SRAM-pressure heatmap (colour-coded cells + gradient legend)."""
    resident, sram = m["resident_bytes"], m["sram_bytes"]
    w, h = _grid_wh(p), _grid_wh(p) + 60
    s = [_svg_open(w, h, "per-PE SRAM pressure heatmap"),
         '<defs><linearGradient id="hg" x1="0" y1="0" x2="1" y2="0">'
         f'<stop offset="0" stop-color="{_heat(0)}"/>'
         f'<stop offset="0.5" stop-color="{_heat(50)}"/>'
         f'<stop offset="1" stop-color="{_heat(100)}"/></linearGradient></defs>']
    for y in range(p):
        for x in range(p):
            pct = per_pe[y * p + x]
            ink = "#0b1020" if pct < 60 else "#ffffff"
            s.append(_cell(x, y, _heat(pct), ink, f"{pct:.1f}%",
                           f"PE({x},{y}): SRAM {pct:.2f}% ({resident:,} / {sram:,} B)"))
    ly = w + 12  # gradient legend bar + ticks below the grid.
    s.append(f'<rect x="{_MARGIN}" y="{ly}" width="{w - 2 * _MARGIN}" height="14" '
             f'rx="4" fill="url(#hg)" stroke="#26304d"/>')
    s.append(f'<text x="{_MARGIN}" y="{ly + 30}" font-size="11" fill="#93a1c4">0%</text>')
    s.append(f'<text x="{w / 2}" y="{ly + 30}" text-anchor="middle" font-size="11" '
             f'fill="#93a1c4">50%</text>')
    s.append(f'<text x="{w - _MARGIN}" y="{ly + 30}" text-anchor="end" font-size="11" '
             f'fill="#93a1c4">&#8805;100% over-subscription</text></svg>')
    return "".join(s)


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


#===----------------------------------------------------------------------===#
# Section renderers (summary / traffic / fusion / critical-path / legends).
#===----------------------------------------------------------------------===#

def render_summary(m: dict, f: dict) -> str:
    bn = m["bottleneck"]
    cls = {"active": "good", "delayed": "warn", "backpressure": "bad",
           "idle": "warn"}.get(bn, "warn")
    return "".join([
        _card("Grid utilization", f'{m["utilization_pct"]:.0f}%',
              f'{m["active_pes"]}/{m["total_pes"]} PEs active'),
        _card("SRAM pressure", f'{m["sram_pressure_pct"]:.2f}%',
              f'{m["resident_bytes"]:,} / {m["sram_bytes"]:,} B'),
        _card("Messages sent", f'{m["messages_sent"]:,}',
              f'A {m["a_wavelets"]:,} + B {m["b_wavelets"]:,}'),
        _card("Avg hop distance", f'{m["avg_hop_distance"]:.1f}', 'hops (1 cycle/hop)'),
        _card("Critical path", f'{m["critical_path_cycles"]:,}', 'cycles (analytic)'),
        _card("Bottleneck", _pill(bn, cls), f'cause: {m["bottleneck_cause"]}'),
        _card("Fusion reduction", f'{f["traffic_reduction_pct"]:.2f}%',
              f'{f["eliminated_wavelets"]:,} wavelets removed'),
    ])


def render_traffic(m: dict) -> str:
    mx = max(m["messages_sent"], 1)
    bars = "".join(_bar(lbl, 100.0 * m[key] / mx, color, f'{m[key]:,}')
                   for lbl, key, color in [
                       ("A wavelets (east broadcast)", "a_wavelets", _A),
                       ("B wavelets (south broadcast)", "b_wavelets", _B),
                       ("Total messages sent", "messages_sent", "#5b8cff")])
    return bars + _kv([
        ("Messages sent (CE fabin injections)", f'{m["messages_sent"]:,} wavelets'),
        ("Average hop distance", f'{m["avg_hop_distance"]:.1f} hops (1 cycle/hop)'),
        ("Panel computes fired", f'{m["panel_computes"]:,}')])


def render_fusion(f: dict) -> str:
    un, fu = f["unfused_wavelets"], f["fused_wavelets"]
    lmax = math.log10(max(un, fu) + 1) or 1.0
    lw = lambda v: 100.0 * math.log10(v + 1) / lmax  # noqa: E731 (log-scale bars)
    bars = (_bar("Unfused (intermediate round-trip + broadcast)", lw(un), "#ef4444",
                 f"{un:,}")
            + _bar("Fused (matmul broadcast only)", lw(fu), "#22c55e", f"{fu:,}"))
    return bars + _kv([
        ("Fused op", f'{_esc(f["fused_op"])} (from {_esc(f["fused_from"])})'),
        ("Eliminated intermediate", f'{_esc(f["eliminated_intermediate"])}'),
        ("Eliminated round-trip", f'{f["eliminated_round_trip_bytes"]:,} B'),
        ("Eliminated wavelets", f'{f["eliminated_wavelets"]:,}'),
        ("Traffic reduction", f'{f["traffic_reduction_pct"]:.4f}%')]) + (
        '<p class="legend">Fusion keeps the eliminated intermediate in local SRAM, so '
        'it crosses the fabric as <strong>zero</strong> wavelets. Bars log-scaled.</p>')


def render_critical(m: dict, p: int) -> str:
    fill, steps = p - 1, m["steps"]
    cp = max(m["critical_path_cycles"], 1)
    bars = (_bar("Broadcast fill (P&#8722;1 hops)", 100.0 * fill / cp, _A, f"{fill}")
            + _bar("Pipelined compute (steps)", 100.0 * steps / cp, _B, f"{steps}"))
    return bars + _kv([
        ("Critical-path cycles", f'{m["critical_path_cycles"]:,}'),
        ("Breakdown", f"broadcast fill (P&#8722;1 = {fill} hops) + pipelined compute "
                      f"({steps} steps) = {fill + steps}"),
        ("Panel computes", f'{m["panel_computes"]:,}'),
        ("Bottleneck", f'{m["bottleneck"]} (cause: {m["bottleneck_cause"]})')])


def route_legend(p: int) -> str:
    return (f'SUMMA mapping on a {p}&times;{p} grid. {_pill("A", "a")} panels broadcast '
            f'<strong>east</strong> along each row (injected at the west edge, '
            f're-emitted east by each PE); {_pill("B", "b")} panels broadcast '
            f'<strong>south</strong> along each column (injected at the north edge); '
            f'C accumulates <em>output-stationary</em> in each PE&rsquo;s local SRAM. '
            f'Ping-pong colours double-buffer A/B. Average broadcast span = '
            f'P&#8722;1 = {p - 1} hops.')


def sram_legend(m: dict) -> str:
    return ('Per-PE resident footprint = output-stationary C tile '
            f'(tileM&times;tileN) + ping-pong A/B double-buffers, vs the 48&nbsp;KB '
            f'SRAM. Colour scale green (low) &rarr; red (&ge;100% = over-subscription '
            f'&rarr; backpressure). This mapping: {m["resident_bytes"]:,} B / '
            f'{m["sram_bytes"]:,} B = {m["sram_pressure_pct"]:.2f}% (uniform across '
            f'the full grid).')


def generated_meta(report: dict) -> str:
    m, sh = report["dataflow_metrics"], report["dataflow_metrics"]["representative_shape"]
    return (f'input: <code>{_esc(report["input"])}</code> &middot; representative shape '
            f'{sh[0]}&times;{sh[1]}&times;{sh[2]} (M,N,K) on a '
            f'{m["grid_p"]}&times;{m["grid_p"]} grid &middot; SUMMA tiles '
            f'{m["tile_m"]}&times;{m["tile_n"]}&times;{m["tile_k"]}, {m["steps"]} steps')


#===----------------------------------------------------------------------===#
# Payload + template population + self-containment gate.
#===----------------------------------------------------------------------===#

def build_payload(report: dict, per_pe: list[float]) -> dict:
    """The embedded #viz-data JSON (single source of truth for the inline JS)."""
    m, f = report["dataflow_metrics"], report["fusion"]
    return {
        "generated_by": "tools/polykernel-report/dataflow_viz.py (Todo 39 / Wave 7)",
        "simulator": report["simulator"], "sdk_analogs": report["sdk_analogs"],
        "input": report["input"],
        "grid": {"p": m["grid_p"], "active_pes": m["active_pes"],
                 "total_pes": m["total_pes"], "utilization_pct": m["utilization_pct"]},
        "tiles": {"m": m["tile_m"], "n": m["tile_n"], "k": m["tile_k"],
                  "steps": m["steps"], "representative_shape": m["representative_shape"]},
        "routes": {"a": "EAST", "b": "SOUTH",
                   "scheme": "SUMMA: A broadcast east along rows, B broadcast south "
                             "along columns, output-stationary C in local SRAM"},
        "sram": {"bytes": m["sram_bytes"], "resident_bytes": m["resident_bytes"],
                 "pressure_pct": m["sram_pressure_pct"],
                 "per_pe_pressure_pct": per_pe},
        "traffic": {"a_wavelets": m["a_wavelets"], "b_wavelets": m["b_wavelets"],
                    "messages_sent": m["messages_sent"],
                    "avg_hop_distance": m["avg_hop_distance"],
                    "panel_computes": m["panel_computes"]},
        "critical_path": {"cycles": m["critical_path_cycles"],
                          "broadcast_fill_hops": m["grid_p"] - 1,
                          "pipelined_steps": m["steps"]},
        "bottleneck": {"state": m["bottleneck"], "cause": m["bottleneck_cause"]},
        "fusion": {k: f[k] for k in (
            "fused_op", "fused_from", "eliminated_intermediate",
            "eliminated_round_trip_bytes", "eliminated_wavelets",
            "unfused_wavelets", "fused_wavelets", "traffic_reduction_pct")},
    }


def render_html(report: dict) -> str:
    """Populate the self-contained template with the metrics + rendered sections."""
    m = report["dataflow_metrics"]
    p = m["grid_p"]
    per_pe = [m["sram_pressure_pct"]] * m["total_pes"]  # SUMMA: uniform across PEs.
    subs = {
        "__GENERATED_META__": generated_meta(report),
        "__SUMMARY_CARDS__": render_summary(m, report["fusion"]),
        "__PE_GRID_SVG__": render_grid_svg(p),
        "__ROUTE_LEGEND__": route_legend(p),
        "__SRAM_HEATMAP_SVG__": render_heatmap_svg(p, per_pe, m),
        "__SRAM_LEGEND__": sram_legend(m),
        "__TRAFFIC_HTML__": render_traffic(m),
        "__FUSION_HTML__": render_fusion(report["fusion"]),
        "__CRITICAL_HTML__": render_critical(m, p),
        "__VIZ_DATA__": json.dumps(build_payload(report, per_pe)),
    }
    html = _TEMPLATE.read_text()
    for k, v in subs.items():
        html = html.replace(k, v)
    return html


def assert_self_contained(html: str) -> None:
    """QA gate: no external refs (offline-safe) + all structural sections present."""
    low = html.lower()
    hits = [t for t in _FORBIDDEN if t in low]
    if hits:
        raise SystemExit(f"error: HTML is NOT self-contained; external refs: {hits}")
    missing = [i for i in _REQUIRED_IDS if f'id="{i}"' not in html]
    if missing:
        raise SystemExit(f"error: HTML missing required sections: {missing}")
    if "<svg" not in html:
        raise SystemExit("error: HTML has no rendered <svg> sections")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polykernel-report",
        description="Dataflow simulator HTML visualizer (self-contained static HTML: "
                    "PE grid / A-east+B-south routes / SRAM-pressure heatmap / message "
                    "traffic / fusion savings / critical path).")
    parser.add_argument("mlir", nargs="?", default="examples/mlp_block.mlir",
                        help="MLIR file to visualize dataflow metrics for")
    parser.add_argument("--backend", default="dataflow", choices=["dataflow"],
                        help="report backend (dataflow simulator metrics)")
    parser.add_argument("--viz", action="store_true",
                        help="generate the self-contained HTML viz (this tool's action; "
                             "accepted for the documented --backend=dataflow --viz form)")
    parser.add_argument("--shape", default="32,32,32",
                        help="representative square SUMMA matmul shape M,N,K")
    parser.add_argument("--grid", type=int, default=4, help="SUMMA grid dimension P")
    parser.add_argument("--tile-k", type=int, default=0, help="K-slab width (0 = auto)")
    parser.add_argument("--opt", default=None, help="path to polykernel-opt")
    parser.add_argument("--html-out", default=str(_DEFAULT_OUT),
                        help="output HTML path (default reports/dataflow_report.html)")
    parser.add_argument("--no-write", action="store_true",
                        help="run the gate but do not write the HTML output")
    args = parser.parse_args(argv)

    vals = [int(x) for x in args.shape.split(",")]
    if len(vals) != 3:
        raise SystemExit(f"error: --shape must be M,N,K (got '{args.shape}')")
    plan = dr.lower_matmul_to_summa(vals[0], vals[1], vals[2], args.grid, args.tile_k)

    # Reuse the Todo 38 metrics (dataflow_report.py) verbatim - do NOT reinvent.
    report = dr.build_report(dr.find_opt(args.opt), Path(args.mlir), plan)
    html = render_html(report)
    assert_self_contained(html)  # self-containment + structural QA gate.

    if not args.no_write:
        Path(args.html_out).write_text(html)
        print(f"wrote {args.html_out}", file=sys.stderr)

    m, red = report["dataflow_metrics"], report["fusion"]["traffic_reduction_pct"]
    print(f"dataflow viz: {m['grid_p']}x{m['grid_p']} grid, "
          f"SRAM pressure {m['sram_pressure_pct']:.2f}%, "
          f"{m['messages_sent']} wavelets, critical path "
          f"{m['critical_path_cycles']} cycles, fusion reduction {red:.4f}%; "
          f"self-contained (no external refs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
