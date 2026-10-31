#!/usr/bin/env python3
"""Generate the PolyKernel report figures as Excalidraw scenes.

Every figure is emitted as a .excalidraw file next to this script and rendered
to PNG with the Excalidraw CLI. The drawing style is deliberately hand drawn:
roughness 2, hachure fills, Virgil, and coloured callouts.

Usage:
    python3 report/figures/gen_figures.py
    EXCALIDRAW=/path/to/excalidraw python3 report/figures/gen_figures.py --render
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

INK = "#1e1e1e"
MUTED = "#5c5f66"

BLUE = "#1971c2"
BLUE_F = "#d0ebff"
ORANGE = "#e8590c"
ORANGE_F = "#ffe8cc"
GREEN = "#2f9e44"
GREEN_F = "#d3f9d8"
RED = "#e03131"
RED_F = "#ffe3e3"
PURPLE = "#9c36b5"
PURPLE_F = "#f3d9fa"
YELLOW = "#f08c00"
YELLOW_F = "#fff3bf"
CYAN = "#0c8599"
CYAN_F = "#c5f6fa"
GRAY = "#495057"
GRAY_F = "#f1f3f5"


class Canvas:
    def __init__(self, width: int, height: int, seed: int = 7):
        self.width = width
        self.height = height
        self.elements: list[dict] = []
        self.rng = random.Random(seed)
        self._n = 0

    # -- low level ---------------------------------------------------------
    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _base(self, kind: str, x: float, y: float, w: float, h: float) -> dict:
        return {
            "id": self._id(kind[:1]),
            "type": kind,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "angle": 0,
            "strokeColor": INK,
            "backgroundColor": "transparent",
            "fillStyle": "hachure",
            "strokeWidth": 2,
            "strokeStyle": "solid",
            "roughness": 2,
            "opacity": 100,
            "groupIds": [],
            "frameId": None,
            "roundness": None,
            "seed": self.rng.randint(1, 2**31 - 1),
            "version": 1,
            "versionNonce": self.rng.randint(1, 2**31 - 1),
            "isDeleted": False,
            "boundElements": None,
            "updated": 1,
            "link": None,
            "locked": False,
        }

    def _text_el(
        self,
        x: float,
        y: float,
        text: str,
        size: int,
        color: str,
        align: str,
        width: float,
        container: str | None = None,
        family: int = 1,
    ) -> dict:
        lines = text.split("\n")
        height = len(lines) * size * 1.25
        el = self._base("text", x, y, width, height)
        el.update(
            {
                "strokeColor": color,
                "backgroundColor": "transparent",
                "fillStyle": "solid",
                "roughness": 1,
                "text": text,
                "fontSize": size,
                "fontFamily": family,
                "textAlign": align,
                "verticalAlign": "middle" if container else "top",
                "containerId": container,
                "originalText": text,
                "lineHeight": 1.25,
                "baseline": int(size * 0.85),
            }
        )
        return el

    # -- shapes ------------------------------------------------------------
    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        stroke: str = INK,
        fill: str = "transparent",
        size: int = 20,
        color: str = INK,
        dashed: bool = False,
        rounded: bool = True,
        fill_style: str = "hachure",
        stroke_width: int = 2,
        align: str = "center",
        family: int = 1,
    ) -> str:
        el = self._base("rectangle", x, y, w, h)
        el.update(
            {
                "strokeColor": stroke,
                "backgroundColor": fill,
                "fillStyle": fill_style,
                "strokeWidth": stroke_width,
                "strokeStyle": "dashed" if dashed else "solid",
                "roundness": {"type": 3} if rounded else None,
            }
        )
        if text:
            tid = self._id("t")
            el["boundElements"] = [{"type": "text", "id": tid}]
            el["id"] = f"r{self._n}" if el["id"].startswith("r") else el["id"]
            style = el["id"]
            self.elements.append(el)
            t = self._text_el(
                x + 10,
                y + h / 2 - (len(text.split("\n")) * size * 1.25) / 2,
                text,
                size,
                color,
                align,
                max(w - 20, 20),
                container=style,
                family=family,
            )
            t["id"] = tid
            self.elements.append(t)
            return style
        self.elements.append(el)
        return el["id"]

    def text(
        self,
        x: float,
        y: float,
        text: str,
        size: int = 18,
        color: str = INK,
        align: str = "left",
        width: float = 400,
        family: int = 1,
    ) -> None:
        self.elements.append(
            self._text_el(x, y, text, size, color, align, width, family=family)
        )

    def arrow(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = INK,
        dashed: bool = False,
        width: int = 2,
        head: bool = True,
        label: str = "",
        label_size: int = 15,
        label_color: str | None = None,
        curve: bool = False,
        label_dx: float = 0.0,
        label_dy: float = -34.0,
    ) -> None:
        el = self._base("arrow", x1, y1, abs(x2 - x1), abs(y2 - y1))
        mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        pts = [[0.0, 0.0], [x2 - x1, y2 - y1]]
        if curve:
            pts = [
                [0.0, 0.0],
                [(x2 - x1) * 0.5, (y2 - y1) * 0.15],
                [x2 - x1, y2 - y1],
            ]
        el.update(
            {
                "points": pts,
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": "arrow" if head else None,
                "strokeColor": color,
                "strokeWidth": width,
                "strokeStyle": "dashed" if dashed else "solid",
                "roundness": {"type": 2} if curve else None,
            }
        )
        self.elements.append(el)
        if label:
            self.text(
                mid[0] - 90 + label_dx,
                mid[1] + label_dy,
                label,
                label_size,
                label_color or color,
                align="center",
                width=180,
            )

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        stroke: str = INK,
        fill: str = "transparent",
        size: int = 18,
        color: str = INK,
    ) -> None:
        el = self._base("ellipse", x, y, w, h)
        el.update({"strokeColor": stroke, "backgroundColor": fill})
        if text:
            tid = self._id("t")
            el["id"] = f"e{self._n}"
            el["boundElements"] = [{"type": "text", "id": tid}]
            self.elements.append(el)
            t = self._text_el(
                x + 6,
                y + h / 2 - (len(text.split("\n")) * size * 1.25) / 2,
                text,
                size,
                color,
                "center",
                w - 12,
                container=el["id"],
            )
            t["id"] = tid
            self.elements.append(t)
            return
        self.elements.append(el)

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        color: str = INK,
        width: int = 2,
        dashed: bool = False,
    ) -> None:
        el = self._base("line", x1, y1, abs(x2 - x1), abs(y2 - y1))
        el.update(
            {
                "points": [[0.0, 0.0], [x2 - x1, y2 - y1]],
                "lastCommittedPoint": None,
                "startBinding": None,
                "endBinding": None,
                "startArrowhead": None,
                "endArrowhead": None,
                "strokeColor": color,
                "strokeWidth": width,
                "strokeStyle": "dashed" if dashed else "solid",
            }
        )
        self.elements.append(el)

    def save(self, path: Path) -> None:
        doc = {
            "type": "excalidraw",
            "version": 2,
            "source": "https://excalidraw.com",
            "elements": self.elements,
            "appState": {
                "gridSize": None,
                "viewBackgroundColor": "#ffffff",
            },
            "files": {},
        }
        path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"wrote {path} ({len(self.elements)} elements)")


# ---------------------------------------------------------------------------
# Figure 1: system pipeline
# ---------------------------------------------------------------------------


def figure_pipeline() -> Canvas:
    c = Canvas(1520, 940, seed=11)
    c.text(40, 28, "One fragment, two GPU backends, two models", 32)
    c.text(
        40,
        74,
        "Solid: built and exercised. Dashed: model, gate, or unverified.",
        18,
        MUTED,
    )

    # Source
    c.rect(
        40,
        165,
        250,
        210,
        "examples/\nmlp_block.mlir\n\n5-op MLP fragment",
        BLUE,
        BLUE_F,
        22,
    )
    c.arrow(290, 270, 345, 270, BLUE, width=3)

    # Compiler
    c.rect(350, 140, 470, 420, "", BLUE, "#f8f9fa", 20)
    c.text(375, 160, "polykernel-opt", 28, BLUE)
    c.text(375, 200, "12 passes", 20, MUTED, width=420)
    c.text(375, 250, "shape and dtype inference", 22, INK, width=430)
    c.text(375, 290, "five single-use fusion passes", 22, INK, width=430)
    c.text(375, 330, "tile layout and memory plan", 22, INK, width=430)
    c.text(375, 370, "custom CUDA/HIP source emit", 22, INK, width=430)
    c.text(375, 440, "no gpu-to-llvm, no LLVM lowering", 18, MUTED, width=430)

    # Backends
    c.arrow(820, 270, 895, 240, BLUE, width=3)
    c.arrow(820, 400, 895, 585, BLUE, width=3)
    c.rect(
        900,
        165,
        300,
        150,
        "CUDA\nnvcc sm_80/89/90\nPTX + ptxas",
        ORANGE,
        ORANGE_F,
        22,
    )
    c.text(900, 320, "measured on RTX 6000 Ada", 18, ORANGE, width=300)
    c.rect(
        900,
        520,
        300,
        150,
        "HIP\nhipcc gfx1101/942\nAMDGPU ISA",
        GREEN,
        GREEN_F,
        22,
    )
    c.text(900, 675, "measured on RX 7800 XT", 18, GREEN, width=300)

    # Template chain
    c.rect(
        1270,
        300,
        210,
        140,
        "one template\nkernel_common.h",
        PURPLE,
        PURPLE_F,
        21,
    )
    c.arrow(1200, 240, 1270, 345, ORANGE, width=3)
    c.arrow(1200, 595, 1270, 420, GREEN, width=3)
    c.rect(
        1270,
        480,
        210,
        110,
        "autotuner\nJSON cache",
        YELLOW,
        YELLOW_F,
        21,
    )
    c.arrow(1375, 440, 1375, 480, PURPLE, width=3)
    c.rect(
        1270,
        630,
        210,
        110,
        "runtime\ndetect, launch",
        CYAN,
        CYAN_F,
        21,
    )
    c.arrow(1375, 590, 1375, 630, YELLOW, width=3)
    c.rect(
        1270,
        780,
        210,
        110,
        "Modal: gated\nnot deployed",
        GRAY,
        GRAY_F,
        19,
        dashed=True,
    )
    c.arrow(1375, 740, 1375, 780, GRAY, dashed=True, width=2)

    # Analyzer
    c.rect(
        350,
        620,
        470,
        180,
        "GPU-free analyzer\nptxas, ISA, roofline\ncontract-H report",
        YELLOW,
        YELLOW_F,
        22,
    )
    c.arrow(1040, 315, 825, 690, YELLOW, dashed=True, width=2)
    c.arrow(900, 610, 825, 720, YELLOW, dashed=True, width=2)

    # Dataflow
    c.rect(
        40,
        450,
        250,
        290,
        "dataflow model\n\n4x4 SUMMA\nwavelets, 24 colors\n\nnot CSL, not silicon",
        RED,
        RED_F,
        20,
        dashed=True,
    )
    c.arrow(165, 375, 165, 450, RED, dashed=True, width=2)

    c.text(
        40,
        900,
        "Status: CUDA and HIP ran on two GPUs. H100, A100, MI300X and Modal did not.",
        18,
        MUTED,
        width=1400,
    )
    return c


# ---------------------------------------------------------------------------
# Figure 2: fusion, traffic, measurement
# ---------------------------------------------------------------------------


def figure_fusion() -> Canvas:
    c = Canvas(1560, 920, seed=23)
    c.text(40, 30, "Fusion saves 32 MiB and costs 10.75 GiB", 34)
    c.text(
        40,
        78,
        "RTX 6000 Ada, bf16, M=2048 K=4096 N=11008. CUDA-event medians.",
        17,
        MUTED,
    )

    # Unfused
    c.text(60, 130, "unfused: two kernels", 22, BLUE)
    c.rect(60, 190, 220, 130, "RMSNorm\nkernel", BLUE, BLUE_F, 21)
    c.rect(430, 190, 220, 130, "MatMul\nkernel", ORANGE, ORANGE_F, 21)
    c.arrow(280, 228, 430, 228, INK, width=3, label="write 16 MiB", label_size=15, label_dy=-30)
    c.arrow(430, 292, 280, 292, INK, width=3, label="read 16 MiB", label_size=15, label_dy=8)
    c.text(70, 350, "one global round trip: 32 MiB", 18, RED, width=420)
    c.text(70, 385, "184.7 GFLOP per matmul", 15, MUTED, width=420)

    # Fused
    c.text(780, 130, "fused: one kernel", 22, RED)
    c.rect(
        780,
        190,
        480,
        140,
        "fused_rmsnorm_matmul\nRMS recomputed for every 16-wide\noutput tile: 688 times per row",
        RED,
        RED_F,
        20,
    )
    c.text(790, 350, "re-reads X: 10.75 GiB", 20, RED, width=460)
    c.text(790, 388, "the 32 MiB saving is 0.29% of that", 15, MUTED, width=460)

    # Measured bars
    c.text(60, 448, "measured wall time, rmsnorm + matmul", 22)
    base_y = 800
    bars = [
        (90, 300, "unfused", 25.52, "1.000x", BLUE, BLUE_F),
        (470, 300, "fused", 30.46, "0.838x, 19.6% slower", RED, RED_F),
        (850, 300, "autotuned (min of runs)", 29.79, "0.857x", YELLOW, YELLOW_F),
    ]
    scale = 10.0  # pixels per ms
    for x, w, label, ms, sp, stroke, fill in bars:
        h = ms * scale
        font = 20 if len(sp) > 8 else 22
        c.rect(x, base_y - h, w, h, f"{ms:.2f} ms\n{sp}", stroke, fill, font)
        c.text(x, base_y + 14, label, 17, MUTED, width=w, align="center")
    c.line(60, base_y, 1240, base_y, INK, 3)

    # Footer
    c.text(
        60,
        865,
        "The fused kernel removes one 32 MiB round trip, then adds one RMS pass over X per output tile. "
        "At this shape the second cost dominates.",
        16,
        MUTED,
        width=1450,
    )
    return c


# ---------------------------------------------------------------------------
# Figure 3: projection arithmetic
# ---------------------------------------------------------------------------


def figure_projection() -> Canvas:
    c = Canvas(1460, 880, seed=31)
    c.text(40, 30, "The 2.84x is a ratio of attainment factors", 32)
    c.text(
        40,
        78,
        "Same roofline, same numerator: only eta changes between the two variants.",
        17,
        MUTED,
    )

    # Formula
    c.rect(
        40,
        150,
        520,
        280,
        "t = max(FLOPs / peak, bytes / BW) / eta\n\n"
        "scalar baseline   eta = 0.12\n"
        "tuned variant      eta = 0.35\n\n"
        "0.35 / 0.12 = 2.92",
        PURPLE,
        PURPLE_F,
        20,
    )
    c.text(
        60,
        455,
        "The autotuned projection is the same model with a\n"
        "different constant. No measured kernel enters it.",
        16,
        MUTED,
        width=490,
    )

    # Bars: effective TFLOPS
    c.text(650, 132, "effective TFLOPS on the up-projection", 22)
    left = 650
    unit = 2.1  # pixels per TFLOP/s
    rows = [
        ("H100 model (0.12 x 989)", 118.7, "peak 66.9"),
        ("A100 model (0.12 x 312)", 37.4, "peak 19.5"),
        ("MI300X model (0.12 x 1307)", 156.9, "peak 163.4"),
        ("Ada measured", 7.24, "7.9% of 91.1"),
    ]
    for i, (label, val, note) in enumerate(rows):
        y = 265 + i * 150
        stroke, fill = (GREEN, GREEN_F) if i == 3 else (ORANGE, ORANGE_F)
        c.text(left, y - 65, label, 17, INK, width=390)
        w = max(6.0, val * unit)
        c.rect(left + 410, y, w, 46, "", stroke, fill, 16)
        c.text(left + 420 + w, y + 8, f"{val:.1f}", 17, stroke, width=90)
        c.text(left + 410, y + 52, note, 14, MUTED, width=220)

    # H100 FP32 ceiling
    xpeak = left + 410 + 66.9 * unit
    c.line(xpeak, 205, xpeak, 775, RED, 3, dashed=True)
    c.text(xpeak + 8, 205, "H100 FP32 vector peak 66.9", 14, RED, width=260)

    c.text(
        40,
        820,
        "A 16x16 tiled scalar kernel cannot exceed the FP32 vector peak. "
        "The H100 and A100 projections assume more than that peak.",
        16,
        MUTED,
        width=1350,
    )
    return c


# ---------------------------------------------------------------------------
# Figure 4: dataflow SUMMA
# ---------------------------------------------------------------------------


def figure_dataflow() -> Canvas:
    c = Canvas(1500, 1000, seed=41)
    c.text(40, 30, "The dataflow backend is a 4x4 SUMMA model", 32)
    c.text(
        40,
        78,
        "Functional and cycle model. It is not CSL, not cslc, and not Cerebras silicon.",
        17,
        MUTED,
    )

    # Grid
    gx, gy, cell, gap = 120, 225, 100, 14
    for r in range(4):
        for col in range(4):
            x = gx + col * (cell + gap)
            y = gy + r * (cell + gap)
            stroke = BLUE if r == 0 else (GREEN if col == 0 else GRAY)
            fill = BLUE_F if r == 0 else (GREEN_F if col == 0 else GRAY_F)
            c.rect(x, y, cell, cell, f"PE\n{r},{col}", stroke, fill, 14)

    # A east, B south
    c.text(gx + 30, gy - 65, "A east", 17, BLUE, width=160)
    c.arrow(gx + 30, gy - 30, gx + 3 * (cell + gap) + 70, gy - 30, BLUE, width=3)
    c.text(0, gy - 55, "B south", 17, GREEN, width=105)
    c.arrow(gx - 40, gy + 10, gx - 40, gy + 3 * (cell + gap) + 90, GREEN, width=3)

    # Steps annotation
    c.text(
        gx,
        gy + 4 * (cell + gap) + 28,
        "step i: column i sends A east, row i sends B south.\n"
        "Each PE keeps C in local SRAM. 4 steps, 2,048 wavelets.",
        17,
        MUTED,
        width=650,
    )

    # PE zoom
    zx = 800
    c.rect(zx, 200, 650, 460, "", PURPLE, PURPLE_F, 20)
    c.text(zx + 22, 215, "one PE", 24, PURPLE)
    c.text(zx + 22, 258, "CE + FMAC, fp32", 18, INK)
    c.text(zx + 22, 294, "5-port router: RAMP, E, W, N, S", 18, INK)
    c.text(zx + 22, 330, "48 KB SRAM: 8 banks x 6 KB", 18, INK)
    c.text(zx + 22, 366, "wavelet: 32 bit, 1 hop per cycle", 18, INK)
    c.text(zx + 22, 402, "color: 5 bit, 24 routable", 18, INK)
    c.rect(
        zx + 22,
        450,
        606,
        100,
        "@activate + @block + @unblock\ncompute fires only when both inputs arrive",
        RED,
        RED_F,
        18,
    )
    c.text(
        zx + 22,
        570,
        "Metrics: 100% utilization, 2.60% SRAM,\nhop distance 3.0, critical path 7 cycles.",
        16,
        MUTED,
        width=600,
    )
    c.arrow(gx + 4 * (cell + gap) + 15, gy + 200, zx - 10, 330, PURPLE, dashed=True, width=2)

    # Critique
    c.rect(
        40,
        790,
        1400,
        165,
        "The '99.99% fusion traffic reduction' compares 16.7M counterfactual wavelets\n"
        "for the eliminated 32 MiB intermediate against 2,048 wavelets from a 32x32x32\n"
        "toy matmul. The unfused baseline assumes that intermediate crosses the fabric.\n"
        "It is a definition, not a measurement.",
        RED,
        RED_F,
        19,
        dashed=True,
    )
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    figs = {
        "system-pipeline": figure_pipeline,
        "fusion-traffic": figure_fusion,
        "projection-gap": figure_projection,
        "dataflow-summa": figure_dataflow,
    }
    for name, fn in figs.items():
        scene = HERE / f"{name}.excalidraw"
        fn().save(scene)
        if not args.render:
            continue
        exe = os.environ.get("EXCALIDRAW", "excalidraw")
        png = HERE / f"{name}.png"
        cmd = [exe, "render", str(scene), "--out", str(png), "--scale", "2",
               "--max-width", "3200", "--max-height", "2000"]
        if os.environ.get("EXCALIDRAW_BROWSER"):
            cmd += ["--browser-path", os.environ["EXCALIDRAW_BROWSER"]]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
