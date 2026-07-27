# PolyKernel Architecture

PolyKernel is a self-contained, **out-of-tree C++/MLIR compiler + runtime** that
lowers a fused transformer-block fragment (the MLP block: RMSNorm → MatMul → GELU →
MatMul → residual Add, plus the attention fragment) into **portable CUDA/HIP
kernels**, analyzes them at compile time, autotunes them, simulates a
**Cerebras-style dataflow** mapping, and deploys the engine on **Modal**.

This document is the map. The depth lives in the per-area docs:

| Area | Doc |
|---|---|
| MLIR pass pipeline + build wiring | [`compiler_pipeline.md`](compiler_pipeline.md) |
| CUDA backend + GPU-free analyzer | [`cuda_backend.md`](cuda_backend.md) |
| HIP/ROCm backend (local gfx1101 + gfx942) | [`hip_backend.md`](hip_backend.md) |
| Cerebras-style dataflow simulator | [`dataflow_backend.md`](dataflow_backend.md) |
| Roofline performance model + projections | [`performance_model.md`](performance_model.md) |

## What it is (and is not)

- **Is:** a compiler/runtime *machinery* demonstration — dialect + passes + codegen
  + compile-time analysis + autotuning + a dataflow simulator + serving. Correctness
  is validated against a NumPy/`ml_dtypes` golden (bf16 round / fp32 accumulate;
  cosine ≥ 0.999, max rel err ≤ 1e-2, PCC ≥ 0.99); **0 failed correctness tests** is
  a success criterion.
- **Is not:** a claim to beat vLLM or any production stack (the goal is the
  machinery, not SOTA). A real Cerebras toolchain (the dataflow backend is a
  functional/cycle **simulator**, not CSL/`cslc`/hardware). A full PyTorch/ONNX
  frontend (input is hand-written `.mlir`). An op zoo (only the named ops).

## System layers

```
                         hand-written .mlir (polykernel dialect)
                                     │
        ┌────────────────────────────┴────────────────────────────┐
        │  Compiler (C1/C2)  —  polykernel-opt + 11 passes          │
        │  infer-shapes · canonicalize · fuse-* · infer-tile-layout │
        │  plan-memory · lower-to-cuda / lower-to-hip · emit-report │
        └───────────────┬───────────────────────────┬───────────────┘
                        │                           │
        ┌───────────────┴───────────┐   ┌───────────┴────────────────┐
        │  CUDA backend (C3a)        │   │  HIP/ROCm backend (C3b)     │
        │  nvcc sm_80/sm_90 + PTX    │   │  hipcc gfx1101 (local run)  │
        │  GPU-free ptxas analyzer   │   │  gfx942 (MI300 cross-compile)│
        │  (regs/smem/occ/roofline)  │   │  AMDGPU ISA + WMMA bf16     │
        └───────────────┬───────────┘   └───────────┬────────────────┘
                        │   shared portable kernel template          │
                        │   (#ifdef POLYKERNEL_CUDA / POLYKERNEL_HIP) │
        ┌───────────────┴───────────────────────────┴───────────────┐
        │  Autotuner + Runtime (C4)                                    │
        │  bounded config grid · correctness-gated bench · JSON cache  │
        │  C++ runtime: detect GPU → select best kernel → load → serve │
        └───────────────┬─────────────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐   ┌────────────────────────────┐
        │  Modal deployment (C6)         │   │  Dataflow simulator (C5)    │
        │  CUDA-toolkit image · /predict │   │  64×64 PE grid (CE+FMAC)    │
        │  /benchmark /kernels /report   │   │  SUMMA matmul (A-east/B-south)│
        │  cold-start · kernel cache     │   │  metrics + self-contained viz│
        └────────────────────────────────┘   └────────────────────────────┘
```

## The compiler (C1/C2)

A custom `polykernel` MLIR dialect defines **exactly** the named transformer-inference
op set (closed: 11 base + 6 fused + the `func`/`return` container) — see
`include/PolyKernel/IR/PolyKernelOps.td`. Eleven passes
(`include/PolyKernel/Passes/Passes.td`) transform the IR:

`--infer-shapes` → `--canonicalize` → `--fuse-rmsnorm-matmul` →
`--fuse-matmul-bias-gelu` → `--fuse-residual-rmsnorm` → `--fuse-softmax-mask` →
`--infer-tile-layout` → `--plan-memory` → `--lower-to-cuda` / `--lower-to-hip` →
`--emit-kernel-report`.

The fusion passes tag each fused op with discardable attributes
(`polykernel.fused_from`, `polykernel.eliminated_type[_N]`) that the traffic report
(`tools/polykernel-report/traffic_report.py`) reads to quantify the eliminated global
round-trips. `--lower-to-cuda`/`--lower-to-hip` are **custom source emitters** (Pinned
contract F): they walk the IR and string-emit portable `.cu` against
`kernels/template/kernel_common.h` — they do **not** use `gpu-to-llvm`/`convert-to-llvm`.
Detail in [`compiler_pipeline.md`](compiler_pipeline.md).

**Tools:** `polykernel-opt` (pass driver), `polykernel-translate` (source emit),
`polykernel-bench` (autotuner + analyzer CLI), `polykernel-report` (traffic / dataflow /
per-kernel / dashboard reports), `polykernel-analyze` (standalone compile-time analyzer).

## The CUDA backend + analyzer (C3a)

Generated kernels (RMSNorm, GELU/SiLU, MatMul, Softmax, fused RMSNorm+MatMul, fused
MatMul+Bias+GELU, + a WMMA variant) use shared-memory tiling, vectorized loads, warp
reductions, and occupancy-aware block sizing. `nvcc` compiles for **sm_80** (A100) and
**sm_90** (H100) and emits PTX. The **compile-time analyzer** (`lib/Analysis/`:
`PtxasParser`, `Occupancy`, `Roofline`, `KernelReport`) parses `ptxas -v`
(registers/smem/spills), computes occupancy from the sm_80/sm_90 constants table, and
classifies the roofline — **all with no NVIDIA GPU present**. Detail in
[`cuda_backend.md`](cuda_backend.md).

## The HIP/ROCm backend (C3b)

The **same** portable template compiles under `hipcc --offload-arch=gfx1101` unchanged
(the portability proof) and **runs on the local RX 7800 XT** (gfx1101, officially
supported since ROCm 7.0). The AMDGPU ISA analyzer (`lib/Analysis/AmdIsaAnalyzer.cpp`)
extracts VGPR/SGPR/LDS/scratch + gfx1101 occupancy; a WMMA bf16 tensor path
(`v_wmma_f32_16x16x16_bf16`) is an additive, correctness-gated variant. `gfx942`
(MI300) is a compile-only cross-compile target for the reports. Detail in
[`hip_backend.md`](hip_backend.md).

## Correctness (the no-NVIDIA-GPU story)

CUDA kernels cannot run locally, so correctness is established three ways: (1) the CUDA
and HIP backends share **one** portable template, so the shared compute logic is
validated by running the HIP build (and the CPU-reference build) on the RX 7800 XT
against the golden; (2) CUDA-specific tensor-core paths are compile-validated locally
and fully validated on rented H100/A100; (3) the dataflow simulator **functionally
executes** the scheduled tile program and is compared to the same golden. The golden
(`tests/golden/`) uses NumPy + `ml_dtypes.bfloat16` only, with the pinned rounding
contract (bf16 RNE inputs, fp32 accumulate, bf16 output) and thresholds. The end-to-end
harness `tests/e2e/test_mlp_correctness.py` reports `N failed / M ops` and is the source
the benchmark dashboard reads for its **failed correctness** stat.

## Autotuner + runtime (C4)

`lib/Autotune/` enumerates a **bounded** config grid (BLOCK_M/N/K, num_warps,
vector_width, unroll, shared-memory stages; pruned to ~100–150 configs), benchmarks each
variant **correctness-gated** (a variant's time is recorded only if it first passes the
golden), and writes a versioned JSON tuning cache keyed by (GPU, op, shape) with a
`scored_by` field (`real-benchmark` vs `compile-time-model`) so reports label
projected-vs-real. The C++ runtime (`lib/Runtime/`: `DeviceDetect`, `KernelCache`,
`Runtime`) detects the GPU arch and loads the best cached kernel.

## The dataflow simulator (C5)

`lib/DataflowSim/` is a self-contained C++ **functional/cycle simulator** of a
Cerebras-style fabric: a 64×64 PE grid (each PE = a CE with FMACs + a 5-port router +
48 KB SRAM), 32-bit wavelets, 5-bit colors (24 routable), wavelet/color routing
(`@set_color_config` rx/tx), and dataflow-triggered compute (`@activate`/`@block`/
`@unblock` rendezvous). The MLP/matmul maps onto the **SUMMA** schedule (A broadcast
east, B broadcast south, output-stationary C in local SRAM). It reports grid
utilization, SRAM pressure, messages sent, average hop distance, communication
bottleneck, critical-path cycles, and fusion traffic reduction, and renders a
**self-contained** HTML visualizer (`reports/dataflow_report.html`). It is **not** real
CSL and **not** Cerebras hardware. Detail in [`dataflow_backend.md`](dataflow_backend.md).

## Modal deployment (C6)

`modal/image.py` defines a CUDA-toolkit image (`nvidia/cuda:12.8.1-devel-ubuntu24.04`,
`add_python="3.12"`) that compiles the engine at image-build time. `modal/app.py`
defines the app + GPU endpoints (`/predict`, `/benchmark`, `/kernels`, `/report`) via
`@modal.fastapi_endpoint`, cold-start measurement (`min_containers`/`scaledown_window`/
`enable_memory_snapshot` + `@modal.enter(snap=)`), and a GPU-aware kernel cache that
reuses the C4 runtime. Cloud deploy is **owner-gated** behind `MODAL_TOKEN`; without a
token the full app + local `modal serve` dev path exist and cloud deploy is marked
SKIPPED (never FAILED).

## Reports + dashboard

`tools/polykernel-report/dashboard.py --dashboard` aggregates the committed per-backend
reports — CUDA (`reports/h100_bench.json` + `reports/kernel_report_example.json`), AMD
(`reports/mi300_bench.json`), dataflow (`reports/dataflow_metrics.json`) — plus compiler
stats (passes / generated kernels / validated / **failed correctness**) into
`reports/benchmark_report.{md,html}` and the per-backend `reports/h100_report.html` /
`reports/mi300_report.html`. Every HTML page is **static + self-contained** (inline
CSS/JS, no external dependencies, renders offline). The H100/A100/MI300 speedups are
**PROJECTED** (roofline + analytic traffic) unless a RunPod rental ran; the dataflow
figures are the simulator's model. See [`performance_model.md`](performance_model.md).

## Directory layout

```
include/PolyKernel/   IR/ (dialect+ops .td), Passes/ (Passes.td + headers),
                      Analysis/, Autotune/, Runtime/, Dataflow/
lib/                  IR/, Passes/, Analysis/, Autotune/, Runtime/, DataflowSim/
kernels/              generated/*.cu (emitted), template/kernel_common.h, cpu/ (refs)
tools/                polykernel-opt/, polykernel-bench/, polykernel-report/
modal/                image.py, app.py, requirements.txt
tests/                golden/, kernels/, e2e/, autotune/   test/ (lit *.mlir)
reports/              per-backend bench JSON/MD + HTML reports + evidence logs
docs/                 this document + per-area docs
```
