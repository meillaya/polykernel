# PolyKernel

### An out-of-tree MLIR compiler that fuses LLM ops into portable CUDA/HIP kernels, autotunes them, simulates a Cerebras-style dataflow fabric, and deploys the engine on Modal GPUs.

**A compiler + runtime + accelerator simulator in one repo.** PolyKernel lowers a fused
transformer-block fragment (RMSNorm → MatMul → GELU → MatMul → residual Add, plus the
attention fragment) through a custom MLIR dialect into portable GPU kernels, analyzes them
at compile time, autotunes them, maps them onto a dataflow fabric, and serves them — built
to demonstrate depth across **GPU kernels, compiler/MLIR lowering, CUDA/PTX, ROCm/HIP,
accelerator-aware scheduling, and inference serving: the overlap that NVIDIA, AMD, Cerebras,
and Modal hire for.**

---

## Resume bullets

- **Out-of-tree C++/MLIR compiler** — a custom `polykernel` dialect (exactly the named
  transformer-inference ops + fused variants) with **11 passes** (shape inference,
  canonicalization, four operator-fusion passes, tile-layout, memory planning, CUDA/HIP
  lowering, kernel-report emission), built against nixpkgs **LLVM/MLIR-21** with
  `polykernel-opt` / `-translate` / `-bench` / `-report` / `-analyze` tools and a lit suite.
- **Portable CUDA/HIP kernels from one template** — a single `#ifdef`-portable kernel
  template compiles under `nvcc` for **sm_80/sm_90** (PTX emit) and under `hipcc` for
  **gfx1101**, and **runs on a local AMD RX 7800 XT**, all golden-validated
  (**0 failed correctness**); a WMMA bf16 tensor-core path is an additive variant.
- **GPU-free compile-time analysis** — a `ptxas -v` parser (both formats) + occupancy model
  (sm_80/sm_90 constants) + roofline classifier produce the full per-kernel report
  (registers/smem/spills/occupancy/traffic/bottleneck/suggested-fixes) with **no NVIDIA GPU
  present**; an AMDGPU ISA analyzer does the same for VGPR/SGPR/LDS/spills on gfx1101/gfx942.
- **Correctness-gated autotuner + runtime** — a bounded config grid, benchmarked only after
  passing the golden, writes a versioned JSON tuning cache keyed by (GPU, op, shape); a C++
  runtime detects the GPU and loads the best cached kernel.
- **Cerebras-style dataflow simulator** — a self-contained C++ **functional/cycle simulator**
  of a 64×64 PE fabric (CE + FMAC, 48 KB SRAM/PE, wavelet/color routing, the **SUMMA**
  matmul mapping) with seven metrics + a **self-contained HTML visualizer**, correctness-
  validated against the same golden — independently of any GPU.
- **Modal GPU deployment** — a CUDA-toolkit image compiling the engine, `fastapi_endpoint`
  `/predict /benchmark /kernels /report`, cold-start measurement, and a GPU-aware kernel
  cache reusing the runtime (cloud deploy owner-gated behind `MODAL_TOKEN`).
- **Benchmark dashboard + docs** — `polykernel-report --dashboard` aggregates CUDA + AMD +
  dataflow + compiler stats into static, self-contained HTML reports (no external
  dependencies), with the roofline performance model and full architecture/backend docs.

## What it is — and is not

| It is | It is not |
|---|---|
| A compiler/runtime **machinery** demonstration | A claim to **beat vLLM** or any production stack |
| A Cerebras-style dataflow **simulator** (correct CSL terminology: CE+FMAC, `@set_color_config`, `fabin_dsd`/`fabout_dsd`) | Real CSL / `cslc` / **Cerebras hardware** |
| Compile + PTX + GPU-free analysis for CUDA; real H100/A100 numbers from owner-gated rental | Running NVIDIA kernels locally (no NVIDIA GPU) |
| Hand-written `.mlir` input in the `polykernel` dialect | A full PyTorch/ONNX/StableHLO frontend |
| Exactly the named ops + fused variants (closed op set) | An op zoo |

## Architecture

```
   .mlir (polykernel dialect)
        │  polykernel-opt · 11 passes
        ├──────────────────────────────┐
   CUDA backend (sm_80/sm_90)     HIP/ROCm backend (gfx1101 local · gfx942)
   nvcc + PTX + GPU-free analyzer  hipcc + AMDGPU ISA + WMMA bf16 · runs on 7800 XT
        └──────────────┬───────────────┘   ← one portable kernel template
        Autotuner (correctness-gated) + JSON cache + C++ runtime (detect → load → serve)
        ├──────────────────────────────┐
   Modal deployment                 Dataflow simulator (64×64 PE grid, SUMMA)
   /predict /benchmark /kernels     metrics + self-contained HTML viz
   /report · cold-start             (functional/cycle model — NOT Cerebras hardware)
```

Full detail in [`docs/architecture.md`](docs/architecture.md).

## Quick start

```bash
# enter the toolchain (LLVM/MLIR-21, CUDA 12.6, ROCm, python3 + numpy/ml_dtypes)
nix develop --impure --accept-flake-config

# configure + build + run the lit suite (13 tests)
cmake -B build -G Ninja \
    -DMLIR_DIR="$(nix eval --raw --impure --expr '(builtins.getFlake "/home/mei/projects/polykernel").inputs.nixpkgs.legacyPackages.x86_64-linux.llvmPackages_21.mlir.dev.outPath')/lib/cmake/mlir" \
    -DLLVM_DIR="$(nix eval --raw --impure --expr '(builtins.getFlake "/home/mei/projects/polykernel").inputs.nixpkgs.legacyPackages.x86_64-linux.llvmPackages_21.llvm.dev.outPath')/lib/cmake/llvm" \
    -DMLIR_TABLEGEN_EXE="$(which mlir-tblgen)"
cmake --build build
cmake --build build --target check-polykernel

# run the full pipeline on the MLP fragment
./build/tools/polykernel-opt/polykernel-opt examples/mlp_block.mlir \
    --infer-shapes --canonicalize --fuse-rmsnorm-matmul --fuse-matmul-bias-gelu \
    --infer-tile-layout --plan-memory --lower-to-cuda
```

See [`docs/compiler_pipeline.md`](docs/compiler_pipeline.md) for the full build wiring
(including the nix-store path resolution) and the pass pipeline.

## Reports + dashboard

```bash
python3 tools/polykernel-report/dashboard.py --dashboard
```

aggregates the committed per-backend reports into:

- `reports/benchmark_report.md` / `.html` — the dashboard (model fragment, backends,
  CUDA + AMD speedups, dataflow metrics, compiler stats, **failed correctness**);
- `reports/h100_report.html` — CUDA H100/A100 detail;
- `reports/mi300_report.html` — AMD MI300 detail;
- `reports/dataflow_report.html` — the dataflow visualizer (linked, not regenerated).

Every HTML page is **static + self-contained** (inline CSS/JS, no external dependencies)
and renders offline. The H100/A100/MI300 speedups are **PROJECTED** (roofline + analytic
traffic) unless a RunPod rental ran — see
[`docs/performance_model.md`](docs/performance_model.md). The dashboard reflects reality:
`--inject-failed-correctness` runs the golden harness with a deliberately-broken kernel and
the report shows `failed correctness: N>0` prominently (`reports/w6_dashboard_neg.log`).

## Correctness

A NumPy + `ml_dtypes.bfloat16` golden (bf16 round-to-nearest-even inputs, fp32 accumulate,
bf16 output) is the single source of truth; thresholds are cosine ≥ 0.999, max relative
error ≤ 1e-2, PCC ≥ 0.99. Because there is no local NVIDIA GPU, CUDA correctness rests on
the **shared** portable template validated by running the HIP + CPU-reference builds on the
RX 7800 XT, compile-validation of the CUDA tensor-core paths (full validation on rental),
and the dataflow simulator's functional execution — all compared to the same golden.
**0 failed correctness tests** is a success criterion.

## Documentation

| Doc | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System map: compiler + backends + simulator + Modal |
| [`docs/compiler_pipeline.md`](docs/compiler_pipeline.md) | MLIR build wiring + the 11-pass pipeline + tools |
| [`docs/cuda_backend.md`](docs/cuda_backend.md) | nvcc/PTX, GPU-free analyzer, contract-H kernel report |
| [`docs/hip_backend.md`](docs/hip_backend.md) | gfx1101 local run, gfx942 cross-compile, AMDGPU ISA, WMMA bf16 |
| [`docs/dataflow_backend.md`](docs/dataflow_backend.md) | The Cerebras-style simulator (CE+FMAC, SUMMA, metrics, viz) |
| [`docs/performance_model.md`](docs/performance_model.md) | Roofline model + real-vs-projected speedups |

## Status

Waves 1–5 (skeleton, CUDA backend, fusion + correctness, HIP backend, autotuner + analyzer +
runtime) and Wave 7 (dataflow simulator + viz) are complete and green; Wave 6 (Modal +
dashboard + reports + writeup) delivers this dashboard, the HTML reports, and these docs.
Wave 8+ (FlashAttention-lite, quantization, kernel-cache hardening, viz polish, optional
upstream PR) is post-MVP and never blocking.
