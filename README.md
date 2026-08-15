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

## What it is — and is not

| It is | It is not |
|---|---|
| A compiler/runtime **machinery** demonstration | A claim to **beat vLLM** or any production stack |
| A Cerebras-style dataflow **simulator** (correct CSL terminology: CE+FMAC, `@set_color_config`, `fabin_dsd`/`fabout_dsd`) | Real CSL / `cslc` / **Cerebras hardware** |
| Compile + PTX + GPU-free analysis for CUDA; a CUDA runtime-validation harness (`cuda_run_main.cpp` + the MMA tensor-core kernel) built and compile-validated for sm_80/89/90, on-GPU run on the RTX 6000 Ada pod pending pod-key authorization; real H100/A100 numbers from owner-gated rental | Running NVIDIA kernels locally (no local NVIDIA GPU; runtime validation targets the remote RTX 6000 Ada pod, currently pending key authorization) |
| Hand-written `.mlir` input in the `polykernel` dialect | A full PyTorch/ONNX/StableHLO frontend |
| Exactly the named ops + fused variants (closed op set) | An op zoo |

## Architecture

```
   .mlir (polykernel dialect)
        │  polykernel-opt · 11 passes
        ├──────────────────────────────┐
   CUDA backend (sm_80/sm_89/sm_90)     HIP/ROCm backend (gfx1101 local · gfx942)
   nvcc + PTX + GPU-free analyzer  hipcc + AMDGPU ISA + WMMA bf16 · runs on 7800 XT
   + MMA kernel + runtime launcher  (CUDA twin: matmul_mma.cu · RTX 6000 Ada)
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

# configure + build + run the lit suite (14 tests)
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
and renders offline. The H100/A100/MI300/RTX 6000 Ada (sm_89) speedups are **PROJECTED**
(roofline + analytic traffic) unless a rental ran; see
[`docs/performance_model.md`](docs/performance_model.md). The dashboard reflects reality:
`--inject-failed-correctness` runs the golden harness with a deliberately-broken kernel and
the report shows `failed correctness: N>0` prominently (`reports/w6_dashboard_neg.log`).

## Correctness

A NumPy + `ml_dtypes.bfloat16` golden (bf16 round-to-nearest-even inputs, fp32 accumulate,
bf16 output) is the single source of truth; thresholds are cosine ≥ 0.999, max relative
error ≤ 1e-2, PCC ≥ 0.99. CUDA correctness rests on the **shared** portable template
validated by running the HIP + CPU-reference builds on the RX 7800 XT, the CUDA
runtime-validation harness (`lib/Runtime/cuda_run_main.cpp` + the MMA kernel
`kernels/generated/matmul_mma.cu`) which is built and compile-validated for sm_80/89/90
with the on-GPU run on the RTX 6000 Ada pod **pending pod-key authorization** (pod gate
SKIPPED), and the dataflow simulator's functional execution — all compared to the same
golden. H100/A100/MI300/RTX 6000 Ada (sm_89) speedups remain **PROJECTED** (see
[`docs/performance_model.md`](docs/performance_model.md)). **0 failed correctness tests**
is a success criterion.

## Status

- **Waves 1-8 (MVP):** the closed `polykernel` dialect + 11-pass pipeline, portable
  CUDA/HIP codegen, HIP runtime-validated on the local RX 7800 XT (scalar + WMMA bf16, 0
  failed correctness), CUDA compile + PTX + GPU-free analysis, the Cerebras-style dataflow
  simulator, the correctness-gated autotuner, the C++ runtime, Modal serving, and the
  Wave 8 attention + quantization work: done and committed.
- **Pass 2 (this pass):** the CUDA runtime-validation harness
  (`lib/Runtime/cuda_run_main.cpp`), the CUDA MMA tensor-core kernel
  (`kernels/generated/matmul_mma.cu`, nvcuda::wmma m16n16k16 bf16, additive +
  correctness-gated), and sm_89 (RTX 6000 Ada) support across the analyzer,
  per-kernel report, bench suite, and dashboard: all **built +
  compile-validated locally** (sm_80/89/90). The **on-GPU run is PENDING pod-key
  authorization** (pass-2 pod gate SKIPPED, `reports/pod_env.log`), so no CUDA number is
  claimed as measured. The dashboard (`reports/benchmark_report.md`) now carries an
  RTX 6000 Ada (sm_89) row next to H100/A100: **PROJECTED**, like
  H100/A100/MI300, which remain **PROJECTED**.
- **Lit:** 14/14 `.mlir` tests green (`check-polykernel`).

<!--## Documentation

| Doc | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System map: compiler + backends + simulator + Modal |
| [`docs/compiler_pipeline.md`](docs/compiler_pipeline.md) | MLIR build wiring + the 11-pass pipeline + tools |
| [`docs/cuda_backend.md`](docs/cuda_backend.md) | nvcc/PTX, GPU-free analyzer, contract-H kernel report |
| [`docs/hip_backend.md`](docs/hip_backend.md) | gfx1101 local run, gfx942 cross-compile, AMDGPU ISA, WMMA bf16 |
| [`docs/dataflow_backend.md`](docs/dataflow_backend.md) | The Cerebras-style simulator (CE+FMAC, SUMMA, metrics, viz) |
| [`docs/performance_model.md`](docs/performance_model.md) | Roofline model + real-vs-projected speedups |-->
