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
| Compile + PTX + GPU-free analysis for CUDA (sm_80/89/90); the runtime-validation harness (`cuda_run_main.cpp` + MMA kernel) is **runtime-validated on the RTX 6000 Ada (sm_89) pod: 0 failed correctness**; H100/A100 numbers are owner-gated **PROJECTED** | Running NVIDIA kernels locally (this machine has no NVIDIA GPU; the kernels are validated on the remote RTX 6000 Ada pod) |
| Hand-written `.mlir` input in the `polykernel` dialect | A full PyTorch/ONNX/StableHLO frontend |
| Exactly the named ops + fused variants (closed op set) | An op zoo |

## Architecture

```
   .mlir (polykernel dialect)
        │  polykernel-opt · 11 passes
        ├────────────────────────────────┐
   CUDA backend (sm_80/sm_89/sm_90)      HIP/ROCm backend (gfx1101 local · gfx942)
   nvcc + PTX + GPU-free analyzer        hipcc + AMDGPU ISA + WMMA bf16 · runs on 7800 XT
   + MMA kernel + runtime launcher       (CUDA twin: matmul_mma.cu · RTX 6000 Ada)
        └────────────────┬───────────────┘   ← one portable kernel template
        Autotuner (correctness-gated) + JSON cache + C++ runtime (detect → load → serve)
        ├────────────────────────────────┐
   Modal deployment                      Dataflow simulator (64×64 PE grid, SUMMA)
   /predict /benchmark /kernels          metrics + self-contained HTML viz
   /report · cold-start                  (functional/cycle model, NOT Cerebras hardware)
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
and renders offline. The RTX 6000 Ada (sm_89) speedups are **MEASURED** on the pod
(1.000×/0.838×/0.857×); H100/A100/MI300 speedups are **PROJECTED** (roofline + analytic
traffic) until a rental runs; see
[`docs/performance_model.md`](docs/performance_model.md). The dashboard reflects reality:
`--inject-failed-correctness` runs the golden harness with a deliberately-broken kernel and
the report shows `failed correctness: N>0` prominently (`reports/w6_dashboard_neg.log`).

## Correctness

The single source of truth is a NumPy + `ml_dtypes.bfloat16` golden: bf16
round-to-nearest-even inputs, fp32 accumulate, bf16 output, with thresholds of cosine ≥
0.999, max relative error ≤ 1e-2, and PCC ≥ 0.99. HIP runs the shared portable template
(scalar + WMMA bf16) golden-validated on the local RX 7800 XT: **75/75 tests passed**.
CUDA correctness rests on the runtime-validation harness (`lib/Runtime/cuda_run_main.cpp`
+ the MMA kernel `kernels/generated/matmul_mma.cu`), built and compile-validated for
sm_80/sm_89/sm_90 and **runtime-validated on the RTX 6000 Ada (sm_89) pod: 0 failed
correctness, 18/18 golden, cosine == pcc == 1.0** (the MMA kernel separately 5/5 on the
pod). The first on-NVIDIA run exposed and fixed two real CUDA bugs: a bf16x2 store
corruption on sm_89 and a gelu erf precision loss (commit `d85866a`), with no HIP
regression. The dataflow simulator's functional execution is checked against the same
golden, independently of any GPU. H100/A100/MI300 speedups remain **PROJECTED** (see
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
  (`kernels/generated/matmul_mma.cu`, `nvcuda::wmma` m16n16k16 bf16, additive +
  correctness-gated), and sm_89 (RTX 6000 Ada) support across the analyzer, per-kernel
  report, bench suite, and dashboard: **built + compile-validated locally** (sm_80/89/90)
  and **runtime-validated on the RTX 6000 Ada pod** (0 failed correctness,
  `reports/pass2_ada_cuda_run.log`). The validation ran on the pod (216.81.200.13, user
  ubuntu) provisioned after the original 65.109.75.15 was terminated by the user; the pod
  has since been terminated. The dashboard (`reports/benchmark_report.md`) carries the
  RTX 6000 Ada (sm_89) row as **MEASURED**: unfused 1.000× / fused 0.838× / autotuned
  0.857× (fused is slower at this compute-bound shape, the plan's shape-dependence
  caveat, confirmed). H100/A100/MI300 remain **PROJECTED**.
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
