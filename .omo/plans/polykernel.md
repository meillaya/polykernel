# polykernel - Work Plan

## TL;DR (For humans)

**What you'll get.** PolyKernel: a self-contained, out-of-tree **C++/MLIR compiler + runtime** that lowers a fused transformer-block fragment (RMSNorm -> QKV MatMul -> RoPE -> Attention -> Output MatMul -> Residual, plus the MLP fragment) into **portable CUDA/HIP kernels**, analyzes them, autotunes them, simulates a **Cerebras-style dataflow** mapping, and deploys the engine on **Modal**. It is built to demonstrate depth across GPU kernels, compiler/MLIR lowering, CUDA/PTX, ROCm/HIP, accelerator-aware scheduling, and inference serving - the overlap that NVIDIA, AMD, Cerebras, and Modal hire for.

Concretely, the plan delivers: a custom `polykernel` MLIR dialect (exactly the named ops + fused variants) with 11 passes and `polykernel-opt`/`-translate`/`-bench`/`-report` tools; a **CUDA backend** that compiles to PTX for sm_80/sm_90 and runs a **fully GPU-free analyzer** (ptxas registers/smem/spills + occupancy + roofline + suggested fixes - no NVIDIA GPU needed); a **HIP/ROCm backend** that compiles for and **runs on your RX 7800 XT (gfx1101, officially supported since ROCm 7.0)** including a WMMA bf16 tensor path and AMDGPU ISA analysis; an **autotuner** with a correctness-gated JSON tuning cache + a C++ runtime that detects the GPU and loads the best cached kernel; a **Cerebras-CSL-faithful dataflow simulator** (64x64 PE grid, wavelet/color routing, the SUMMA matmul mapping, utilization/SRAM/messages/hops/bottleneck/fusion-savings metrics) with an HTML visualizer; and a **Modal deployment** (`/predict /benchmark /kernels /report`, cold-start measurement, autoscaling) plus a benchmark dashboard, HTML reports, and a technical writeup. Correctness is validated against a NumPy/`ml_dtypes` golden (bf16 round / fp32 accumulate; cosine >= 0.999, rel <= 1e-2, PCC >= 0.99); **0 failed correctness tests** is a success criterion.

**Why this approach.** Research de-risked the three scary parts: (1) out-of-tree MLIR against nixpkgs `llvmPackages_21` is well-trodden (nixpkgs' own circt/flang wiring + the Studio-Todos/Lin flake); (2) the entire CUDA analyzer works at compile time with **no GPU** (Triton runs `ptxas` as a subprocess); (3) your **gfx1101 is officially supported** in ROCm 7.0+, so HIP is a first-class locally-runnable path. Because CUDA and HIP share one portable kernel template, the shared compute logic is correctness-validated by running the HIP build on your 7800 XT against the golden; CUDA-specific tensor-core paths are compile-validated locally and fully validated on rented H100/A100. The Cerebras piece is a faithful **simulator** (the official `gemm-collectives_2d` SUMMA example matches your A-east/B-south/local-C model exactly), not a real Cerebras toolchain.

**What it will NOT do.** Beat vLLM (the goal is the machinery, not SOTA). Compile to real Cerebras hardware (simulator only, with correct CSL terminology - CE+FMAC, `@set_color_config`, not the spec's incorrect FMU/PMU/fdata/fcast/fmove). Run NVIDIA kernels locally (none available). Spend on cloud GPU or Modal without your credentials (RunPod + Modal token are gated; the build degrades to clearly-labeled projections + SKIPPED, never blocks). Build a full PyTorch/ONNX frontend (input is hand-written `.mlir`). Grow an op zoo (only the named ops).

**Effort.** Large: **44 todos across 8 waves + a 4-way final verification wave (F1-F4)**. Waves 1-6 are your 6-week MVP (skeleton, CUDA backend, fusion+correctness, HIP backend, autotuner+analyzer+runtime, Modal+reports); wave 7 is the Cerebras dataflow simulator + HTML viz; waves 8+ are elite (FlashAttention-lite, quantization, kernel-cache hardening, viz polish, optional upstream PR). Waves 4 (HIP) and 7 (dataflow sim) can run in parallel after wave 3.

**Risk (all mitigated).** R1 MLIR-via-nix build -> a wave-1 **toolchain spike gates all dialect work** + the circt/Lin wiring pattern + `MLIR_TABLEGEN_EXE` patch. R2 local ROCm -> official ROCm 7.0 support + `HSA_OVERRIDE_GFX_VERSION=11.0.1` fallback + a verification gate. R3 CUDA correctness without a GPU -> shared template validated via HIP-on-7800XT + a C++ CPU reference impl + rental validation. R4 autotuner cost on rented GPUs -> bounded grid (~100-150 configs) + correctness-gated benchmarking + a ~$50-100 ceiling. R5 Modal API drift -> pinned modal client + current documented API.

**Decisions (locked).** CMake+Ninja (not Bazel); `llvmPackages_21` (pin nixpkgs rev via committed `flake.lock`); `CC/CXX = llvmPackages_21.clang`; C++20 with C++17 fallback; `nixpkgs.config.allowUnfree/cudaSupport/rocmSupport = true`; CUDA compile/analyze-only locally with real H100 numbers from **RunPod rental (NVIDIA + MI300, ~$50-100 ceiling)**; HIP first-class local run on gfx1101; Cerebras-CSL-faithful SUMMA simulator; **current Modal API** (`fastapi_endpoint`, `enter(snap=)`, `min_containers`, `gpu="A10"/"H100"`); **provide a Modal token for real deploy**; tests-alongside (lit/FileCheck + gtest + golden).

**How it runs.** `nix develop` (enters the toolchain), then `cmake -B build -G Ninja -DMLIR_DIR=... -DLLVM_DIR=... -DMLIR_TABLEGEN_EXE=... -DLLVM_EXTERNAL_LIT=... && cmake --build build && cmake --build build --target check-polykernel`. Every todo is Implementation+Test together with agent-executed happy+failure QA and an evidence path; external-dependency waves (Modal, RunPod) degrade to SKIPPED if credentials are absent. Execute with `/start-work`.

## Scope

**What you'll get.** PolyKernel: a self-contained, out-of-tree C++/MLIR compiler + runtime that lowers a fused transformer-block fragment into portable CUDA/HIP kernels, analyzes them, autotunes them, simulates a Cerebras-style dataflow mapping, and deploys the engine on Modal. Concretely:

- **Compiler (C1/C2):** a custom `polykernel` MLIR dialect (`func`, `rmsnorm`, `matmul`, `bias`, `gelu`/`silu`, `add`, `softmax`, `rope`, `attention`, `kv_cache_update`, + fused variants `fused_rmsnorm_matmul`, `fused_matmul_bias_gelu`, `fused_residual_rmsnorm`, `fused_softmax_mask`, `qkv_projection`, `fused_kv_append_attention`); passes `--infer-shapes`, `--canonicalize`, `--fuse-rmsnorm-matmul`, `--fuse-matmul-bias-gelu`, `--fuse-residual-rmsnorm`, `--fuse-softmax-mask`, `--infer-tile-layout`, `--plan-memory`, `--lower-to-cuda`, `--lower-to-hip`, `--emit-kernel-report`; shape+dtype inference; `polykernel-opt`/`polykernel-translate`/`polykernel-bench`/`polykernel-report` tools; examples `mlp_block.mlir`, `rmsnorm_matmul.mlir`, `attention_prefill.mlir`.
- **CUDA backend + analyzer (C3a):** generated CUDA kernels (RMSNorm, GELU/SiLU, MatMul, Softmax, fused RMSNorm+MatMul, fused MatMul+Bias+GELU) with shared-memory tiling, vectorized loads, warp reductions, WMMA/MMA tensor-core path, occupancy-aware block sizing; nvcc compile for sm_80/sm_90; PTX dump; fully **compile-time** analyzer (ptxas -v registers/smem/spills + occupancy arithmetic + roofline memory-traffic + suggested fixes) — **no GPU required**.
- **HIP/ROCm backend + analyzer (C3b):** portable kernels via `#ifdef POLYKERNEL_CUDA`/`POLYKERNEL_HIP`; hipcc `--offload-arch=gfx1101`; **local execution on the RX 7800 XT** (officially supported ROCm 7.0+); AMDGPU ISA analysis (llvm-objdump, `.amdhsa_next_free_vgpr/sgpr`, LDS, scratch spills); WMMA bf16 path; separate AMD tuning DB.
- **Autotuner + runtime (C4):** variant grid (BLOCK_M/N/K, num_warps, vector_width, unroll, shared-mem stages), correctness-gated benchmarking, JSON tuning cache keyed by (GPU, op, shape), C++ runtime that detects the GPU and loads the cached kernel.
- **Cerebras-CSL dataflow simulator (C5):** 64x64 PE grid (CE+router+48KB SRAM/PE), wavelet/color routing (`@set_color_config` rx/tx), dataflow-triggered compute (`@activate`/`@block`/`@unblock` rendezvous), **SUMMA matmul** (A broadcast east, B broadcast south, local output-stationary C accumulate, row/col reduction); metrics (grid utilization, SRAM pressure, messages sent, avg hop distance, comm bottleneck, critical-path cycles, fusion traffic reduction); HTML visualizer. Functionally executes the tile program and is validated against the same CPU golden.
- **Modal deployment + reports (C6):** CUDA-toolkit Image compiling the engine; `@modal.fastapi_endpoint` `/predict` `/benchmark` `/kernels` `/report`; cold-start measurement (`min_containers`/`scaledown_window`/`enable_memory_snapshot`/`@modal.enter(snap=)`); autoscaling; GPU-aware kernel cache; benchmark dashboard; HTML reports (`h100_report.html`, `mi300_report.html`, `dataflow_report.html`); technical writeup + README positioning.
- **Correctness:** NumPy/ml_dtypes golden (bf16 round, fp32 accumulate; cosine >= 0.999, max rel err <= 1e-2, PCC >= 0.99); 0 failed correctness tests is a success criterion.

**Waves (full scope, user's phasing).** 1 skeleton (nix/CMake/MLIR21 + dialect + shape inference + `polykernel-opt` + examples + lit) -> 2 CUDA backend (unfused kernels + nvcc + PTX + golden harness + compile-time analyzer) -> 3 fusion passes (+ tile/layout + memory plan + correctness vs golden + before/after traffic report) -> 4 HIP backend (portable template + hipcc gfx1101 + verify local ROCm + run on 7800 XT + AMDGPU ISA analysis + WMMA bf16) -> 5 autotuner + analyzer + C++ runtime + `polykernel-bench` -> 6 Modal + dashboard + reports + writeup (owner-gated: Modal token) -> 7 Cerebras dataflow simulator + HTML viz -> 8+ elite (FlashAttention-lite, quantization int8/fp8-sim, kernel-cache hardening, viz polish, optional upstream PR).

**What it will NOT do (Must-NOT-Have).**
- Will NOT beat vLLM or any production stack — the goal is to prove the compiler/runtime machinery, not SOTA performance.
- Will NOT compile to real Cerebras hardware — the dataflow backend is a faithful CSL-concept **simulator** (simfabric-style) using correct terminology (CE+FMAC, `@set_color_config`, `fabin_dsd`/`fabout_dsd`), explicitly NOT the spec's incorrect "FMU/PMU", `fdata`/`fcast`/`fmove`, or `@compute`/`@data` terms.
- Will NOT run NVIDIA kernels locally (no NVIDIA GPU) — CUDA is compile + PTX + compile-time analysis; real H100/A100 numbers come from owner-gated RunPod rental (else clearly-labeled perf-model projections).
- Will NOT spend on cloud GPU or Modal without the user's credentials — RunPod + Modal token are hard external dependencies gating wave 6 and the real benchmark numbers; the plan keeps building past their absence.
- Will NOT build a full PyTorch/ONNX/StableHLO frontend — input is hand-written `.mlir` in the `polykernel` dialect.
- Will NOT grow an op zoo — only the named ops/fused patterns; no speculative operators.
- Will NOT reuse tenstorrent/tt-metal code — the dataflow sim is Cerebras-CSL-flavored, distinct from the separate `tensixflow` project.
- Upstream PR (elite) is optional, never blocking.

**Effort.** Large (8 waves, ~50 todos). Low technical risk after research (MLIR-via-nix is well-trodden; gfx1101 is officially supported; the CUDA analyzer is GPU-free). The only hard gates are external credentials (RunPod, Modal token).

**Risk.** (R1) llvmPackages_21 out-of-tree MLIR build in the nix sandbox — mitigated by the nixpkgs circt/flang wiring pattern + MLIR_TABLEGEN_EXE patch + Studio-Todos/Lin reference flake. (R2) Local ROCm on 7800 XT — mitigated by official ROCm 7.0 support + HSA_OVERRIDE_GFX_VERSION=11.0.1 fallback + an early verification gate. (R3) CUDA correctness without a GPU — mitigated by the shared portable template validated via HIP-on-7800XT against the golden, CUDA-specific paths compile-validated + fully validated on rented H100/A100. (R4) autotuner cost on rented GPUs — mitigated by a bounded search grid + correctness-gated benchmarking + a hard spend ceiling. (R5) Modal API drift — mitigated by pinning the modal client version + using the current documented API.

**Decisions (locked).** CMake+Ninja (not Bazel); `llvmPackages_21` (pin nixpkgs rev); C++20 (C++17 fallback); CUDA compile/analyze-only locally with real H100 numbers from RunPod rental; HIP first-class local run on gfx1101; Cerebras-CSL-faithful simulator with SUMMA mapping; current Modal API; tests-alongside; rent NVIDIA+MI300 for real numbers; provide Modal token for real deploy.

## Verification strategy

**Philosophy: tests-alongside.** Every pass/kernel/module ships with its tests in the SAME todo (Implementation + Test = ONE todo). Agent-executed QA is mandatory for every todo: a happy path AND a failure path, each with an exact tool invocation and an evidence path. Zero human-intervention verification.

**Three test layers.**
1. **IR passes — lit/FileCheck.** Each `polykernel-opt` pass gets a `test/<pass>.mlir` with `// RUN: polykernel-opt %s -<pass> | FileCheck %s` checking the IR transform (e.g. `polykernel.matmul` + `polykernel.rmsnorm` -> `polykernel.fused_rmsnorm_matmul`; shape inference produces the expected `tensor<...>` type). Wired via `test/lit.cfg.py` + `lit.site.cfg.py.in` (LLVM_EXTERNAL_LIT=`${lit}/bin/.lit-wrapped`).
2. **Modules — gtest.** Analyzer (ptxas-parser, occupancy calculator, roofline), autotuner (config enumeration, JSON cache r/w), dataflow simulator (router, color routing, SUMMA schedule, metrics), runtime (GPU-detect, kernel selection) each get gtest unit tests. The occupancy calculator is unit-tested against HAND-COMPUTED values for sm_80/sm_90 (e.g. regs=128,threads=256,smem=32KB -> known blocks/SM + occupancy). The ptxas parser is tested against captured real `ptxas -v` stderr fixtures (registers/smem/spills/gmem).
3. **Kernel correctness — NumPy/ml_dtypes golden.** For every generated kernel, a Python golden computes the reference (bf16 round, fp32 accumulate) and the test asserts cosine >= 0.999, max relative error <= 1e-2, PCC >= 0.99. The golden is the single source of truth.

**The no-NVIDIA-GPU correctness story (critical).** CUDA kernels cannot run locally. Correctness is established by THREE complementary means:
- The CUDA and HIP backends share ONE portable kernel template (`#ifdef POLYKERNEL_CUDA`/`POLYKERNEL_HIP`). The shared compute logic is correctness-validated by building the HIP variant and RUNNING it on the RX 7800 XT against the golden (waves 2/4). This validates the algorithmic core for both backends.
- CUDA-specific paths (WMMA/MMA intrinsics, CUDA-only launch config) are compile-validated (nvcc compiles clean for sm_80/sm_90, PTX emits, ptxas -v parses) and structurally reviewed in waves 2/3, then FULLY correctness-validated on rented H100/A100 during the wave-5/6 benchmark passes (owner-gated). Until then, CUDA correctness for tensor-core paths is marked "compile-validated; runtime-validated on rental".
- The dataflow simulator (C5) FUNCTIONALLY EXECUTES the scheduled tile program, so its numerical output is compared against the same CPU golden (same thresholds) — the simulator is correctness-validated independently of any GPU.

**Correctness-gated benchmarking.** The autotuner records a variant's time ONLY if that variant first passes the golden correctness check. A variant that fails correctness is discarded (logged), never benchmarked as "best". This guarantees the tuning cache only ever contains correct kernels.

**Compile-only CUDA QA (no GPU).** `polykernel-bench --backend=cuda --mode=analyze` runs nvcc/ptxas, parses `ptxas -v` (registers/smem/spills), computes occupancy from the sm_80/sm_90 constants table, estimates memory traffic + roofline classification from shapes+dtype, and emits the JSON/text kernel report — all without a device. QA asserts the report fields are populated and match hand-computed values for a known kernel.

**External-dependency gates.** Wave 6 (Modal) requires `MODAL_TOKEN`; the real H100/MI300 benchmark numbers require RunPod credentials. Both are detected at runtime; if absent, the wave produces the fully-implemented artifact + a clearly-labeled "projected / local-only" report + the opt-in deploy/benchmark scripts, and logs a SKIPPED (not FAILED) status with the exact missing credential. The plan never blocks on these — it degrades gracefully.

**Final verification wave (F1-F4)** runs in parallel after all todos and ALL must APPROVE: F1 plan-compliance audit, F2 code-quality review, F3 real manual QA (actually run polykernel-opt end-to-end, run HIP kernels on the 7800 XT, run the dataflow sim, hit the Modal endpoint if a token is present), F4 scope-fidelity (no Must-NOT-Have violations, no op zoo, correct Cerebras terminology).

### Pinned contracts, schemas & per-backend acceptance rules (BINDING on every todo)

These resolve every cross-cutting judgment call. Where a todo and this section differ, THIS SECTION WINS.

**A. Build model.** The deliverable is a devenv **devShell** (devenv.sh / `nix develop`) + a MANUAL `cmake -B build -G Ninja && cmake --build build` inside the shell. It is NOT a sandboxed `nix build` of polykernel itself (that would fight the nix sandbox over network/paths); only the nixpkgs LLVM/MLIR derivation is sandboxed and cached. The devShell inputs (todo 1) MUST include `pkgs.gtest` (the gtest unit tests will not configure without it) in addition to the LLVM/CUDA/ROCm/python packages.

**B. Kernel I/O contract.** Every generated CUDA/HIP kernel is wrapped in a small host driver that reads input `.npy` files (NumPy format), runs the kernel, and writes an output `.npy`. The golden harness is a Python driver `tests/golden/run_kernel_test.py` that (1) invokes the compiled kernel binary (or ctypes `.so`), (2) loads the output `.npy`, (3) computes the golden reference, (4) asserts the thresholds, (5) writes an evidence JSON to `reports/`. `.npy` is the single I/O contract between kernels and the golden (no ad-hoc per-kernel wiring). The C++ host driver reads/writes `.npy` via `cnpy` (header-only) or a hand-rolled minimal `.npy` parser (simple header + raw buffer); pin the choice in W2 (todo 8).

**C. Golden definition + rounding contract.** The golden uses **NumPy + `ml_dtypes.bfloat16` ONLY** (PyTorch is NOT used in the golden - avoid mixing bf16 implementations). Rounding contract (exact): (1) inputs/weights rounded to bf16 round-to-nearest-even (RNE, the ml_dtypes default); (2) all reductions/accumulation in fp32; (3) each op's OUTPUT rounded back to bf16 (so chained ops see bf16 inputs); (4) comparison is on the final bf16 output upcast to fp32. Metric definitions (exact): `cosine` = dot(a,b)/(||a|| ||b||) over the FLATTENED fp32-upcast tensors; `pcc` = Pearson correlation over the flattened tensors; `max_rel_err` = max over elements of |a-ref|/(|ref|+eps) with **eps = 1e-6 (pinned)**. Thresholds: cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99 - ALL three must hold.

**D. Canonical shapes (consistent across golden, autotuner, simulator, reports).** MLP/matmul: M=2048, N=11008, K=4096 (the spec fragment) for end-to-end; [256,512,256] as the small canonical matmul shape; [128,128,128] for unit tests; plus a non-multiple-of-16 shape (e.g. [130,127,65]) to exercise bounds checks. Attention: seq=512, d_model=4096, heads=32, head_dim=128. All shapes are STATIC; if a dynamic dim (`?`) appears, lowering fails gracefully with a diagnostic.

**E. Op-zoo boundary.** The ONLY `polykernel.` ops are the named ones (base + fused). Lowering intermediates use UPSTREAM MLIR dialects only (`arith`, `tensor`, `memref`, `gpu`, `scf`, `llvm`) - NO new `polykernel.*` helper ops (no `polykernel.transpose`/`cast`/`broadcast`/`reduce`). The dialect test (todo 3) asserts the registered `polykernel.` op set is exactly the named set.

**F. Kernel emission + analyzer split.** `--lower-to-cuda`/`--lower-to-hip` lower `polykernel` ops to UPSTREAM dialects only (`scf`/`arith`/`memref`/`tensor`, per contract E). `polykernel-translate --mlir-to-cuda-source` / `--mlir-to-hip-source` is a CUSTOM SOURCE EMITTER that walks the lowered IR and generates CUDA/HIP C++ text (`__global__`, shared-memory tiling, `#ifdef` guards) via string emission - it does NOT use the upstream `gpu-to-llvm`/`convert-to-llvm` paths; the generated `.cu`/`.hip` is then compiled by nvcc/hipcc as ordinary source. The lit tests for codegen check the emitted text for markers (`__global__`, `#ifdef POLYKERNEL_CUDA`/`POLYKERNEL_HIP`, the WMMA builtin). The compile-time analyzer is built in W2 (todo 11) as a LIBRARY + a minimal standalone CLI (`polykernel-analyze`); W5's `polykernel-bench` (todo 25/27) WRAPS that library - the analyzer is not built twice.

**G. Per-backend disjunctive ACCEPTANCE rules (absent hardware/credentials = SKIPPED, NEVER FAILED).**
- **CUDA ACCEPTS iff** (nvcc compiles clean for sm_80 AND sm_90) AND (PTX emitted + ptxas -v parses) AND (host/CPU reference impl passes the golden) AND [ (rented H100/A100 runs + passes golden) OR (marked "compile-validated; runtime-validated on rental" + projected report) ]. Local-compile + host-ref golden are MANDATORY; real-GPU run is OPTIONAL (rental).
- **HIP ACCEPTS iff** (hipcc compiles clean for gfx1101) AND [ (runs on local 7800 XT + passes golden) OR (HSA_OVERRIDE_GFX_VERSION=11.0.1 run + passes golden) OR (if `/dev/kfd` is absent: compile-clean + local-run marked SKIPPED, NOT FAILED) ]. Compile is MANDATORY; local run is best-effort with the override fallback; absent GPU degrades to compile-only SKIPPED.
- **Dataflow simulator ACCEPTS iff** (sim functionally executes the MLP/matmul fragment) AND (assembled output passes the golden) AND (all 7 metrics populated) AND (HTML viz structurally valid). No hardware - MANDATORY full pass.
- **Modal ACCEPTS iff** [ (MODAL_TOKEN present -> deployed + cold-start measured + endpoints return 200) OR (token absent -> full app + local `modal serve` boots + endpoints 200 locally + cloud deploy marked SKIPPED) ].
- **Benchmark numbers ACCEPT iff** [ (RunPod rental ran -> REAL numbers in the report) OR (no rental -> clearly-labeled PROJECTIONS from the compile-time analyzer + roofline) ].

**H. Schemas (authoritative).**
- *JSON tuning cache* (`cache/tuning_cache.json`): `{version:1, entries:[{gpu:"gfx1101"|"sm_90"|"sm_80"|"gfx942", op, shape:{M,N,K,dtype}, best:{BLOCK_M,BLOCK_N,BLOCK_K,num_warps,vector_width,unroll,shared_memory_stages,kernel_path,scored_by:"real-benchmark"|"compile-time-model",time_ms(null if compile-time-model),occupancy,registers,smem_bytes,correctness:{cosine,max_rel_err,pcc}}, searched_configs, correct_configs}]}`. The `scored_by` field lets reports label projected-vs-real.
- *Per-kernel report*: `{kernel, backend:"cuda"|"hip", arch, registers_per_thread, smem_per_block_bytes, spill_stores_bytes, spill_loads_bytes, occupancy:{active_warps_per_sm,max_warps_per_sm,occupancy_pct,limiter}, traffic:{global_read_bytes,global_write_bytes,arithmetic_intensity_flop_per_byte,roofline}, bottleneck, suggested_fixes:[], path}` with ENUM values pinned: `limiter ∈ {registers, smem, warps, blocks}`, `roofline ∈ {compute-bound, memory-bound}`, `bottleneck ∈ {compute, memory, latency}`, `path ∈ {scalar, wmma, mma}`.

**I. Autotuner budget guards.** Max ~200 configs per (op, shape); 2-3 representative shapes per op; max ~30 min wall-clock per rental session; `benchmarks/rent_runpod.sh` sets a RunPod pod timeout + writes a budget log; the ~$50-100 ceiling is enforced. The autotuner has TWO scoring modes - local compile-time-model (no GPU) and real-benchmark (rented) - recorded via `scored_by`. The rental benchmark bundle INCLUDES the golden harness (so rented runs are correctness-gated too).

**J. Dataflow simulator fidelity.** The simulator is SELF-CONTAINED C++; it does NOT depend on, link against, or shell out to the Cerebras SDK, `cslc`, or simfabric (the research notes cite real SDK APIs only to ground the concepts - they are reproduced as a functional model, never integrated). The CE compute model executes `@fmacs` as a REAL fp32 fma on bf16-rounded inputs (matching the golden's rounding contract in C) - NOT literal fp16, which would spuriously fail the golden. Assembled per-PE tile outputs are numerically compared to the golden (same thresholds). Non-matmul ops (rmsnorm, gelu/silu, softmax, add, bias) map to LOCAL PE compute - each PE applies the op to its resident tile in local SRAM with NO fabric wavelets; only matmul/fused-matmul uses the SUMMA fabric schedule. Fusion savings (todo 38) arise because fused intermediates (e.g. the rmsnorm output feeding matmul) remain in local SRAM and never traverse the fabric. The HTML viz is a SINGLE self-contained `.html` (inline CSS/JS + an embedded `<script id="viz-data" type="application/json">` data blob); no CDN/network fetch, no framework, no build step. Viz QA = STRUCTURAL validation (file exists + non-empty + parses as valid HTML + contains the expected `viz-data` JSON payload + expected DOM sections), NOT human visual confirmation.

**K. WMMA path.** Recommend **rocWMMA** (`rocmPackages.rocwmma`, abstracts the fragment layout - less error-prone than the raw builtin) for the bf16 tensor path. First check (todo 22): assemble a single `v_wmma_f32_16x16x16_bf16` for gfx1101 (compile-only) before building the full kernel. WMMA QA is NUMERICAL (a 16x16x16 bf16 matmul through the WMMA path vs the golden), not merely "compiles". The scalar/tiled matmul (todo 10) remains the correctness anchor; WMMA is an additive, correctness-gated variant.

**L. Two distinct traffic-reduction metrics (do not conflate).** W3 (todo 17) traffic reduction = GPU GLOBAL-MEMORY bytes before/after fusion. W7 (todo 38) fusion traffic reduction = dataflow WAVELET/message count before/after fusion. Different schemas, different reports.

**M. MVP attention boundary.** The MVP implements `polykernel.attention` as a correct-but-SIMPLE (non-Flash) kernel; the FlashAttention-lite tiled fusion is elite (todo 40, W8). Do not pull Flash tiling into W2/W3.

**N. C++17 fallback trigger.** The W1 toolchain spike (todo 2) first verifies the pinned nixpkgs rev has a working `llvmPackages_21.mlir` (`nix build nixpkgs#llvmPackages_21.mlir`), then builds the minimal out-of-tree dialect + lit under C++20. Spike PASSES if that builds + lit is green under C++20; FAILS (header/std incompatibility) -> set `CMAKE_CXX_STANDARD=17` project-wide and record it.

## Execution strategy

**Wave dependency graph.**
```
W1 skeleton (dialect+shape+opt+lit)
  -> W2 CUDA backend (unfused kernels + nvcc + PTX + golden harness + compile-time analyzer)
  -> W3 fusion + tile/layout + memory plan + correctness + traffic report
       -> W4 HIP backend (portable template + hipcc gfx1101 + verify ROCm + run on 7800 XT + ISA analysis + WMMA bf16)
            -> W5 autotuner + analyzer + runtime + polykernel-bench (needs W2+W4 kernels)
                 -> W6 Modal + dashboard + reports + writeup (OWNER-GATED: Modal token; real numbers need RunPod)
       -> W7 Cerebras dataflow simulator + HTML viz (independent of GPU backends; can run in PARALLEL with W4/W5/W6)
            -> W8+ elite (FlashAttention-lite, quantization, kernel-cache hardening, viz polish, optional upstream PR)
```

**Parallelism.** W1->W2->W3 are strictly sequential (each needs the prior's IR/codegen). After W3, **W4 (HIP) and W7 (dataflow sim) can proceed in parallel** (the simulator depends only on the dialect/ops + golden, not on GPU codegen). W5 (autotuner) needs W2+W4 kernels. W6 (Modal) needs the W5 engine. W8+ needs the MVP. A worker can fan out W4 || W7 to compress the schedule.

**Owner-gated external dependencies (do not block the build).**
- `MODAL_TOKEN` gates the real `modal deploy` + live cold-start measurement in W6. If absent: implement the full app + local `modal serve` dev path + documented `modal deploy`, mark cloud deploy SKIPPED with the missing credential.
- RunPod credentials gate the REAL H100/A100 + MI300 benchmark numbers (W5/W6). If absent: produce clearly-labeled perf-model projections (compile-time analyzer + roofline) + opt-in rental scripts (`benchmarks/rent_runpod.sh`), mark real numbers SKIPPED. Spend ceiling ~$50-100; autotuner search bounded (reduced grid ~100-150 configs, few shapes) to respect it.

**Environment setup (W1, then verified each wave).** nix flake + devenv.sh against nixpkgs-unstable, PIN the nixpkgs rev in flake.lock. Inputs: `llvmPackages_21` (mlir + mlir.dev + tblgen + llvm), `pkgs.lit`, `clang`, `cmake`, `ninja`, `cudaPackages_12_6` (nvcc/ptxas/cuobjdump for the compile-only CUDA path), `rocmPackages.clr` (+ rocminfo/rocm-smi for HIP), python3 (NumPy/ml_dtypes for the golden). CMake wiring: `MLIR_DIR=${mlir.dev}/lib/cmake/mlir`, `LLVM_DIR=${llvm.dev}/lib/cmake/llvm`, `MLIR_TABLEGEN_EXE=${tblgen}/bin/mlir-tblgen`, `LLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped`. C++20 (fallback C++17).

**Worktree/branch discipline.** One git repo at /home/mei/projects/polykernel (init in W1). Conventional commits per todo (see Commit strategy). No WIP commits; each todo = one atomic green commit. Dirty/unrelated paths (the stale `.codegraph/`) are left untouched and out of scope.

**Subagent strategy for the worker.** Sequential waves 1-3 (foundation, must be solid). Parallel fan-out W4 || W7. Each todo is Implementation+Test together. Every todo's QA is agent-executed (happy + failure) with an evidence path under `reports/` or `build/`.

## Todos

<!-- Implementation task rows MUST be column-zero and match: - [ ] N. <title> -->
<!-- APPEND todo batches below; 50+ todos is fine. Each todo: References, Acceptance criteria, QA (happy + failure with evidence path), Commit. -->

### Wave 1 - Foundation (C1): nix/CMake/MLIR21 + dialect + shape inference + polykernel-opt + examples + lit

- [ ] 1. Repo init + nix flake + devenv.sh toolchain
  - **Where:** `/home/mei/projects/polykernel/{flake.nix,devenv.yaml,devenv.nix,.gitignore,.envrc}`; `git init`.
  - **References:** user convention `tensorforge/devenv.nix` (cudaPackages_12_6, allowUnfree/cudaSupport, cudaCapabilities, clang, CUDA_HOME/LD_LIBRARY_PATH incl /run/opengl-driver) and `tensorforge/devenv.yaml` (nixpkgs-unstable); research bg_26a62b16 (llvmPackages_21 = nixpkgs default; attrs `llvmPackages_21.{mlir,mlir.dev,tblgen,llvm,clang}`, `pkgs.lit`, FileCheck in `llvm`); bg_774c9f47 (`rocmPackages.{clr,rocminfo,rocm-smi}`, gpuTargets includes gfx1101). nixpkgs-unstable in 2026 ships ROCm 7.x.
  - **Acceptance:** `git init` done; `.gitignore` excludes `.codegraph/`, `build/`, `result`, `.devenv*`, `__pycache__`; `devenv.yaml` inputs nixpkgs-unstable; `devenv.nix` sets `nixpkgs.config.allowUnfree=true`, `cudaSupport=true`, `rocmSupport=true`; packages = `llvmPackages_21.{mlir,tblgen,llvm,clang}`, `pkgs.lit`, `cmake`, `ninja`, `cudaPackages_12_6.{cuda_nvcc,cuda_cudart,cuda_nvrtc,libcublas,cuda_cuobjdump}`, `rocmPackages.{clr,rocminfo,rocm-smi}`, `pkgs.gtest` (REQUIRED for the gtest unit tests - see Pinned contracts A), `python3` (+ `numpy`, `ml_dtypes`, `pytest` via uv/pip); `env.CC`/`env.CXX` = `llvmPackages_21.clang` (NOT clang_18 - avoid MLIR-21 header skew); `flake.lock` committed (pins nixpkgs rev; `nix flake update` is the only mover). `nix develop` enters a shell where `mlir-tblgen --version`, `nvcc --version`, `hipcc --version`, `cmake --version`, `lit --version`, `FileCheck --version` all succeed.
  - **QA happy:** `nix develop -c bash -c 'mlir-tblgen --version && nvcc --version && hipcc --version && FileCheck --version && python3 -c "import numpy, ml_dtypes"'` exits 0 -> evidence: `reports/w1_toolchain.log`.
  - **QA failure:** temporarily set `allowUnfree=false` -> `nix develop` fails to evaluate cudaPackages (proves the EULA gate is real) -> revert -> evidence: `reports/w1_allowunfree_negative.log`.
  - **Commit:** `build: nix flake + devenv.sh toolchain (llvmPackages_21 MLIR, cudaPackages_12_6, rocmPackages.clr, allowUnfree)`

- [ ] 2. Toolchain spike: minimal out-of-tree MLIR dialect + polykernel-opt skeleton (GATE)
  - **Where:** `CMakeLists.txt` (top-level), `cmake/`, `tools/polykernel-opt/polykernel-opt.cpp`, `include/PolyKernel/IR/`, `lib/IR/`, `test/lit.cfg.py`, `test/lit.site.cfg.py.in`.
  - **References:** research bg_26a62b16 - CMake wiring `MLIR_DIR=${mlir.dev}/lib/cmake/mlir`, `LLVM_DIR=${llvm.dev}/lib/cmake/llvm`, `MLIR_TABLEGEN_EXE=${tblgen}/bin/mlir-tblgen` (REQUIRED: nixpkgs patches MLIRConfig.cmake per llvm/llvm-project#150986), `LLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped`; llvm-project `mlir/examples/standalone` CMake skeleton (`add_mlir_dialect`, `mlir_tablegen --gen-pass-decls`, `add_mlir_dialect_library`, `add_llvm_executable`+`mlir_check_all_link_libraries`, DialectRegistry.insert+registerAllPasses+MlirOptMain); real flake Studio-Todos/Lin. C++20 first, fall back to C++17 (MLIR minimum) at the first header error.
  - **Acceptance (GATE - must pass before any polykernel op work):** top-level CMake configures with Ninja against `llvmPackages_21.mlir`; a minimal `PolyKernel` dialect (empty, just registered) builds; `polykernel-opt --help` lists the dialect; one trivial lit test (`test/smoke.mlir`: parse an empty module) passes via `check-polykernel`. C++ standard recorded (20 or 17 fallback) in `docs/compiler_pipeline.md`.
  - **QA happy:** `nix develop -c cmake -B build -G Ninja -DMLIR_DIR=$MLIR_DIR ... && cmake --build build && cmake --build build --target check-polykernel` -> `polykernel-opt --help` shows PolyKernel, lit 1/1 passes -> evidence: `reports/w1_spike.log`.
  - **QA failure:** build with C++20 against a deliberately incompatible flag; if MLIR headers reject C++20, switch `CMAKE_CXX_STANDARD` to 17 and rebuild green (proves the fallback path) -> evidence: `reports/w1_cppstd.log`.
  - **Commit:** `build: out-of-tree MLIR toolchain spike (polykernel-opt skeleton + lit) against llvmPackages_21`

- [ ] 3. `polykernel` dialect ODS: exactly the named ops
  - **Where:** `include/PolyKernel/IR/PolyKernelDialect.td`, `include/PolyKernel/IR/PolyKernelOps.td`, `include/PolyKernel/IR/CMakeLists.txt`, `lib/IR/PolyKernelDialect.cpp`, `lib/IR/PolyKernelOps.cpp`, `lib/IR/CMakeLists.txt`.
  - **References:** llvm-project standalone `StandaloneDialect.td`/`StandaloneOps.td` ODS pattern (research bg_26a62b16); spec op list. Op-zoo guardrail: define EXACTLY `polykernel.func, rmsnorm, matmul, bias, gelu, silu, add, softmax, rope, attention, kv_cache_update` + fused `fused_rmsnorm_matmul, fused_matmul_bias_gelu, fused_residual_rmsnorm, fused_softmax_mask, qkv_projection, fused_kv_append_attention` - no others.
  - **Acceptance:** `add_mlir_dialect(PolyKernelOps polykernel)` generates `PolyKernelOps.h.inc/.cpp.inc`; `add_mlir_dialect_library(MLIRPolyKernel ... DEPENDS MLIRPolyKernelOpsIncGen LINK_LIBS PUBLIC MLIRIR MLIRFuncDialect MLIRInferTypeOpInterface)` builds; all named ops parse in a `.mlir` round-trip (`polykernel-opt` prints them back). A lit test asserts the op set is exactly the named set (no extra ops registered).
  - **QA happy:** `polykernel-opt test/dialect_roundtrip.mlir` round-trips every named op -> evidence: `reports/w1_dialect_roundtrip.log`.
  - **QA failure:** `polykernel-opt` on a `.mlir` using an undefined op `polykernel.conv2d` -> error "unknown op" (proves op-zoo is closed) -> evidence: `reports/w1_unknown_op.log`.
  - **Commit:** `feat(ir): polykernel dialect + ops (ODS/TableGen) - exactly the named transformer-inference op set`

- [ ] 4. Shape + dtype inference pass (`--infer-shapes`)
  - **Where:** `include/PolyKernel/Passes/InferShapes.h`, `lib/Passes/InferShapes.cpp`, `include/PolyKernel/Passes/Passes.td` (`mlir_tablegen --gen-pass-decls`), `test/infer_shapes.mlir`.
  - **References:** MLIR `InferTypeOpInterface` / `InferIntRangeInterface`; standalone `StandalonePasses.td` pass-decl pattern (bg_26a62b16). Rules: matmul `tensor<MxKxd>xtensor<KxNxd>->tensor<MxNxd>`; rmsnorm/gelu/silu/softmax preserve shape; add requires equal shapes; bias broadcasts `[N]` over `[...,N]`; fused ops inherit.
  - **Acceptance:** `--infer-shapes` replaces unresolved result types with inferred `tensor<...xbf16>` types; registered via `registerInferShapesPass`; lit/FileCheck asserts inferred types for matmul/rmsnorm/gelu/add/fused on bf16 + fp16.
  - **QA happy:** `polykernel-opt test/infer_shapes.mlir --infer-shapes | FileCheck test/infer_shapes.mlir` passes (checks `tensor<2048x11008xbf16>` etc.) -> evidence: `reports/w1_infer_shapes.log`.
  - **QA failure:** matmul with mismatched K dims (`tensor<4x8xbf16> x tensor<16x4xbf16>`) -> `--infer-shapes` emits a shape-mismatch diagnostic, non-zero exit -> evidence: `reports/w1_shape_mismatch.log`.
  - **Commit:** `feat(pass): --infer-shapes (shape+dtype inference for all polykernel ops)`

- [ ] 5. `--canonicalize` patterns + basic folds
  - **Where:** `lib/Passes/Canonicalize.cpp` (or per-op `getCanonicalizationPatterns` in `PolyKernelOps.cpp`), `test/canonicalize.mlir`.
  - **References:** MLIR canonicalization (`RewritePatternSet`, `getCanonicalizationPatterns`, `--canonicalize` from MLIRTransforms). Patterns: identity residual-add elimination (`add(x, 0) -> x`), constant-fold of pure elementwise on dense constants, no-op rmsnorm eps handling.
  - **Acceptance:** `--canonicalize` folds `add(x, zero_const) -> x` and removes dead ops; lit/FileCheck asserts the simplifications.
  - **QA happy:** `polykernel-opt test/canonicalize.mlir --canonicalize | FileCheck test/canonicalize.mlir` (CHECK-NOT for the eliminated zero-add) -> evidence: `reports/w1_canonicalize.log`.
  - **QA failure:** a non-foldable op (add of two non-constant tensors) is left unchanged (CHECK the op survives) -> evidence: `reports/w1_canonicalize_noop.log`.
  - **Commit:** `feat(pass): --canonicalize (identity-add elimination, constant folding, DCE)`

- [ ] 6. Examples + lit harness + end-to-end parse
  - **Where:** `examples/{mlp_block.mlir,rmsnorm_matmul.mlir,attention_prefill.mlir}`, `test/lit.cfg.py`, `test/lit.site.cfg.py.in`, `test/CMakeLists.txt`, `test/e2e_parse.mlir`.
  - **References:** llvm-project standalone `test/lit.cfg.py`+`lit.site.cfg.py.in` (bg_26a62b16); `LLVM_EXTERNAL_LIT=${lit}/bin/.lit-wrapped`; spec examples. mlp_block.mlir = the spec's MLP fragment (rmsnorm->matmul->gelu->matmul->add); rmsnorm_matmul.mlir = the fused target; attention_prefill.mlir = the transformer-block fragment (documented example, ops parsed even if full lowering lands later).
  - **Acceptance:** `check-polykernel` CMake target runs lit over `test/`; `polykernel-opt examples/mlp_block.mlir --infer-shapes --canonicalize` succeeds end-to-end; all 3 examples parse without error.
  - **QA happy:** `cmake --build build --target check-polykernel` -> all lit tests pass; `polykernel-opt examples/*.mlir --infer-shapes` exits 0 -> evidence: `reports/w1_check.log`.
  - **QA failure:** a malformed example (unterminated `polykernel.func`) -> `polykernel-opt` emits a parse error with line/col, non-zero exit -> evidence: `reports/w1_parse_error.log`.
  - **Commit:** `test: examples (mlp_block/rmsnorm_matmul/attention_prefill) + lit harness + e2e parse`

### Wave 2 - CUDA backend (C3a) + golden correctness harness

- [ ] 7. Golden correctness harness (NumPy + ml_dtypes) + rounding contract
  - **Where:** `tests/golden/golden.py`, `tests/golden/metrics.py`, `tests/golden/conftest.py`, `tests/golden/test_golden_self.py`.
  - **References:** user tensixflow golden approach (bg research; bf16 round / fp32 accumulate; rel/abs/PCC/cosine). NumPy has NO native bfloat16 -> use `ml_dtypes.bfloat16`. Rounding contract (PINNED): inputs/weights rounded to bf16 (round-to-nearest-even), accumulation in fp32, output rounded to bf16. Thresholds (PINNED): cosine similarity >= 0.999, max relative error <= 1e-2, PCC >= 0.99.
  - **Acceptance:** `golden.py` provides reference impls for rmsnorm/matmul/bias/gelu/silu/add/softmax (+fused) in NumPy+ml_dtypes honoring the rounding contract; `metrics.py` provides `cosine`, `max_rel_err`, `pcc`, and `assert_correct(actual, ref)` applying the thresholds; a self-test (`test_golden_self.py`) verifies the reference against hand-computed tiny cases.
  - **QA happy:** `nix develop -c pytest tests/golden/test_golden_self.py -q` -> all pass (reference matches hand-computed bf16 results) -> evidence: `reports/w2_golden_self.log`.
  - **QA failure:** `assert_correct` given a deliberately perturbed output (add 5% noise) -> raises with cosine/rel/PCC printed (proves the test detects wrong kernels) -> evidence: `reports/w2_golden_negative.log`.
  - **Commit:** `test: golden correctness harness (NumPy+ml_dtypes bf16, rounding contract, cosine/rel/PCC thresholds)`

- [ ] 8. Portable kernel template + --lower-to-cuda scaffolding + nvcc driver + CPU reference impl
  - **Where:** `include/PolyKernel/Codegen/KernelTemplate.h`, `kernels/template/kernel_common.h` (`#ifdef POLYKERNEL_CUDA`/`#ifdef POLYKERNEL_HIP` includes + macros mapping `__global__`/shared/launch to CUDA vs HIP), `lib/CodegenCUDA/LowerToCuda.cpp`, `lib/Passes/LowerToCuda.cpp`, `kernels/cpu/cpu_reference.{h,cpp}`, `tools/polykernel-bench/nvcc_driver.py`.
  - **References:** spec `#ifdef POLYKERNEL_CUDA/POLYKERNEL_HIP` pattern; HIP=C++ runtime closely aligned to CUDA (bg_774c9f47). CRITICAL ordering decision: design the template PORTABLE now (W2) so W4 reuses it - do NOT retrofit. Follow user tensorforge GEMM conventions (bg via Metis: dtype-templated FP32/FP16/BF16, FP32 accumulate, `launch_*(const void*..., Dtype, void* stream)` ABI, bounds-checked). CPU reference impl = host-compilable C++ implementing the SAME math (= local executable spec + CPU fallback backend). nvcc driver compiles generated `.cu` for `-arch=sm_80` and `-arch=sm_90`, emits PTX (`-ptx`/`--keep`), runs `ptxas -v`.
  - **Acceptance:** `--lower-to-cuda` pass registered, emits CUDA C++ for the ops into `kernels/generated/`; the template compiles under BOTH `nvcc -DPOLYKERNEL_CUDA` (W2) and is structured so `hipcc -DPOLYKERNEL_HIP` (W4) works unchanged; `cpu_reference` builds and exposes the same `launch_*`-style entry points; nvcc driver compiles a generated kernel for sm_80+sm_90 and dumps PTX.
  - **QA happy:** generate RMSNorm CUDA -> `nvcc_driver.py --arch sm_80,sm_90 --ptx` compiles clean + emits `.ptx`; `cpu_reference` rmsnorm matches `golden.py` rmsnorm -> evidence: `reports/w2_template.log`.
  - **QA failure:** compile a generated kernel with an undefined macro (simulate a codegen bug) -> nvcc fails with a clear error captured (proves the driver surfaces codegen bugs) -> evidence: `reports/w2_nvcc_fail.log`.
  - **Commit:** `feat(cuda): portable CUDA/HIP kernel template + --lower-to-cuda scaffolding + nvcc driver + CPU reference impl`

- [ ] 9. CUDA kernels + CPU ref: RMSNorm + GELU/SiLU
  - **Where:** `kernels/generated/rmsnorm.cu`, `kernels/generated/gelu.cu`, `kernels/cpu/{rmsnorm,gelu}.cpp`, `tests/kernels/test_rmsnorm_gelu.py`.
  - **References:** RMSNorm = warp-level reduction for the RMS + vectorized loads; GELU/SiLU elementwise (tensorforge elementwise.cu conventions). WMMA not needed here (elementwise/reduction). Correctness model: CPU reference validated vs golden NOW; CUDA kernel compile-validated now, runtime-validated via HIP-on-7800XT in W4 (shared template) + rental.
  - **Acceptance:** generated RMSNorm uses a warp-shuffle reduction + vectorized (float4/half4) loads; GELU/SiLU are vectorized elementwise; nvcc compiles both for sm_80+sm_90 + PTX emitted; CPU reference passes `assert_correct` vs golden for shapes [2048,4096] bf16 (and a non-multiple-of-vector-width shape to exercise bounds).
  - **QA happy:** `pytest tests/kernels/test_rmsnorm_gelu.py -q` (CPU ref vs golden, incl. odd shape) passes; nvcc compiles + PTX present -> evidence: `reports/w2_rmsnorm_gelu.log`.
  - **QA failure:** a deliberately broken RMSNorm CPU ref (wrong eps) fails `assert_correct` (cosine < 0.999) -> evidence: `reports/w2_rmsnorm_neg.log`.
  - **Commit:** `feat(cuda): RMSNorm (warp-reduce + vectorized) + GELU/SiLU kernels + CPU reference, golden-validated`

- [ ] 10. CUDA kernels + CPU ref: MatMul (tiled/vectorized) + Softmax
  - **Where:** `kernels/generated/matmul.cu`, `kernels/generated/softmax.cu`, `kernels/cpu/{matmul,softmax}.cpp`, `tests/kernels/test_matmul_softmax.py`.
  - **References:** tensorforge gemm.cuh conventions (tiled 16x16 bank-padded `As[16][17]`, vectorized float4/half4/bfloat164, double-buffered smem, bounds-checked, FP32 accumulate). Tensor-core/WMMA path is STUBBED here (landed in W4 for HIP WMMA + elite for CUDA MMA) - the tiled scalar path is the correctness baseline. Softmax = online (safe) softmax with warp/block reduction.
  - **Acceptance:** MatMul tiled shared-memory kernel + CPU ref pass `assert_correct` vs golden for [M,N,K]=[128,128,128] and [256,512,256] (small canonical) and the full [2048,11008,4096] bf16 incl. non-tile-multiple dims; Softmax passes for a [tokens, vocab] shape; nvcc compiles both sm_80+sm_90 + PTX.
  - **QA happy:** `pytest tests/kernels/test_matmul_softmax.py -q` passes (incl. non-multiple-of-16 dims); nvcc compiles + PTX -> evidence: `reports/w2_matmul_softmax.log`.
  - **QA failure:** MatMul CPU ref with a transposed-index bug fails golden (proves bounds/tiling tests catch it) -> evidence: `reports/w2_matmul_neg.log`.
  - **Commit:** `feat(cuda): MatMul (tiled+vectorized, FP32 accum) + Softmax kernels + CPU reference, golden-validated`

- [ ] 11. Compile-time analyzer v1: ptxas parser + occupancy + roofline + kernel report
  - **Where:** `include/PolyKernel/Analysis/PtxasParser.h`, `lib/Analysis/{PtxasParser,Occupancy,Roofline,KernelReport}.cpp`, `tools/polykernel-bench/analyze.py`, `lib/Analysis/tests/{ptxas_parser_test,occupancy_test,roofline_test}.cpp` (gtest).
  - **References:** research bg_fbed4570 - `ptxas -v` stderr format OLD (`Used N registers, N bytes smem, N bytes cmem[0]`) AND NEW (`Used N registers, used N barriers, N bytes smem`); parse regexes (NVIDIA/Fuser codediff.py): `(\d+) bytes stack frame`, `(\d+) bytes spill stores/loads`, `(\d+) registers`, `(\d+) bytes smem`, `(\d+) bytes gmem`. Occupancy constants sm_80 (smem/SM=164KB) & sm_90 (smem/SM=228KB): regs/SM=65536, regs/thread=255, warps/SM=64, blocks/SM=32, threads/SM=2048, warp=32; blocks=min(reg_limit,smem_limit,warp_limit,32), occupancy=active_warps/64. Roofline: bytes=(M*K+K*N+M*N)*dtype_bytes, flops=2*M*N*K, AI=flops/bytes, ridge=peak/bw (H100 989TFLOPS/3.35TB/s). All GPU-FREE.
  - **Acceptance:** PtxasParser handles BOTH old+new formats (gtest vs captured fixtures in `lib/Analysis/tests/fixtures/`); Occupancy calculator gtest vs HAND-COMPUTED values (e.g. regs=128,threads=256,smem=32KB on sm_90 -> known blocks/SM + occupancy); Roofline classifies a large GEMM compute-bound and a small GEMM memory-bound; `polykernel-bench --backend=cuda --mode=analyze` emits a populated JSON/text kernel report (registers, smem, spills, occupancy, bytes_read, bytes_write, arithmetic_intensity, roofline) with NO GPU present.
  - **QA happy:** `ctest -R 'ptxas_parser|occupancy|roofline'` passes; `polykernel-bench --backend=cuda --mode=analyze --kernel matmul --shape 128,128,128 --dtype bf16` prints a populated report matching hand-computed occupancy -> evidence: `reports/w2_analyze.log`.
  - **QA failure:** feed PtxasParser a malformed/garbage ptxas log -> returns explicit parse-error (not a crash, not silent zeros) -> evidence: `reports/w2_ptxas_parse_neg.log`.
  - **Commit:** `feat(analyze): compile-time analyzer v1 (ptxas parser old+new, occupancy sm_80/sm_90, roofline, GPU-free kernel report)`

### Wave 3 - Fusion + tile/layout + memory planning + correctness + traffic report (C2)

- [ ] 12. `--fuse-rmsnorm-matmul` fusion pass
  - **Where:** `lib/Passes/FuseRmsnormMatmul.cpp`, `include/PolyKernel/Passes/Passes.td`, `test/fuse_rmsnorm_matmul.mlir`.
  - **References:** MLIR DRR/TableGen rewrite patterns or C++ `RewritePattern` (OpRewritePattern matching `polykernel.rmsnorm` -> `polykernel.matmul` with single use -> `polykernel.fused_rmsnorm_matmul`); track eliminated intermediate tensor (the rmsnorm output) for the traffic report.
  - **Acceptance:** `--fuse-rmsnorm-matmul` rewrites `rmsnorm(%x); matmul(%rms,%w)` -> `fused_rmsnorm_matmul(%x,%w)` when the rmsnorm result has a single use; records the eliminated intermediate in a side-channel (attribute/report) for traffic accounting; lit/FileCheck asserts the rewrite + that a multi-use rmsnorm is NOT fused.
  - **QA happy:** `polykernel-opt test/fuse_rmsnorm_matmul.mlir --fuse-rmsnorm-matmul | FileCheck ...` (CHECK `polykernel.fused_rmsnorm_matmul`, CHECK-NOT the standalone rmsnorm) -> evidence: `reports/w3_fuse_rmsnorm_matmul.log`.
  - **QA failure:** rmsnorm feeding TWO matmuls (multi-use) -> NOT fused (CHECK both ops survive) -> evidence: `reports/w3_fuse_multiuse_neg.log`.
  - **Commit:** `feat(pass): --fuse-rmsnorm-matmul (single-use pattern + eliminated-intermediate tracking)`

- [ ] 13. `--fuse-matmul-bias-gelu` + `--fuse-residual-rmsnorm` + `--fuse-softmax-mask` passes
  - **Where:** `lib/Passes/{FuseMatmulBiasGelu,FuseResidualRmsnorm,FuseSoftmaxMask}.cpp`, `test/fuse_matmul_bias_gelu.mlir`, `test/fuse_residual_rmsnorm.mlir`, `test/fuse_softmax_mask.mlir`.
  - **References:** same rewrite-pattern approach as todo 12. matmul->bias->gelu (single-use chain) -> `fused_matmul_bias_gelu`; add(residual)->rmsnorm -> `fused_residual_rmsnorm`; softmax with an additive mask -> `fused_softmax_mask`.
  - **Acceptance:** each pass rewrites its pattern (single-use chain) and is lit/FileCheck-tested, including a negative case (broken chain / multi-use -> not fused).
  - **QA happy:** `polykernel-opt test/fuse_matmul_bias_gelu.mlir --fuse-matmul-bias-gelu | FileCheck ...` (CHECK `polykernel.fused_matmul_bias_gelu`) + analogous for the other two -> evidence: `reports/w3_fuse_mb g.log`.
  - **QA failure:** matmul->gelu WITHOUT bias -> not fused into fused_matmul_bias_gelu (CHECK chain survives) -> evidence: `reports/w3_fuse_nobias_neg.log`.
  - **Commit:** `feat(pass): --fuse-matmul-bias-gelu, --fuse-residual-rmsnorm, --fuse-softmax-mask`

- [ ] 14. Fused CUDA kernel codegen + CPU ref: fused_rmsnorm_matmul + fused_matmul_bias_gelu
  - **Where:** `kernels/generated/fused_rmsnorm_matmul.cu`, `kernels/generated/fused_matmul_bias_gelu.cu`, `kernels/cpu/fused_*.cpp`, `tests/kernels/test_fused.py`.
  - **References:** epilogue fusion (matmul tile computes, then bias+gelu applied in-register before store -> eliminates the intermediate global write/read); prologue fusion (rmsnorm computed on-the-fly as the matmul A-operand is loaded). Follows todo 8 template + tensorforge GEMM tiling. CPU ref implements the fused math.
  - **Acceptance:** `--lower-to-cuda` lowers `fused_rmsnorm_matmul` and `fused_matmul_bias_gelu` to single fused CUDA kernels; CPU refs pass `assert_correct` vs the golden fused reference; nvcc compiles sm_80+sm_90 + PTX; the kernel report shows the eliminated intermediate.
  - **QA happy:** `pytest tests/kernels/test_fused.py -q` passes (fused CPU ref vs golden); nvcc compiles fused kernels + PTX -> evidence: `reports/w3_fused_kernels.log`.
  - **QA failure:** fused_matmul_bias_gelu CPU ref that applies gelu BEFORE bias (wrong order) fails golden -> evidence: `reports/w3_fused_order_neg.log`.
  - **Commit:** `feat(cuda): fused_rmsnorm_matmul + fused_matmul_bias_gelu kernels (epilogue/prologue fusion), golden-validated`

- [ ] 15. `--infer-tile-layout` pass
  - **Where:** `lib/Passes/InferTileLayout.cpp`, `test/infer_tile_layout.mlir`.
  - **References:** assign tile shapes (BLOCK_M/BLOCK_N/BLOCK_K) + operand layout (row-major, transposed) per matmul/fused-matmul based on shape heuristics (e.g. round up to tile multiples); attach as op attributes (`{tile_m, tile_n, tile_k, layout}`) consumed by codegen + autotuner.
  - **Acceptance:** `--infer-tile-layout` annotates each matmul/fused-matmul with tile + layout attributes; lit/FileCheck asserts attributes for representative shapes.
  - **QA happy:** `polykernel-opt test/infer_tile_layout.mlir --infer-tile-layout | FileCheck ...` (CHECK tile attrs) -> evidence: `reports/w3_tile_layout.log`.
  - **QA failure:** a matmul with a dim smaller than the min tile -> pass clamps/pads the tile attr sensibly (CHECK the clamped value, no crash) -> evidence: `reports/w3_tile_small.log`.
  - **Commit:** `feat(pass): --infer-tile-layout (per-matmul tile shape + layout attributes)`

- [ ] 16. `--plan-memory` pass
  - **Where:** `lib/Passes/PlanMemory.cpp`, `test/plan_memory.mlir`.
  - **References:** compute per-kernel shared-memory budget (from tile attrs + dtype + pipeline stages), workspace/scratch allocation, and a memory plan (which intermediates are fused away vs need workspace); attach a memory-plan attribute / emit a plan report.
  - **Acceptance:** `--plan-memory` produces a memory plan (smem bytes/kernel, workspace bytes, fused-away intermediates) and lit/FileCheck asserts the plan for the MLP fragment.
  - **QA happy:** `polykernel-opt test/plan_memory.mlir --infer-tile-layout --plan-memory | FileCheck ...` (CHECK smem/workspace figures) -> evidence: `reports/w3_plan_memory.log`.
  - **QA failure:** a tile config whose smem exceeds the sm_90 per-block limit (227KB) -> pass flags an over-budget diagnostic / reduces stages (CHECK the guard) -> evidence: `reports/w3_smem_overbudget.log`.
  - **Commit:** `feat(pass): --plan-memory (smem budget, workspace, fused-away intermediate accounting)`

- [ ] 17. End-to-end correctness + before/after fusion memory-traffic report
  - **Where:** `tools/polykernel-report/traffic_report.py`, `tests/e2e/test_mlp_correctness.py`, `reports/mlp_traffic.{json,md}`.
  - **References:** spec "correctness tests against PyTorch/NumPy" (implemented as the NumPy/ml_dtypes golden per contract C) + "before/after memory traffic report". Traffic = sum of global bytes read/written per op (from shapes+dtype); fusion eliminates the intermediate's global round-trip.
  - **Acceptance:** run the full pipeline on `examples/mlp_block.mlir` unfused vs fused; ALL ops (unfused + fused) pass golden via CPU ref (0 failed correctness); the traffic report quantifies global bytes before vs after fusion (e.g. "fused_matmul_bias_gelu eliminates N MB intermediate write+read"); report written to `reports/mlp_traffic.{json,md}`.
  - **QA happy:** `pytest tests/e2e/test_mlp_correctness.py -q` (0 failures) + `polykernel-report --traffic examples/mlp_block.mlir` shows a traffic reduction > 0 from fusion -> evidence: `reports/mlp_traffic.md`.
  - **QA failure:** inject a wrong fused kernel (broken CPU ref) into the e2e run -> the harness reports the failing op + cosine/rel/PCC and exits non-zero (0-failures criterion is enforced) -> evidence: `reports/w3_e2e_neg.log`.
  - **Commit:** `feat(report): end-to-end MLP correctness (0 failures) + before/after fusion memory-traffic report`

### Wave 4 - HIP/ROCm backend + AMDGPU ISA analysis + WMMA bf16 (C3b)

- [ ] 18. Local ROCm verification gate (GATE for local HIP run)
  - **Where:** `scripts/check_rocm.sh`, `docs/hip_backend.md` (prereqs section), `reports/w4_rocm_check.log`.
  - **References:** research bg_774c9f47 - gfx1101 OFFICIALLY SUPPORTED since ROCm 7.0; `rocmPackages.clr` auto-sets HIP_PATH/ROCM_PATH/HIP_CLANG_PATH/DEVICE_LIB_PATH/HSA_PATH; NixOS prereqs `hardware.amdgpu.opencl.enable=true`, user in `video`+`render` groups, `amdgpu` driver loaded, `nixpkgs.config.rocmSupport=true`; fallback `HSA_OVERRIDE_GFX_VERSION=11.0.1` (or `11.0.0` to emulate gfx1100 - the widely-recognized ISA-compatible target for runtimes that don't know gfx1101; gfx1101==gfx1100 768/32KiB VGPR/SGPR so either is safe).
  - **Acceptance:** `check_rocm.sh` reports: rocmPackages version (assert >= 7.0, else set HSA_OVERRIDE_GFX_VERSION=11.0.1 + warn), `rocminfo` detects a gfx1101 agent, user is in video/render groups, amdgpu driver present; writes a pass/fail gate report. If the gate fails, the script sets the HSA_OVERRIDE fallback and re-checks; if still failing, HIP-run todos degrade to "compile-only + rental-validated" (SKIPPED-local, not FAILED) with the exact missing prereq documented.
  - **QA happy:** `nix develop -c ./scripts/check_rocm.sh` -> detects gfx1101, gate PASS (or PASS-with-HSA_OVERRIDE) -> evidence: `reports/w4_rocm_check.log`.
  - **QA failure:** run `rocminfo` with `HSA_OVERRIDE_GFX_VERSION=99.0.0` (bogus) -> detection fails and the script reports the exact remedy (proves the gate diagnoses failures) -> evidence: `reports/w4_rocm_neg.log`.
  - **Commit:** `feat(hip): local ROCm verification gate (gfx1101 detect, ROCm>=7.0 check, HSA_OVERRIDE fallback, NixOS prereqs)`

- [ ] 19. `--lower-to-hip` pass + hipcc driver: build the portable template's HIP variant
  - **Where:** `lib/Passes/LowerToHip.cpp`, `lib/CodegenHIP/`, `tools/polykernel-bench/hipcc_driver.py`, `kernels/generated/*.hip` (or reuse `.cu` compiled by hipcc).
  - **References:** HIP closely aligned to CUDA (bg_774c9f47); `hipcc --offload-arch=gfx1101`; the portable template (todo 8) compiles under `hipcc -DPOLYKERNEL_HIP` UNCHANGED. hipcc driver compiles all ops (rmsnorm, gelu, matmul, softmax + fused) for gfx1101.
  - **Acceptance:** `--lower-to-hip` registered (shares the codegen template with --lower-to-cuda, differing only in the `POLYKERNEL_HIP` define + hipcc driver); hipcc compiles every generated kernel for gfx1101 without source edits to the template (proves portability).
  - **QA happy:** `hipcc_driver.py --arch gfx1101 --all` compiles all kernels clean -> evidence: `reports/w4_hipcc_compile.log`.
  - **QA failure:** compile with a CUDA-only intrinsic left unguarded in the template (simulate a portability bug) -> hipcc errors, proving the `#ifdef` discipline is enforced -> evidence: `reports/w4_hipcc_portability_neg.log`.
  - **Commit:** `feat(hip): --lower-to-hip + hipcc driver (gfx1101) - portable template compiles unchanged under hipcc`

- [ ] 20. HIP runtime launches + run on the RX 7800 XT + golden correctness (shared-template runtime validation)
  - **Where:** `lib/Runtime/HipRuntime.cpp`, `tests/kernels/test_hip_run.py`, `reports/w4_hip_correctness.log`.
  - **References:** HIP runtime launch (`hipMalloc`/`hipMemcpy`/`hipLaunchKernel`/`hipDeviceSynchronize`); this is WHERE the shared portable template's RUNTIME correctness is validated for BOTH backends (CUDA-specific paths validated here via the shared logic + on rental). Correctness vs golden (todo 7), same thresholds.
  - **Acceptance:** the HIP kernels run on the local 7800 XT (or HSA_OVERRIDE fallback) and ALL pass `assert_correct` vs golden (rmsnorm, gelu, matmul, softmax + fused) - 0 failed correctness; this validates the shared compute logic that the CUDA backend also uses. If todo 18 gate failed-local, this todo runs on the first available rental GPU and is otherwise SKIPPED-local (documented).
  - **QA happy:** `pytest tests/kernels/test_hip_run.py -q` on the 7800 XT -> all kernels pass golden (0 failures) -> evidence: `reports/w4_hip_correctness.log`.
  - **QA failure:** launch a kernel with an out-of-bounds grid (too-large block) -> HIP returns an error code that the runtime catches + reports (no silent corruption) -> evidence: `reports/w4_hip_launch_neg.log`.
  - **Commit:** `feat(hip): HIP runtime launches + golden correctness on RX 7800 XT (validates shared template for both backends)`

- [ ] 21. AMDGPU ISA analyzer: VGPR/SGPR/LDS/spills + gfx1101 occupancy + AMD report
  - **Where:** `lib/Analysis/ AmdIsaAnalyzer.cpp`, `tools/polykernel-bench/amd_analyze.py`, `lib/Analysis/tests/amd_isa_test.cpp` (gtest), `lib/Analysis/tests/fixtures/*.s`.
  - **References:** research bg_774c9f47 - `llvm-objdump --triple=amdgcn-amd-amdhsa --mcpu=gfx1101 -d`; `hipcc --save-temps` -> `.s` with `.amdhsa_next_free_vgpr`, `.amdhsa_next_free_sgpr`, `.amdhsa_group_segment_fixed_size` (LDS), `.amdhsa_private_segment_fixed_size` (per-thread scratch=spills); spills also = `scratch_store_*/scratch_load_*` insn count; `llvm-readobj --notes` for ELF metadata. gfx1101 occupancy: per-CU VGPR/SGPR/LDS file sizes read from `rocminfo` at runtime (research cites ~768KiB VGPR file, 32KiB SGPR, 128KiB LDS, wave32); occupancy=min(VGPR-limited, SGPR-limited, LDS-limited, wave-limited). Separate AMD tuning DB namespace.
  - **Acceptance:** AmdIsaAnalyzer parses `.amdhsa_next_free_vgpr/sgpr`, LDS, scratch from a `.s`/object (gtest vs captured fixtures); computes gfx1101 occupancy; detects spills (`scratch_*` count + private_segment_fixed_size>0); `polykernel-bench --backend=hip --mode=analyze --arch gfx1101` emits the AMD per-kernel report (VGPR/SGPR/LDS/spills/occupancy/bottleneck); results stored in the separate AMD tuning DB namespace.
  - **QA happy:** `ctest -R amd_isa` passes; `polykernel-bench --backend=hip --mode=analyze --kernel matmul --arch gfx1101` prints VGPR/LDS/occupancy matching the fixture -> evidence: `reports/w4_amd_analyze.log`.
  - **QA failure:** feed a `.s` fixture WITH `scratch_store`/`private_segment_fixed_size>0` -> analyzer reports spills>0 + flags "register pressure / spill" bottleneck (proves spill detection) -> evidence: `reports/w4_amd_spill.log`.
  - **Commit:** `feat(analyze): AMDGPU ISA analyzer (VGPR/SGPR/LDS/spills + gfx1101 occupancy) + separate AMD tuning DB`

- [ ] 22. WMMA bf16 tensor-core path for MatMul (RDNA3) - additive to the scalar baseline
  - **Where:** `kernels/generated/matmul_wmma.cu` (HIP WMMA), `kernels/cpu/matmul.cpp` (baseline unchanged), `tests/kernels/test_wmma.py`.
  - **References:** research bg_774c9f47 - gfx1101 has `v_wmma_f32_16x16x16_bf16`; use `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (wave32, 16x16x16, 16 elems/lane input, 8xf32/lane accum) and/or rocWMMA (`rocmPackages.rocwmma`); GPUOpen "WMMA on RDNA3" guide; vLLM ships production WMMA on RDNA3. NEW territory (no tensorforge pattern) -> the scalar/tiled matmul (todo 10) REMAINS the correctness baseline; WMMA is an additive variant the autotuner can select.
  - **Acceptance:** a WMMA bf16 matmul variant compiles with hipcc gfx1101 + runs on the 7800 XT; passes `assert_correct` vs golden (same thresholds); the kernel report notes which path (wmma vs scalar) was used. If WMMA correctness fails locally, it is excluded from the tuning cache (correctness-gated) and the scalar baseline remains - never blocks the wave.
  - **QA happy:** `pytest tests/kernels/test_wmma.py -q` on 7800 XT -> WMMA bf16 matmul passes golden -> evidence: `reports/w4_wmma.log`.
  - **QA failure:** a WMMA fragment with a wrong lane-mapping fails golden -> autotuner discards it (validated=false), scalar baseline still passes (proves WMMA is safely additive) -> evidence: `reports/w4_wmma_neg.log`.
  - **Commit:** `feat(hip): WMMA bf16 matmul path (RDNA3 v_wmma, additive to scalar baseline), golden-validated`

- [ ] 23. gfx942 (MI300) compile target for reports + AMD tuning DB finalization
  - **Where:** `tools/polykernel-bench/hipcc_driver.py` (`--arch gfx942`), `lib/Autotune/AmdTuningDb.cpp`, `reports/mi300_compile.log`.
  - **References:** MI300 = gfx942 (CDNA3, the report headline AMD part); compile-only locally (no MI300 hardware), runtime on rental (W5). The AMD tuning DB stores per-(gpu,op,shape) best configs separately for gfx1101 (local) and gfx942 (rental).
  - **Acceptance:** hipcc compiles all kernels for gfx942 (compile-only, no run) + AMDGPU ISA captured for the report; AMD tuning DB schema finalized + gtest for r/w; MI300 report figures are clearly-labeled "compile-only locally; runtime on rental" until W5 rental.
  - **QA happy:** `hipcc_driver.py --arch gfx942 --all` compiles clean + ISA dumped -> evidence: `reports/mi300_compile.log`.
  - **QA failure:** attempt to RUN a gfx942 binary locally (no MI300) -> runtime reports "no gfx942 device, compile-only" gracefully (not a crash) -> evidence: `reports/w4_gfx942_norun.log`.
  - **Commit:** `feat(hip): gfx942 (MI300) compile target + AMD tuning DB finalization (separate gfx1101/gfx942 namespaces)`

### Wave 5 - Autotuner + analyzer + C++ runtime + bench (C4)

- [ ] 24. Autotuner config space + enumerator + JSON tuning-cache schema
  - **Where:** `include/PolyKernel/Autotune/ConfigSpace.h`, `lib/Autotune/{ConfigSpace,TuningCache}.cpp`, `lib/Autotune/tests/{config_space_test,tuning_cache_test}.cpp` (gtest), `tuning/schema.json`.
  - **References:** research bg_fbed4570 - Triton/CUTLASS config space BLOCK_M{16,32,64,128}, BLOCK_N{16,32,64,128}, BLOCK_K{32,64,128}, num_warps{4,8}, vector_width{1,2,4,8}, unroll{1,2,4,8}, shared_memory_stages{2,3,4}; prune to ~100-150 configs (drop BLOCK_M*BLOCK_N<4096 with num_warps=8, smem-over-budget stages, etc.). JSON tuning-cache schema: see Pinned contract H (AUTHORITATIVE) - implement the versioned-envelope format (`version`, `entries[]`) with `shape:{M,N,K,dtype}`, `scored_by`, `time_ms` (null if compile-time-model), and a `correctness:{cosine,max_rel_err,pcc}` sub-object; do NOT redefine the schema here.
  - **Acceptance:** ConfigSpace enumerates the bounded grid + applies pruning (gtest asserts the pruned count is within ~100-150 and excludes invalid configs); TuningCache reads/writes the pinned JSON schema (gtest round-trip); the schema is documented in `tuning/schema.json`.
  - **QA happy:** `ctest -R 'config_space|tuning_cache'` passes; enumerator emits a bounded, pruned config list; a sample cache entry round-trips through JSON -> evidence: `reports/w5_configspace.log`.
  - **QA failure:** TuningCache given a JSON entry missing `scored_by` (or any contract-H field) -> rejects with a schema error (not a silent default) -> evidence: `reports/w5_schema_neg.log`.
  - **Commit:** `feat(autotune): bounded config space + pruner + JSON tuning-cache schema (gpu/op/shape/best/time/validated)`

- [ ] 25. Correctness-gated benchmarking harness (`polykernel-bench`)
  - **Where:** `tools/polykernel-bench/bench.py`, `lib/Autotune/Benchmark.cpp`, `tests/autotune/test_bench_gate.py`.
  - **References:** correctness-gated benchmarking (Verification strategy): a variant's time is recorded ONLY if it first passes the golden `assert_correct`; failing variants are discarded + logged, never "best". Timing via HIP events on the 7800 XT (`hipEventRecord`/`hipEventElapsedTime`); CUDA compile-time scoring locally (analyzer estimate) + runtime timing on rental.
  - **Acceptance:** `polykernel-bench --autotune --op fused_matmul_bias_gelu --shape 2048,4096,11008 --dtype bf16 --backend hip --arch gfx1101` runs each pruned variant, correctness-gates it, times passing variants, and writes the best to the tuning cache with `validated:true`; a variant that fails correctness is logged + excluded (gtest/pytest asserts a deliberately-broken variant is never selected).
  - **QA happy:** `pytest tests/autotune/test_bench_gate.py -q` -> the harness selects the fastest VALIDATED variant; cache entry has validated:true -> evidence: `reports/w5_bench.log`.
  - **QA failure:** inject a fast-but-wrong variant (passes timing, fails golden) -> the harness EXCLUDES it (validated:false, not selected) and logs the rejection -> evidence: `reports/w5_bench_gate_neg.log`.
  - **Commit:** `feat(autotune): correctness-gated benchmarking harness (golden before timing; validated-only cache entries)`

- [ ] 26. C++ runtime: GPU detect -> select best cached kernel -> load -> serve
  - **Where:** `include/PolyKernel/Runtime/{Runtime,KernelCache,DeviceDetect}.h`, `lib/Runtime/{Runtime,KernelCache,DeviceDetect}.cpp`, `lib/Runtime/tests/{runtime_test,device_detect_test}.cpp` (gtest).
  - **References:** spec runtime kernel cache ("detect GPU -> select best compiled kernel -> load cached binary -> serve"). DeviceDetect uses `hipGetDeviceProperties` (HIP) / `cudaGetDeviceProperties` (CUDA) to get the arch (gfx1101/gfx942/sm_80/sm_90); KernelCache maps (gpu,op,shape)->best variant from the tuning DB and loads the compiled variant (dlopen a cached `.so` or select a compiled symbol). gtest mocks the device-detect + selection logic (no GPU needed for the logic test).
  - **Acceptance:** Runtime detects the device arch, selects the best cached kernel for a (gpu,op,shape) request, loads it, and executes; gtest (with a mocked device) asserts the selection logic picks the cache entry for the detected arch; on the 7800 XT, Runtime runs a fused op end-to-end and matches the golden.
  - **QA happy:** `ctest -R 'runtime|device_detect'` passes (mocked selection); on 7800 XT, `Runtime` executes fused_matmul_bias_gelu via the cached best variant + matches golden -> evidence: `reports/w5_runtime.log`.
  - **QA failure:** request a (gpu,op,shape) with NO cache entry -> Runtime returns a clear "no tuned kernel; run autotuner" error (not a wrong kernel, not a crash) -> evidence: `reports/w5_runtime_miss_neg.log`.
  - **Commit:** `feat(runtime): C++ runtime (GPU detect -> select best cached kernel -> load -> serve) + kernel cache`

- [ ] 27. Full per-kernel report (`--emit-kernel-report` + `polykernel-report`)
  - **Where:** `lib/Passes/EmitKernelReport.cpp`, `tools/polykernel-report/report.py`, `lib/Analysis/tests/kernel_report_test.cpp` (gtest), `reports/kernel_report_example.{json,txt}`.
  - **References:** spec per-kernel report format (registers/thread, smem/block, occupancy, global r/w, bottleneck, suggested fixes). Per-kernel report schema: see Pinned contract H (AUTHORITATIVE) - nested `occupancy:{active_warps_per_sm,max_warps_per_sm,occupancy_pct,limiter}` and `traffic:{global_read_bytes,global_write_bytes,arithmetic_intensity_flop_per_byte,roofline}` objects, split `spill_stores_bytes`/`spill_loads_bytes`, `roofline` enum (NOT `bound_type`), `path ∈ {scalar,wmma,mma}`; do NOT redefine the schema here. Merges CUDA compile-time analysis (todo 11) + AMD ISA analysis (todo 21) + autotuner result (todo 25). Suggested-fix rules: high registers->reduce unroll/split accumulator; spills->reduce register pressure; low occupancy->smaller tile/fewer regs; memory-bound->wider vectorized loads/fusion.
  - **Acceptance:** `polykernel-opt ... --emit-kernel-report` + `polykernel-report` produce the spec-format report for each generated kernel (CUDA + AMD); gtest asserts the suggested-fix rules fire for crafted inputs (e.g. a high-register kernel -> "reduce unroll"); the example report matches the spec's illustrative output shape.
  - **QA happy:** `polykernel-report --kernel fused_rmsnorm_matmul --backend cuda --arch sm_90` prints the full report (registers/smem/occupancy/traffic/bottleneck/suggested-fixes) matching the schema -> evidence: `reports/kernel_report_example.json`.
  - **QA failure:** generate a report for a kernel with spills>0 -> suggested_fixes includes a register-pressure/spill remedy (proves the rules fire) -> evidence: `reports/w7_report_spill_fix.log`.
  - **Commit:** `feat(report): full per-kernel report (CUDA+AMD analysis + autotuner + suggested-fix rules) + pinned schema`

- [ ] 28. Real benchmark on rented GPUs (OWNER-GATED: RunPod) + speedup numbers
  - **Where:** `benchmarks/rent_runpod.sh`, `benchmarks/bench_cuda.cpp`, `benchmarks/bench_hip.cpp`, `benchmarks/run_bench_suite.py`, `reports/{h100,mi300}_bench.{json,md}`.
  - **References:** user decision: rent NVIDIA H100/A100 + AMD MI300 on RunPod, ~$50-100 ceiling, bounded autotuner search. RunPod = ad-hoc per-hour GPU pods (docker + per-second billing). The bench suite measures unfused baseline vs fused generated vs best autotuned for the MLP fragment on H100/A100 (CUDA) + MI300 (HIP gfx942). Budget guard: the script caps total runtime + logs spend; the autotuner search is the bounded grid (todo 24).
  - **Acceptance:** `benchmarks/rent_runpod.sh` provisions H100/A100 + MI300 pods (requires RunPod credentials), builds the engine in-container, runs the bounded autotuner + bench suite, and writes real unfused/fused/autotuned speedups to `reports/{h100,mi300}_bench.{json,md}`; if RunPod credentials are ABSENT, the suite is SKIPPED (not FAILED) and the reports are populated with clearly-labeled perf-model PROJECTIONS (from the compile-time analyzer + roofline, todo 11/27) + the opt-in rental script is documented. Spend ceiling enforced (~$50-100).
  - **QA happy:** with credentials, `./benchmarks/rent_runpod.sh --suite mlp` produces real speedup numbers (unfused 1.00x baseline, fused > 1.0x, autotuned >= fused) within budget -> evidence: `reports/h100_bench.md`, `reports/mi300_bench.md`.
  - **QA failure:** run without RunPod credentials -> suite exits SKIPPED with the missing credential named + writes clearly-labeled projected numbers (proves graceful degradation, no crash, no surprise spend) -> evidence: `reports/w5_rent_skipped.log`.
  - **Commit:** `feat(bench): RunPod benchmark suite (H100/A100 + MI300 real speedups, budget guard) + projected-number fallback`

### Wave 6 - Modal deployment + dashboard + reports + writeup (C6, OWNER-GATED: Modal token)

- [ ] 29. Modal Image + app skeleton (CURRENT API)
  - **Where:** `modal/image.py`, `modal/app.py`, `modal/requirements.txt` (pins `modal==<version>`).
  - **References:** research bg_fd807b34 - CURRENT Modal API. Image: `modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12").entrypoint([]).apt_install("cmake","build-essential").add_local_dir("./", remote_path="/app").run_commands("cd /app && cmake -B build && cmake --build build")`. add_python REQUIRED (NVIDIA images lack Python). Pin the modal client version to avoid API drift.
  - **Acceptance:** `modal/image.py` builds a CUDA-toolkit image that compiles the PolyKernel engine `.so` at image-build time; `modal/app.py` defines `modal.App("polykernel", image=...)`; `modal build` (or `modal serve` dry-run) succeeds locally WITHOUT a token (image definition is valid). Modal client version pinned in requirements.
  - **QA happy:** `python -c "import modal.image" ` imports + the image definition is constructible; `modal serve modal/app.py --dry-run` (or local image build) validates the image spec -> evidence: `reports/w6_image.log`.
  - **QA failure:** an image step with a bad nvcc flag -> image build fails with the compile error surfaced (proves build-time compilation is real) -> evidence: `reports/w6_image_build_neg.log`.
  - **Commit:** `feat(modal): Modal CUDA-toolkit Image (nvidia/cuda devel + add_python) compiling the engine + app skeleton (pinned client)`

- [ ] 30. GPU endpoints /predict /benchmark /kernels /report
  - **Where:** `modal/app.py` (the `@app.cls` + endpoints).
  - **References:** bg_fd807b34 - `@app.cls(gpu="H100"/"A10", timeout=..., startup_timeout=...)`, `@modal.enter()` to ctypes-load the engine `.so` + load weights from a `modal.Volume`, `@modal.fastapi_endpoint(method="POST")` (NOT web_endpoint) for /predict /benchmark; /kernels lists tuned kernels; /report returns the per-kernel report. ctypes.CDLL for the `.so`; subprocess for the bench binary.
  - **Acceptance:** the class exposes `/predict` (run the fused block), `/benchmark` (run a kernel bench, return JSON), `/kernels` (list tuned kernels from the cache), `/report` (return the per-kernel report); endpoints are wired with `@modal.fastapi_endpoint`; locally testable via `modal serve` (dev URL) without cloud spend.
  - **QA happy:** `modal serve modal/app.py` (dev) + `curl -XPOST $URL/predict` returns a prediction; `/kernels` returns the tuned-kernel list; `/report` returns the schema-conformant report -> evidence: `reports/w6_endpoints.log`.
  - **QA failure:** `/predict` with a malformed payload -> returns a 422/400 with a clear error (not a 500 crash) -> evidence: `reports/w6_predict_badinput.log`.
  - **Commit:** `feat(modal): GPU endpoints /predict /benchmark /kernels /report (fastapi_endpoint, ctypes engine load, Volume weights)`

- [ ] 31. Cold-start measurement + autoscaling
  - **Where:** `modal/app.py` (autoscaling params + `@modal.enter(snap=)`), `modal/cold_start.py` (client-side measurement).
  - **References:** bg_fd807b34 - `min_containers` (replaces keep_warm), `max_containers`, `buffer_containers`, `scaledown_window`, `enable_memory_snapshot=True`, `experimental_options={"enable_gpu_snapshot":True}`; `@modal.enter(snap=True)` runs BEFORE snapshot (no GPU), `@modal.enter(snap=False)` runs AFTER restore (GPU available, do warmup). Containers boot ~1s; measure cold start client-side by sleeping > scaledown_window then timing the first vs warm request.
  - **Acceptance:** the class sets `min_containers`/`scaledown_window`/`enable_memory_snapshot` + a `@modal.enter(snap=True)` load + `@modal.enter(snap=False)` GPU warmup; `modal/cold_start.py` measures cold-vs-warm latency client-side and writes `reports/cold_start.json`; autoscaling params are documented.
  - **QA happy:** deployed (or local `modal serve`), `python modal/cold_start.py --url $URL` reports cold_ms + warm_ms (cold > warm) -> evidence: `reports/cold_start.json`.
  - **QA failure:** set `scaledown_window` to the min (2s) and measure -> containers scale down fast and cold start is observed reliably (proves the measurement captures cold starts) -> evidence: `reports/w6_coldstart_neg.log`.
  - **Commit:** `feat(modal): cold-start measurement (enter(snap=) warmup, min_containers/scaledown_window/memory-snapshot) + autoscaling`

- [ ] 32. GPU-aware kernel cache on Modal (reuse C4 runtime)
  - **Where:** `modal/app.py` (`@modal.enter` calls the C4 Runtime), `lib/Runtime/` (reused).
  - **References:** spec "detect GPU -> select best compiled kernel -> load cached binary -> serve". Reuse the C++ Runtime (todo 26): at container `@modal.enter`, detect the Modal GPU (H100/A10), select the best cached kernel for that arch from the tuning DB, load it. The tuning DB + compiled variants are baked into the image (or on a Volume).
  - **Acceptance:** on container start, the Runtime detects the Modal GPU arch + loads the matching best cached kernel; `/predict` uses the selected kernel; if no tuned kernel exists for the detected arch, the endpoint reports "run autotuner for this GPU" (graceful). gtest of the selection logic is reused from todo 26.
  - **QA happy:** container start logs "detected <arch>, loaded best cached kernel for <op,shape>"; `/predict` succeeds using it -> evidence: `reports/w6_kernel_cache.log`.
  - **QA failure:** deploy on a GPU arch with NO tuning entry -> `/predict` returns the "no tuned kernel" error gracefully (not a wrong result) -> evidence: `reports/w6_kernel_cache_miss.log`.
  - **Commit:** `feat(modal): GPU-aware kernel cache on Modal (detect -> select best cached kernel -> load, reuses C4 runtime)`

- [ ] 33. Deploy (OWNER-GATED: MODAL_TOKEN) + live cold-start/autoscaling verification
  - **Where:** `modal/deploy.py`, `reports/w6_deploy.log`.
  - **References:** bg_fd807b34 - `modal setup` (token), `modal deploy modal/app.py` (persistent), `modal secret create`, `modal volume put`. User decision: provide a Modal token for REAL deploy. Modal free tier/credits likely cover a small demo (~$0).
  - **Acceptance:** if `MODAL_TOKEN` is present, `modal/deploy.py` runs `modal deploy`, verifies the deployed endpoints respond, measures live cold-start + autoscaling, and writes `reports/w6_deploy.log` (deploy URL, cold/warm latency, autoscaling behavior); if `MODAL_TOKEN` is ABSENT, the full app + local `modal serve` dev path + documented `modal deploy` exist and cloud deploy is marked SKIPPED with the missing credential (never FAILED, never blocks).
  - **QA happy:** with a token, `python modal/deploy.py` deploys + the deployed `/predict` responds + cold-start is measured -> evidence: `reports/w6_deploy.log`.
  - **QA failure:** run `modal/deploy.py` with no token -> exits SKIPPED, names `MODAL_TOKEN` as missing, documents the exact `modal setup` + `modal deploy` commands (proves graceful degradation) -> evidence: `reports/w6_deploy_skipped.log`.
  - **Commit:** `feat(modal): deploy flow (modal deploy + live cold-start/autoscaling verification) with SKIPPED fallback on missing token`

- [ ] 34. Benchmark dashboard + HTML reports + docs + README writeup
  - **Where:** `tools/polykernel-report/dashboard.py`, `reports/{h100_report,mi300_report,dataflow_report}.html`, `reports/benchmark_report.{md,html}`, `docs/{architecture,compiler_pipeline,cuda_backend,hip_backend,dataflow_backend,performance_model}.md`, `README.md`.
  - **References:** spec demo report format (model fragment; backends enabled; CUDA H100 unfused/fused/autotuned speedups; AMD MI300; dataflow simulator metrics; compiler stats: passes/generated kernels/validated/failed correctness=0). Static self-contained HTML + inline JS (NO web framework - scope guardrail). README headline/subtitle/resume positioning per spec.
  - **Acceptance:** `polykernel-report --dashboard` generates the benchmark report (text + HTML) aggregating CUDA (real or projected), AMD (real or projected), dataflow metrics, + compiler stats (0 failed correctness); `h100_report.html`/`mi300_report.html`/`dataflow_report.html` render as static self-contained HTML; the 6 docs are written; README has the spec headline/subtitle/resume bullets. All numbers are clearly labeled real-vs-projected.
  - **QA happy:** `polykernel-report --dashboard` writes the reports; opening `reports/benchmark_report.html` shows the model fragment + backends + speedups + "failed correctness: 0"; HTML is self-contained (no external CDN) -> evidence: `reports/benchmark_report.html`.
  - **QA failure:** generate the dashboard with a deliberately injected failed correctness test -> the report shows "failed correctness: N>0" prominently (proves the dashboard reflects reality, doesn't hide failures) -> evidence: `reports/w6_dashboard_neg.log`.
  - **Commit:** `docs: benchmark dashboard + HTML reports (h100/mi300/dataflow) + architecture/backend/perf docs + README positioning`

### Wave 7 - Cerebras-CSL dataflow simulator + HTML visualizer (C5)

- [ ] 35. Dataflow simulator core: PE grid + wavelet/color/router model
  - **Where:** `include/PolyKernel/Dataflow/{Pe,Wavelet,Color,Router,Grid}.h`, `lib/DataflowSim/{Pe,Wavelet,Color,Router,Grid}.cpp`, `lib/DataflowSim/tests/{router_test,color_test,grid_test}.cpp` (gtest).
  - **References:** research bg_ffb419c4 - PE = CE (4x FP16 FMAC, 64-bit datapath) + 5-port router (RAMP + E/W/N/S, 32b/link, single-cycle hop, lossless backpressure) + 48KB SRAM (8 banks x 6KB, 2 reads+1 write/cycle). Wavelet = 32-bit (16b data + 16b control/index). Color = 5-bit tag, 24 routable colors (IDs 0-23), virtual channels over a shared physical link. NOTE correct terminology: CE+FMAC (NOT FMU/PMU), `@set_color_config` rx/tx (NOT set_color), `fabin_dsd`/`fabout_dsd` (NOT fdata/fcast/fmove). Default grid 64x64.
  - **Acceptance:** Grid models a 64x64 PE mesh; each PE has 48KB SRAM (8 banks) + a 5-port router; Wavelet is 32-bit (16b data+16b control); Color is a 5-bit tag (24 routable); Router routes a wavelet one hop per cycle in E/W/N/S/RAMP with `rx` (one dir) / `tx` (set of dirs = multicast); gtest asserts single-cycle hop, multicast tx, and color isolation (congestion on one color doesn't block another).
  - **QA happy:** `ctest -R 'router|color|grid'` passes (single-cycle hop, multicast, 24-color isolation) -> evidence: `reports/w7_sim_core.log`.
  - **QA failure:** route a wavelet off-grid (east from the rightmost column) -> the router drops/flags it (no out-of-bounds crash) -> evidence: `reports/w7_router_oob_neg.log`.
  - **Commit:** `feat(dataflow): simulator core (64x64 PE grid, CE+router+48KB SRAM, 32b wavelet, 24 colors, single-cycle routing)`

- [ ] 36. Dataflow task model + scheduling (dataflow-triggered compute)
  - **Where:** `include/PolyKernel/Dataflow/{Task,TaskPicker,Scheduler}.h`, `lib/DataflowSim/{Task,Scheduler}.cpp`, `lib/DataflowSim/tests/scheduler_test.cpp` (gtest).
  - **References:** bg_ffb419c4 - 3 task types: data task (wavelet-triggered, bound to a color/data_task_id), local task (self-triggered via activate), control task. Task picker selects a task when activated AND unblocked; `@activate`/`@block`/`@unblock` rendezvous. Dataflow = compute triggered by data arrival. (No `@compute`/`@data` decorators - `compute` is an exported host-callable convention.)
  - **Acceptance:** Scheduler models data tasks (fire on wavelet arrival on a bound color), local tasks (fire on activate), the activate/block/unblock rendezvous (a task runs only when activated AND unblocked), and a task picker; gtest asserts a compute task fires only after BOTH its inputs arrive (rendezvous), and a blocked task does not fire until unblocked.
  - **QA happy:** `ctest -R scheduler` passes (compute fires only after both wavelets arrive; block/unblock gates firing) -> evidence: `reports/w7_task_model.log`.
  - **QA failure:** activate a task but leave it blocked -> it does NOT fire until unblocked (gtest asserts no premature fire) -> evidence: `reports/w7_task_blocked_neg.log`.
  - **Commit:** `feat(dataflow): task model + scheduler (wavelet-triggered data tasks, activate/block/unblock rendezvous, task picker)`

- [ ] 37. SUMMA matmul mapping + lowering of polykernel ops + golden correctness
  - **Where:** `lib/DataflowSim/Summa.cpp`, `lib/DataflowSim/LowerToDataflow.cpp`, `lib/DataflowSim/tests/{summa_test,dataflow_correctness_test}.cpp` (gtest + golden).
  - **References:** bg_ffb419c4 - the official Cerebras GEMM (csl-examples gemm-collectives_2d) is SUMMA: P x P grid; step i: column i broadcasts its A tile EAST along the row, row i broadcasts its B tile SOUTH along the column; each PE accumulates C_tile += Ap*Bp (output-stationary C in local SRAM) via an @fmacs-style outer-product; 2 colors/axis (ping-pong double-buffer); dataflow rendezvous (x_done -> activate(compute), y_done -> unblock(compute)). This EXACTLY matches the spec's "A east / B south / local C accumulate / row-col reduction". The simulator FUNCTIONALLY EXECUTES the tile program, so its numerical output is validated vs the NumPy golden (todo 7, same thresholds).
  - **Acceptance:** `--lower-to-dataflow` (or a simulator lowering) maps `polykernel.matmul`/fused-matmul onto the SUMMA schedule (A broadcast east, B broadcast south, local C accumulate, 2 colors/axis ping-pong, rendezvous-triggered compute); the simulator executes the schedule and the resulting C matches the golden `assert_correct` (cosine>=0.999/rel<=1e-2/PCC>=0.99) for a representative matmul; gtest asserts the routing pattern (A east, B south) + rendezvous.
  - **QA happy:** `ctest -R 'summa|dataflow_correctness'` passes (simulator-executed matmul matches golden; routing is A-east/B-south) -> evidence: `reports/w7_summa.log`.
  - **QA failure:** break the rendezvous (fire compute on only ONE input) -> simulator output is wrong + fails golden (proves the correctness test catches dataflow bugs) -> evidence: `reports/w7_summa_neg.log`.
  - **Commit:** `feat(dataflow): SUMMA matmul mapping (A-east/B-south, output-stationary C, ping-pong colors, rendezvous) + golden-validated execution`

- [ ] 38. Simulator metrics + fusion traffic reduction + `polykernel-report --backend=dataflow`
  - **Where:** `lib/DataflowSim/Metrics.cpp`, `tools/polykernel-report/dataflow_report.py`, `lib/DataflowSim/tests/metrics_test.cpp` (gtest), `reports/dataflow_metrics.json`.
  - **References:** bg_ffb419c4 metrics: grid utilization (active PEs / total), local SRAM pressure (resident tiles+double-buffers vs 48KB), messages sent (total wavelets = sum of fabout extents), average hop distance (1 cycle/hop; broadcast along width P = P-1 hops), communication bottleneck (Active/Delayed/Backpressure/Idle states), critical-path cycles, fusion traffic reduction (wavelet count before vs after fusion - fused MLP keeps intermediates in local SRAM -> zero fabric wavelets for them). cslc `--out-routes` / `calculate_cycles` are the real-SDK analogs we model.
  - **Acceptance:** the simulator reports grid utilization, SRAM pressure, messages sent, avg hop distance, comm bottleneck, critical-path cycles, AND fusion traffic reduction (wavelet count unfused vs fused); gtest asserts metrics against HAND-COMPUTED values for a small (e.g. 4x4) grid; `polykernel-report --backend=dataflow examples/mlp_block.mlir` emits the spec's dataflow metrics (utilization, traffic reduction from fusion, bottleneck) to `reports/dataflow_metrics.json`.
  - **QA happy:** `ctest -R metrics` passes (hand-computed 4x4 metrics match); `polykernel-report --backend=dataflow` shows utilization%, traffic-reduction%>0 from fusion, + bottleneck -> evidence: `reports/dataflow_metrics.json`.
  - **QA failure:** simulate a config that over-subscribes local SRAM (tiles > 48KB) -> SRAM pressure > 100% is reported + flagged as the bottleneck (proves pressure detection) -> evidence: `reports/w7_sram_over Neg.log`.
  - **Commit:** `feat(dataflow): simulator metrics (utilization/SRAM/messages/hops/bottleneck/critical-path/fusion-traffic-reduction)`

- [ ] 39. Dataflow HTML visualizer (static, self-contained)
  - **Where:** `tools/polykernel-report/dataflow_viz.py`, `reports/dataflow_report.html`, `lib/DataflowSim/viz_template.html`.
  - **References:** spec dataflow HTML viz (PE grid, tile routes, local SRAM pressure, message traffic, fusion savings, critical path) + `reports/dataflow_report.html`. Scope guardrail: STATIC self-contained HTML + inline JS (no web framework, no CDN).
  - **Acceptance:** `polykernel-report --backend=dataflow --viz` generates `reports/dataflow_report.html` rendering the PE grid, tile routes (A-east/B-south), SRAM-pressure heatmap, message traffic, fusion savings, and critical path; the HTML is self-contained (opens offline, no external deps).
  - **QA happy:** generate the viz; the HTML contains the grid + routes + SRAM heatmap + traffic + critical-path elements (grep the HTML for the expected SVG/canvas/data sections) -> evidence: `reports/dataflow_report.html`.
  - **QA failure:** generate the viz with no external network -> it still renders fully (proves self-containment; assert no `http://`/CDN references in the HTML) -> evidence: `reports/w7_viz_offline.log`.
  - **Commit:** `feat(dataflow): static self-contained HTML visualizer (grid/routes/SRAM-pressure/traffic/fusion-savings/critical-path)`

### Wave 8+ - Elite (post-MVP; starts only after waves 1-7 are green)

- [ ] 40. FlashAttention-lite: QKT + masked softmax + PV + KV-cache r/w
  - **Where:** `kernels/generated/attention_prefill.cu`, `kernels/generated/kv_cache_update.cu`, `kernels/cpu/attention.cpp`, `lib/Passes/FuseKvAppendAttention.cpp`, `tests/kernels/test_attention.py`.
  - **References:** spec elite #1 (FlashAttention-lite: QKT, masked softmax, PV, KV-cache read/write). Build on `polykernel.attention` (prefill) + `polykernel.kv_cache_update` + `fused_kv_append_attention` ops (defined in W1). Online softmax (flash-style tiling) to avoid materializing the full attention matrix. CPU ref + golden (with attention-appropriate thresholds).
  - **Acceptance:** attention-prefill kernel computes QKT -> masked softmax -> PV with flash-style tiling; kv_cache_update appends K/V to the cache; `fused_kv_append_attention` fuses append+attention; CUDA compiles sm_80/sm_90 + HIP compiles/runs gfx1101; CPU ref passes golden; lit tests the fusion pass.
  - **QA happy:** `pytest tests/kernels/test_attention.py -q` (prefill + kv-cache vs golden) passes; HIP runs on 7800 XT -> evidence: `reports/w8_attention.log`.
  - **QA failure:** attention with a wrong causal mask leaks future tokens -> golden fails (proves mask correctness is tested) -> evidence: `reports/w8_attention_mask_neg.log`.
  - **Commit:** `feat(elite): FlashAttention-lite (QKT + masked softmax + PV + KV-cache r/w) + fused_kv_append_attention`

- [ ] 41. Quantized inference: int8 weight-only + fp8 simulation
  - **Where:** `kernels/generated/matmul_int8.cu`, `lib/Codegen/Quantize.cpp`, `kernels/cpu/matmul_quant.cpp`, `tests/kernels/test_quant.py`.
  - **References:** spec elite #2 (fp16, bf16, int8 weight-only quantization, fp8 simulation). int8 weight-only: weights quantized to int8 (per-channel scale), activations bf16, dequantize-and-multiply; fp8 (e4m3/e5m2) simulated via ml_dtypes float8 in the golden. Quantization-aware golden thresholds (relaxed: cosine>=0.99, rel<=5e-2 for int8).
  - **Acceptance:** int8 weight-only matmul kernel (CUDA + HIP) + fp8-sim path; quantization-aware golden (`tests/golden` extended with int8/fp8 references + relaxed thresholds); CPU ref passes the quantized golden; the kernel report notes the dtype/quant path.
  - **QA happy:** `pytest tests/kernels/test_quant.py -q` (int8 weight-only + fp8-sim vs quantization-aware golden) passes -> evidence: `reports/w8_quant.log`.
  - **QA failure:** int8 matmul with a wrong per-channel scale fails the quantized golden (proves scale correctness is tested) -> evidence: `reports/w8_quant_scale_neg.log`.
  - **Commit:** `feat(elite): quantized inference (int8 weight-only + fp8 simulation) + quantization-aware golden`

- [ ] 42. Runtime kernel cache hardening (persistent + multi-GPU + invalidation)
  - **Where:** `lib/Runtime/KernelCache.cpp` (extend), `lib/Runtime/tests/kernel_cache_persist_test.cpp` (gtest).
  - **References:** spec elite #3 (runtime kernel cache: detect GPU -> select best compiled kernel -> load cached binary -> serve). Harden todo 26: persistent on-disk cache (the tuning DB + compiled `.so`s), cache invalidation (keyed by gpu+op+shape+dtype+kernel-hash), multi-GPU selection, fallback when a cached binary is stale/missing.
  - **Acceptance:** the runtime persists the kernel cache to disk, invalidates stale entries (hash mismatch), selects per-GPU in a multi-GPU setup, and falls back to re-autotuning on a cache miss; gtest covers persist/reload/invalidation/multi-GPU selection.
  - **QA happy:** `ctest -R kernel_cache_persist` passes (cache persists across runs, reloads, selects per-GPU) -> evidence: `reports/w8_cache_persist.log`.
  - **QA failure:** corrupt a cached binary (hash mismatch) -> runtime invalidates it + falls back to re-autotune (not a wrong/stale kernel) -> evidence: `reports/w8_cache_stale_neg.log`.
  - **Commit:** `feat(elite): hardened runtime kernel cache (persistent on-disk, hash invalidation, multi-GPU selection, miss fallback)`

- [ ] 43. Dataflow HTML viz polish (interactive routes/timeline/critical-path)
  - **Where:** `tools/polykernel-report/dataflow_viz.py` (extend), `reports/dataflow_report.html`.
  - **References:** spec elite #4 (dataflow backend report: PE grid, tile routes, local SRAM pressure, message traffic, fusion savings, critical path). Extend todo 39 with interactivity (hover a PE for its instruction/wavelet trace, animate wavelet routes, timeline of the critical path) - still STATIC self-contained HTML + inline JS (no framework).
  - **Acceptance:** the polished viz adds interactive PE inspection (per-PE trace on hover), animated wavelet routes, and a critical-path timeline; remains self-contained (offline, no CDN); gtest/lint asserts no external references.
  - **QA happy:** generate the polished viz; it includes interactive trace + animated routes + timeline elements; opens offline -> evidence: `reports/dataflow_report.html`.
  - **QA failure:** assert the polished HTML has zero `http://`/CDN references (proves self-containment held after polish) -> evidence: `reports/w8_viz_selfcontained.log`.
  - **Commit:** `feat(elite): polished interactive dataflow HTML viz (per-PE trace, animated routes, critical-path timeline)`

- [ ] 44. Optional upstream PR (NEVER blocking)
  - **Where:** `docs/upstream_pr.md`, a fork/branch of one target repo.
  - **References:** spec elite #5 (small contribution to ROCm, Triton, MLIR, vLLM, Modal examples, or Cerebras SDK examples). Optional, never a blocking deliverable; only attempted after the MVP (waves 1-7) is green.
  - **Acceptance:** a small, self-contained contribution (e.g. a PolyKernel-style fused-kernel example, a doc fix, or a test) is prepared against ONE of the named upstreams with a clear description; documented in `docs/upstream_pr.md`. If not pursued, `docs/upstream_pr.md` records "deferred" with rationale - the MVP is complete regardless.
  - **QA happy:** the contribution builds/passes the target repo's relevant check locally; `docs/upstream_pr.md` documents the target + diff + submission status -> evidence: `docs/upstream_pr.md`.
  - **QA failure:** the upstream contribution is declined/unmerged -> documented as such; MVP success criteria are unaffected (proves this is non-blocking) -> evidence: `docs/upstream_pr.md`.
  - **Commit:** `docs(elite): optional upstream PR (one of ROCm/Triton/MLIR/vLLM/Modal/Cerebras examples) - non-blocking`

## Final verification wave

<!-- Final-verifier rows MUST be column-zero and match: - [ ] F<number>. <title> -->
<!-- Runs in parallel after ALL todos; ALL must APPROVE; surface results and wait for the user's explicit okay. -->

- [ ] F1. Plan-compliance audit
  - **What:** verify every wave/todo was implemented as specified - all 11 passes exist + run, all named ops + fused ops are in the dialect (and NO others - op-zoo closed), the JSON tuning-cache + per-kernel report schemas match the pinned definitions, the golden rounding contract + thresholds are honored, commits are atomic + conventional + green, tags per wave exist.
  - **Acceptance:** a written audit (path: `reports/f1_compliance.md`) maps each todo to its evidence path + commit; ALL todos have green evidence; no todo is missing or substituted; the dialect op set is exactly the named set.
  - **QA happy:** the audit lists every todo with a green evidence path + matching commit -> `reports/f1_compliance.md` verdict APPROVE.
  - **QA failure:** any todo lacking green evidence (or a missing pass/op) -> audit verdict CHANGES_REQUESTED with the specific gap -> `reports/f1_compliance.md`.

- [ ] F2. Code-quality review
  - **What:** review the C++/MLIR/CUDA/HIP/Python for quality - no dead code, the portable template has no unguarded backend-specific intrinsics, error handling on all GPU/HIP calls, no leaked secrets/credentials, gtest/lit coverage is meaningful (not trivial asserts), the analyzer parsers handle both ptxas formats + malformed input, C++ standard consistent.
  - **Acceptance:** a written review (`reports/f2_quality.md`) with no blocker/high findings open; clang-tidy (or equivalent) clean on the changed code; no hardcoded credentials; all GPU/HIP calls error-checked.
  - **QA happy:** review finds no blocker/high issues; clang-tidy clean -> `reports/f2_quality.md` verdict APPROVE.
  - **QA failure:** an unguarded CUDA intrinsic in the portable template (breaks hipcc) or an unchecked hipMalloc -> flagged as blocker -> `reports/f2_quality.md` CHANGES_REQUESTED.

- [ ] F3. Real manual QA (agent-executed, zero human intervention)
  - **What:** actually RUN the system end-to-end: `polykernel-opt` full pipeline on all 3 examples; build CUDA kernels (nvcc sm_80/sm_90) + emit the compile-time report (no GPU); build + RUN HIP kernels on the RX 7800 XT + golden correctness (0 failures); run the autotuner (correctness-gated) + load via the runtime; run the dataflow simulator + golden + metrics; if a MODAL_TOKEN is present, deploy + hit `/predict` + measure cold start, else verify the SKIPPED path; if RunPod creds present, run the real bench, else verify projected numbers. Every command run by the agent with captured output.
  - **Acceptance:** a QA log (`reports/f3_qa.md`) with the exact commands + outputs: polykernel-opt e2e green; CUDA compile+analyze green (no GPU); HIP run on 7800 XT with 0 failed correctness; autotuner+runtime green; dataflow sim golden+metrics green; Modal/bench either real (with creds) or clean SKIPPED (without). Verdict APPROVE only if all runnable paths pass and all gated paths degrade cleanly.
  - **QA happy:** all runnable paths pass + gated paths SKIPPED cleanly -> `reports/f3_qa.md` verdict APPROVE.
  - **QA failure:** any runnable path fails (e.g. a HIP kernel fails golden on the 7800 XT) OR a gated path crashes instead of SKIPPING -> `reports/f3_qa.md` CHANGES_REQUESTED with the failing command+output.

- [ ] F4. Scope-fidelity audit (Must-NOT-Have enforcement)
  - **What:** verify NO Must-NOT-Have violations: no op zoo beyond the named ops; the dataflow simulator uses CORRECT Cerebras terminology (CE+FMAC, `@set_color_config`, `fabin_dsd`/`fabout_dsd`) and is NOT claimed to be a real CSL/cslc compiler or to run on Cerebras hardware; no claim of beating vLLM/SOTA; no full PyTorch/ONNX frontend; no cloud spend without credentials; no tenstorrent/tt-metal reuse; elite (waves 8+) did not start before waves 1-7 were green; upstream PR is non-blocking.
  - **Acceptance:** a written audit (`reports/f4_scope.md`) confirming each Must-NOT-Have holds; README/docs claims are accurate (simulator = simulator, projected numbers labeled as such, no SOTA claim).
  - **QA happy:** all Must-NOT-Have items hold; docs claims are accurate -> `reports/f4_scope.md` verdict APPROVE.
  - **QA failure:** an extra op in the dialect, a "beats vLLM" claim, FMU/PMU terminology, or an unlabeled projected number presented as real -> `reports/f4_scope.md` CHANGES_REQUESTED with the violation.

## Commit strategy

- **One atomic commit per todo**, conventional-commit format scoped by area:
  - `feat(ir): ...` (dialect, ops, shape inference)
  - `feat(pass): ...` (fusion, tile/layout, memory planning)
  - `feat(cuda): ...` (CUDA codegen, kernels, PTX)
  - `feat(hip): ...` (HIP codegen, gfx1101, WMMA)
  - `feat(analyze): ...` (ptxas parser, occupancy, roofline, kernel report)
  - `feat(autotune): ...` (variant search, JSON cache)
  - `feat(runtime): ...` (GPU detect, kernel selection/load)
  - `feat(dataflow): ...` (simulator, routing, SUMMA, metrics, HTML viz)
  - `feat(modal): ...` (image, endpoints, cold-start, dashboard)
  - `feat(bench): ...` / `feat(report): ...` (bench harness, reports, writeup)
  - `test: ...` (when a todo is test-only)
  - `docs: ...` (architecture.md, backend docs, README)
  - `build: ...` (CMake, nix flake, devenv)
  - `chore: ...` (repo init, .gitignore)
- **Each commit must be green**: the todo's tests pass before committing. No WIP, no "fix later" commits. The commit message body cites the todo number + the evidence path (test output / report artifact).
- **Branch discipline:** work on `main` (or a `dev` branch merged per-wave if the worker prefers); never force-push; never commit unrelated/dirty paths (the stale `.codegraph/` stays untracked/ignored).
- **Tag per wave:** `v0.1-skeleton`, `v0.2-cuda`, `v0.3-fusion`, `v0.4-hip`, `v0.5-autotune`, `v0.6-modal`, `v0.7-dataflow`, `v1.0-mvp`, `v1.x-elite`.

## Success criteria

**Functional (all must hold).**
1. `polykernel-opt` parses `examples/mlp_block.mlir`, `examples/rmsnorm_matmul.mlir`, `examples/attention_prefill.mlir` and runs the full pass pipeline (`--infer-shapes --canonicalize --fuse-rmsnorm-matmul --fuse-matmul-bias-gelu --infer-tile-layout --plan-memory --lower-to-cuda/--lower-to-hip --emit-kernel-report`) without error; lit/FileCheck tests for every pass pass.
2. Generated CUDA kernels compile clean with nvcc for sm_80 AND sm_90; PTX is emitted; the compile-time analyzer produces a populated per-kernel report (registers/smem/occupancy/traffic/spills/bottleneck/suggested-fixes) matching hand-computed values for a reference kernel — all with NO GPU present.
3. Generated HIP kernels compile with hipcc `--offload-arch=gfx1101` AND run on the local RX 7800 XT (or, if the local ROCm gate fails, compile clean + run with HSA_OVERRIDE_GFX_VERSION=11.0.1); AMDGPU ISA analysis extracts VGPR/SGPR/LDS/spills; WMMA bf16 path compiles + runs.
4. **0 failed correctness tests**: every generated kernel (HIP-on-7800XT, and the dataflow simulator's functional execution) passes the NumPy/ml_dtypes golden (cosine >= 0.999, max rel err <= 1e-2, PCC >= 0.99, bf16 round / fp32 accumulate).
5. The autotuner searches the bounded variant grid, correctness-gates each variant, and writes a JSON tuning cache keyed by (GPU, op, shape); the C++ runtime detects the GPU and loads the cached best kernel.
6. The Cerebras-CSL dataflow simulator maps the MLP/matmul fragment onto a 64x64 PE grid using the SUMMA mapping and reports grid utilization, SRAM pressure, messages sent, avg hop distance, comm bottleneck, critical-path cycles, and fusion traffic reduction; the HTML visualizer renders grid/routes/SRAM/traffic/critical-path. Simulator output passes the golden.
7. The Modal app (current API) builds the CUDA-toolkit image, exposes `/predict` `/benchmark` `/kernels` `/report`, and — given a `MODAL_TOKEN` — deploys and measures cold-start + autoscaling; without a token, the full app + local `modal serve` + documented deploy exist and cloud deploy is marked SKIPPED.
8. The benchmark report (text + HTML: `h100_report.html`, `mi300_report.html`, `dataflow_report.html`) shows the model fragment, backends enabled, unfused/fused/autotuned speedups (REAL numbers if RunPod rental ran, else clearly-labeled projections), dataflow simulator metrics, and compiler stats (passes, generated kernels, validated kernels, failed correctness = 0).
9. Docs complete: `docs/architecture.md`, `compiler_pipeline.md`, `cuda_backend.md`, `hip_backend.md`, `dataflow_backend.md`, `performance_model.md`; README with the specified headline/subtitle/resume positioning.

**Quality gates.**
- All lit + gtest + golden tests pass in CI (the nix devShell `check` phase).
- No Must-NOT-Have violations (F4): no op zoo beyond named ops, correct Cerebras terminology (CE+FMAC, `@set_color_config`, no FMU/PMU/fdata/fcast/fmove/@compute/@data), no real-Cerebras-hardware claim, no SOTA-performance claim.
- The final verification wave F1-F4 all APPROVE.

**Stretch (elite, post-MVP).** FlashAttention-lite (QKT, masked softmax, PV, KV-cache r/w); quantized inference (fp16/bf16/int8 weight-only/fp8 sim); hardened runtime kernel cache; polished dataflow HTML viz; optional small upstream PR (ROCm/Triton/MLIR/vLLM/Modal/Cerebras examples) — never blocking.
