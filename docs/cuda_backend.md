# PolyKernel CUDA Backend + GPU-Free Analyzer

The CUDA backend (C3a) generates CUDA kernels for the `polykernel` ops, compiles them to
PTX for **sm_80** (A100), **sm_89** (RTX 6000 Ada) and **sm_90** (H100), and analyzes them
**entirely at compile time, with no device attached to the analysis** (the heart of the
backend). The local dev machine has **no NVIDIA GPU**; the CUDA runtime-validation
target is the **remote RTX 6000 Ada (sm_89) pod instance**, and the on-GPU run there is
**PENDING pod-key authorization** (the pass-2 pod gate was SKIPPED with
`Permission denied (publickey)`, `reports/pod_env.log`). Real H100/A100 *runtime*
numbers would come only from an owner-gated RunPod rental, and until then stay
clearly-labeled **projections** (see [`performance_model.md`](performance_model.md)).

## The portable kernel template

CUDA and HIP share **one** portable template, `kernels/template/kernel_common.h`. It maps
the backend-specific primitives through `#ifdef`:

```c
#ifdef POLYKERNEL_CUDA
  #include <cuda_runtime.h>
  #define PK_GLOBAL __global__
  #define PK_SHARED __shared__
  // ... cudaLaunchKernel, __syncthreads, warp shuffles
#elif defined(POLYKERNEL_HIP)
  #include <hip/hip_runtime.h>
  #define PK_GLOBAL __global__
  #define PK_SHARED __shared__
  // ... hipLaunchKernel, the HIP equivalents
#endif
```

The compute logic is written **once** against these macros. The `POLYKERNEL_CUDA` /
`POLYKERNEL_HIP` define is a **driver** concern (applied by `nvcc`/`hipcc`), never an
emitter concern. This is what lets the HIP build run on the local RX 7800 XT and validate
the *shared* compute logic for both backends (the shared-logic correctness story; the
CUDA-only runtime story is covered in the correctness section below).

## Generated kernels

`--lower-to-cuda` emits these into `kernels/generated/`:

| Kernel | Technique |
|---|---|
| `rmsnorm.cu` | warp-shuffle reduction for the RMS + vectorized (`float4`/`half4`) loads |
| `gelu.cu` / `silu.cu` | vectorized elementwise (exact erf-based GELU) |
| `matmul.cu` | tiled shared-memory GEMM (16×16 bank-padded `As[16][17]`), double-buffered smem, FP32 accumulate, bounds-checked |
| `softmax.cu` | online (safe) softmax with warp/block reduction |
| `fused_rmsnorm_matmul.cu` | prologue fusion: RMS computed on-the-fly as the matmul A-operand loads |
| `fused_matmul_bias_gelu.cu` | epilogue fusion: bias+GELU applied in-register before the store (eliminates the intermediate global write/read) |
| `matmul_mma.cu` | CUDA tensor-core variant: `nvcuda::wmma` m16n16k16 bf16 (sm_80+; RTX 6000 Ada target = sm_89), additive + correctness-gated, `path=mma` |

`matmul_wmma.cu` also lives in `kernels/generated/` but is the **HIP/RDNA3 sibling** (the
`v_wmma_f32_16x16x16_bf16` WMMA path), **not** a CUDA kernel; its own header and the
`-DPOLYKERNEL_HIP` guards make it HIP-only by design (see
[`hip_backend.md`](hip_backend.md)). The table above is the CUDA view: the scalar/tiled
`matmul.cu` is the **correctness anchor**; `matmul_mma.cu` is an additive,
correctness-gated tensor-core variant the autotuner may select.

## Compile + PTX (no GPU needed)

`tools/polykernel-bench/nvcc_driver.py` drives the compile:

```bash
nvcc -arch=sm_80 -DPOLYKERNEL_CUDA -ptx kernels/generated/matmul.cu   # PTX for A100
nvcc -arch=sm_89 -DPOLYKERNEL_CUDA -ptx kernels/generated/matmul.cu   # PTX for RTX 6000 Ada
nvcc -arch=sm_90 -DPOLYKERNEL_CUDA -ptx kernels/generated/matmul.cu   # PTX for H100
ptxas -v matmul.ptx                                                    # registers/smem/spills
```

`nvcc` compiling clean for **sm_80 / sm_89 / sm_90** + PTX emitting + `ptxas -v` parsing is
the CUDA *compile* acceptance gate (Pinned contract G). It needs no device.

The **runtime-validation harness** `lib/Runtime/cuda_run_main.cpp` (the CUDA twin of
`hip_run_main`) links clean for the same three archs (build `build/cuda_run/cuda_run_{sm_80,
sm_89,sm_90}`) and includes a `dev` probe that prints `DEVICE <name> COMPUTE_CAPABILITY
<major>.<minor>`. It is built and compile-validated locally; launching it on the RTX 6000
Ada pod is pending pod-key authorization.

## The GPU-free compile-time analyzer

`lib/Analysis/` implements the analyzer as a library (reused by `polykernel-analyze`,
`polykernel-bench`, and `polykernel-report`):

- **`PtxasParser.cpp`** — parses `ptxas -v` stderr in **both** the old format
  (`Used N registers, N bytes smem, N bytes cmem[0]`) and the new format
  (`Used N registers, used N barriers, N bytes smem`), extracting registers/thread, smem,
  spill stores/loads, stack frame, and gmem. Malformed input yields an explicit parse
  error, never silent zeros.
- **`Occupancy.cpp`** — computes occupancy from the arch constants table. sm_80:
  smem/SM = 164 KB; sm_90: smem/SM = 228 KB; both share 65536 regs/SM, 255 regs/thread,
  64 warps/SM, 32 blocks/SM, 2048 threads/SM, warp = 32. **sm_89 (RTX 6000 Ada) differs on
  every parallelism dimension**: 48 warps / 24 blocks / 1536 threads per SM, 100 KB
  smem/SM (99 KB/block max). `blocks = min(reg_limit, smem_limit, warp_limit, 32)` (24 for
  sm_89); `occupancy = active_warps / warps_per_sm`. Unit-tested against **hand-computed**
  values (e.g. regs=128, threads=256, smem=32 KB on sm_90; warp-limited on sm_89).
- **`Roofline.cpp`** — `bytes = (M·K + K·N + M·N)·dtype_bytes`, `flops = 2·M·N·K`,
  `AI = flops/bytes`, ridge = peak_flop/peak_bw (H100: 989 TFLOPS / 3.35 TB/s ≈ 295
  flop/byte; RTX 6000 Ada sm_89: scalar bf16 91.1 TFLOPS / 960 GB/s ≈ 95 flop/byte).
  Classifies each op `compute-bound` (AI > ridge) or `memory-bound`.
- **`KernelReport.cpp`** — assembles the **contract-H** per-kernel report (below).

## The contract-H per-kernel report

`polykernel-report --kernel <k> --backend cuda --arch sm_90` merges the CUDA compile-time
analysis + the autotuner result into the authoritative report (schema pinned in contract H):

```json
{
  "kernel": "fused_rmsnorm_matmul", "backend": "cuda", "arch": "sm_90",
  "registers_per_thread": 31, "smem_per_block_bytes": 1024,
  "spill_stores_bytes": 0, "spill_loads_bytes": 0,
  "occupancy": { "active_warps_per_sm": 64, "max_warps_per_sm": 64,
                 "occupancy_pct": 100, "limiter": "registers" },
  "traffic": { "global_read_bytes": 106954752, "global_write_bytes": 45088768,
               "arithmetic_intensity_flop_per_byte": 1214.68, "roofline": "compute-bound" },
  "bottleneck": "compute", "suggested_fixes": [], "path": "scalar"
}
```

(the live copy is `reports/kernel_report_example.json`). Enum values are pinned:
`limiter ∈ {registers, smem, warps, blocks}`, `roofline ∈ {compute-bound, memory-bound}`,
`bottleneck ∈ {compute, memory, latency}`, `path ∈ {scalar, wmma, mma}`.

**Suggested-fix rules** (the heart of Todo 27): spills > 0 → reduce register pressure;
register-limited with occupancy < 100% → reduce unroll / split the accumulator;
occupancy < 50% → smaller tile / fewer registers; memory-bound → wider vectorized loads
or fusion. The example kernel above is compute-bound at 100% occupancy with no spills, so
`suggested_fixes` is empty.

## CUDA tensor-core path (sm_89)

`kernels/generated/matmul_mma.cu` is the CUDA tensor-core matmul (added in pass 2, todo
10). It closes the old gap where CUDA had only the scalar baseline, and is the CUDA twin
of the HIP/RDNA3 WMMA path:

- **API / instruction:** `nvcuda::wmma` from `<mma.h>` driving
  `mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32` on sm_80+; the RTX 6000 Ada target is
  **sm_89**. One warp (32 lanes) computes one 16×16 fp32 output tile; K is reduced in
  16-wide WMMA tiles staged through shared memory.
- **Fragments:** `matrix_a` 16×16×16 bf16 **row-major**, `matrix_b` 16×16×16 bf16
  **col-major** (so the shared B tile is staged transposed, `Bs[n][k] = B[k0+k][n0+n]`),
  `accumulator` 16×16×16 **fp32**. This is the one deliberate staging deviation from
  `matmul_wmma.cu`: staging B naturally with a `col_major` fragment would transpose the
  operand and fail the golden.
- **Rounding contract:** bf16 RNE inputs, fp32 accumulate, bf16 RNE store, the same
  golden contract as the scalar baseline and the HIP WMMA path. The fp32 accumulation
  order differs from NumPy, so at large K an isolated element can land 1 bf16 ULP off at
  small magnitude (the same documented reduction-order class as the scalar baseline).
- **Additive + correctness-gated:** the scalar `matmul.cu` remains the correctness anchor;
  MMA is a candidate the autotuner may select (`path=mma`, a report annotation, contract
  H). A bad variant (the `BadStore` negative path) swaps `mem_row_major` for
  `mem_col_major`, transposing C; it fails the golden decisively and is discarded
  (`validated=false`) while the scalar baseline still passes, proving MMA is safely
  additive.
- **HIP guard:** under `-DPOLYKERNEL_HIP` the file is a hard `#error`; the HIP tensor-core
  sibling is `matmul_wmma.cu`.
- **Status:** built + compile-validated locally (nvcc sm_80/89/90, PTX contains the real
  `mma.sync` instruction, `ptxas -v` parses, 0 spills). The on-GPU runtime validation on
  the RTX 6000 Ada (sm_89) pod is **PENDING pod-key authorization** (pod gate SKIPPED,
  `reports/pod_env.log`); the harness (`lib/Runtime/cuda_run_main.cpp` +
  `tests/kernels/test_mma.py`) is ready to run once the key is authorized.

## Correctness model

- The **shared** compute logic is validated by running the HIP build (and the CPU-reference
  build, `kernels/cpu/`) on the RX 7800 XT against the golden, which validates the
  algorithmic core for *both* backends.
- CUDA-specific paths (the MMA tensor-core kernel `matmul_mma.cu` and the runtime launcher
  `lib/Runtime/cuda_run_main.cpp`) are **built and compile-validated locally**: nvcc links
  clean for sm_80/sm_89/sm_90, the PTX contains the real
  `mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32` instruction, and `ptxas -v` parses. The
  **on-GPU runtime validation on the RTX 6000 Ada (sm_89) pod is PENDING pod-key
  authorization** (the pass-2 pod gate was SKIPPED, `reports/pod_env.log`); the harness
  is ready and will run on the pod once the SSH key is authorized. H100/A100 runtime
  numbers remain **projected** (owner-gated RunPod rental).

## See also

- [`hip_backend.md`](hip_backend.md) — the HIP sibling that compiles the same template.
- [`performance_model.md`](performance_model.md) — how the compile-time analyzer feeds the
  projected H100/A100 speedups.
- [`compiler_pipeline.md`](compiler_pipeline.md) — `--lower-to-cuda` custom source emission.
