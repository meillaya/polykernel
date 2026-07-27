# PolyKernel CUDA Backend + GPU-Free Analyzer

The CUDA backend (C3a) generates CUDA kernels for the `polykernel` ops, compiles them to
PTX for **sm_80** (A100) and **sm_90** (H100), and — the heart of the backend — analyzes
them **entirely at compile time, with no NVIDIA GPU present**. There is no local NVIDIA
GPU; real H100/A100 *runtime* numbers come only from an owner-gated RunPod rental, and
until then are clearly-labeled **projections** (see [`performance_model.md`](performance_model.md)).

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
the *shared* compute logic for both backends (the no-NVIDIA-GPU correctness story).

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
| `matmul_wmma.cu` | additive tensor-core variant (WMMA/MMA path) |

The scalar/tiled matmul is the **correctness anchor**; the WMMA/MMA path is an additive,
correctness-gated variant the autotuner may select.

## Compile + PTX (no GPU)

`tools/polykernel-bench/nvcc_driver.py` drives the compile:

```bash
nvcc -arch=sm_80 -DPOLYKERNEL_CUDA -ptx kernels/generated/matmul.cu   # PTX for A100
nvcc -arch=sm_90 -DPOLYKERNEL_CUDA -ptx kernels/generated/matmul.cu   # PTX for H100
ptxas -v matmul.ptx                                                    # registers/smem/spills
```

`nvcc` compiling clean for **both** sm_80 and sm_90 + PTX emitting + `ptxas -v` parsing is
the CUDA *compile* acceptance gate (Pinned contract G). It needs no device.

## The GPU-free compile-time analyzer

`lib/Analysis/` implements the analyzer as a library (reused by `polykernel-analyze`,
`polykernel-bench`, and `polykernel-report`):

- **`PtxasParser.cpp`** — parses `ptxas -v` stderr in **both** the old format
  (`Used N registers, N bytes smem, N bytes cmem[0]`) and the new format
  (`Used N registers, used N barriers, N bytes smem`), extracting registers/thread, smem,
  spill stores/loads, stack frame, and gmem. Malformed input yields an explicit parse
  error, never silent zeros.
- **`Occupancy.cpp`** — computes occupancy from the arch constants table. sm_80:
  smem/SM = 164 KB; sm_90: smem/SM = 228 KB; both: 65536 regs/SM, 255 regs/thread,
  64 warps/SM, 32 blocks/SM, 2048 threads/SM, warp = 32. `blocks = min(reg_limit,
  smem_limit, warp_limit, 32)`; `occupancy = active_warps / 64`. Unit-tested against
  **hand-computed** values (e.g. regs=128, threads=256, smem=32 KB on sm_90).
- **`Roofline.cpp`** — `bytes = (M·K + K·N + M·N)·dtype_bytes`, `flops = 2·M·N·K`,
  `AI = flops/bytes`, ridge = peak_flop/peak_bw (H100: 989 TFLOPS / 3.35 TB/s ≈ 295
  flop/byte). Classifies each op `compute-bound` (AI > ridge) or `memory-bound`.
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

## Correctness model

- The **shared** compute logic is validated by running the HIP build (and the CPU-reference
  build, `kernels/cpu/`) on the RX 7800 XT against the golden — this validates the
  algorithmic core for *both* backends.
- CUDA-specific tensor-core paths (WMMA/MMA, CUDA-only launch config) are
  **compile-validated** locally (nvcc clean for sm_80/sm_90, PTX emits, `ptxas -v` parses)
  and **fully validated on rented H100/A100** (owner-gated). Until then they are marked
  "compile-validated; runtime-validated on rental".

## See also

- [`hip_backend.md`](hip_backend.md) — the HIP sibling that compiles the same template.
- [`performance_model.md`](performance_model.md) — how the compile-time analyzer feeds the
  projected H100/A100 speedups.
- [`compiler_pipeline.md`](compiler_pipeline.md) — `--lower-to-cuda` custom source emission.
