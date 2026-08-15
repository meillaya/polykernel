# PolyKernel rmsnorm+matmul prologue fusion benchmark — RTX 6000 Ada (sm_89)

> ## ✅ MEASURED — REAL GPU RUN
> MEASURED on the RTX 6000 Ada pod (216.81.200.13) with CUDA events.
> Real wall times (warmup=5, iters=20, tune_iters=50), not roofline projections.
> nvcc 12.2 (V12.2.140) `-arch=sm_89`, device: NVIDIA RTX 6000 Ada Generation.

- **Task:** T14 (pass 2): real headline benchmark on the pod (`bench_cuda.cpp --arch sm_89`)
- **Fragment:** `rmsnorm+matmul prologue fusion` (examples/rmsnorm_matmul.mlir target)
- **dtype / shape:** bf16, M=2048, K=4096, N=11008
- **Generated:** 2026-08-15T14:49Z by `benchmarks/bench_cuda.cpp` on the pod (CUDA events)

## Measured speedup (unfused = 1.00x baseline)

| arch | backend | unfused | fused | autotuned |
|---|---|---|---|---|
| NVIDIA RTX 6000 Ada (sm_89) | cuda (sm_89) | 1.000x | 0.838x | 0.857x |

## Measured fragment wall time (ms)

| arch | unfused min / median | fused min / median | autotuned min / median | ridge (flop/byte) |
|---|---|---|---|---|
| NVIDIA RTX 6000 Ada (sm_89) | 25.177 / 25.518 | 29.891 / 30.459 | 29.785 / 30.127 | 94.9 |

## Methodology (reproducible)

- **Tool:** `benchmarks/bench_cuda.cpp` (CUDA events; the same fragment bench as the
  RunPod rental path, run directly on the pod).
- **Equation:** `speedup = unfused.median_ms / variant.median_ms`; autotuned uses the
  min over the extended budget (tune_iters=50). Speedups vs unfused = 1.00x.
- **Variants:**
  - `unfused` = `launch_rmsnorm(X -> N)` + `launch_matmul(N @ W -> C)`: two kernels; the
    normalized tensor N is materialized in global (the 32 MiB round-trip).
  - `fused` = `launch_fused_rmsnorm_matmul(X, W -> C)`: one kernel; the round-trip is
    eliminated, but every output tile recomputes the per-row RMS.
  - `autotuned` = the fused realization timed over the extended budget; best (min) is
    the autotuned figure (>= fused by selection).
- **Build on the pod (the exact command, with two pod-specific fixes):**
  `/usr/local/cuda-12.2/bin/nvcc -ccbin /usr/bin/g++-11 -x cu -DPOLYKERNEL_CUDA -include
  cstdio -std=c++20 -O2 -Ikernels/template -arch=sm_89 benchmarks/bench_cuda.cpp
  kernels/generated/rmsnorm.cu kernels/generated/matmul.cu
  kernels/generated/fused_rmsnorm_matmul.cu -o /tmp/bench`
  - Fix 1: the pod's default `gcc` is gcc-12 but its `cc1plus` is missing, so nvcc's
    host-compiler probe fails; `-ccbin /usr/bin/g++-11` (whose cc1plus exists) fixes it.
  - Fix 2: `/usr/local/bin/nvcc` is a symlink and resolves its CUDA include path to
    `/usr/local/bin/../include` (which lacks cuda_runtime.h); the direct
    `/usr/local/cuda-12.2/bin/nvcc` binary resolves the real toolkit path.
  - `-include cstdio` remains required (kernel_common.h's PK_CHECK/fprintf uses stderr
    without including `<cstdio>`; cuda_runtime.h does not pull it in transitively).

## Caveats

- MEASURED on the RTX 6000 Ada pod with CUDA events (warmup=5, iters=20, tune_iters=50);
  min + median over the timed launches.
- The fused kernel is SLOWER than unfused at this shape (0.84x): the generated fused
  kernel recomputes the per-row RMS in every output tile, and at this compute-bound shape
  (AI~1215 >> ridge) the redundant work outweighs the 32 MiB round-trip elimination. The
  projected report predicted exactly this shape dependence.
- The scalar matmul baseline attains ~8% of the 91.1 TFLOPS sm_89 roofline (25.5 ms vs
  ~2.03 ms ideal), so these are SCALAR-kernel numbers; the tensor-core (mma) path is not
  exercised by bench_cuda.cpp.
- run-to-run variance is small: unfused median 25.5-25.8 ms, fused median 30.5-31.7 ms,
  autotuned min 29.8-30.0 ms across two runs on the pod.

## Reproduce

```bash
ssh -i private_key.pem ubuntu@216.81.200.13 'cd ~/polykernel && \
  /usr/local/cuda-12.2/bin/nvcc -ccbin /usr/bin/g++-11 -x cu -DPOLYKERNEL_CUDA -include cstdio \
  -std=c++20 -O2 -Ikernels/template -arch=sm_89 benchmarks/bench_cuda.cpp \
  kernels/generated/rmsnorm.cu kernels/generated/matmul.cu \
  kernels/generated/fused_rmsnorm_matmul.cu -o /tmp/bench && /tmp/bench --json --arch sm_89'
```
