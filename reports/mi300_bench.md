# PolyKernel MLP block (examples/mlp_block.mlir): rmsnorm+matmul+gelu+matmul+add benchmark — MI300

> ## ⚠ PROJECTED — NOT MEASURED
> PROJECTED (no rental; run benchmarks/rent_runpod.sh with RUNPOD_API_KEY set for real measured numbers).
> Numbers are perf-model projections (roofline + analytic traffic), not a rented-GPU run. No pod was provisioned; no money was spent.

- **Task:** T28 (Wave 5): real benchmark on rented GPUs (OWNER-GATED: RunPod)
- **Fragment:** `MLP block (examples/mlp_block.mlir): rmsnorm+matmul+gelu+matmul+add`
- **dtype / shape:** bf16, M=2048, K=4096, N_up=11008, N_dn=4096
- **Generated:** 2026-07-27T12:13:51Z by `benchmarks/run_bench_suite.py (roofline + analytic traffic)`

## Projected speedup (unfused = 1.00x baseline)

| arch | backend | unfused | fused | autotuned |
|---|---|---|---|---|
| AMD Instinct MI300X (gfx942) | hip (gfx942) | 1.000x | 1.003x | 2.850x |

## Projected fragment wall time (ms)

| arch | unfused | fused | autotuned | ridge (flop/byte) |
|---|---|---|---|---|
| AMD Instinct MI300X (gfx942) | 2.3954 | 2.3875 | 0.8403 | 246.7 |

## Methodology (reproducible)

- **Model:** GEMM roofline (lib/Analysis/Roofline.cpp) + Todo 17 analytic traffic
- **Equation:** `t = max(flops/peak_flop, bytes/peak_bw) / eta; speedup = t_unfused / t_variant`
- **Attainment assumptions (modeling knobs):** scalar GEMM η=0.12, autotuned vectorized/wmma η=0.35, elementwise η=0.8
- **Traffic source:** reports/mlp_traffic.json (Todo 17)

  - Todo 17 fusion eliminates 33,554,432 bytes (7.018% of fragment traffic).

## Per-op roofline (AMD Instinct MI300X (gfx942), unfused)

| op | flops | bytes | AI (flop/byte) | bound | projected ms |
|---|---|---|---|---|---|
| rmsnorm | 25,000,000 | 33,554,432 | 0.75 | memory-bound | 0.00791 |
| matmul_up | 184,683,593,728 | 152,043,520 | 1214.68 | compute-bound | 1.17717 |
| gelu | 200,000,000 | 90,177,536 | 2.22 | memory-bound | 0.02127 |
| matmul_dn | 184,683,593,728 | 152,043,520 | 1214.68 | compute-bound | 1.17717 |
| add_residual | 8,400,000 | 50,331,648 | 0.17 | memory-bound | 0.01187 |

## Caveats

- PROJECTED, not measured: derived from the roofline model + analytic traffic, not from a rented GPU run.
- The fragment is compute-bound (the two matmuls, AI~1215 >> ridge), so fusion's traffic elimination projects a modest speedup; the headline gain is the scalar->autotuned attainment jump.
- The generated fused kernel recomputes the per-row RMS in every output tile, so the REAL fusion speedup is shape-dependent (verified on the local gfx1101: fused can be slower at small, compute-light shapes).
- Attainment factors are documented modeling assumptions, not measured efficiencies; real numbers require the opt-in RunPod rental.

## Opt-in real measurement

```bash
export RUNPOD_API_KEY=<your-key>   # owner-gated; ~$50-100 ceiling
./benchmarks/rent_runpod.sh --suite mlp
```
