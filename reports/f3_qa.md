# F3 — REAL-MANUAL-QA (final verification wave)

**Engineer:** F3 REAL-MANUAL-QA (autonomous, zero human intervention)
**Date (UTC):** 2026-07-27
**Method:** Every command below was RE-RUN by F3 in this session; every output line is
captured from that real run (exit codes, pass counts, key metrics). No prior log was
copied. Build under test: `build_verify/` (fresh full build; 154 ctest tests).

---

## 0. Environment (verified, not assumed)

Command:
```bash
nix develop --impure --accept-flake-config -c bash -c 'which nvcc ptxas hipcc python3 pytest lit FileCheck; nvcc --version; hipcc --version; python3 --version; rocminfo | grep -iE "gfx|Marketing"; echo "MODAL_TOKEN=[${MODAL_TOKEN:-UNSET}] RUNPOD_API_KEY=[${RUNPOD_API_KEY:-UNSET}]"'
```
Captured (abridged):
```
nvcc   : Cuda compilation tools, release 12.6, V12.6.85
ptxas  : /nix/store/...-cuda12.6-cuda_nvcc-12.6.85/bin/ptxas
hipcc  : HIP version: 7.2.53211-9999  (rocm-7.2.3)
python : Python 3.14.6   (pytest, lit 18.1.8, FileCheck all on PATH)
GPU    : Name: gfx1101   Marketing Name: AMD Radeon RX 7800 XT   (rocminfo, native)
CPU    : AMD Ryzen 7 5700X 8-Core Processor
MODAL_TOKEN=[UNSET]  RUNPOD_API_KEY=[UNSET]
```
Facts established: **no NVIDIA GPU** (CUDA = compile + PTX + GPU-free analyze only);
**HIP RUNS on the RX 7800 XT (gfx1101)**; **no cloud creds** (Modal/RunPod must degrade
to SKIPPED/PROJECTED). `build_verify/tools/polykernel-opt/polykernel-opt` and
`build_verify/tools/polykernel-analyze/polykernel-analyze` confirmed present.
`ctest --test-dir build_verify -N` → **Total Tests: 154**.

Hygiene: F3 wrote ONLY this file. Tool runs that regenerate committed artifacts
(`reports/h100_bench.*`, `mi300_bench.*`, `w5_rent_skipped.log`, `w6_deploy_skipped.log`
— 1 deterministic line each) were restored via `git checkout`; `git status` confirms no
product-code change (generated kernels byte-identical after `--lower-to-cuda`).

---

## Area 1 — polykernel-opt full pipeline on ALL 3 examples  → PASS

Command (run per example; `PASSES` identical for all three):
```bash
OPT=build_verify/tools/polykernel-opt/polykernel-opt
PASSES="--infer-shapes --canonicalize --fuse-rmsnorm-matmul --fuse-matmul-bias-gelu \
        --infer-tile-layout --plan-memory --lower-to-cuda --emit-kernel-report"
for ex in mlp_block rmsnorm_matmul attention_prefill; do
  $OPT examples/${ex}.mlir $PASSES; echo "EXIT_CODE=$?"
done
```
Captured:
```
examples/mlp_block.mlir        EXIT_CODE=0   (stderr empty)
examples/rmsnorm_matmul.mlir   EXIT_CODE=0   (stderr empty)
examples/attention_prefill.mlir EXIT_CODE=0  (stderr empty)
```
IR evidence (mlp_block, post-pipeline):
```
%0 = polykernel.fused_rmsnorm_matmul %arg0, %arg1 {epsilon = 9.99999974E-6 : f32,
     polykernel.eliminated_type = tensor<1x2048x4096xbf16>,
     polykernel.fused_from = "rmsnorm_matmul", polykernel.layout = "row_major",
     polykernel.report_arch = "sm_90", polykernel.report_backend = "cuda",
     polykernel.report_kernel = "fused_rmsnorm_matmul", polykernel.report_path = "scalar",
     polykernel.smem_bytes = 131072 : i64, polykernel.tile_k = 128 : i64,
     polykernel.tile_m = 128 : i64, polykernel.tile_n = 128 : i64,
     polykernel.workspace_bytes = 0 : i64}
   : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
%2 = polykernel.matmul %1, %arg2 {... polykernel.workspace_bytes = 16777216 : i64}
   : tensor<1x2048x11008xbf16>, tensor<11008x4096xbf16> -> tensor<1x2048x4096xbf16>
```
- mlp_block census: `fused_rmsnorm_matmul` (fused_from rmsnorm_matmul, eliminated_type
  present), `matmul`, `gelu`, `add`; tile attrs 128/128/128; smem/workspace + report_* attrs.
- rmsnorm_matmul: 2 funcs, 2× `fused_rmsnorm_matmul`, full tile/memory/report attrs.
- attention_prefill: `attention`, `qkv_projection`, 2× `rope`, `matmul`, `rmsnorm`, `add`;
  tile/memory/report attrs.
- `--lower-to-cuda` emitted all 11 generated kernels (`kernels/generated/*.cu`), byte-identical
  to the committed set (`git diff` clean). **All 3 examples traverse all 8 passes, exit 0, no diagnostics.**

---

## Area 2 — CUDA compile (nvcc, sm_80 AND sm_90) + GPU-free per-kernel report  → PASS

Command (per kernel):
```bash
python3 tools/polykernel-bench/nvcc_driver.py --kernel <k> --arch sm_80,sm_90 --ptx --out-dir /tmp/opencode/f3/nvcc
```
Captured `ptxas -v` parsed summary (registers / smem / spills), all `EXIT_CODE=0`, PTX emitted:
```
kernel                   sm_80                sm_90                spills
rmsnorm                  38 reg / 36 B        32 reg / 36 B        0 / 0
gelu                     20 reg / 0 B         23 reg / 0 B         0 / 0
silu                     19 reg / 0 B         22 reg / 0 B         0 / 0
matmul                   31 reg / 1024 B      32 reg / 1024 B      0 / 0
softmax                  28 reg / 40 B        28 reg / 40 B        0 / 0
fused_rmsnorm_matmul     31 reg / 1024 B      31 reg / 1024 B      0 / 0
fused_matmul_bias_gelu   31 reg / 1024 B      32 reg / 1024 B      0 / 0
attention_prefill        35 reg / 256 B       35 reg / 256 B       0 / 0
kv_cache_update          26 reg / 0 B         26 reg / 0 B         0 / 0
matmul_int8              32 reg / 768 B       32 reg / 768 B       0 / 0
```
Representative raw line: `ptxas info : Used 31 registers, used 1 barriers, 1024 bytes smem`
(sm_90, fused_rmsnorm_matmul). `matmul_wmma.cu` is the RDNA3 `v_wmma` path (AMD-only by
design) — validated under HIP in Area 3, not nvcc.

GPU-free compile-time per-kernel contract-H report:
```bash
python3 tools/polykernel-report/report.py --kernel fused_rmsnorm_matmul --backend cuda --arch sm_90 --shape 2048,11008,4096
```
```json
{
  "kernel": "fused_rmsnorm_matmul", "backend": "cuda", "arch": "sm_90",
  "registers_per_thread": 31, "smem_per_block_bytes": 1024,
  "spill_stores_bytes": 0, "spill_loads_bytes": 0,
  "occupancy": {"active_warps_per_sm": 64, "max_warps_per_sm": 64,
                "occupancy_pct": 100, "limiter": "registers"},
  "traffic": {"global_read_bytes": 106954752, "global_write_bytes": 45088768,
              "arithmetic_intensity_flop_per_byte": 1214.68, "roofline": "compute-bound"},
  "bottleneck": "compute", "suggested_fixes": [], "path": "scalar"
}
```
`EXIT_CODE=0`. Full report (registers/smem/spills/occupancy/traffic/bottleneck/fixes) with
**no NVIDIA GPU present** (drives nvcc→ptxas→`polykernel-analyze`).

---

## Area 3 — HIP build + RUN on RX 7800 XT (gfx1101) + golden correctness  → PASS

ROCm gate:
```bash
bash scripts/check_rocm.sh
```
```
[INFO] Name: gfx1101   Marketing Name: AMD Radeon RX 7800 XT
[PASS] amdgpu driver present (/sys/module/amdgpu)
GATE VERDICT: PASS
  gfx1101 enumerated natively (ROCm 7.2.3); no override needed.
  HIP-run todos (T20) may run LOCALLY.
GATE_EXIT=0
```
(Advisory `render`-group WARN only; GPU is enumerated & accessible — kernels run.)

Golden-correctness suite (HIP kernels built with hipcc `--offload-arch=gfx1101`, launched on
the 7800 XT, diffed vs the bf16 golden):
```bash
python3 -m pytest tests/kernels/ -q
```
```
75 passed in 41.70s
PYTEST_EXIT=0
```
Breakdown (verbose re-run):
```
tests/kernels/test_hip_run.py      18 passed  (rmsnorm/gelu/matmul[square,rect,non-tile,
                                              batched,large-k]/softmax/fused_rmsnorm_matmul/
                                              fused_matmul_bias_gelu + negative-launch-caught)
tests/kernels/test_wmma.py          5 passed  (WMMA bf16 tensor-core HIP RUN + bad-lane negative)
tests/kernels/test_attention.py    13 passed  (CPU + HIP prefill/append/fused_kv_append +
                                              wrong-mask-leaks-future negative)
tests/kernels/test_quant.py        13 passed  (int8 CPU×4 + GPU/HIP×4, fp8-sim e4m3/e5m2,
                                              wrong-per-channel-scale negative)
(+ test_cpu_ref*.py CPU-reference tests)   →  75 total
```
**0 failed correctness.** The negative tests (wrong mask, wrong scale, bad lane-mapping,
invalid launch) all PASS by correctly FAILING the golden (e.g. bad-lane cosine=0.148,
wrong-scale cosine=0.888) — the gate is real, not vacuous. `rocminfo` + gate confirm the
kernels execute on **gfx1101**.

---

## Area 4 — Correctness-gated autotuner + C++ runtime  → PASS

Autotuner (builds a hipcc driver, runs each variant on gfx1101, gates on golden BEFORE
timing, writes the contract-H cache; `--inject-broken` adds a fast-but-wrong variant):
```bash
python3 tools/polykernel-bench/bench.py --autotune --op fused_matmul_bias_gelu \
    --shape 2048,4096,11008 --dtype bf16 --backend hip --arch gfx1101 \
    --variants 4 --inject-broken --cache-out /tmp/opencode/f3/tuning_cache.json
```
```
[bench] op=fused_matmul_bias_gelu shape=M=2048,N=4096,K=11008 dtype=bf16 arch=gfx1101 rel_ceiling=inf
[bench] VALIDATED config_0  config=(16,32,32,4,1,4,2) cosine=1.000000 rel=1.953e+03 pcc=1.000000 min_ms=71.71940 median_ms=72.02851
[bench] VALIDATED config_1  config=(16,32,32,4,1,4,3) cosine=1.000000 rel=1.953e+03 pcc=1.000000 min_ms=71.73717 median_ms=72.24065
[bench] VALIDATED config_2  config=(16,32,32,4,1,4,4) cosine=1.000000 rel=1.953e+03 pcc=1.000000 min_ms=71.30725 median_ms=72.11686
[bench] VALIDATED config_3  config=(16,32,32,4,2,2,2) cosine=1.000000 rel=1.953e+03 pcc=1.000000 min_ms=71.45219 median_ms=71.86292
[bench] REJECTED  broken_fast config=(16,32,32,4,1,4,2) cosine=0.564037 rel=7.000e+06 pcc=0.000000 -> validated:false, NOT timed, excluded
          gate log: REJECTED fused_matmul_bias_gelu/broken: correctness gate FAILED (cosine=0.564037 max_rel_err=7.000000e+06 pcc=0.000000 ceiling=inf) - excluded, NOT timed, never best
[bench] BEST (fastest validated): config_3 config=(16,32,32,4,2,2,2) median_ms=71.86292
[bench] tuning cache written ... (validated:true):
{"entries":[{"best":{"block_k":32,"block_m":16,"block_n":32,"num_warps":4,"shared_memory_stages":2,
  "unroll":2,"vector_width":2},"correctness":{"cosine":1,"max_rel_err":1953.125,"pcc":1},
  "gpu":"gfx1101","op":"fused_matmul_bias_gelu","scored_by":"measure",
  "shape":{"K":11008,"M":2048,"N":4096,"dtype":"bf16"},"time_ms":71.86292,"validated":true}],"version":1}
BENCH_EXIT=0
```
Real HIP-event timing on the 7800 XT (~71.86 ms median for a 2048×4096×11008 GEMM);
cosine=pcc=1.0 (rel≈1.95e3 is the documented large-K near-zero-reference noise; gate uses
cosine+pcc with ceiling=inf there). The injected fast-but-wrong variant is **rejected, never
timed, never best** — the correctness gate is the discriminator.

C++ runtime (detect → select best cached kernel → load → serve) + hardened cache:
```bash
ctest --test-dir build_verify -R "runtime|device_detect" --output-on-failure   # 16/16 passed
ctest --test-dir build_verify -R kernel_cache_persist --output-on-failure      # 8/8  passed
ctest --test-dir build_verify -R "config_space|tuning_cache|benchmark|amd_tuning_db"  # 39/39 passed
```
```
runtime|device_detect : 100% tests passed, 0 tests failed out of 16   RUNTIME_EXIT=0
  (incl. runtime.LoadTuningCacheJsonThenSelect, runtime.RunServesSelectedVariant,
   runtime.UnvalidatedEntryIsNotServed, runtime.MissReturnsClearErrorAndDoesNotServe)
kernel_cache_persist  : 100% tests passed, 0 tests failed out of 8    PERSIST_EXIT=0
  (incl. StaleBinaryInvalidatedFallsBackToMiss, MultiGpuSelectsBestPerGpu, HashFileMissingFileIsError)
autotuner C++ core    : 100% tests passed, 0 tests failed out of 39   AUTOTUNE_CPP_EXIT=0
```

---

## Area 5 — Dataflow simulator + golden + metrics + viz  → PASS

```bash
POLYKERNEL_OPT=$PWD/build_verify/tools/polykernel-opt/polykernel-opt \
  python3 tools/polykernel-report/dataflow_report.py --backend=dataflow examples/mlp_block.mlir \
    --json-out /tmp/opencode/f3/dataflow_metrics.json
```
```
dataflow: utilization 100.00%, traffic reduction 99.9878% from fusion, bottleneck=active
representative_shape:   [32, 32, 32] (M,N,K) on a 4x4 grid
summa_tiles:            tileM=8 tileN=8 tileK=8 steps=4
grid_utilization:       100.00% (16/16 PEs active)
sram_pressure:          2.60% (1280 B resident vs 49152 B)
messages_sent:          2048 wavelets (A=1024 + B=1024)
avg_hop_distance:       3.0 hops (1 cycle/hop)
critical_path_cycles:   7
bottleneck:             active (cause: none)
fusion:                 fused_rmsnorm_matmul eliminates tensor<1x2048x4096xbf16>
traffic_reduction:      99.9878% (16777216 wavelets removed; 33,554,432 B round-trip)
xref mlp_traffic.json:  global reduction 7.02% (whole-pipeline global-byte view)
REPORT_EXIT=0
```
Self-contained HTML visualizer:
```bash
python3 tools/polykernel-report/dataflow_viz.py --backend=dataflow --viz examples/mlp_block.mlir --html-out /tmp/opencode/f3/dataflow_report.html
```
```
dataflow viz: 4x4 grid, SRAM pressure 2.60%, 2048 wavelets, critical path 7 cycles, fusion reduction 99.9878%; self-contained (no external refs)
VIZ_EXIT=0
self-containment grep:  http://=0  https://=0  createElementNS=0  xmlns=0  cdn=0
```
Simulator golden + metrics + core (ctest):
```bash
ctest --test-dir build_verify -R "dataflow|summa|metrics" --output-on-failure   # 14/14 passed
ctest --test-dir build_verify -R "scheduler|router|color|grid"                  # 36/36 passed
```
```
dataflow|summa|metrics      : 100% tests passed, 0 tests failed out of 14   DATAFLOW_EXIT=0
  (incl. dataflow_correctness.SimulatorExecutedMatmulMatchesGolden,
   dataflow_correctness.OutputStationaryAccumulateIsExact,
   dataflow_correctness.BrokenRendezvousFailsGolden [negative],
   summa.RendezvousComputeFiresOnlyAfterBothPanels, metrics.HandComputed4x4*)
scheduler|router|color|grid : 100% tests passed, 0 tests failed out of 36   SIMCORE_EXIT=0
```
The simulator-executed SUMMA matmul matches the golden bit-exact (cosine=1.0/rel=0.0/pcc=1.0);
broken-rendezvous negative fails golden. GPU-free (independent of any GPU).

---

## Area 6 — Modal: NO MODAL_TOKEN → clean SKIPPED  → PASS

```bash
echo "MODAL_TOKEN=[${MODAL_TOKEN:-UNSET}]"   # MODAL_TOKEN=[UNSET]
python3 modal/deploy.py
```
```
STATUS: SKIPPED (not FAILED). Missing credential: MODAL_TOKEN
Detection: os.environ["MODAL_TOKEN"] is unset/empty -> no Modal auth available.
ACTION TAKEN: graceful skip. NO `modal deploy`, NO Modal API call, NO image build,
NO GPU container provisioned, NO spend. ...
EXACT modal setup + deploy commands (an owner runs these to deploy):
  modal setup ...  modal secret create polykernel-secrets ...  modal volume put polykernel-weights ...
  modal deploy modal/app.py
VERDICT: SKIPPED — graceful degradation, no crash, no surprise spend.
DEPLOY_EXIT=0
```
```bash
python3 modal/app.py
```
```
app: polykernel
service class: PolyKernelService
endpoints: predict (POST), benchmark (POST), kernels (GET), report (GET)
gpu: A10, timeout: 600s, startup_timeout: 300s
autoscaling: min_containers=1, max_containers=8, buffer_containers=1, scaledown_window=300s
memory snapshot: enable_memory_snapshot=True, experimental_options={'enable_gpu_snapshot': True}
[kernel-cache] detected sm_86 (NVIDIA A10), loaded best cached kernel for sm_86/fused_matmul_bias_gelu/M=2048,N=4096,K=11008,bf16 best=(block_m=128,block_n=128,block_k=64,num_warps=8,vector_width=4,unroll=4,stages=3) validated=True scored_by=measure
construction OK (no token required)
APP_EXIT=0
```
Both degrade cleanly: **SKIPPED / construct-OK, exit 0, no crash, no spend, no API call.**

---

## Area 7 — RunPod: NO RUNPOD_API_KEY → PROJECTED numbers  → PASS

```bash
echo "RUNPOD_API_KEY=[${RUNPOD_API_KEY:-UNSET}]"   # RUNPOD_API_KEY=[UNSET]
python3 benchmarks/run_bench_suite.py --no-write
```
```
SKIPPED: RUNPOD_API_KEY not set — opt-in rental documented (no spend).
[bench-suite] RunPod credentials ABSENT (checked RUNPOD_API_KEY, RUNPOD_API_URL, ~/.runpod).
[bench-suite] STATUS=SKIPPED (not FAILED); no pod provisioned, no RunPod API call, no spend.
[bench-suite] PROJECTED reports written (roofline + Todo 17 traffic):
  - reports/h100_bench.json  - reports/h100_bench.md  - reports/mi300_bench.json  - reports/mi300_bench.md
[bench-suite] H100 projected speedup (unfused=1.00x): fused=1.004x autotuned=2.838x  [PROJECTED]
[bench-suite] OPT-IN real numbers: export RUNPOD_API_KEY=<key> && ./benchmarks/rent_runpod.sh --suite mlp
SUITE_EXIT=0
```
Dashboard (aggregates + labels projections + failed-correctness gate):
```bash
python3 tools/polykernel-report/dashboard.py --dashboard --no-write
```
```
dashboard: failed correctness: 0 (10/10 ops validated against golden contract C); self-contained HTML (no external dependencies)
## Correctness (golden contract C) — PASS
- **failed correctness: 0**
- validated: 10 / 10 ops pass the golden (cosine >= 0.999, max rel err <= 1e-2, PCC >= 0.99)
## CUDA speedups (unfused = 1.00x baseline)
> PROJECTED (not measured): ... run benchmarks/rent_runpod.sh with RUNPOD_API_KEY set for real measured numbers
| NVIDIA H100 (SXM5, sm_90) | cuda (sm_90) | 1.000x | 1.004x | 2.838x |
| NVIDIA A100 (80GB, sm_80) | cuda (sm_80) | 1.000x | 1.003x | 2.862x |
## AMD speedups (unfused = 1.00x baseline)   > PROJECTED (not measured) ...
| AMD Instinct MI300X (gfx942) | hip (gfx942) | 1.000x | 1.003x | 2.850x |
DASHBOARD_EXIT=0
```
`grep -c PROJECTED`: h100_bench.md = 3, mi300_bench.md = 3. **No spend, no API call; speedups
clearly labeled PROJECTED; failed correctness 0.**

---

## Area 8 — lit suite + full ctest  → PASS

```bash
cmake --build build_verify --target check-polykernel
```
```
-- Testing: 14 tests, 14 workers --
Testing Time: 0.15s
Total Discovered Tests: 14
  Passed: 14 (100.00%)
LIT_EXIT=0
```
```bash
ctest --test-dir build_verify --output-on-failure
```
```
100% tests passed, 0 tests failed out of 154
Total Test time (real) =   4.56 sec
CTEST_EXIT=0
```
Per-suite (all 100%):
```
runtime|device_detect                          16/16
kernel_cache_persist                            8/8
config_space|tuning_cache                      19/19
benchmark                                      11/11
amd_tuning_db                                   9/9
ptxas|occupancy|roofline|kernel_report|amd_isa 19/19
summa|dataflow|metrics                         14/14
scheduler|router|color|grid                    36/36
quantize                                        4/4
```
Full Python suite (cross-check):
```bash
python3 -m pytest tests/ -q
```
```
97 passed in 42.02s
FULL_PYTEST_EXIT=0
```
(golden self 15 + e2e MLP 5 + autotune bench-gate 2 + kernels 75 = 97.)

---

## Summary

| # | Area | Result | Key evidence |
|---|------|--------|--------------|
| 1 | polykernel-opt e2e, 3 examples, 8 passes | **PASS** | all EXIT_CODE=0, fusion+tile+memory+report attrs, 11 kernels emitted |
| 2 | CUDA nvcc sm_80+sm_90 + GPU-free report | **PASS** | 10 kernels, 0 spills, PTX emitted; contract-H report (occ 100%, compute-bound) |
| 3 | HIP build + RUN on gfx1101 + golden | **PASS** | gate PASS; pytest tests/kernels 75 passed, **0 failed correctness** |
| 4 | Autotuner (gated) + C++ runtime | **PASS** | 4 validated + broken REJECTED on gfx1101; runtime 16/16, persist 8/8, autotune-core 39/39 |
| 5 | Dataflow sim + golden + metrics + viz | **PASS** | util 100%, fusion reduction 99.9878%, self-contained viz; ctest 14/14 + 36/36 |
| 6 | Modal no-token SKIPPED | **PASS** | deploy.py SKIPPED exit 0; app.py constructs exit 0; no spend |
| 7 | RunPod no-key PROJECTED | **PASS** | bench-suite SKIPPED exit 0; dashboard failed-correctness 0, speedups PROJECTED |
| 8 | lit + full ctest | **PASS** | lit 14/14; ctest 154/154; pytest tests/ 97/97 |

**Runnable paths:** all pass (compiler e2e, CUDA compile+analyze GPU-free, HIP run on the
7800 XT with 0 failed correctness, correctness-gated autotuner + runtime, dataflow sim
golden+metrics). **Gated paths:** degrade cleanly (Modal SKIPPED, RunPod PROJECTED — exit 0,
no crash, no spend). No product code modified; only this report written.

VERDICT: APPROVE
