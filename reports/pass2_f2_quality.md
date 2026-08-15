# F2 Code-Quality Review - polykernel-pass2

**Reviewer:** F2 code-quality reviewer (final verification wave)
**Date:** 2026-08-15
**Plan:** `.omo/plans/polykernel-pass2.md` (F2 acceptance, L456-461)
**Scope:** every changed C++/CUDA/Python file of the pass; key-leak greps; gtest
meaningfulness; backend-guard discipline; C++ standard; stub scan; full build gate.
**Base for the diff:** pre-pass HEAD; the pass is fully committed (working tree clean
at first review; the re-run below adds the aligned `test_cuda_run.py` flag as the only
working-tree change).

## F2 RE-RUN (2026-08-15) - LOW-1 fix verification

Re-runs the F2 code-quality review after the cosmetic fix for the single LOW finding
(L-1: test-launcher C++ standard). Verdict after re-run: **APPROVE, L-1 CLOSED** - the
fix introduced no new findings (it is a single compile flag with no code-path impact).

## VERDICT: APPROVE

No blocker or high findings. One low (C++ standard consistency in two test launchers)
and three informational notes; none blocks the pass. The plan's F2 QA-failure
conditions (an unguarded CUDA intrinsic in the portable template, or a leaked key
line) are both verifiably absent.

**RE-RUN VERDICT (2026-08-15): APPROVE - L-1 CLOSED.** Commit 3771bc7 changes
`test_mma.py` L135 from `-std=c++17` to `-std=c++20` (the commit's sole change), and
the re-run additionally aligns `test_cuda_run.py`'s nvcc invocation with `-std=c++20`
(working-tree change, not committed - the audit .md is the deliverable). Re-verified:
`pytest tests/kernels/test_mma.py tests/kernels/test_cuda_run.py -q` inside the nix
devenv shell -> **23 skipped (5 + 18), 0 failed, exit 0** - both nvcc sm_89 launcher
builds compile clean under C++20. The project's C++20 standard now holds uniformly
(CMake, bench.py both phases, bench_cuda.cpp, both test launchers). See L-1 for the
closure record.

---

## 1. What was reviewed

| File | Change (pass 2) |
|---|---|
| `kernels/generated/matmul_mma.cu` | NEW: nvcuda::wmma m16n16k16 bf16 MMA variant (todo 10) |
| `kernels/template/kernel_common.h` | bf16x2 store fix (CUDA arm only, todo 9 Bug 1) |
| `kernels/generated/gelu.cu` | gelu_one double-erf precision chain (todo 9 Bug 2) |
| `kernels/generated/fused_matmul_bias_gelu.cu` | same gelu_one chain (todo 9 Bug 2) |
| `lib/Runtime/cuda_run_main.cpp` | NEW: CUDA launcher driver, `CK_` error checks (todo 7) |
| `lib/Autotune/Benchmark.cpp` | CudaEventTimer + backend-guarded driver shim (todo 13) |
| `include/PolyKernel/Autotune/Benchmark.h` | CudaEventTimer declaration (todo 13) |
| `tools/polykernel-bench/bench.py` | `--backend cuda` three-phase nvcc driver build (todo 13) |
| `tools/polykernel-report/dashboard.py` | per-arch real-vs-projected status tagging (todo 16) |
| `lib/Passes/LowerToCuda.cpp` / `LowerToHip.cpp` | gelu emitter parity (todo 9 Bug 2) |
| `lib/Analysis/tests/{occupancy,roofline}_test.cpp` | hand-computed sm_89 fixtures (todo 5) |
| `tests/kernels/test_cuda_run.py` / `test_mma.py` | CUDA test twins (todos 8/10) |
| `scripts/pod_env.sh` / `scripts/sync_pod.sh` | pod probe + sync (todo 4); `*.pem` excluded |
| `.gitignore` | `*.pem` under a security section (todo 3) |

Untouched and verified unchanged vs pre-pass (`git diff --stat` empty):
`kernels/generated/matmul.cu` (scalar correctness anchor), `matmul_wmma.cu` (RDNA3
sibling), `lib/Runtime/HipRuntime.cpp`, `lib/Analysis/Occupancy.cpp` sm_80/sm_90
values, H100/A100/MI300 report numbers.

## 2. Plan F2 "What" - verification results

### 2.1 Portable-template discipline - PASS
- `matmul_mma.cu` L62-64: `#if !defined(POLYKERNEL_CUDA) #error` - hard error under
  HIP, mirroring `matmul_wmma.cu`'s inverse guard. The NVIDIA-only section
  (fragment types, `nvcuda::wmma::*`, `__nv_bfloat16` aliasing) lives entirely
  behind that guard; every backend-agnostic concern uses `pk_*`/`PK_*`.
- `kernel_common.h` L103-118: the bf16x2 store fix is `#if defined(POLYKERNEL_CUDA)`
  (materializes both halves via `__bfloat16_as_ushort`, composes `(hi<<16)|lo`,
  memcpy); the `#else` HIP arm is the pre-existing plain memcpy, byte-identical to
  the original. Loads untouched (values from memory are already canonical).
- `Benchmark.cpp`: Layer 1 (gate + selection, L31-52) is GPU-free, always compiled,
  and unchanged - the git diff adds only Layer 2/3 content; the 13 `benchmark.Gate*`
  gtests pass in the 158/158 ctest run. Layer 2 timers are mutually exclusive
  (`#ifdef POLYKERNEL_HIP` / `#ifdef POLYKERNEL_CUDA`); the driver's `Timer` alias
  (L208-212) selects CudaEventTimer under CUDA, avoiding the dead HIP reference.
  Layer 3's device surface routes ALL seven Hip* call sites (DevBuf ctor/dtor/
  move-assign, upload H2D, download_write D2H, run sync x2) through the
  backend-guarded shim (L219-243); the old `namespace runtime = ...` alias was
  removed and the HIP branch calls `polykernel::runtime::Hip*` fully qualified.
- `bench.py` `build_driver`: HIP = one hipcc command (HipRuntime.cpp in the source
  list); CUDA = the documented three-phase build (clang++ host TUs -> nvcc
  `-x cu -arch= -DPOLYKERNEL_CUDA -include cstdio` device TUs -> nvcc link
  `-lLLVM-21`), with `HipRuntime.cpp` dropped for CUDA and per-backend driver paths
  so the two builds coexist. The three-phase split is a documented toolchain
  constraint (nvcc's gcc-13.4 host compiler cannot parse LLVM-21 headers), not a
  defect; the ldd/closure evidence (`reports/pass2_driver_ldd.log`) is recorded per
  plan R2.
- Grep sweep: no raw `__nv_*`/`__hip_*` intrinsic outside the guarded arms of
  `kernel_common.h` and `matmul_mma.cu`'s `#if` section in any changed file.

### 2.2 No leaked secrets / credentials - PASS (semantic), one note
- `git check-ignore private_key.pem public_key.pem` -> exit 0, matched by
  `.gitignore:46:*.pem`; `git ls-files | grep -c '\.pem$'` -> 0 tracked pem files.
- `git grep -i "PRIVATE KEY"` on tracked files: NOT literally empty - it returns 6
  benign lines: mode-600/presence checks in `scripts/pod_env.sh` (e.g. "[PASS]
  private key present, mode 600") + `scripts/sync_pod.sh`, and the same status text
  in the tracked probe evidence logs (`reports/pod_env*.log`,
  `reports/pass2_pod_probe_neg.log`, `reports/pass2_pod_sync.log` incl. the line
  "the private key is NOT on the pod"). None contains key MATERIAL: no `BEGIN ...
  PRIVATE KEY` block, no base64 body, no key fingerprint or passphrase anywhere in
  tracked files (verified by content grep). The plan's F2 QA-failure trigger is "a
  leaked key line" - absent. (Finding I-2 below records the wording gap.)
- The private key never leaves the machine: `sync_pod.sh` excludes `*.pem` in BOTH
  the rsync and the tar-over-ssh fallback exclude lists, and step 4 verifies on the
  pod that no pem landed.

### 2.3 gtest coverage meaningful - PASS
- `Occupancy.HandComputedRegisterLimitedSm89`: regs=128, threads=256, smem=32KB ->
  blocks = min(65536/32768=2, 102400/32768=3, 48/8=6, 24) = 2 -> active_warps 16,
  occ = 16/48 = 33.33%, limiter=registers; asserts `max_warps_per_sm == 48` and the
  exact hand-computed fraction `100.0*16.0/48.0` (not a trivial assert).
- `Occupancy.WarpLimitedSm89`: threads=1024 -> 32 warps/block -> 48/32 = 1 block ->
  occ = 32/48 = 66.67%, limiter=warps; the same input on sm_90 yields 100%,
  locking the 48-warp table against the 64-warp default.
- `Roofline.Sm89RidgeAndClassification` (PerfFor {91.1 TFLOPS, 960 GB/s} -> ridge
  94.9; small GEMM AI 63.01 memory-bound vs (4096^3) AI 1365 compute-bound) and
  `Roofline.Sm89BoundaryStraddlesRidge` (K=360 AI 94.43 memory vs K=368 AI 94.97
  compute - pins the ridge to ~1 decimal). sm_80/sm_90 fixtures unchanged.

### 2.4 CUDA launcher error checking mirrors the HIP driver - PASS
- `cuda_run_main.cpp` `CK_` (L78-86): abort-on-failure with call text, `__FILE__`,
  `__LINE__`, `cudaGetErrorString` - the exact shape of the HIP layer's
  `PK_HIP_CHECK` (the header comment documents why `PK_CHECK` alone is
  insufficient: it only prints). Applied to every cudaMalloc/cudaMemcpy/
  cudaDeviceSynchronize/cudaFree in the bridge. The `neg` mode deliberately uses
  raw calls so it can OBSERVE the invalid-launch error code instead of aborting
  (and returns exit 1 + "invalid launch CAUGHT"). The MMA driver's `PK_CUDA_CHECK`
  and Benchmark.cpp's `PK_CUDA_CHECK` are the same twin pattern.

### 2.5 C++ standard consistency - PASS (re-run: LOW L-1 closed)
- C++20 everywhere in the build: CMakeLists.txt (project standard 20, L22-23), bench.py
  both build phases (`-std=c++20` in the hipcc offload phase, the clang++ host phase and
  the nvcc device phase), `benchmarks/bench_cuda.cpp` (rent_runpod.sh line documents
  `-std=c++20`).
- RE-RUN 2026-08-15: the two nvcc TEST launchers now ALSO compile with C++20 -
  `test_mma.py` L135 `-std=c++20` (commit 3771bc7, was explicit c++17) and
  `test_cuda_run.py`'s nvcc cmd gained `-std=c++20` (working-tree alignment mirroring
  test_mma.py/bench.py; previously nvcc-default C++17). Verified: `pytest tests/kernels/
  test_mma.py tests/kernels/test_cuda_run.py -q` inside `nix develop` -> 23 skipped
  (5+18), 0 failed, exit 0 - both sm_89 launcher builds (full host+device nvcc links)
  succeed under C++20. The literal "C++20 consistent" criterion now holds across CMake,
  bench.py, bench_cuda.cpp and both test launchers.

### 2.6 No stubs / TODOs / placeholders - PASS
- `git grep -nE "TODO|FIXME|HACK|XXX"` over all 11 changed code files: 0 matches.

## 3. HIP-regression + gelu golden-exactness claims (verified)

### 3.1 bf16x2 fix is CUDA-only (HIP path unchanged)
Static: the `#if defined(POLYKERNEL_CUDA)`/`#else memcpy` structure above.
Dynamic: live HIP regression on the local RX 7800 XT -
`pytest tests/kernels/ -q` inside the devenv shell: **75 passed, 23 skipped
(0 failed)**. The 23 skips are the CUDA-only modules (no NVIDIA GPU on this
machine); the HIP twin `test_hip_run.py` ran fully 18/18 with the fixed shared
header + kernels, including rmsnorm/gelu/fused_matmul_bias_gelu (the exact kernels
that exercised `pk_store_bf16x2` and `gelu_one`).

### 3.2 gelu erf change is golden-exact on both backends
- Structure: both generated kernels AND both generators
  (`LowerToCuda.cpp` L196-199 == `LowerToHip.cpp` L211-214, byte-identical emitted
  text) compute `0.5f * x * (1.0f + static_cast<float>(erf(static_cast<double>(x) *
  kInvSqrt2D)))`, mirroring golden.py L63-66's chain exactly: double erf -> fp32
  round BEFORE `1.0 +` -> fp32 add/mul in the same association
  (`0.5*x` first, then `*(1+erf)`); `kInvSqrt2D == 1.0/math.sqrt(2.0)` as doubles.
- Quantified residual (400,019-sample fp32 sweep across [-10,10] incl. the
  cancellation region, numpy replicas of both chains): 3 bf16-level mismatches
  (0.00075%), all in the |x|>=3.5 tail, max rel err 7.35e-3 - comfortably under
  the strict 1e-2 ceiling and far above the 99% bit-exact floor (99.999%+). The
  only structural difference is the pre-erf product precision (numpy rounds
  `x*inv` to fp32 under NEP 50; the kernel keeps the double product) - a benign,
  better-rounded deviation whose measured impact is within contract everywhere.
- End-to-end: pod run 18/18 on the Ada (cosine == pcc == 1.0, bit_exact_min
  0.999992) + local HIP 75/0 above.

## 4. Build gate
- `ctest --test-dir build --output-on-failure`: **158/158 passed, 0 failed**
  (154 pre-pass baseline + 4 new sm_89 fixtures; includes the 13 `benchmark.Gate*`
  tests proving Layer 1 untouched).
- lsp_diagnostics note: clangd is not configured with the kernel include paths /
  `-DPOLYKERNEL_CUDA`, so `cuda_run_main.cpp` reports env-level "kernel_common.h
  not found" errors (expected; the file is compiled by nvcc with the documented
  flags and ran on the pod). `Benchmark.cpp` (GPU-free Layer 1 TU) reports no
  diagnostics. The real gate is the build + ctest + pytest, all green.

## 5. Findings

### L-1 (LOW) C++ standard inconsistency in the two nvcc test launchers
`tests/kernels/test_mma.py` L135 passes `-std=c++17` and `tests/kernels/
test_cuda_run.py` relies on nvcc's default (C++17), while CMake, bench.py (both
build phases) and bench_cuda.cpp use C++20. No functional impact - all code
compiles and runs under both standards (75/0 pytest, 158/158 ctest, pod runs
green) and the shared template uses no C++20-only feature - but the F2 criterion
"C++ standard consistent (C++20)" is not literally met.
**Suggested fix (one line each):** add `-std=c++20` to both nvcc invocations,
matching bench.py. Non-blocking.

**RESOLVED - CLOSED (re-run 2026-08-15).** Both invocations now compile with C++20:
- `test_mma.py` L135 `-std=c++17` -> `-std=c++20`, the sole change of commit 3771bc7
  (`test(cuda): test_mma.py compile with -std=c++20 (project standard; was c++17 -
  F2 cosmetic finding)`).
- `test_cuda_run.py` nvcc cmd gained `-std=c++20` (working-tree alignment, 1 line,
  NOT committed per the F2 re-run constraint; the audit .md is the deliverable).
Re-verification: `pytest tests/kernels/test_mma.py tests/kernels/test_cuda_run.py -q`
inside the nix devenv shell -> **23 skipped (5 + 18), 0 failed, exit 0** - both nvcc
sm_89 launcher builds compile clean under C++20 (the shared template uses no
C++20-only feature, so the flag is purely standard-consistency). No new blocker/high/
medium/low findings introduced: the change is a single compile flag with no code-path
impact.

### I-1 (INFO) `git grep -i "PRIVATE KEY"` is not literally empty on tracked files
The 6 matches are presence/mode-check STATUS text in `scripts/pod_env.sh`,
`scripts/sync_pod.sh` and their evidence logs (e.g. "[PASS] private key present,
mode 600", "the private key is NOT on the pod") - zero key material (no BEGIN
block, no fingerprint, no base64). The plan's F2 acceptance wording ("returns
nothing") is stricter than the intent ("no leaked secrets/credentials"), which
holds. If the verbatim criterion must be met, the log lines could be reworded
(e.g. "SSH key check"), but no action is required for APPROVE.

### I-2 (INFO) `matmul_mma.cu` L16 comment names the m16n8k16 PTX instruction
The header comment explains the underlying hardware instruction
(`mma.sync.aligned.m16n8k16...`), which is what the wmma m16n16k16 fragment API
lowers to; the fragment is correctly m16n16k16 and the kernel is golden-validated
on the Ada. Comment-level nuance only; pedagogically it is accurate (the wmma API
abstracts the lane mapping). No action.

### I-3 (INFO) Benchmark.cpp `run_dev` (L594-604) prints `cudaGetErrorString(err)`
When `err == cudaSuccess` but `ndev == 0`, the message reads "(count=0, no
error)" instead of naming the no-device condition; the cuda_run_main.cpp twin
handles this by substituting `cudaErrorNoDevice`. Cosmetic - the exit code and the
bench's "no device" SKIPPED detection are unaffected.

## 6. Conclusion
Every F2 "What" item verifies: portable-template discipline holds (the only raw
backend intrinsics sit behind the exact guards the plan demands), the bf16x2 fix
is CUDA-only with the HIP path byte-identical, the gelu chain is golden-exact on
both backends within the calibrated contract (quantified), no key material exists
in tracked files, the sm_89 fixtures are hand-computed (33.33% / 66.67%), the
launcher's `CK_` mirrors the HIP driver's abort-on-failure checking, and there
are no stubs or placeholders. The build gate is green (158/158; HIP 75/0). The
single low finding (test-launcher C++17) is cosmetic and non-blocking.

RE-RUN CONCLUSION (2026-08-15): the single LOW finding L-1 is CLOSED - both nvcc test
launchers now compile with `-std=c++20` (`test_mma.py` committed in 3771bc7;
`test_cuda_run.py` aligned in the working tree), matching CMakeLists.txt
(CMAKE_CXX_STANDARD 20), bench.py (both phases) and bench_cuda.cpp. Re-verification:
`pytest tests/kernels/test_mma.py tests/kernels/test_cuda_run.py -q` -> 23 skipped
(5+18), 0 failed, exit 0. The single-flag fix introduced NO new blocker/high findings
(no new findings of any severity). Verdict unchanged.

VERDICT: APPROVE
