# F1 Plan-Compliance Audit — polykernel-pass2

- **Audit:** F1 (final verification wave, `polykernel-pass2.md` L449-454)
- **Date:** 2026-08-15 (UTC)
- **RE-RUN (F1 fix verified):** the single gap from the first run (stale README/docs
  PENDING/SKIPPED/PROJECTED claims, CHANGES_REQUESTED) is **FIXED by commit `be10c8f`**
  (docs re-sync to the measured reality) — re-audited against the current committed tree.
  **VERDICT: APPROVE** (details in §4/§5).
- **Method:** evidence-based — every todo mapped to its committed evidence + commit; key claims
  re-verified by executing the verification commands directly (not trusting summaries):
  `python3 -c "import json; ..."` on the bench JSON, `grep` on the tuning cache / dashboard /
  pod logs, `git check-ignore` / `stat` on the keys, `git cat-file -e` on every commit, source
  greps (`flake.nix`, `devenv.yaml`, `.gitignore`, `schema.json`, `Occupancy.cpp`, `Roofline.cpp`,
  `matmul_mma.cu`, `test_cuda_run.py` vs `test_hip_run.py`, `bench.py`, `dashboard.py`).
- **Plan checkboxes:** all 16 todos are `[x]` in `.omo/plans/polykernel-pass2.md` (verified by
  reading the plan; no checkbox was modified by this audit). F1-F4 rows remain `[ ]`.

---

## 1. Per-todo mapping: todo → evidence → commit

| # | Todo | Evidence path(s) (committed) | Commit(s) | Green |
|---|------|------------------------------|-----------|-------|
| 1 | `devenv.yaml` flat `nixpkgs:` block (allow_unfree + cuda_support + cuda_capabilities) | `reports/pass2_devenv_yaml.log` (nvcc 12.6 / ptxas / hipcc on PATH via `nix develop`), `reports/pass2_devenv_yaml_neg.log` (CLI-mode unfree error without the block, flake.nix operative both ways) | `e357545` | ✅ |
| 2 | devenv v2.1.2 → v2.2.2 bump + lockfile + full green gate | `reports/pass2_devenv_bump.log` (lit **14/14**, ctest **154/154**), `reports/pass2_devenv_bump_neg.log` (revert proves lock moved: v2.2.2 rev b8030c58 vs v2.1.2 rev ea3d94ac) | `e8c05c9` | ✅ |
| 3 | `*.pem` gitignored + mode 600 | `reports/pass2_key_hygiene.log` + `.gitignore` L44-46 (security section, `*.pem`); re-verified: `git check-ignore private_key.pem public_key.pem` → both ignored; `stat` = **600**; `git ls-files | grep -i pem` → none; `git grep "BEGIN.*PRIVATE"` on tracked files → no key material | `f8c6305` | ✅ |
| 4 | Pod sync + env probe (gate for pod todos) | `reports/pod_env.log` (initial gate SKIPPED — old pod rejected the key, exact cause named), `reports/pod_env_v2.log` (**GATE VERDICT: PASS** — new pod `ubuntu@216.81.200.13`, RTX 6000 Ada detected, nvcc 12.2 ≥ 11.8, numpy/ml_dtypes/pytest OK, glibc 2.35, rsync present), `reports/pass2_pod_probe_neg.log` (port 2222 → clean SKIPPED, no hang), `reports/pass2_pod_sync.log` (rsync PASS, key excluded from pod) | `6a82a4a`, `31af130` | ✅ |
| 5 | sm_89 in Occupancy + Roofline + hand-computed gtest fixtures | `reports/pass2_sm89_analyzer.log` (**21/21 tests pass** incl. new sm_89 fixtures, sm_80/90 regression unchanged), `reports/pass2_sm89_analyzer_neg.log` (2048-thread block clamps to 0, no crash); source re-checked: `Occupancy.cpp` sm_89 = warps 48 / blocks 24 / threads 1536 / smem 100·1024 / 99·1024; `Roofline.cpp` sm_89 = 960 GB/s + BPG 91.1 TFLOPS scalar-bf16 convention annotated (tensor-dense 364.25 / fp16 182 distinguished) | `0129933` | ✅ |
| 6 | Tools plumbing: nvcc_driver / analyze / report / run_bench_suite / dashboard / tuning schema | `reports/pass2_sm89_plumbing.log` (nvcc `--arch sm_80,sm_89,sm_90 --ptx` + ptxas -v parses), `reports/kernel_report_sm89.json` (arch sm_89, occupancy 48/48 = 100%, limiter warps, roofline compute-bound, path=scalar), `reports/pass2_sm89_plumbing_neg.log` (`--arch sm_88` rejected by nvcc fatal AND analyze.py validate_arch — no silent default); `tuning/schema.json` L27-29 documents `gpu ∈ {gfx1101, gfx942, sm_80, sm_89, sm_90}` (contract H) | `2ca118f` | ✅ |
| 7 | `lib/Runtime/cuda_run_main.cpp` (CUDA twin of hip_run_main) | `reports/pass2_cuda_run_build.log` (nvcc sm_80/89/90 clean; ptxas -v spill check: **0 spill stores / 0 spill loads** on all 3 archs, all kernels), `reports/pass2_cuda_run_neg.log` (removed launch extern → nvcc link error, ABI contract enforced); `dev` subcommand probe evidence in the pod logs (device name + 8.9) | `eae0be8` | ✅ |
| 8 | `tests/kernels/test_cuda_run.py` (CUDA twin of test_hip_run.py) | `reports/pass2_cuda_run_skip.log` (no-device host: nvcc builds, `_gpu_probe` skips **18/18** tests with named reason, exit 0); source re-checked: same session fixture / `_gpu_probe` / `_check` / per-op fixtures / `test_cuda_negative_launch_caught` (1024-thread-per-block limit, "invalid configuration argument" asserted) as the HIP twin | `8e5ecc2` | ✅ |
| 9 | Run CUDA validation on the pod (every kernel vs golden on Ada) | `reports/pass2_ada_cuda_run.log` (**18/18 passed**, verdict "0 failed correctness — cosine==pcc==1.0, ≥ 99.99% bit-exact on the RTX 6000 Ada (sm_89)"; device probe "NVIDIA RTX 6000 Ada Generation COMPUTE_CAPABILITY 8.9"), `reports/pass2_ada_cuda_run.json` (17/17 harness cases, `cosine_min=1.0`, `bit_exact_min=0.999992` ≥ 0.9999, all `passed:true`, large-K rel ceiling 0.05 honored), `reports/pass2_ada_cuda_run_neg.log` (scaled-GELU variant FAILS golden on the pod — gate is real, not vacuous) | `3f9b760`, `d85866a` | ✅ |
| 10 | `kernels/generated/matmul_mma.cu` (nvcuda::wmma m16n16k16 bf16, additive, correctness-gated) | `reports/pass2_mma_local.log` (nvcc sm_80/89/90 + PTX clean; pytest 5/5 SKIPPED on no-device host cleanly), `reports/pass2_mma_neg.log` (bad-store variant cosine 0.0156 < 0.999 → excluded; scalar baseline STANDS); source re-checked: `#if !defined(POLYKERNEL_CUDA) → #error`, `path=mma` annotation, m16n16k16 fragment contract | `d84b77f` | ✅ |
| 11 | MMA validated on the pod + `path=mma` annotation | `reports/pass2_ada_mma.log` (**5/5 pytest PASS** on RTX 6000 Ada; `path=mma` stderr annotation verbatim; mma_bad negative EXCLUDED cosine 0.0475, scalar STANDS bit-exact), `reports/pass2_ada_mma_neg.log` (on-pod: K=130 CORRECT via zeroed-tail, K padded to 160 in smem, documented) | `9d7fea4` | ✅ |
| 12 | Pedagogical docs/README claim audit | `reports/pass2_docs_audit.log` (sweeps: "fully validated on rented H100/A100" gone, "runtime-validated on rental" gone, "13 tests" gone → **14**, matmul_wmma.cu explicitly HIP-only, MMA path documented with col-major B staging), `reports/pass2_docs_audit_neg.log` (pre-fix catches: cuda_backend.md L118-119 false claims found + fixed). Re-run note: audit was accurate at commit time; the post-pod-unblock stale prose ("PENDING pod-key authorization") was the F1 Finding-1 gap, now closed by `be10c8f` (re-verified below) | `e29a2c2`, `be10c8f` | ✅ |
| 13 | CUDA backend for the correctness-gated benchmark (CudaEventTimer + nvcc driver) | `reports/pass2_driver_ldd.log` (R2 closure decision: full GLIBC_ symbol list recorded; pod chose build-on-pod with clang++-15 + apt llvm-15-dev, glibc 2.35 OK), `reports/pass2_bench_cuda_local.log` (no-device host: clear "SKIPPED: no device" degradation), `reports/pass2_bench_cuda_gate_neg.log` (broken constant-fill variant **REJECTED** — cosine 0.564, "excluded, NOT timed, never best"; gtest `NeverSelectsFasterButUnvalidated` OK) | `388ed57` | ✅ |
| 14 | Real headline benchmark on the pod → `reports/ada6000_bench.{json,md}` | `reports/ada6000_bench.json` — **re-verified: `measured: True | arch: sm_89 | device: NVIDIA RTX 6000 Ada Generation`**, unfused 25.5181 ms / fused 30.4589 ms / autotuned 29.7851 ms → 1.000x / 0.838x / 0.857x; `reports/ada6000_bench.md` (MEASURED banner); `reports/pass2_ada_bench_neg.log` (genuine OOM: `cudaMalloc ... out of memory`, EXIT_CODE=1, no hang — bounds-check proven) | `c3bc7ad` | ✅ |
| 15 | Full correctness-gated CUDA autotuner on the pod → real sm_89 cache | `reports/pass2_ada_autotune.log` (8/8 configs VALIDATED, real CUDA-event ms; injected broken **REJECTED**; cache written with `gpu:"sm_89"`, `scored_by:"measure"`, `validated:true`, `time_ms:27.04896`), `reports/ada6000_tuning_cache.json` (archived copy — re-verified identical fields), `reports/pass2_ada_autotune_consume.log` (**consumption check PASS**: `LoadTuningCache` size=1 + `Select("sm_89", fused_matmul_bias_gelu, 2048x4096x11008)` → returns the written best; wrong-gpu `Select(sm_90)` → clean miss), `reports/pass2_ada_autotune_neg.log` (`--rel-ceiling 1e-8` → all rejected, no empty-best fallback) | `745e73b` | ✅ |
| 16 | Dashboard + reports regeneration: MEASURED sm_89 section, projections labeled, README status | `reports/pass2_dashboard.log` (PROJECTED-era regeneration + negative missing-file→SKIPPED path), `reports/pass2_dashboard_v2.log` (**MEASURED-era regeneration**: `| NVIDIA RTX 6000 Ada (sm_89) | cuda (sm_89) | MEASURED | 1.000x | 0.838x | 0.857x |`, wall times, fusion-slower NOTE, `failed correctness: 0`, H100/A100/MI300 rows byte-identical PROJECTED, HTML self-contained, embedded JSON `"measured": true`), `reports/benchmark_report.md` re-verified (MEASURED row L32, PROJECTED labels L30-31/L42, `failed correctness: 0` L20/L64), `reports/pass2_dashboard_neg.log` (deleted ada6000_bench.json → no crash, row degrades to SKIPPED + missing-file note, never fake MEASURED). Re-run note: the README-status acceptance (todo 16) is now MET via `be10c8f` (README L21/L82-83/L96-101/L112-124 say runtime-validated + MEASURED; H100/A100/MI300 remain PROJECTED) | `4ebd15c`, `dc0537a`, `be10c8f` | ✅ |

**⚠️ = partial gap — see Finding 1 below.**

> **Re-run status (2026-08-15, after `be10c8f`):** the ⚠️ marks are lifted — todos 12 and 16
> are now fully green. Finding 1 is RESOLVED (see §4).

---

## 2. Key claim verification (commands executed during this audit)

| Claim | Command | Result |
|-------|---------|--------|
| `ada6000_bench.json` is measured | `python3 -c "import json; d=json.load(open('reports/ada6000_bench.json')); print(d['measured'], d['arch'], d['device'])"` | `True sm_89 NVIDIA RTX 6000 Ada Generation` ✅ |
| sm_89 tuning cache is measured | grep `gpu`/`scored_by`/`validated`/`time_ms` in `reports/ada6000_tuning_cache.json` | `"gpu":"sm_89"` `"scored_by":"measure"` `"validated":true` `"time_ms":27.04896` ✅ |
| Dashboard MEASURED row | `grep -n "MEASURED" reports/benchmark_report.md` | Ada sm_89 row `MEASURED | 1.000x | 0.838x | 0.857x` + wall-time line + fusion-slower NOTE; H100/A100/MI300 `PROJECTED`; `failed correctness: 0` ✅ |
| Pod gate verdict | `grep -n "GATE VERDICT" reports/pod_env_v2.log` | `GATE VERDICT: PASS` (checks ssh=0 gpu=0 nvcc=0 python3=0 pkgs=0) ✅ |
| 0 failed correctness on the pod | `pass2_ada_cuda_run.log` L1-2/L29-31 + `.json` summary | `18 passed` (17 harness golden cases + negative-launch test); harness 17/17, `bit_exact_min=0.999992` ≥ 0.9999, `cosine_min=1.0`; MMA 5/5 ✅ |
| `*.pem` gitignored, keys mode 600, no leak | `git check-ignore private_key.pem public_key.pem`; `stat -c '%a' private_key.pem`; `git ls-files | grep -i pem`; `git grep "BEGIN.*PRIVATE"` | both ignored (exit 0); mode 600; no tracked pem; no key material in any tracked file ✅ |
| devenv v2.2.2 + flake.nix intact | `flake.nix` L8 `devenv.url = "github:cachix/devenv/v2.2.2"`; `devenv.yaml` flat `nixpkgs:` block with allow_unfree/cuda_support/cuda_capabilities; `flake.nix` pkgs config `allowUnfree/cudaSupport/rocmSupport` present and operative | ✅ (both entry paths kept, per plan) |
| Contract H gpu key | `tuning/schema.json` L27-29 | `gpu` description: `gpu ∈ {gfx1101, gfx942, sm_80, sm_89, sm_90}` ✅ |
| All commits exist | `git cat-file -e <hash>^{commit}` for all 16 todo commits + `31af130` + `d85866a` | all OK ✅ |
| Scalar anchor unchanged | `git log --oneline 0129933~1..HEAD -- kernels/generated/matmul.cu` | empty — `matmul.cu` untouched in pass 2 ✅ |

---

## 3. Contracts check (A / C / G / H / K)

- **Contract A (build):** devenv v2.2.2 in `flake.nix` + lockfile committed; full gate green after bump (lit 14/14, ctest 154/154 — `pass2_devenv_bump.log`). Toolset versions stay (llvmPackages_21 / cudaPackages_12_6 / ROCm unchanged). ✅
- **Contract C (golden):** single NumPy + ml_dtypes bf16 golden; `_check` enforces cosine ≥ 0.999 AND max_rel_err ≤ 1e-2 AND pcc ≥ 0.99, large-K ceiling 5e-2; pod JSON shows every case passing with `rel_ceiling` recorded per test. ✅
- **Contract G (per-backend acceptance, CUDA runtime row):** nvcc compiles clean sm_80/89/90 (`pass2_cuda_run_build.log`, `pass2_sm89_plumbing.log`), PTX emitted + ptxas -v parses, host/CPU ref passes, and the Ada pod run passes golden for every kernel (18/18 + 5/5) — the pod-unavailable SKIPPED carve-out was not needed (gate passed). ✅
- **Contract H (schemas):** `gpu` key documented for sm_89 in `tuning/schema.json` (verified above); `path=mma` report annotation operational (MMA kernel stderr + reports). ✅
- **Contract K (tensor-core CUDA twin):** `matmul_mma.cu` = `nvcuda::wmma` m16n16k16 bf16, A row-major / B col-major fragments, `load_matrix_sync`/`mma_sync`/`store_matrix_sync`, `#if !defined(POLYKERNEL_CUDA) → #error` guard, OOB zeroed (K padded to 160 for K=130, verified on-GPU), additive + correctness-gated, scalar anchor untouched. ✅

---

## 4. Finding 1 (the one gap) — README/docs claims contradict the committed MEASURED state

> **STATUS: RESOLVED — commit `be10c8f` (docs re-sync) closed this gap. Re-verified on the
> current committed tree (2026-08-15 re-run) — all checks below pass.**

**What the committed final state says (true):**
- `reports/pod_env_v2.log`: `GATE VERDICT: PASS` on the new pod (`ubuntu@216.81.200.13`) — the old pod's key rejection was unblocked (`31af130`).
- `reports/pass2_ada_cuda_run.log/json` + `pass2_ada_mma.log`: 18/18 + 5/5 golden on the RTX 6000 Ada.
- `reports/ada6000_bench.json`: `measured: true`, arch `sm_89`.
- `reports/benchmark_report.md`: `| NVIDIA RTX 6000 Ada (sm_89) | cuda (sm_89) | MEASURED | 1.000x | 0.838x | 0.857x |` (`dc0537a`).

**What README.md + docs/*.md still say (stale, now false):**
- `README.md` L21: "on-GPU run on the RTX 6000 Ada pod **pending pod-key authorization**"; L96-98: "**pending pod-key authorization** (pod gate SKIPPED) ... RTX 6000 Ada (sm_89) speedups remain **PROJECTED**"; L112-117: "The **on-GPU run is PENDING pod-key authorization** (pass-2 pod gate SKIPPED, `reports/pod_env.log`), **so no CUDA number is claimed as measured**. The dashboard ... now carries an RTX 6000 Ada (sm_89) row next to H100/A100: **PROJECTED**".
- `docs/architecture.md` L101, L122, L174-175: "**PENDING pod-key authorization** (pod gate SKIPPED, `reports/pod_env.log`)".
- `docs/cuda_backend.md` L8, L77, L159, L172: same PENDING claims.
- `docs/performance_model.md` L10, L113: "on-GPU run is **PENDING pod-key authorization** (the pass-2 pod gate was SKIPPED...)".
- `docs/upstream_pr.md` L330: "pending pod-key authorization".

**Why this is a gap, not a nit:** `README.md`/`docs/*.md` were last touched by `e29a2c2` (todo 12) and `4ebd15c` (todo 16's PROJECTED-era step) — both accurate *at their commit time* (pod genuinely SKIPPED). But the pod unblock (`31af130`), the real golden runs (`3f9b760`, `9d7fea4`, `d85866a`), the measured bench (`c3bc7ad`) and the MEASURED dashboard (`dc0537a`) all landed **after** — and `dc0537a` updated only `reports/*` + `dashboard.py`, **not** `README.md`/`docs/`. The repo now contains a direct contradiction: the dashboard says the Ada row is MEASURED while the README says "no CUDA number is claimed as measured" and "RTX 6000 Ada (sm_89) speedups remain PROJECTED". This fails todo 16's acceptance ("README Status says CUDA is runtime-validated on an RTX 6000 Ada (sm_89) with H100/A100/MI300 still projected") and the plan's pedagogical definition of done ("no claim requires trusting the author").

**Required fix (one commit):** re-sync `README.md` (Status section + "It is / It is not" table + Correctness section) and `docs/{architecture,cuda_backend,performance_model,upstream_pr}.md` PENDING/SKIPPED/PROJECTED claims to post-pass reality: pod gate PASSED (`reports/pod_env_v2.log`), CUDA runtime-validated on the RTX 6000 Ada (sm_89) with 0 failed correctness (18/18 + 5/5), sm_89 row MEASURED (1.000x/0.838x/0.857x, `reports/ada6000_bench.json`), H100/A100/MI300 remain PROJECTED.

### 4.1 Re-run verification (fix `be10c8f` — all commands executed against the current tree)

| Check | Command | Result |
|-------|---------|--------|
| Stale-claims sweep is EMPTY | `grep -rn "pending pod-key\|no CUDA number\|no measured\|pod gate SKIPPED" README.md docs/` | no matches, exit 1 ✅ |
| README L21 (It is/It is not table) | `sed -n '21p' README.md` | "runtime-validated on the RTX 6000 Ada pod (0 failed correctness)" ✅ |
| README L96-101 (Correctness) | `sed -n '96,101p' README.md` | "runtime-validated on the RTX 6000 Ada (sm_89) pod: 0 failed correctness (18/18 golden, cosine==pcc==1.0)" + bug fixes d85866a ✅ |
| README L112-124 (Status) | `sed -n '112,124p' README.md` | Pass-2 bullet: runtime-validated (0 failed correctness, `reports/pass2_ada_cuda_run.log`), pod story (216.81.200.13, ubuntu), row **MEASURED**: unfused 1.000× / fused 0.838× / autotuned 0.857×, "H100/A100/MI300 remain **PROJECTED**" ✅ |
| Measured numbers match the JSON | `python3 -c "import json; d=json.load(open('reports/ada6000_bench.json')); print(d['measured'], d['speedup'])"` | `True {'unfused': 1.0, 'fused': 0.8378, 'autotuned': 0.8567}` — README's 1.000x/0.838x/0.857x = the JSON ✅ |
| Dashboard ground truth | `grep -n "MEASURED\|PROJECTED" reports/benchmark_report.md` | L32 `| NVIDIA RTX 6000 Ada (sm_89) | cuda (sm_89) | MEASURED | 1.000x | 0.838x | 0.857x |`; H100/A100 PROJECTED (L30-31); MI300 PROJECTED (L42); `failed correctness: 0` ✅ |
| No residual PENDING anywhere in docs | `grep -rn "PENDING\|SKIPPED\|no measured" docs/` | only legit non-pod uses: hip_backend's `SKIPPED-local` semantics, architecture L167 Modal owner-gating, performance_model L137 runpod-no-credentials path — zero Ada-pod claims ✅ |
| Only new commits since first audit | `git log --oneline -25` | `be10c8f` (docs re-sync — the F1 fix) + `3771bc7` (test_mma.py `-std=c++20`, the F2 cosmetic finding) — nothing else; per-todo evidence map unchanged ✅ |

---

## 5. Verdict

VERDICT: APPROVE (re-run, 2026-08-15)

**Re-run outcome.** The sole gap from the first F1 run — README.md + docs claiming the pass-2
pod gate was SKIPPED / the on-GPU run PENDING pod-key authorization / "no CUDA number is
claimed as measured" / the RTX 6000 Ada (sm_89) row PROJECTED, contradicting the committed
MEASURED dashboard — is **FIXED by `be10c8f`** and re-verified against the current tree:
- `grep -rn "pending pod-key|no CUDA number|no measured|pod gate SKIPPED" README.md docs/`
  is **EMPTY** (exit 1);
- README L21 / L96-101 / L112-124 say runtime-validated on the RTX 6000 Ada pod (0 failed
  correctness, 18/18 golden, `reports/pass2_ada_cuda_run.log`), row **MEASURED**
  1.000×/0.838×/0.857× (matching `ada6000_bench.json`: `measured:true`,
  `{1.0, 0.8378, 0.8567}`), H100/A100/MI300 remain **PROJECTED**;
- `reports/benchmark_report.md` (the dashboard ground truth) shows MEASURED for sm_89
  (L32) + PROJECTED for H100/A100 (L30-31) + MI300 (L42), `failed correctness: 0`;
- the only commits since the first audit are `be10c8f` (the F1 fix) and `3771bc7` (the F2
  cosmetic test_mma.py `-std=c++20` fix) — nothing regressed, the per-todo evidence map
  holds unchanged.

**Full verdict (unchanged from the first run, now all-green):** todos 1-16 all have green
committed evidence (todos 12/16 fully green after `be10c8f`); contracts A/C/G/H/K honored;
all 16 todos present, none substituted; "0 failed correctness on the pod" verified true from
the raw logs/JSON; the measured numbers are real (`measured: true`, sm_89, RTX 6000 Ada,
CUDA events); no doc/README claim contradicts the committed state. Todo 16's README-status
acceptance and the plan's pedagogical definition of done ("no claim requires trusting the
author") are now met.

**Residual notes (non-blocking, unchanged):** the plan text's stale figures (lit "13 tests"
-> 14; ctest 154/154 -> 158/158; "AI ≈ 1.2e6" -> 1214.68; "11 base + 6 fused" -> 5 fused)
were recorded by the todo workers and F2/F4; they are plan-text slips, not implementation
gaps, and do not affect compliance.
