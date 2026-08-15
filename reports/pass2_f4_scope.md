# Pass 2 - F4 Scope-Fidelity Audit (Must-NOT-Have enforcement)

**Date:** 2026-08-15 · **Plan:** `.omo/plans/polykernel-pass2.md` (Must-NOT-Have section, L91-104)
**Pass base commit:** `cba03af` (wave-8 head, commit before `e357545`); pass span = `e357545..dc0537a` (19 commits).
**Method:** every check run by the agent with captured output; all diffs are against the pass base unless stated.

---

## 1. Op set stays CLOSED — no new `polykernel` ops — ✅ HOLDS

- `grep -c "polykernel\." include/PolyKernel/IR/PolyKernelOps.td` → **25** (mnemonic-string occurrences).
  Decoded: exactly **18 distinct mnemonics** = the named wave-8 set — 11 base ops
  (`add, attention, bias, gelu, kv_cache_update, matmul, qkv_projection, rmsnorm, rope, silu, softmax`),
  5 fused ops (`fused_kv_append_attention, fused_matmul_bias_gelu, fused_residual_rmsnorm,
  fused_rmsnorm_matmul, fused_softmax_mask`), plus `func` + `return` (the extra occurrences of
  func/return are the assembly-format strings of every op). Total = **18 op definitions**, matching the
  wave-8 F4 audit's "exactly 18 op definitions (16 compute ops + func/return)" (reports/f4_scope.md L23-34).
  NOTE: the plan's "11 base + 6 fused" arithmetic miscounts the fused group (5, not 6) — a plan-text
  bookkeeping slip, immaterial to the guardrail: the set is byte-identical to the audited wave-8 set.
- **Decisive proof:** `git diff cba03af -- include/PolyKernel/IR/PolyKernelOps.td lib/Dialect/ test/op_set_closed.mlir`
  → **0 lines**. The dialect source and its closed-set test were NOT touched anywhere in the pass.
- `matmul_mma.cu` adds **no** dialect op: it is a codegen VARIANT of the existing `polykernel.matmul`
  op with the `path=mma` report annotation only (contract H). `path=mma` confirmed verbatim in the pod
  run evidence: `[polykernel-mma] path=mma M=16 N=16 K=16` (reports/pass2_ada_mma.log).
- **Lit gate:** `llvm-lit -q build/test` → `PASS: POLYKERNEL :: op_set_closed.mlir (3 of 14)` (negative:
  `polykernel.conv2d` rejected as "custom op 'polykernel.conv2d' is unknown") and
  `PASS: POLYKERNEL :: dialect_roundtrip.mlir (6 of 14)` (positive: every named op round-trips).
  Full gate **14/14 (100%)**.

## 2. No SOTA / vLLM-beating claim — ✅ HOLDS (all hits are negated assertions)

Command: `grep -rinE "beat(s)? vllm|sota|state-of-the-art|faster than vllm" README.md docs/ reports/`.
Raw output is NOT empty — it returns 14 hits. **Every single hit was read in context and is a NEGATED
assertion** (the enforcement mechanism of this very rule), zero affirmative claims:

| Hit | Context |
|---|---|
| README.md:19 | "It is **not**" column: "A claim to **beat vLLM** or any production stack" |
| docs/architecture.md:26-27 | "**Is not:** a claim to beat vLLM … not SOTA" |
| docs/performance_model.md:16 | "**No SOTA-performance claim is made**: the goal is the machinery, not beating vLLM" |
| reports/w6_dashboard_neg.log:96 | "No SOTA-performance claim: … not beating vLLM" |
| reports/f4_scope.md (wave-8 audit) | the prior audit's own classification text (same ruling) |
| reports/benchmark_report.md:77 + 3 HTML disclaimers | "No SOTA-performance claim" (dashboard-generated) |

Per the wave-8 F4 precedent and the audit charter ("SOTA may appear only in a 'we do NOT claim this'
context"), these are **COMPLIANT, not violations**. The `| grep -v pass2` filter in the task's command
only drops pass-2 log lines; the compliant negations in README/docs remain and are correctly classified.

## 3. H100/A100/MI300 stay PROJECTED; every report honestly labeled — ✅ HOLDS

- `reports/ada6000_bench.json` → `measured: true`, `arch: "sm_89"`, `device: "NVIDIA RTX 6000 Ada
  Generation"`, `status: "MEASURED"`, `projected: false` — backed by a **real pod run** (todo 14:
  CUDA-event timings, unfused 25.52 / fused 30.46 / autotuned 29.79 ms median, speedups
  1.000x/0.838x/0.857x). **No fake MEASURED anywhere without a real run.**
- `reports/h100_bench.json` → `measured: false` / `status: PROJECTED`; `reports/mi300_bench.json` →
  `measured: false` / `status: PROJECTED`; `reports/h100_bench.md` + `mi300_bench.md` →
  "⚠ PROJECTED — NOT MEASURED". Both wave-8 files byte-identical (git diff empty).
- Dashboard (`reports/benchmark_report.md`): MEASURED appears for **sm_89 only** —
  `| NVIDIA RTX 6000 Ada (sm_89) | cuda (sm_89) | MEASURED | 1.000x | 0.838x | 0.857x |` while
  H100 (1.004x/2.838x), A100 (1.003x/2.862x), MI300 (1.003x/2.850x) rows are all PROJECTED with
  the per-arch status line `H100: PROJECTED; A100: PROJECTED; RTX 6000 Ada (sm_89): MEASURED`.
- **The negative path proves the label is real:** `reports/pass2_dashboard_neg.log` — with
  `ada6000_bench.json` deleted, the dashboard exits 0 and renders the row `SKIPPED | n/a | n/a` +
  "reports/ada6000_bench.json missing - no pod gate run, no sm_89 numbers on disk". Never a fake
  MEASURED row.
- The honest <1.0x fused result is surfaced with an explicit NOTE (fusion slower at this
  compute-bound shape), not hidden or massaged.

## 4. Cloud spend — ✅ HOLDS with ONE documented OWNER-OVERRIDE (recorded, not a violation)

- **The intended deviation.** The plan's "no new cloud spend" Must-NOT-Have was written when the
  existing pod (`root@65.109.75.15:22`, the plan's SSH target) was assumed usable. That instance was
  **terminated by the user**; `reports/pod_env.log` (13:27Z) shows `ssh to root@65.109.75.15:22 FAILED
  (rc=255)` — the old instance is gone. The user then explicitly authorized provisioning a replacement
  ("you'll need to create a new one"). A NEW pod was created **at that explicit authorization**:
  - **pod `76f2feccd9bc448ba8c33a5195b2f15c`**, name `pk-pass2-ada`, RTX6000Ada_48GB x1,
    IP **216.81.200.13**, **$0.75/hr**, offer **03c728**, created 2026-08-15 14:37:09 UTC.
  - This is an **OWNER-OVERRIDE of the plan's phrasing, not a pass violation** — the pass itself
    executed no provisioning: no `prime pods create`, no `prime pods terminate`, no RunPod, no Modal.
- Verified: `grep -rlE "prime pods (create|terminate)"` over the repo → **no matches** in pass-2
  files (only the pre-existing `projects/tensors` convention scripts, outside this repo's scope);
  `scripts/pod_env.sh`/`sync_pod.sh` state in-header "direct SSH (no provisioning, no `prime pods`)"
  and contain no `prime` CLI calls; `git diff cba03af -- benchmarks/` = only `run_bench_suite.py`
  ada6000 additions (23 lines; `rent_runpod.sh`/`bench_cuda.cpp` RunPod/modal references are
  wave-8 originals, untouched).
- **The pod is LEFT RUNNING, exactly as planned:** `prime pods status 76f2feccd9bc448ba8c33a5195b2f15c`
  → **Status: ACTIVE**, Installation Status: FINISHED, GPU RTX6000Ada_48GB (read-only status query;
  no termination or restart performed). All pass-2 pod evidence (pod_env_v2.log, pass2_pod_sync.log,
  todos 9/11/14/15 logs) targets the SAME instance: `ubuntu@216.81.200.13`.

## 5. Toolset versions STAY PUT — ✅ HOLDS

- `git diff cba03af -- flake.nix devenv.nix` → flake.nix: **only** the devenv pin
  `v2.1.2 → v2.2.2` (+ the comment); **devenv.nix: zero diff**. flake.lock: devenv node
  update only (cachix/crate2nix input restructure, internal to devenv).
- Toolset pins byte-identical: `devenv.nix` still `llvmPkgs = pkgs.llvmPackages_21` (L9),
  `cudaPackages = pkgs.cudaPackages_12_6` (L12), `rocmPackages = pkgs.rocmPackages` (L17) +
  `gfx1101` (L16), `gfx942` unchanged. `devenv.yaml` gained only the flat `nixpkgs:` block
  (`cuda_support`/`cuda_capabilities` — config, not a version change).

## 6. `matmul.cu` (scalar anchor) unchanged — ✅ HOLDS

- `git diff HEAD~8 -- kernels/generated/matmul.cu` → **empty** (the task's exact command).
- Stronger: `git diff cba03af -- kernels/generated/matmul.cu` → **empty** (whole-pass span).

## 7. Dataflow sim untouched — ✅ HOLDS

- Pass file list (`git diff cba03af --name-only`, 79 files) contains **no** `lib/DataflowSim/`,
  `LowerToDataflow.cpp`, `lib/Simulation/` or any dataflow-sim file. The pass touched analyzer
  (sm_89), runtime (cuda launcher), kernels (gelu fix + new mma), generators, tools, docs, reports.

## 8. Stale `.codegraph/` symlink untouched — ✅ HOLDS

- `.codegraph -> /home/mei/.omo/codegraph/projects/polykernel-87712ad18a7d5eb7` — still a symlink,
  dated Jul 21 (predates the pass), absent from every pass diff/status. Untouched.

## 9. No leaked keys — ✅ HOLDS

- `git log --all --oneline -- private_key.pem public_key.pem` → **empty** (no commit ever touched them).
- `git check-ignore private_key.pem public_key.pem` → exit 0 for BOTH (`*.pem` in .gitignore since
  `f8c6305`); `git ls-files | grep -c '\.pem$'` → **0** tracked pem files.
- `git grep -iE "PRIVATE KEY|BEGIN RSA|BEGIN OPENSSH"` → only benign prose in pass-2 log files
  ("private key present, mode 600", "private key NOT synced"); zero key material. No ssh public-key
  material in any code file (0 matches).
- `reports/pass2_pod_sync.log` confirms the sync excludes `*.pem` — the private key is NOT on the pod.

---

## Summary table

| # | Must-NOT-Have | Result |
|---|---|---|
| 1 | Op set closed (no new dialect ops; MMA = codegen variant) | ✅ HOLDS (td diff 0; op_set_closed PASS; 18 ops) |
| 2 | No SOTA/vLLM claim | ✅ HOLDS (14 grep hits, all negated assertions) |
| 3 | H100/A100/MI300 PROJECTED; honest MEASURED labels | ✅ HOLDS (MEASURED only for sm_89; neg-path proof) |
| 4 | No new cloud spend | ✅ HOLDS w/ 1 OWNER-OVERRIDE (user-authorized new pod, recorded) |
| 5 | Toolset versions stay put | ✅ HOLDS (only devenv v2.2.2 bump) |
| 6 | `matmul.cu` scalar anchor unchanged | ✅ HOLDS (diff empty, HEAD~8 and pass-base) |
| 7 | Dataflow sim untouched | ✅ HOLDS (no sim file in pass diff) |
| 8 | Stale `.codegraph/` symlink untouched | ✅ HOLDS (untouched, Jul 21) |
| 9 | No leaked keys | ✅ HOLDS (empty pem log; gitignored; 0 tracked) |

**Pass-level note (not a violation, per inherited wisdom):** the first on-NVIDIA run exposed two real
CUDA bugs (bf16x2 store corruption on sm_89; fp32 erff saturation of the GELU (1+erf) tail) — FIXED in
`d85866a` and validated on the Ada (18/18 golden) + HIP (49/49). This is exactly the pass's
pedagogical-correctness goal, not a scope violation.

**Pod instance state:** left RUNNING (`prime pods status` = ACTIVE). No termination script exists or
ran; no restart performed. The user retains control of termination.

---

VERDICT: APPROVE
