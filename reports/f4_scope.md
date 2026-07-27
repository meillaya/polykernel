# F4 — SCOPE-FIDELITY AUDIT (Must-NOT-Have Guardrails)

**Auditor:** F4 Scope-Fidelity Auditor (final verification wave)
**Subject:** PolyKernel — out-of-tree C++/MLIR compiler + correctness-gated autotuner + Cerebras-style dataflow **simulator** + Modal deployment
**Date:** 2026-07-27
**Method:** Read/Grep/Glob + `git log`/`git for-each-ref`/`git merge-base`. Whole-repo case-insensitive greps (excluding `.git`, `build*`, `.cache`, `.devenv`, `.codegraph`, `.pytest_cache` binary/generated noise), with every hit read in context. codegraph NOT used (0 files indexed).
**Scope of this audit:** Must-NOT-Have (scope fidelity) ONLY. No product code was modified; this file is the only artifact written. No git commit/tag/push; no plan checkboxes touched.

---

## Summary

Every Must-NOT-Have guardrail **HOLDS**. The op set is closed at exactly the named transformer-inference ops + fused variants; the dataflow simulator uses correct Cerebras terminology (CE+FMAC, `@set_color_config`, `fabin_dsd`/`fabout_dsd`, wavelet/color routing, SUMMA) and is explicitly a functional/cycle **model**, never claimed to be CSL/`cslc`/hardware; there is no SOTA/vLLM-beating claim; no full PyTorch/ONNX/StableHLO frontend; no cloud spend without credentials (Modal deploy + RunPod bench both SKIPPED); no tenstorrent/tt-metal reuse; Wave 8 elite commits are descendants of v1.0-mvp; the upstream PR (T44) is prepared-not-submitted and non-blocking; and README/docs claims are accurate (simulator = simulator, projections labeled PROJECTED).

**Transparency note (read before the checklist):** a literal grep for the forbidden Cerebras strings (`FMU`/`PMU`/`fdata`/`fcast`/`fmove`/`@compute`/`@data`) does return hits in this repo. **Every single hit is a negated guardrail assertion** of the form "we do **NOT** use FMU/PMU; we use CE+FMAC" — i.e. the enforcement mechanism of the very rule being audited. There is **zero affirmative use** of any forbidden term as actual terminology. Per the audit charter's own guidance ("Do NOT report a false-positive violation from a benign substring … read context"; SOTA may appear "only in a 'we do NOT claim this' context"), these are classified **COMPLIANT**, not violations. Full per-hit evidence is in §3.

---

## Checklist (each Must-NOT-Have with evidence)

### 1. NO op zoo beyond the named ops — ✅ HOLDS

The dialect is defined in `include/PolyKernel/IR/PolyKernelOps.td`. `grep 'def PolyKernel_'` over `include/` returns **exactly 18 op definitions** (plus the single `def PolyKernel_Dialect` in `PolyKernelDialect.td:23`), all in `PolyKernelOps.td`:

**Function container (2 — explicitly NOT compute ops):**
- `func` (line 104), `return` (line 172) — modeled on `func.func`/`func.return`; the header (lines 32–35) states they "are NOT counted among the 'compute' ops, so the compute op-zoo guardrail still holds."

**Named base transformer-inference ops (10):**
- `rmsnorm` (194), `matmul` (243), `bias` (252), `gelu` (206), `silu` (213), `add` (220), `softmax` (227), `rope` (265), `attention` (285), `kv_cache_update` (297).

**Fused variants (6):**
- `fused_rmsnorm_matmul` (312), `fused_matmul_bias_gelu` (324), `fused_residual_rmsnorm` (336), `fused_softmax_mask` (348), `qkv_projection` (358), `fused_kv_append_attention` (377).

That is exactly the named transformer-inference op set + fused variants — **16 compute ops + func/return = 18 total**, matching the dashboard's `dialect ops=18` (`.omo/start-work/ledger.jsonl` T34). No quantize/dequantize or other extra ops were added to the dialect (elite quantization is implemented as kernels + golden, not new dialect ops; `grep` for `quantize|dequantize|def PolyKernel_` in the IR sources finds no extra op defs).

**Guardrail is self-documented** in the file header (`PolyKernelOps.td:7–8`): *"This file defines EXACTLY the PolyKernel transformer-inference op set and nothing else (op-zoo guardrail)"*, and line 35: *"No other ops are defined."*

### 2. Dataflow simulator: CORRECT Cerebras terminology + NOT claimed to be real CSL/`cslc`/hardware — ✅ HOLDS

**Correct terminology used throughout (affirmative):**
- `docs/dataflow_backend.md:11–16`: "a PE's compute unit is a **CE** with **FMACs** … routing color is configured with **`@set_color_config`** rx/tx … the fabric move DSDs are **`fabin_dsd`** / **`fabout_dsd`** … compute is triggered by data arrival via **`@activate`/`@block`/`@unblock`** rendezvous."
- SUMMA mapping documented: `docs/dataflow_backend.md:66–85` ("map `polykernel.matmul` … onto the **SUMMA** schedule … A tile EAST … B tile SOUTH … output-stationary C"); implemented in `lib/DataflowSim/Summa.cpp` / `LowerToDataflow.cpp`.
- Code headers: `include/PolyKernel/Dataflow/Pe.h:11` (CE with FMACs), `Summa.h:29–31`, `Metrics.h:29`, `Color.h:20,25`.
- Viz: `tools/polykernel-report/dataflow_viz.py:25–26` ("CE + FMAC; fabin_dsd/fabout_dsd; @set_color_config rx/tx"); `lib/DataflowSim/viz_template.html` (CE/wavelet/color/SRAM/SUMMA terms present).

**Explicitly a simulator / model, NOT CSL/`cslc`/hardware:**
- `docs/dataflow_backend.md:3–9`: "a self-contained C++ **functional / cycle SIMULATOR** … It is **NOT** real CSL, **NOT** a `cslc` compiler, and **NOT** Cerebras hardware. It does not link against, depend on, or shell out to the Cerebras SDK or simfabric. … The real-SDK analogs **modelled** (not integrated) are `cslc --out-routes` and `calculate_cycles`."
- `README.md:49` ("It is not" column): "Real CSL / `cslc` / **Cerebras hardware**".
- `docs/architecture.md:142–143`: "It is **not** real CSL and **not** Cerebras hardware."
- `tools/polykernel-report/dataflow_viz.py:25`: "THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras hardware)."
- `lib/DataflowSim/viz_template.html:21–22,133–134`: "It is NOT real CSL, NOT a cslc compiler, and NOT Cerebras hardware."
- `reports/dataflow_metrics.json:4–8`: `"simulator": "PolyKernel dataflow simulator (functional/cycle model; NOT real CSL, NOT Cerebras hardware)"`, with `"sdk_analogs": ["cslc --out-routes", "calculate_cycles"]` — i.e. `cslc` appears only as a **modelled analog**, never as something the simulator IS or runs.

**`cslc` context check:** every `cslc` occurrence (`docs/dataflow_backend.md:9`, `docs/architecture.md:143`, `dataflow_report.py:215`, `dataflow_metrics.json:6`, `viz_template.html:22,134`, `Metrics.h:31`) is in a "modelled analog / NOT cslc" disclaimer context. **No positive claim** of being a `cslc` compiler or running on Cerebras hardware exists anywhere.

### 3. FORBIDDEN terms ABSENT (FMU/PMU/fdata/fcast/fmove/@compute/@data) — ✅ HOLDS (zero affirmative use)

Whole-repo case-insensitive grep results, classified by context:

| Forbidden term | Affirmative use? | Where the literal string appears (all negated guardrails) |
|---|---|---|
| `FMU` / `PMU` | **NONE** | `include/PolyKernel/Dataflow/Color.h:20` (`no "FMU"/"PMU" here`), `Pe.h:11` (`there is no "FMU"/"PMU"`), `Summa.h:29–30` (`no FMU/PMU`), `Metrics.h:29` (`CE + FMAC (no FMU/PMU)`), `lib/DataflowSim/tests/summa_test.cpp:18`, `tools/polykernel-report/dataflow_report.py:25`, `docs/dataflow_backend.md:12` (`not "FMU"/"PMU"`), `reports/w7_summa.log:19`, plus `.omo/` planning docs — **all "NOT FMU/PMU"**. |
| `fdata` / `fcast` / `fmove` | **NONE** | `include/PolyKernel/Dataflow/Color.h:25` (`NOT "fdata"/"fcast"/"fmove"`), `docs/dataflow_backend.md:14` (`not "fdata"/"fcast"/"fmove"`), plus `.omo/` planning — **all negated**. |
| `@compute` / `@data` (CSL decorators) | **NONE** | Precise grep `@data([^a-zA-Z]|$)|@compute([^a-zA-Z]|$)` returns only negated guardrails: `include/PolyKernel/Dataflow/Task.h:26` (`there are NO @compute / @data decorators in CSL`), `Summa.h:31`, `lib/DataflowSim/tests/scheduler_test.cpp:19`, `summa_test.cpp:19`, `docs/dataflow_backend.md:15`, `reports/w7_task_model.log:11`, `w7_summa.log:20–21`, plus `.omo/`. **All state the decorators do NOT exist.** |

**False-positive exclusion (per charter):** the broad `@data` substring match also hit Python `@dataclass(frozen=True)` decorators in `tools/polykernel-bench/{nvcc_driver.py,bench.py}`, `tests/e2e/test_mlp_correctness.py`, `benchmarks/run_bench_suite.py`, `modal/{app.py,deploy.py,cold_start.py}`. These are the standard-library `@dataclass` decorator, **not** the CSL `@data` decorator (the precise `@data\b`-style grep excludes them). Likewise "dataflow" contains "data" but is not "@data". These are **not violations**.

**Conclusion:** the forbidden literals appear **only** as explicit "we do NOT use this incorrect term" guardrail comments/docs (the enforcement of this rule). The simulator's actual terminology — code identifiers, doc prose, viz labels, JSON fields — is uniformly CE/FMAC/`@set_color_config`/`fabin_dsd`/`fabout_dsd`/`@activate`/`@block`/`@unblock`/SUMMA. **No affirmative forbidden-term usage exists.** ✅

### 4. NO claim of beating vLLM/SOTA performance — ✅ HOLDS

Grep `beats vllm|outperform|state.of.the.art|\bsota\b|faster than vllm|beat vllm` (case-insensitive). **Every hit is negated:**
- `README.md:48` ("It is not" column): "A claim to **beat vLLM** or any production stack".
- `docs/architecture.md:26–27`: "**Is not:** a claim to beat vLLM … not SOTA".
- `docs/performance_model.md:9–10`: "**No SOTA-performance claim is made** — the goal is the compiler/runtime machinery, not beating vLLM."
- `docs/upstream_pr.md:52–53,100–101`: "PolyKernel explicitly does not claim to compete with it [vLLM]"; "No performance claim is made against vLLM … The tutorial's only claim is correctness versus a PyTorch reference."
- Dashboard generator emits the disclaimer: `tools/polykernel-report/dashboard.py:294,597` ("No SOTA-performance claim"); rendered into `reports/benchmark_report.{md,html}`, `h100_report.html`, `mi300_report.html` ("No SOTA-performance claim").
- `kernels/generated/attention_prefill.cu:20`: "no SOTA claim".
- `.omo/` planning docs: "Will NOT beat vLLM … not SOTA".

**No affirmative SOTA/vLLM-beating claim exists.** ✅

### 5. NO full PyTorch/ONNX/StableHLO frontend — ✅ HOLDS

- Input is **hand-written `.mlir`**: `examples/` contains exactly `mlp_block.mlir`, `rmsnorm_matmul.mlir`, `attention_prefill.mlir` (hand-authored `polykernel`-dialect IR).
- `grep -liE 'onnx|stablehlo|torch.load|import_onnx|frontend'` over `lib/ include/ tools/` → **NONE**. There is no importer/frontend code.
- `README.md:51` ("It is not"): "A full PyTorch/ONNX/StableHLO frontend".
- `docs/architecture.md:28`: "A full PyTorch/ONNX [frontend is not built]".
- Remaining `pytorch` mentions are benign: (a) negated README/architecture statements; (b) `docs/upstream_pr.md` describing the **Triton tutorial** validating against a PyTorch reference (that is the upstream Triton example's reference, not a PolyKernel frontend); (c) `.omo/` planning notes about the golden (contract C pins the golden to NumPy + `ml_dtypes` ONLY — PyTorch NOT used).

**No full frontend exists.** ✅

### 6. NO cloud spend without credentials — ✅ HOLDS

**Modal deploy** (`modal/deploy.py`): owner-gated on `MODAL_TOKEN`.
- `has_token()` (line 97–99) checks `os.environ.get("MODAL_TOKEN")`.
- `main()` (line 306–308): `if not has_token(): print(write_skipped()); return 0  # SKIPPED, never FAILED, never blocks, no spend`.
- Skip log text (lines 113–135): "STATUS: SKIPPED (not FAILED). Missing credential: MODAL_TOKEN … ACTION TAKEN: graceful skip. NO `modal deploy`, NO Modal API call, NO image build … VERDICT: SKIPPED — graceful degradation, no crash, no surprise spend." Writes `reports/w6_deploy_skipped.log`.

**RunPod bench** (`benchmarks/run_bench_suite.py`): owner-gated on `RUNPOD_API_KEY`.
- Credential detection (lines 297–300): empty ⇒ SKIPPED path.
- Lines 381–389: no credentials → projections + skip log, **exit 0 (not FAILED)**; "SKIPPED: RUNPOD_API_KEY not set — opt-in rental documented (no spend) … no pod provisioned".
- Skip log (lines 326–355): "STATUS: SKIPPED (not FAILED) … NO pod provisioned, NO RunPod API create/start … VERDICT: SKIPPED — graceful degradation, no crash, no surprise spend." Writes `reports/w5_rent_skipped.log`.
- Projections carry `status: "PROJECTED"`, `projected: True`, `measured: false` (lines 187–190).

`docs/performance_model.md:4`: "No GPU was rented; no pod was provisioned; no money was spent." **No cloud spend occurs without credentials.** ✅

### 7. NO tenstorrent/tt-metal reuse — ✅ HOLDS

Grep `tenstorrent|tt-metal|tt_metal` (case-insensitive). **All hits are in `.omo/` planning docs, in negated context:**
- `.omo/plans/polykernel.md:42`: "Will NOT reuse tenstorrent/tt-metal code — the dataflow sim is Cerebras-CSL-flavored, distinct from the separate `tensixflow` project."
- `.omo/drafts/polykernel.md:128–129`: "No reproduction of the full Cerebras/Tenstorrent stack"; "No tenstorrent/tt-metal code reuse".
- `.omo/drafts/polykernel.md:42`: "tensixflow's TT-Metalium sim was never built" (i.e. built fresh, not reused).

**Zero hits in product code** (`lib/`, `include/`, `tools/`, `modal/`, `kernels/`, `benchmarks/`). The dataflow simulator is built fresh (Cerebras-CSL-flavored), with no tt-metal reuse. ✅

### 8. Elite (Wave 8+) did NOT start before Waves 1–7 were green — ✅ HOLDS

Git evidence (`git for-each-ref refs/tags` + `git merge-base --is-ancestor`):
- `v1.0-mvp` resolves to commit **`6bde128`** (`git rev-parse v1.0-mvp^{commit}` == `git rev-parse 6bde128` → **MATCH**).
- Wave 8 elite commits in `git log --oneline --reverse`: `e98cd1b` (FlashAttention-lite), `f51d403` (quantization), `49517d2` (hardened kernel cache), `5e34d21` (viz polish), `7090822` (upstream PR docs) — all appear **after** `6bde128`.
- `git merge-base --is-ancestor 6bde128 e98cd1b` → **YES** (first elite commit is a descendant of v1.0-mvp).
- `git merge-base --is-ancestor 6bde128 7090822` → **YES** (last elite commit is a descendant of v1.0-mvp).
- Tag ordering: `v1.0-mvp` (3f1e201 tag obj → 6bde128 commit) precedes `v1.x-elite` (→ 7090822).

**Elite (Wave 8+) began strictly after v1.0-mvp (Waves 1–7 green).** ✅

### 9. Upstream PR (T44) is non-blocking — ✅ HOLDS

`docs/upstream_pr.md`:
- Line 3: "**Status: PREPARED, NOT SUBMITTED. This deliverable is optional and never blocking.**"
- Lines 5–8: "PolyKernel's MVP (Waves 1 through 7) is complete and green independent of this file. Nothing here was pushed, forked, or opened as a pull request. This environment has no GitHub credentials and no cloud budget, and the plan marks this task 'NEVER blocking.'"
- Line 20: "Prepared, NOT submitted (no credentials here; non-blocking)".
- Carries the scope guardrails forward (lines 97–104): simulator is a model not CSL/`cslc`/hardware; no vLLM/SOTA claim; projected numbers labeled PROJECTED; "This file makes no speedup claim at all."

**T44 is prepared-not-submitted and explicitly non-blocking.** ✅

### 10. README/docs claims accurate (simulator = simulator; projections labeled PROJECTED; no SOTA) — ✅ HOLDS

- **Simulator = simulator:** `README.md:49` + "What it is — and is not" table, `docs/dataflow_backend.md:3–9`, `docs/architecture.md:142–143`, viz files — all state functional/cycle simulator, NOT hardware/CSL/`cslc` (§2).
- **Projected numbers labeled PROJECTED:**
  - `README.md:110`: "The H100/A100/MI300 speedups are **PROJECTED** (roofline + analytic traffic) unless a RunPod rental ran."
  - `docs/performance_model.md:3–10,44,56,88`: "**PROJECTED — NOT MEASURED**"; speedup tables headed "The projected speedups" / "Projected fragment wall time"; caveats "PROJECTED, not measured".
  - `docs/architecture.md:165`, `docs/cuda_backend.md:7,125`, `docs/hip_backend.md:213`, `docs/upstream_pr.md:103` — all label projections.
  - HTML reports (`benchmark_report.html`, `h100_report.html`, `mi300_report.html`) carry the disclaimer banner: "H100/A100/MI300 speedups are PROJECTED (roofline + analytic traffic, not measured on rented hardware)."
  - Bench JSONs carry `status: "PROJECTED"`, `projected: true`, `measured: false`.
- **No SOTA claim:** §4.

**README/docs claims are accurate and consistent with the implementation.** ✅

---

## Violations found

**None.** No Must-NOT-Have guardrail is violated. The only literal-string matches for forbidden terms are negated guardrail assertions (documented in §3), which are the enforcement of the rule, not violations of it.

---

VERDICT: APPROVE
