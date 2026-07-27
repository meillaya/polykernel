# PolyKernel HIP / ROCm Backend — Prereqs & Local-Run Gate

> Todo 18 (Wave 4): the **local ROCm verification gate**. This is a **GATE** — it
> decides whether the later HIP-run todos (Todo 20) run **locally** on the gfx1101
> GPU or **degrade** to "compile-only + rental-validated". It detects and reports
> only; it never modifies system state and never hard-fails the project.
>
> Local target: **AMD Radeon RX 7800 XT (Navi 32) = `gfx1101`**, officially
> supported since **ROCm 7.0**. Local toolchain: `rocmPackages.clr` = **ROCm 7.2.3**.

## Status

| Check | Result (local host) |
|---|---|
| `rocmPackages` / ROCm version `>= 7.0` | ✅ 7.2.3 |
| `rocminfo` detects a `gfx1101` agent | ✅ AMD Radeon RX 7800 XT |
| user in `video` **and** `render` groups | ⚠️ `video` yes, `render` no (advisory — GPU world-accessible) |
| `amdgpu` kernel driver loaded | ✅ `/sys/module/amdgpu` |
| **Gate verdict** | ✅ **PASS** (gfx1101 enumerated natively; no override needed) |

Evidence: `reports/w4_rocm_check.log` (happy → PASS), `reports/w4_rocm_neg.log`
(bogus `HSA_OVERRIDE_GFX_VERSION=99.0.0` → SKIPPED-local + exact remedy).

## Prerequisites (local HIP run on gfx1101)

### 1. ROCm `>= 7.0`

`gfx1101` (RDNA3, RX 7700 XT / 7800 XT) is **officially supported since ROCm 7.0**.
The gate asserts the `rocmPackages` major version is `>= 7`; on `< 7.0` it exports
the `HSA_OVERRIDE_GFX_VERSION=11.0.1` fallback (below) and warns. The local
`rocmPackages.clr` is **7.2.3**, so the assert passes without the override.

### 2. NixOS prereqs

For the GPU to be enumerable by `rocminfo` / usable by HIP on NixOS:

- `nixpkgs.config.rocmSupport = true` — enables the ROCm package set. In this repo
  it is set where nixpkgs is instantiated (`flake.nix`:
  `pkgs = import nixpkgs { config = { allowUnfree = true; cudaSupport = true;
  rocmSupport = true; ... }; }`), because the devenv module system has no
  `nixpkgs.config` option (see `devenv.nix`).
- `hardware.amdgpu.opencl.enable = true` — installs the OpenCL/ROCm runtime + ICD so
  the HSA runtime can see the device.
- the **`amdgpu` driver loaded** — `lsmod | grep amdgpu` or `/sys/module/amdgpu`.
  (NixOS: `boot.initrd.kernelModules += [ "amdgpu" ]` if not auto-loaded.)
- the **user in the `video` and `render` groups** — grants access to `/dev/kfd` and
  `/dev/dri/renderD*`:
  ```bash
  sudo usermod -aG video,render $USER   # then re-login
  ```
  The gate **detects and reports** group membership; it does **not** run `usermod`.
  If the GPU is nonetheless enumerable (e.g. world-writable `/dev/kfd`,
  `crw-rw-rw- root render`), a missing `render` group is downgraded to an advisory
  `[WARN]` and the gate still passes — the GPU is empirically accessible.

### 3. `rocmPackages.clr` auto-set env vars

Entering the dev shell (`nix develop --impure --accept-flake-config`) puts
`hipcc`/`rocminfo` on `PATH` and auto-sets the HIP/ROCm environment from
`rocmPackages.clr`, so no manual paths are needed:

| Var | Points at |
|---|---|
| `HIP_PATH` | the `clr` store path (HIP runtime + `hipcc`) |
| `ROCM_PATH` | the ROCm root |
| `HIP_CLANG_PATH` | the HIP clang (`hipClang`) used to compile device code |
| `DEVICE_LIB_PATH` / `HIP_DEVICE_LIB_PATH` | the `rocm-device-libs` amdgcn bitcode |
| `HSA_PATH` | the `rocm-runtime` (HSA) store path |

## `HSA_OVERRIDE_GFX_VERSION` fallback rationale

Some HSA runtimes do not recognize `gfx1101` by name. The fallback

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.1   # or 11.0.0 to emulate gfx1100
```

tells the runtime to treat the device as a known ISA target. This is **safe** for the
RX 7800 XT because **`gfx1101` is ISA-compatible with `gfx1100`** — same RDNA3 ISA and
the same register file (**768 VGPRs / 32 KiB**, **768 SGPRs / 32 KiB**), so code
generated for `gfx1100` runs correctly on `gfx1101`. `11.0.0` emulates the
widely-recognized `gfx1100` target; `11.0.1` names `gfx1101` directly.

The gate applies this fallback **automatically** when `gfx1101` is not enumerated and
the caller did **not** pre-set `HSA_OVERRIDE_GFX_VERSION`, then re-checks
(→ `PASS-with-HSA_OVERRIDE`). If the caller **did** pre-set it, the gate **honors** the
caller value (so a bogus value such as `99.0.0` is diagnosed, not silently masked).

## The gate: `scripts/check_rocm.sh`

```bash
nix develop --impure --accept-flake-config -c ./scripts/check_rocm.sh
```

Checks (each a greppable `[PASS]`/`[FAIL]`/`[WARN]`/`[INFO]` line):

1. `rocmPackages` / ROCm version `>= 7.0` (else export `HSA_OVERRIDE_GFX_VERSION=11.0.1` + warn);
2. `rocminfo` detects a `gfx1101` GPU agent;
3. user is in the `video` **and** `render` groups;
4. the `amdgpu` driver is loaded.

Verdicts (final `GATE VERDICT:` line):

- **`PASS`** — `gfx1101` enumerated natively, ROCm `>= 7.0`. HIP-run todos run locally.
- **`PASS-with-HSA_OVERRIDE`** — `gfx1101` enumerated only after the `11.0.1`
  fallback. HIP-run todos run locally (export the override first).
- **`SKIPPED-local`** (compile-only + rental-validated) — `gfx1101` not enumerated
  even after the fallback. The gate names the **exact missing prereq** + **remedy**
  and HIP-run todos degrade to compile/analyze locally + validate on a gfx1101 rental.

The gate is **robust and non-fatal**: `set -euo pipefail` with every check guarded, so
a failed check records a verdict instead of aborting; it always `exit 0` and never
crashes the calling shell or blocks the rest of Wave 4. It only **detects and reports**
— it never runs `usermod`, never loads drivers, never modifies system state.

### Negative test (proves the gate diagnoses failure)

```bash
HSA_OVERRIDE_GFX_VERSION=99.0.0 nix develop --impure --accept-flake-config -c \
    env HSA_OVERRIDE_GFX_VERSION=99.0.0 ./scripts/check_rocm.sh
```

Under the bogus `99.0.0` target, `rocminfo` returns `HSA_STATUS_ERROR_OUT_OF_RESOURCES`
(exit 8) and the `gfx1101` agent disappears; the gate reports `SKIPPED-local` with the
exact remedy (unset the override, or use `11.0.1`/`11.0.0`). Evidence:
`reports/w4_rocm_neg.log`.

---

## Todo 19 (Wave 4): `--lower-to-hip` + hipcc driver — the portability proof

`--lower-to-hip` is the HIP sibling of `--lower-to-cuda`. It is the **same custom source
emitter** and emits **byte-identical** portable compute against
`kernels/template/kernel_common.h`; the two passes differ only in which driver compiles
the output. The `POLYKERNEL_HIP` define is applied by the **hipcc driver**
(`tools/polykernel-bench/hipcc_driver.py`), not the emitter:

```bash
hipcc --offload-arch=gfx1101 -DPOLYKERNEL_HIP kernels/generated/matmul.cu
```

hipcc compiles **every** generated kernel (rmsnorm, gelu, silu, matmul, softmax,
fused_rmsnorm_matmul, fused_matmul_bias_gelu) for gfx1101 **with no source edits** to the
template — that is the portability proof (`reports/w4_hipcc_compile.log`). The negative
(`reports/w4_hipcc_portability_neg.log`) leaves a CUDA-only intrinsic unguarded and hipcc
errors, proving the `#ifdef` discipline is enforced.

## Todo 20 (Wave 4): HIP runtime launches + golden correctness on the RX 7800 XT

`lib/Runtime/HipRuntime.cpp` launches the kernels (`hipMalloc` / `hipMemcpy` /
`hipLaunchKernel` / `hipDeviceSynchronize`). This is **where the shared portable
template's runtime correctness is validated for BOTH backends**: the HIP kernels run on the
local 7800 XT (or the `HSA_OVERRIDE_GFX_VERSION=11.0.1` fallback) and **all pass** the
golden (`assert_correct`, contract C) — rmsnorm, gelu, matmul, softmax + fused, **0 failed
correctness** (`reports/w4_hip_correctness.log`). Because the CUDA backend shares this
compute logic, this validates the algorithmic core for CUDA too. The negative
(`reports/w4_hip_launch_neg.log`) launches an out-of-bounds grid and the runtime catches +
reports the HIP error code (no silent corruption).

## Todo 21 (Wave 4): AMDGPU ISA analyzer

`lib/Analysis/AmdIsaAnalyzer.cpp` (+ `tools/polykernel-bench/amd_analyze.py`) parses the
AMDGPU ISA from `hipcc --save-temps` (`.s`) / `llvm-objdump --triple=amdgcn-amd-amdhsa
--mcpu=gfx1101 -d`:

- `.amdhsa_next_free_vgpr` / `.amdhsa_next_free_sgpr` — VGPR/SGPR usage;
- `.amdhsa_group_segment_fixed_size` — LDS (shared memory);
- `.amdhsa_private_segment_fixed_size` + `scratch_store_*`/`scratch_load_*` count — spills;
- gfx1101 occupancy = `min(VGPR-limited, SGPR-limited, LDS-limited, wave-limited)` from the
  per-CU file sizes (wave32; ~768 VGPRs / 32 KiB SGPR / 128 KiB LDS).

`polykernel-bench --backend=hip --mode=analyze --arch gfx1101` emits the AMD per-kernel
report (VGPR/SGPR/LDS/spills/occupancy/bottleneck); results live in a **separate AMD tuning
DB namespace** (`lib/Autotune/AmdTuningDb.cpp`). The spill fixture
(`reports/w4_amd_spill.log`) has `scratch_store`/`private_segment_fixed_size > 0` and the
analyzer flags a register-pressure/spill bottleneck.

## Todo 22 (Wave 4): WMMA bf16 tensor-core path (RDNA3)

gfx1101 has `v_wmma_f32_16x16x16_bf16`. The WMMA matmul variant
(`kernels/generated/matmul_wmma.cu`) uses the wave32 builtin
`__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` (16×16×16, 16 elems/lane input, 8×f32/lane
accum). It is **additive**: the scalar/tiled matmul (Todo 10) remains the correctness
baseline, and WMMA is a variant the autotuner can select. WMMA QA is **numerical** — a
16×16×16 bf16 matmul through the WMMA path vs the golden (`reports/w4_wmma.log`). A WMMA
fragment with a wrong lane-mapping fails the golden and the autotuner discards it
(`validated=false`), leaving the scalar baseline intact (`reports/w4_wmma_neg.log`) — WMMA
is safely additive, never blocking.

## Todo 23 (Wave 4): gfx942 (MI300) cross-compile + AMD tuning DB

MI300 = **gfx942** (CDNA3), the AMD headline arch for the benchmark reports. It is
**compile-only locally** (no MI300 hardware); runtime is on rental (Wave 5). hipcc compiles
all kernels for gfx942 + the AMDGPU ISA is captured for the report
(`reports/mi300_compile.log`); attempting to *run* a gfx942 binary locally reports "no
gfx942 device, compile-only" gracefully (`reports/w4_gfx942_norun.log`). The AMD tuning DB
stores per-(gpu, op, shape) best configs in **separate** gfx1101 (local) and gfx942
(rental) namespaces.

## Cross-reference: the same matmul on the dataflow fabric

The matmul the HIP backend computes (and validates on the 7800 XT) is the **same** matmul
the Cerebras-style dataflow simulator maps onto its PE grid via the **SUMMA** schedule
(A broadcast east, B broadcast south, output-stationary C in local SRAM). The simulator is
a separate backend — see [`dataflow_backend.md`](dataflow_backend.md) — but sharing the one
matmul definition (and the one golden) is what lets the dataflow path be
correctness-validated independently of any GPU.

## See also

- [`cuda_backend.md`](cuda_backend.md) — the CUDA sibling sharing this template.
- [`dataflow_backend.md`](dataflow_backend.md) — the SUMMA dataflow mapping.
- [`performance_model.md`](performance_model.md) — the projected MI300 (gfx942) speedups.
