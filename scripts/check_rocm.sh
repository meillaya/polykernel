#!/usr/bin/env bash
#
# scripts/check_rocm.sh — PolyKernel Todo 18 (Wave 4): LOCAL ROCm VERIFICATION GATE.
#
# GATE (not a build step): decides whether later HIP-run todos (T20) run LOCALLY
# on the gfx1101 GPU (RX 7800 XT) or DEGRADE to "compile-only + rental-validated".
# It DETECTS and REPORTS only — it NEVER modifies system state (no usermod, no
# modprobe) and NEVER hard-fails the project: a missing GPU/prereq degrades to a
# SKIPPED-local verdict naming the EXACT missing prereq + remedy.
#
# Checks (each emits a greppable [PASS]/[FAIL]/[WARN]/[INFO] line):
#   1. rocmPackages / ROCm version  — assert >= 7.0 (gfx1101 official since 7.0);
#                                     if < 7.0, export HSA_OVERRIDE_GFX_VERSION=11.0.1 + warn.
#   2. rocminfo detects a gfx1101 GPU agent (the RX 7800 XT).
#   3. current user is in the `video` AND `render` groups.
#   4. the `amdgpu` kernel driver is loaded.
#
# Fallback: if gfx1101 is NOT enumerated and the caller did NOT pre-set
# HSA_OVERRIDE_GFX_VERSION, the gate sets HSA_OVERRIDE_GFX_VERSION=11.0.1 (emulate
# gfx1100 — ISA-compatible with gfx1101: 768/32KiB VGPR/SGPR) and RE-CHECKS.
# If the caller DID pre-set HSA_OVERRIDE_GFX_VERSION, the gate HONORS it (so a bogus
# value such as 99.0.0 is diagnosed, not silently masked) and reports the remedy.
#
# Verdicts: PASS | PASS-with-HSA_OVERRIDE | SKIPPED-local (compile-only + rental-validated).
#
# Usage:  nix develop --impure --accept-flake-config -c ./scripts/check_rocm.sh
# Exit:   always 0 (a GATE degrades; it does not abort the calling shell/pipeline).

set -euo pipefail

# ---------------------------------------------------------------------------
# Report helpers — every line is greppable; the verdict block is machine-readable.
# ---------------------------------------------------------------------------
say()  { printf '%s\n' "$*"; }
info() { say "[INFO] $*"; }
pass() { say "[PASS] $*"; }
warn() { say "[WARN] $*"; }
fail() { say "[FAIL] $*"; }

TARGET_GFX="gfx1101"                 # RX 7800 XT (Navi 32), official since ROCm 7.0
HSA_FALLBACK="11.0.1"                # emulate gfx1100 (ISA-compatible with gfx1101)
ROCM_MIN_MAJOR=7                     # gfx1101 officially supported since ROCm 7.0

# Per-check results (0=ok, 1=problem). Group/driver are advisory when the GPU is
# empirically accessible; they become the named cause only if detection fails.
rc_version=1
rc_detect=1
rc_groups=1
rc_driver=1
used_override=0                      # set if the gate itself applied the fallback
honored_caller_override=0            # set if caller pre-set HSA_OVERRIDE_GFX_VERSION
missing_prereq=""                    # exact missing prereq (for SKIPPED-local)
remedy=""                            # exact remedy text (for SKIPPED-local)

say "##############################################################################"
say "# PolyKernel Todo 18 (Wave 4): LOCAL ROCm VERIFICATION GATE"
say "# Host: $(uname -srm)   Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "# Target GPU agent: ${TARGET_GFX} (AMD Radeon RX 7800 XT, Navi 32)"
say "##############################################################################"
say ""

# ---------------------------------------------------------------------------
# CHECK 1: rocmPackages / ROCm version (assert >= 7.0).
# ---------------------------------------------------------------------------
say "=== CHECK 1: rocmPackages / ROCm version (require >= ${ROCM_MIN_MAJOR}.0) ==="
rocm_version=""
# Preferred source: the nix store path of rocminfo/hipcc encodes clr-<ver> /
# rocm-runtime-<ver>. Fall back to `hipcc --version` (HIP version: X.Y.Z).
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo_bin="$(command -v rocminfo || true)"
  rocm_version="$(printf '%s\n' "$rocminfo_bin" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | tail -1 || true)"
fi
if [[ -z "$rocm_version" ]] && command -v hipcc >/dev/null 2>&1; then
  rocm_version="$(hipcc --version 2>/dev/null | grep -oE 'HIP version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
fi
if [[ -z "$rocm_version" && -n "${HIP_PATH:-}" ]]; then
  rocm_version="$(printf '%s\n' "$HIP_PATH" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | tail -1 || true)"
fi

if [[ -z "$rocm_version" ]]; then
  fail "ROCm version: could not determine rocmPackages version (rocminfo/hipcc/HIP_PATH absent)"
  missing_prereq="rocmPackages (ROCm/HIP toolchain) not on PATH"
  remedy="enter the dev shell: nix develop --impure --accept-flake-config (provides rocmPackages.clr + rocminfo)"
else
  rocm_major="${rocm_version%%.*}"
  if (( rocm_major >= ROCM_MIN_MAJOR )); then
    pass "ROCm version ${rocm_version} >= ${ROCM_MIN_MAJOR}.0 (${TARGET_GFX} officially supported since ROCm 7.0)"
    rc_version=0
  else
    warn "ROCm version ${rocm_version} < ${ROCM_MIN_MAJOR}.0 — ${TARGET_GFX} not officially supported; exporting HSA_OVERRIDE_GFX_VERSION=${HSA_FALLBACK}"
    export HSA_OVERRIDE_GFX_VERSION="${HSA_FALLBACK}"
    used_override=1
    # Version too old is not fatal for detection: the override may still enumerate the GPU.
    rc_version=0
  fi
fi
info "rocminfo: $(command -v rocminfo 2>/dev/null || echo 'NOT ON PATH')"
info "hipcc:    $(command -v hipcc 2>/dev/null || echo 'NOT ON PATH')"
say ""

# ---------------------------------------------------------------------------
# CHECK 2: rocminfo detects a gfx1101 GPU agent.
# Guarded: rocminfo may exit non-zero (e.g. bogus HSA_OVERRIDE) — never abort.
# ---------------------------------------------------------------------------
detect_gfx() {
  # Returns 0 if a ${TARGET_GFX} agent is enumerated, 1 otherwise.
  local out
  out="$(rocminfo 2>&1 || true)"
  if printf '%s\n' "$out" | grep -q "${TARGET_GFX}"; then
    return 0
  fi
  return 1
}

say "=== CHECK 2: rocminfo detects a ${TARGET_GFX} GPU agent ==="
if ! command -v rocminfo >/dev/null 2>&1; then
  fail "rocminfo not on PATH — cannot enumerate GPU agents"
  rc_detect=1
  [[ -z "$missing_prereq" ]] && missing_prereq="rocminfo not on PATH"
  [[ -z "$remedy" ]] && remedy="enter the dev shell: nix develop --impure --accept-flake-config (provides rocmPackages.rocminfo)"
else
  # Honor a caller-provided HSA_OVERRIDE_GFX_VERSION (so a bogus value is diagnosed,
  # not silently masked). Otherwise, on failure, apply the 11.0.1 fallback + re-check.
  if [[ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
    honored_caller_override=1
    info "HSA_OVERRIDE_GFX_VERSION already set by caller = ${HSA_OVERRIDE_GFX_VERSION} (honoring; not auto-overriding)"
  fi

  if detect_gfx; then
    pass "rocminfo detected ${TARGET_GFX} agent (AMD Radeon RX 7800 XT)"
    rc_detect=0
    rocminfo 2>/dev/null | grep -E "Name:|Marketing Name:" | grep -A1 -i "gfx1101" | head -4 | sed 's/^/[INFO]   /' || true
  else
    if (( honored_caller_override )); then
      fail "rocminfo did NOT enumerate a ${TARGET_GFX} agent under caller-set HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
      rc_detect=1
      missing_prereq="no ${TARGET_GFX} GPU agent enumerated by rocminfo (HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} is bogus/unsupported)"
      remedy="unset HSA_OVERRIDE_GFX_VERSION, or set it to a supported ISA target: export HSA_OVERRIDE_GFX_VERSION=11.0.1 (or 11.0.0 to emulate gfx1100)"
    else
      warn "rocminfo did NOT enumerate ${TARGET_GFX}; applying fallback HSA_OVERRIDE_GFX_VERSION=${HSA_FALLBACK} and re-checking"
      export HSA_OVERRIDE_GFX_VERSION="${HSA_FALLBACK}"
      used_override=1
      if detect_gfx; then
        pass "rocminfo detected ${TARGET_GFX} agent AFTER HSA_OVERRIDE_GFX_VERSION=${HSA_FALLBACK} fallback"
        rc_detect=0
      else
        fail "rocminfo still does NOT enumerate a ${TARGET_GFX} agent after HSA_OVERRIDE_GFX_VERSION=${HSA_FALLBACK}"
        rc_detect=1
        missing_prereq="no ${TARGET_GFX} GPU agent enumerated by rocminfo even with HSA_OVERRIDE_GFX_VERSION=${HSA_FALLBACK}"
        remedy="verify an AMD RDNA3 GPU is present (lspci | grep -i amd), the amdgpu driver is loaded, and /dev/kfd + /dev/dri/renderD* are accessible; on a GPU-less host HIP runs degrade to compile-only + rental-validated"
      fi
    fi
  fi
fi
say ""

# ---------------------------------------------------------------------------
# CHECK 3: current user is in the `video` AND `render` groups.
# ---------------------------------------------------------------------------
say "=== CHECK 3: user '${USER:-$(id -un)}' is in the video AND render groups ==="
user_groups="$(id -nG 2>/dev/null || true)"
have_video=0; have_render=0
[[ " ${user_groups} " == *" video "* ]] && have_video=1
[[ " ${user_groups} " == *" render "* ]] && have_render=1
if (( have_video && have_render )); then
  pass "user is in both video and render groups"
  rc_groups=0
else
  # Advisory when the GPU is empirically accessible (e.g. world-writable /dev/kfd);
  # becomes the named cause only if detection (CHECK 2) also failed.
  grp_missing=""
  (( have_video ))  || grp_missing="video"
  (( have_render )) || grp_missing="${grp_missing:+${grp_missing}+}render"
  if (( rc_detect == 0 )); then
    warn "user NOT in group(s): ${grp_missing} — but ${TARGET_GFX} is accessible (GPU enumerated); advisory only"
    warn "  remedy: sudo usermod -aG ${grp_missing//+/,} ${USER:-$(id -un)} && re-login"
    rc_groups=0   # GPU proven accessible -> does not block the gate
  else
    fail "user NOT in group(s): ${grp_missing}"
    fail "  remedy: sudo usermod -aG ${grp_missing//+/,} ${USER:-$(id -un)} && re-login"
    rc_groups=1
    [[ -z "$missing_prereq" ]] && missing_prereq="user not in group(s): ${grp_missing}"
    [[ -z "$remedy" ]] && remedy="sudo usermod -aG ${grp_missing//+/,} ${USER:-$(id -un)} && re-login"
  fi
fi
info "groups: ${user_groups}"
say ""

# ---------------------------------------------------------------------------
# CHECK 4: the `amdgpu` kernel driver is loaded.
# ---------------------------------------------------------------------------
say "=== CHECK 4: amdgpu kernel driver is loaded ==="
if [[ -d /sys/module/amdgpu ]]; then
  pass "amdgpu driver present (/sys/module/amdgpu)"
  rc_driver=0
elif lsmod 2>/dev/null | grep -q '^amdgpu '; then
  pass "amdgpu driver present (lsmod)"
  rc_driver=0
else
  if (( rc_detect == 0 )); then
    warn "amdgpu driver not detected via /sys/module or lsmod — but ${TARGET_GFX} is accessible; advisory only"
    warn "  remedy: sudo modprobe amdgpu  (and ensure boot.kernelModules / hardware.amdgpu.enable on NixOS)"
    rc_driver=0
  else
    fail "amdgpu driver NOT loaded"
    fail "  remedy: sudo modprobe amdgpu  (NixOS: boot.initrd.kernelModules += [ \"amdgpu\" ])"
    rc_driver=1
    [[ -z "$missing_prereq" ]] && missing_prereq="amdgpu kernel driver not loaded"
    [[ -z "$remedy" ]] && remedy="sudo modprobe amdgpu (NixOS: boot.initrd.kernelModules += [ \"amdgpu\" ])"
  fi
fi
say ""

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
say "=== GATE VERDICT ==="
if (( rc_detect == 0 && rc_version == 0 )); then
  if (( used_override )); then
    verdict="PASS-with-HSA_OVERRIDE"
    say "GATE VERDICT: ${verdict}"
    say "  ${TARGET_GFX} enumerated only with HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}."
    say "  HIP-run todos (T20) may run LOCALLY (export HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION} first)."
  else
    verdict="PASS"
    say "GATE VERDICT: ${verdict}"
    say "  ${TARGET_GFX} enumerated natively (ROCm ${rocm_version:-?}); no override needed."
    say "  HIP-run todos (T20) may run LOCALLY."
  fi
  say "  checks: version=${rc_version} detect=${rc_detect} groups=${rc_groups} driver=${rc_driver} (0=ok)"
else
  verdict="SKIPPED-local"
  say "GATE VERDICT: ${verdict} (compile-only + rental-validated)"
  say "  HIP-run todos (T20) DEGRADE: compile/analyze locally, validate on a gfx1101 rental."
  say "  EXACT missing prereq: ${missing_prereq:-unknown}"
  say "  REMEDY:               ${remedy:-see per-check [FAIL] lines above}"
  say "  checks: version=${rc_version} detect=${rc_detect} groups=${rc_groups} driver=${rc_driver} (0=ok)"
fi
say ""
info "evidence: append this output to reports/w4_rocm_check.log (happy) or reports/w4_rocm_neg.log (negative)"

# A GATE never aborts the calling shell/pipeline.
exit 0
