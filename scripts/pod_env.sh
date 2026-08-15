#!/usr/bin/env bash
#
# scripts/pod_env.sh — PolyKernel pass2 (todo 4): POD ENVIRONMENT PROBE (GATE).
#
# First REAL contact with the user's existing Prime Intellect RTX 6000 Ada
# instance via DIRECT SSH (no provisioning, no `prime pods`, no state change on
# the pod beyond the pinned `pip install numpy ml_dtypes pytest` — or the
# PEP-668 venv fallback /root/pkvenv). The instance is a paid per-hour pod:
# every check is a single bounded ssh round trip, nothing is left running.
#
# Records a PASS/FAIL verdict per item to reports/pod_env.log. A probe is a
# GATE, not a build step: if SSH fails or nvcc < 11.8 (or any gating check
# fails), the verdict is SKIPPED with the EXACT cause named (never FAILED, no
# crash, no hang — ConnectTimeout=15 bounds every connection). This is the
# decision input for todos 9/11/13/14/15 and for W5's R2 closure strategy
# (glibc + apt llvm-21 availability are recorded on purpose).
#
# Overrides (all optional):
#   POD_HOST    (default 65.109.75.15)   POD_PORT (default 22)   POD_USER (default root)
#   POD_KEY     (default <repo>/private_key.pem)
#   POD_ENV_LOG (default <repo>/reports/pod_env.log) — the negative QA points
#               this at reports/pass2_pod_probe_neg.log with POD_PORT=2222.
#
# Usage:  ./scripts/pod_env.sh
# Exit:   always 0 (a GATE degrades; it does not abort the calling shell/pipeline).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_HOST="${POD_HOST:-65.109.75.15}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$REPO_ROOT/private_key.pem}"
LOG="${POD_ENV_LOG:-$REPO_ROOT/reports/pod_env.log}"

SSH_FLAGS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$POD_KEY" -p "$POD_PORT")
POD_ADDR="${POD_USER}@${POD_HOST}"
NVCC_MIN="11.8"   # native sm_89 (Ada) requires CUDA >= 11.8

# Report helpers: every line goes to stdout AND the log file.
out()  { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$LOG"; }
info() { out "[INFO] $*"; }
pass() { out "[PASS] $*"; }
warn() { out "[WARN] $*"; }
fail() { out "[FAIL] $*"; }

# Per-check results (0=ok, 1=problem).
rc_ssh=1; rc_gpu=1; rc_nvcc=1; rc_python=1; rc_pkgs=1
missing_ssh=""; missing_gpu=""; missing_nvcc=""; missing_python=""; missing_pkgs=""

: > "$LOG"

out "##############################################################################"
out "# PolyKernel pass2 todo 4: POD ENVIRONMENT PROBE (GATE for todos 9/11/13/14/15)"
out "# Host: $(uname -srm)   Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
out "# Target: ${POD_ADDR}:${POD_PORT}  (existing Prime Intellect RTX 6000 Ada instance - direct SSH, no provisioning)"
out "# Key: ${POD_KEY}  mode: $(stat -c '%a' "$POD_KEY" 2>/dev/null || echo 'MISSING')  (contents never printed)"
out "##############################################################################"
out ""

# ---------------------------------------------------------------------------
# CHECK 0: local prerequisites (ssh client + private key present, mode 600).
# ---------------------------------------------------------------------------
out "=== CHECK 0: local prerequisites (ssh client + private key mode 600) ==="
local_ok=1
if [[ ! -f "$POD_KEY" ]]; then
  fail "private key not found: ${POD_KEY}"
  missing_ssh="private key missing at ${POD_KEY}"
  local_ok=0
elif [[ "$(stat -c '%a' "$POD_KEY")" != "600" ]]; then
  fail "private key mode is $(stat -c '%a' "$POD_KEY") - OpenSSH requires 600"
  missing_ssh="private key mode is $(stat -c '%a' "$POD_KEY"), expected 600"
  local_ok=0
else
  pass "private key present, mode 600"
fi
if ! command -v ssh >/dev/null 2>&1; then
  fail "ssh client not on PATH"
  missing_ssh="${missing_ssh:+${missing_ssh}; }ssh not on PATH"
  local_ok=0
else
  pass "ssh client: $(command -v ssh)"
fi
out ""

if (( local_ok == 0 )); then
  out "=== GATE VERDICT ==="
  out "GATE VERDICT: SKIPPED"
  out "  Pod probe not attempted (local prerequisite problem)."
  out "  EXACT cause: ${missing_ssh}"
  out "  REMEDY: fix the local prerequisite, then re-run ./scripts/pod_env.sh"
  info "evidence: ${LOG}"
  exit 0
fi

# ---------------------------------------------------------------------------
# CHECK 1: SSH connectivity (the one-line smoke `echo POD_OK` is the first
#          real contact; ConnectTimeout=15 bounds a blackholed pod).
# ---------------------------------------------------------------------------
out "=== CHECK 1: SSH connectivity (${POD_ADDR}:${POD_PORT}, ConnectTimeout=15) ==="
if ssh_out="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'echo POD_OK; uname -srmo' 2>&1)"; then
  pass "ssh session established to ${POD_ADDR}:${POD_PORT}"
  while IFS= read -r l; do info "  $l"; done <<<"$ssh_out"
  rc_ssh=0
else
  ssh_rc=$?
  # Skip the "Permanently added ... known hosts" warning that precedes the real error on first contact.
  first_err="$(printf '%s\n' "$ssh_out" | grep -iE 'denied|refused|timed out|no route|unreachable|not found|invalid|closed|reset|offering' | head -1 || true)"
  [[ -z "$first_err" ]] && first_err="$(printf '%s\n' "$ssh_out" | tail -1)"
  fail "ssh to ${POD_ADDR}:${POD_PORT} FAILED (rc=${ssh_rc})"
  info "  ${first_err}"
  missing_ssh="cannot ssh to ${POD_ADDR}:${POD_PORT} (rc=${ssh_rc}): ${first_err}"
  rc_ssh=1
fi
out ""

# ---------------------------------------------------------------------------
# CHECK 2: nvidia-smi -L (expect "RTX 6000 Ada Generation" = cc 8.9).
# ---------------------------------------------------------------------------
check_gpu() {
  out "=== CHECK 2: nvidia-smi -L (expect RTX 6000 Ada Generation, compute capability 8.9) ==="
  if (( rc_ssh != 0 )); then
    out "[SKIP] nvidia-smi -L (no ssh session)"
    out ""
    return
  fi
  local out_nv
  out_nv="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'nvidia-smi -L 2>&1; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>&1' 2>&1 || true)"
  if [[ "$out_nv" == *"RTX 6000 Ada"* ]]; then
    pass "GPU: NVIDIA RTX 6000 Ada Generation detected (sm_89)"
    while IFS= read -r l; do info "  $l"; done <<<"$out_nv"
    rc_gpu=0
  else
    fail "nvidia-smi did not enumerate an RTX 6000 Ada: ${out_nv:-no output}"
    missing_gpu="nvidia-smi -L did not enumerate an RTX 6000 Ada (${out_nv:-no output})"
    rc_gpu=1
  fi
  out ""
}

# ---------------------------------------------------------------------------
# CHECK 3: nvcc --version (require >= 11.8 for native sm_89).
# ---------------------------------------------------------------------------
check_nvcc() {
  out "=== CHECK 3: nvcc --version (require >= ${NVCC_MIN} for native sm_89) ==="
  if (( rc_ssh != 0 )); then
    out "[SKIP] nvcc --version (no ssh session)"
    out ""
    return
  fi
  local out_nv ver
  out_nv="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'nvcc --version 2>&1 || echo NVCC_MISSING' 2>&1 || true)"
  ver="$(printf '%s\n' "$out_nv" | grep -oE 'release [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
  if [[ -z "$ver" ]]; then
    fail "nvcc not found or no version reported"
    while IFS= read -r l; do info "  $l"; done <<<"$out_nv"
    missing_nvcc="nvcc absent/undetectable on the pod (sm_89 needs CUDA >= ${NVCC_MIN}; Prime Intellect CUDA images ship 12.x)"
    rc_nvcc=1
  elif [[ "$(printf '%s\n' "$NVCC_MIN" "$ver" | sort -V | head -1)" == "$NVCC_MIN" ]]; then
    pass "nvcc ${ver} >= ${NVCC_MIN} (native sm_89 supported)"
    while IFS= read -r l; do info "  $l"; done <<<"$out_nv"
    rc_nvcc=0
  else
    fail "nvcc ${ver} < ${NVCC_MIN} (no native sm_89)"
    missing_nvcc="nvcc ${ver} is older than ${NVCC_MIN} (no native sm_89)"
    rc_nvcc=1
  fi
  out ""
}

# ---------------------------------------------------------------------------
# CHECK 4: python3 --version.
# ---------------------------------------------------------------------------
check_python() {
  out "=== CHECK 4: python3 --version ==="
  if (( rc_ssh != 0 )); then
    out "[SKIP] python3 --version (no ssh session)"
    out ""
    return
  fi
  local out_py
  out_py="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'python3 --version 2>&1 || echo PYTHON3_MISSING' 2>&1 || true)"
  if [[ "$out_py" == PYTHON3_MISSING* ]]; then
    fail "python3 not on PATH"
    missing_python="python3 not on PATH on the pod"
    rc_python=1
  else
    pass "python3: ${out_py}"
    rc_python=0
  fi
  out ""
}

# ---------------------------------------------------------------------------
# CHECK 5: python packages numpy/ml_dtypes/pytest. Pinned installs only:
#          `pip install numpy ml_dtypes pytest`; PEP-668 fallback:
#          python3 -m venv /root/pkvenv && /root/pkvenv/bin/pip install ...
#          Later pod todos use /root/pkvenv/bin/python if that path is echoed.
# ---------------------------------------------------------------------------
check_pkgs() {
  out "=== CHECK 5: python packages numpy/ml_dtypes/pytest (pinned; PEP-668 venv fallback) ==="
  if (( rc_ssh != 0 )); then
    out "[SKIP] python packages (no ssh session)"
    out ""
    return
  fi
  local out_pk
  out_pk="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'bash -s' 2>&1 <<'REMOTE_PKGS' || true
set -u
rm -f /tmp/pk_pip.log /tmp/pk_pip_venv.log
if python3 -c 'import numpy, ml_dtypes, pytest' 2>/dev/null; then
  python3 -c 'import numpy, ml_dtypes, pytest; print("numpy", numpy.__version__, "ml_dtypes", ml_dtypes.__version__, "pytest", pytest.__version__)'
  echo "PKGS_OK system-python3 (already importable, no install needed)"
  exit 0
fi
if python3 -m pip install --quiet numpy ml_dtypes pytest >/tmp/pk_pip.log 2>&1; then
  python3 -c 'import numpy, ml_dtypes, pytest; print("numpy", numpy.__version__, "ml_dtypes", ml_dtypes.__version__, "pytest", pytest.__version__)'
  echo "PKGS_OK system-pip"
  exit 0
fi
if grep -qi 'externally-managed-environment' /tmp/pk_pip.log; then
  echo "PEP668: system pip blocked (externally-managed-environment) -> venv fallback"
  if python3 -m venv /root/pkvenv \
     && /root/pkvenv/bin/pip install --quiet numpy ml_dtypes pytest >/tmp/pk_pip_venv.log 2>&1 \
     && /root/pkvenv/bin/python -c 'import numpy, ml_dtypes, pytest' 2>/dev/null; then
    /root/pkvenv/bin/python -c 'import numpy, ml_dtypes, pytest; print("numpy", numpy.__version__, "ml_dtypes", ml_dtypes.__version__, "pytest", pytest.__version__)'
    echo "PKGS_OK venv /root/pkvenv"
    echo "PYTHON_FOR_POD=/root/pkvenv/bin/python"
    exit 0
  fi
  echo "venv fallback failed:"; tail -5 /tmp/pk_pip_venv.log 2>/dev/null || true
  echo "PKGS_FAILED venv-fallback"
  rm -f /tmp/pk_pip.log /tmp/pk_pip_venv.log
  exit 1
fi
echo "pip install failed:"; tail -5 /tmp/pk_pip.log 2>/dev/null || true
echo "PKGS_FAILED pip"
rm -f /tmp/pk_pip.log /tmp/pk_pip_venv.log
exit 1
REMOTE_PKGS
)"
  if [[ "$out_pk" == *"PKGS_OK"* ]]; then
    pass "python packages usable (numpy/ml_dtypes/pytest)"
    while IFS= read -r l; do info "  $l"; done <<<"$out_pk"
    rc_pkgs=0
  else
    fail "numpy/ml_dtypes/pytest are NOT usable on the pod"
    while IFS= read -r l; do info "  $l"; done <<<"$out_pk"
    missing_pkgs="pip install numpy ml_dtypes pytest failed on the pod (incl. the PEP-668 venv fallback); see CHECK 5"
    rc_pkgs=1
  fi
  out ""
}

# ---------------------------------------------------------------------------
# CHECK 6: environment record for the R2 closure decision (gcc / disk free /
#          glibc via ldd --version / apt llvm-21 + libllvm-21-dev / pod rsync).
#          One bounded round trip; informational (PASS/WARN), never gates.
# ---------------------------------------------------------------------------
check_info() {
  out "=== CHECK 6: R2 closure decision inputs (gcc / disk free / glibc / apt llvm-21 / rsync) ==="
  if (( rc_ssh != 0 )); then
    out "[SKIP] environment record (no ssh session)"
    out ""
    return
  fi
  local out_in
  out_in="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'bash -s' 2>&1 <<'REMOTE_INFO' || true
set -u
echo "GCC_VERSION: $(gcc --version 2>/dev/null | head -1 || echo MISSING)"
echo "DISK_ROOT: $(df -h /root 2>/dev/null | tail -1 || echo MISSING)"
echo "GLIBC: $(ldd --version 2>/dev/null | head -1 || echo MISSING)"
echo "APT_LLVM21: $(apt-cache policy llvm-21 libllvm-21-dev 2>&1 | tr '\n' '|' | sed 's/||*/|/g' | head -c 700 || echo MISSING)"
echo "POD_RSYNC: $(command -v rsync 2>/dev/null || echo MISSING)"
REMOTE_INFO
)"
  while IFS= read -r l; do
    case "$l" in
      GCC_VERSION:*)   [[ "$l" == *MISSING* ]] && warn "${l#GCC_VERSION: } (gcc)" || pass "${l#GCC_VERSION: } (gcc)" ;;
      DISK_ROOT:*)     [[ "$l" == *MISSING* ]] && warn "${l#DISK_ROOT: } (disk free /root)" || pass "${l#DISK_ROOT: } (disk free /root)" ;;
      GLIBC:*)         [[ "$l" == *MISSING* ]] && warn "${l#GLIBC: } (glibc)" || pass "${l#GLIBC: } (glibc - R2 closure decision input)" ;;
      APT_LLVM21:*)    apt_line="${l#APT_LLVM21: }"
                       if [[ "$apt_line" == *"Candidate: "*[0-9]* ]]; then
                         pass "apt llvm-21/libllvm-21-dev: candidate present - R2 path (a) (build-on-pod w/ apt LLVM) viable"
                       elif [[ "$apt_line" == *"Unable to locate"* || "$apt_line" == *"(none)"* || -z "$apt_line" ]]; then
                         warn "apt llvm-21/libllvm-21-dev: no candidate in the current apt lists - R2 path (a) needs an apt update or falls to (b)/(c)"
                       else
                         info "apt llvm-21/libllvm-21-dev output: ${apt_line}"
                       fi ;;
      POD_RSYNC:*)     [[ "$l" == *MISSING* ]] && warn "${l#POD_RSYNC: } (rsync on pod - sync_pod.sh will tar-fallback)" || pass "${l#POD_RSYNC: } (rsync on pod)" ;;
      *)               info "pod: $l" ;;
    esac
  done <<<"$out_in"
  out ""
}

check_gpu
check_nvcc
check_python
check_pkgs
check_info

# ---------------------------------------------------------------------------
# VERDICT: PASS | SKIPPED (exact cause named). Never FAILED, always exit 0.
# ---------------------------------------------------------------------------
out "=== GATE VERDICT ==="
if (( rc_ssh != 0 )); then
  out "GATE VERDICT: SKIPPED"
  out "  Pod-runnable todos (9/11/13/14/15) DEGRADE to SKIPPED with the exact cause named:"
  out "    - SSH: ${missing_ssh}"
  out "  REMEDY: verify the instance is RUNNING (it is a paid per-hour instance) and that"
  out "          port ${POD_PORT} is reachable from here; do NOT provision a new pod."
elif (( rc_gpu != 0 || rc_nvcc != 0 || rc_python != 0 || rc_pkgs != 0 )); then
  out "GATE VERDICT: SKIPPED"
  out "  Pod-runnable todos (9/11/13/14/15) DEGRADE to SKIPPED with the exact cause named:"
  (( rc_gpu != 0 ))    && out "    - GPU: ${missing_gpu}"
  (( rc_nvcc != 0 ))   && out "    - NVCC: ${missing_nvcc}"
  (( rc_python != 0 )) && out "    - PYTHON3: ${missing_python}"
  (( rc_pkgs != 0 ))   && out "    - PYTHON-PKGS: ${missing_pkgs}"
  out "  REMEDY: see the per-check [FAIL]/[WARN] lines above; re-run ./scripts/pod_env.sh after fixing."
else
  out "GATE VERDICT: PASS"
  out "  Pod env is ready for the pod-runnable todos (9/11/13/14/15)."
  out "  R2 closure decision inputs recorded above: glibc, apt llvm-21, gcc, disk free."
fi
out "  checks: ssh=${rc_ssh} gpu=${rc_gpu} nvcc=${rc_nvcc} python3=${rc_python} pkgs=${rc_pkgs} (0=ok)"
out ""
info "evidence: ${LOG} (happy) or the POD_ENV_LOG path with POD_PORT set wrong (negative)"

# A GATE never aborts the calling shell/pipeline.
exit 0
