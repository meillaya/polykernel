#!/usr/bin/env bash
#
# scripts/sync_pod.sh — PolyKernel pass2 (todo 4): PUSH THE REPO TO THE POD.
#
# rsyncs the repo (the committed state at HEAD; uncommitted .omo state stays
# local) to /root/polykernel on the existing Prime Intellect RTX 6000 Ada
# instance via DIRECT SSH (no provisioning, no `prime pods`). Excludes
# build/, build_*/, .git/, .omo/, .codegraph and *.pem (live SSH keys never
# leave this machine). Falls back to tar-over-ssh if rsync is absent on the
# pod (or locally).
#
# Re-sync rule (plan): todos 9/11/13/14/15 re-run ./scripts/sync_pod.sh before
# every pod run — this script pushes the working tree minus the excludes at
# call time, so it stays correct as later todos add new files.
#
# Overrides (all optional):
#   POD_HOST (default 65.109.75.15)  POD_PORT (default 22)  POD_USER (default root)
#   POD_KEY  (default <repo>/private_key.pem)
#   POD_SYNC_LOG (default <repo>/reports/pass2_pod_sync.log)
#
# Usage:  ./scripts/sync_pod.sh
# Exit:   always 0 (gate style; reports PASS | FAIL | SKIPPED, never aborts the pipeline).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_HOST="${POD_HOST:-65.109.75.15}"
POD_PORT="${POD_PORT:-22}"
POD_USER="${POD_USER:-root}"
POD_KEY="${POD_KEY:-$REPO_ROOT/private_key.pem}"
LOG="${POD_SYNC_LOG:-$REPO_ROOT/reports/pass2_pod_sync.log}"
# Destination on the pod: $HOME/polykernel for the $POD_USER (e.g.
# /home/ubuntu/polykernel), NOT /root/polykernel — the pod user may not have
# permission to create /root (the pass-2 pod runs as `ubuntu`). `~` in a remote
# path is expanded by the pod's shell, so DEST is a literal tilde expression.
DEST="~/polykernel"

SSH_FLAGS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i "$POD_KEY" -p "$POD_PORT")
POD_ADDR="${POD_USER}@${POD_HOST}"
RSYNC_EXCLUDES=(--exclude 'build/' --exclude 'build_*/' --exclude '.git/' --exclude '.omo/' --exclude '.codegraph' --exclude '*.pem')
TAR_EXCLUDES=(--exclude='build' --exclude='build_*' --exclude='.git' --exclude='.omo' --exclude='.codegraph' --exclude='*.pem')

out()  { printf '%s\n' "$*"; printf '%s\n' "$*" >> "$LOG"; }
info() { out "[INFO] $*"; }
pass() { out "[PASS] $*"; }
warn() { out "[WARN] $*"; }
fail() { out "[FAIL] $*"; }

: > "$LOG"

out "##############################################################################"
out "# PolyKernel pass2 todo 4: REPO SYNC TO POD (direct SSH)"
out "# Host: $(uname -srm)   Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
out "# Source: ${REPO_ROOT} -> ${POD_ADDR}:${POD_PORT}:${DEST}/"
out "# Key: ${POD_KEY} (mode $(stat -c '%a' "$POD_KEY" 2>/dev/null || echo '?')) - contents never printed"
out "##############################################################################"
out ""

# --- local prerequisites -----------------------------------------------------
local_ok=1
local_missing=""
if [[ ! -f "$POD_KEY" || "$(stat -c '%a' "$POD_KEY")" != "600" ]]; then
  local_ok=0
  local_missing="private key missing or not mode 600 (${POD_KEY})"
fi
if ! command -v ssh >/dev/null 2>&1; then
  local_ok=0
  local_missing="${local_missing:+${local_missing}; }ssh not on PATH"
fi
if ! command -v rsync >/dev/null 2>&1 && ! command -v tar >/dev/null 2>&1; then
  local_ok=0
  local_missing="${local_missing:+${local_missing}; }neither rsync nor tar on PATH"
fi
if (( local_ok == 0 )); then
  out "=== SYNC VERDICT ==="
  out "SYNC VERDICT: SKIPPED (local prerequisite problem)"
  out "  EXACT cause: ${local_missing}"
  out "  REMEDY: fix the local prerequisite, then re-run ./scripts/sync_pod.sh"
  info "evidence: ${LOG}"
  exit 0
fi
pass "local prerequisites ok (ssh + rsync/tar + key mode 600)"
out ""

# --- STEP 1: reach the pod + ensure the destination exists --------------------
out "=== STEP 1: reach the pod + ensure ${DEST} exists ==="
if ! mkdir_out="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" "mkdir -p ${DEST} && echo DIR_OK" 2>&1)"; then
  fail "cannot ssh to ${POD_ADDR}:${POD_PORT}: $(printf '%s\n' "$mkdir_out" | head -1)"
  out "=== SYNC VERDICT ==="
  out "SYNC VERDICT: SKIPPED (cannot reach the pod)"
  out "  EXACT cause: ssh to ${POD_ADDR}:${POD_PORT} failed"
  out "  REMEDY: verify the instance is RUNNING; do NOT provision a new pod"
  info "evidence: ${LOG}"
  exit 0
fi
pass "ssh ok, ${DEST} ready"
out ""

# --- STEP 2: pick the transfer method -----------------------------------------
sync_method="rsync"
if ! command -v rsync >/dev/null 2>&1; then
  warn "local rsync absent - using tar-over-ssh fallback"
  sync_method="tar"
else
  remote_rsync="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" 'command -v rsync >/dev/null 2>&1 && echo RSYNC_OK || echo RSYNC_MISSING' 2>&1 || true)"
  if [[ "$remote_rsync" != *"RSYNC_OK"* ]]; then
    warn "rsync absent on the pod - using tar-over-ssh fallback"
    sync_method="tar"
  else
    pass "rsync available on both ends"
  fi
fi
out ""

# --- STEP 3: push -------------------------------------------------------------
out "=== STEP 3: push the repo (method: ${sync_method}) ==="
push_ok=0
if [[ "$sync_method" == "rsync" ]]; then
  if rsync_out="$(rsync -az -e "ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -i ${POD_KEY} -p ${POD_PORT}" \
       "${RSYNC_EXCLUDES[@]}" "$REPO_ROOT/" "${POD_ADDR}:${DEST}/" 2>&1)"; then
    push_ok=1
    info "rsync completed (quiet -a mode; zero bytes shown means no changes)"
  else
    rsync_rc=$?
    fail "rsync failed (rc=${rsync_rc}): $(printf '%s\n' "$rsync_out" | head -3 | tr '\n' ' ')"
  fi
else
  if ( tar -czf - "${TAR_EXCLUDES[@]}" -C "$REPO_ROOT" . \
       | ssh "${SSH_FLAGS[@]}" "$POD_ADDR" "mkdir -p ${DEST} && tar -xzf - -C ${DEST}" 2>&1 ); then
    push_ok=1
    info "tar-over-ssh completed"
  else
    fail "tar-over-ssh failed (see stderr above)"
  fi
fi
out ""

# --- STEP 4: verify on the pod -------------------------------------------------
out "=== STEP 4: verify on the pod (CMakeLists present + private key NOT synced) ==="
if verify_out="$(ssh "${SSH_FLAGS[@]}" "$POD_ADDR" "test -f ${DEST}/CMakeLists.txt && echo CMakeLists_OK || echo CMakeLists_MISSING; test -e ${DEST}/private_key.pem && echo KEY_LEAKED || echo KEY_EXCLUDED_OK; printf 'files_on_pod: '; find ${DEST} -type f | wc -l" 2>&1)"; then
  while IFS= read -r l; do info "  $l"; done <<<"$verify_out"
  cmake_ok=0; key_ok=0
  [[ "$verify_out" == *"CMakeLists_OK"* ]] && cmake_ok=1
  [[ "$verify_out" == *"KEY_EXCLUDED_OK"* ]] && key_ok=1
else
  fail "verification ssh failed after the push"
  cmake_ok=0; key_ok=0; push_ok=0
fi
out ""

# --- verdict (a sync is a GATE: PASS | FAIL | SKIPPED, always exit 0) ----------
out "=== SYNC VERDICT ==="
if (( push_ok && cmake_ok && key_ok )); then
  out "SYNC VERDICT: PASS (repo pushed to ${POD_ADDR}:${DEST} via ${sync_method})"
  out "  excludes honored: build/ build_*/ .git/ .omo/ .codegraph *.pem - the private key is NOT on the pod"
else
  out "SYNC VERDICT: FAIL (push_ok=${push_ok} cmake_ok=${cmake_ok} key_ok=${key_ok})"
  out "  EXACT cause: see the [FAIL]/[WARN] lines above"
fi
out ""
info "evidence: ${LOG}"

exit 0
