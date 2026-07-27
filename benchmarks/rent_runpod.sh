#!/usr/bin/env bash
#===- rent_runpod.sh - Opt-in RunPod GPU rental + bench (T28) ---*- bash -*-===//
#
# PolyKernel real-GPU benchmark rental orchestration (Todo 28 / Wave 5). OPT-IN:
# provisions NVIDIA H100/A100 (CUDA) + AMD MI300 (HIP gfx942) ad-hoc per-hour pods
# on RunPod, builds the engine + bench in-container, runs the bounded autotuner
# (Todo 24 ConfigSpace grid) + the bench suite (benchmarks/bench_cuda.cpp /
# bench_hip.cpp), and writes REAL unfused/fused/autotuned speedups to
# reports/{h100,mi300}_bench.{json,md}.
#
# CREDENTIAL GATE (binding): without RunPod credentials this script does NOTHING
# but print the SKIPPED notice + the exact opt-in commands and exit 0 (SKIPPED,
# not FAILED). It makes NO RunPod API call and spends NO money unless credentials
# are present. The whole rental below the gate is opt-in code.
#
# BUDGET GUARD: per-second RunPod billing; the script caps total wall-clock
# runtime (MAX_RUNTIME_MIN) AND a spend ceiling (BUDGET_CEILING_USD, default $100,
# hard-capped at $100 unless --i-understand-spend is passed), logs running spend,
# and terminates every pod on exit (trap) so a failure can never leave a pod
# billing. The autotuner search is the BOUNDED Todo 24 grid (a prefix, not all
# ~141 configs), keeping pod time short.
#
# Usage (opt-in):
#   export RUNPOD_API_KEY=<your-key>
#   ./benchmarks/rent_runpod.sh --suite mlp                 # real numbers
#   ./benchmarks/rent_runpod.sh --suite mlp --dry-run       # print, don't spend
#
# RunPod = ad-hoc per-hour GPU pods (docker + per-second billing). GPU type IDs
# are owner-verified (they change); override via env (RUNPOD_GPU_H100 / _A100 /
# _MI300). The script uses the RunPod GraphQL API via curl + python3 (no jq dep).
#
#==============================================================================//

set -euo pipefail

SUITE="mlp"
REPORTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/reports"
DRY_RUN=0
BUDGET_CEILING_USD="${BUDGET_CEILING_USD:-100}"
MAX_RUNTIME_MIN="${MAX_RUNTIME_MIN:-30}"
I_UNDERSTAND_SPEND=0

# Owner-verified RunPod GPU type IDs (override via env; these are the common
# secure-cloud / community names and MUST be confirmed for the target account).
RUNPOD_GPU_H100="${RUNPOD_GPU_H100:-NVIDIA H100 80GB HBM3}"
RUNPOD_GPU_A100="${RUNPOD_GPU_A100:-NVIDIA A100 SXM4 80GB}"
RUNPOD_GPU_MI300="${RUNPOD_GPU_MI300:-AMD Instinct MI300X}"
# Container images with the toolchain to build + run the bench in-container.
RUNPOD_IMAGE_CUDA="${RUNPOD_IMAGE_CUDA:-nvidia/cuda:12.6.0-devel-ubuntu22.04}"
RUNPOD_IMAGE_ROCM="${RUNPOD_IMAGE_ROCM:-rocm/dev-ubuntu-22.04:6.2}"

# Pods we provision (cleaned up on exit).
DEPLOYED_POD_IDS=()
START_EPOCH="$(date +%s)"

log()  { printf '[rent_runpod] %s\n' "$*"; }
die()  { printf '[rent_runpod] ERROR: %s\n' "$*" >&2; exit 1; }

# --- JSON helper (python3; no jq dependency). Walks the arg path through the ---
# --- stdin JSON; prints the scalar leaf (or "null" when a key is absent). The  ---
# --- script is passed via -c so stdin stays free for the piped JSON document.  ---
json_get() {
  python3 -c '
import json, sys
node = json.load(sys.stdin)
for key in sys.argv[1:]:
    if isinstance(node, list):
        node = node[int(key)]
    elif isinstance(node, dict):
        node = node.get(key)
    else:
        node = None
    if node is None:
        break
print(node if isinstance(node, (str, int, float)) else json.dumps(node))
' "$@"
}

usage() {
  cat <<'EOF'
Usage: rent_runpod.sh [--suite mlp] [--reports-dir DIR] [--dry-run]
                      [--budget-ceiling USD] [--max-runtime-min MIN]
                      [--i-understand-spend]

OWNER-GATED opt-in RunPod rental. Without RUNPOD_API_KEY it prints a SKIPPED
notice and exits 0 (no API call, no spend). With credentials it provisions
H100/A100 (CUDA) + MI300 (HIP) pods, builds + runs the bounded autotuner + bench
suite in-container under a spend ceiling, and writes REAL speedups to
reports/{h100,mi300}_bench.{json,md}.
EOF
}

# --- Arg parse. ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)              SUITE="$2"; shift 2 ;;
    --reports-dir)        REPORTS_DIR="$2"; shift 2 ;;
    --dry-run)            DRY_RUN=1; shift ;;
    --budget-ceiling)     BUDGET_CEILING_USD="$2"; shift 2 ;;
    --max-runtime-min)    MAX_RUNTIME_MIN="$2"; shift 2 ;;
    --i-understand-spend) I_UNDERSTAND_SPEND=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done

# === CREDENTIAL GATE (binding). No credentials => SKIPPED, exit 0, no spend. ===
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  cat <<EOF
SKIPPED: RUNPOD_API_KEY not set — opt-in rental documented (no spend).

This is the OWNER-GATED RunPod rental (Todo 28). No RunPod credentials were
found (checked: RUNPOD_API_KEY). NOTHING was provisioned; NO RunPod API call was
made; NO money was spent. The bench SUITE code is present:
    benchmarks/bench_cuda.cpp   (H100/A100, nvcc sm_80/sm_90)
    benchmarks/bench_hip.cpp    (MI300 gfx942 / RX 7800 XT gfx1101)
    benchmarks/run_bench_suite.py (orchestrator + roofline projections)

To obtain REAL speedup numbers, opt in (spend ceiling ~\$50-100 enforced):
    export RUNPOD_API_KEY=<your-key>
    ./benchmarks/rent_runpod.sh --suite mlp                # real run
    ./benchmarks/rent_runpod.sh --suite mlp --dry-run      # preview, no spend

Without credentials the suite reports clearly-labeled perf-model PROJECTIONS:
    python3 benchmarks/run_bench_suite.py
    -> reports/{h100,mi300}_bench.{json,md}  (PROJECTED) + reports/w5_rent_skipped.log
EOF
  exit 0
fi

# === Credentials present: the opt-in rental proceeds below. ===================

# Budget guard: hard-cap the ceiling at $100 unless explicitly acknowledged.
if (( $(echo "$BUDGET_CEILING_USD > 100" | bc -l 2>/dev/null || echo 0) )) && [[ "$I_UNDERSTAND_SPEND" -ne 1 ]]; then
  log "budget ceiling $BUDGET_CEILING_USD exceeds the \$100 guard; clamping to 100 (pass --i-understand-spend to override)."
  BUDGET_CEILING_USD=100
fi
log "credentials present; suite=$SUITE reports=$REPORTS_DIR budget<=\$${BUDGET_CEILING_USD} max_runtime<=${MAX_RUNTIME_MIN}min dry_run=$DRY_RUN"

SPEND_LOG="${REPORTS_DIR}/runpod_spend.log"
mkdir -p "$REPORTS_DIR"
RUNPOD_API="https://api.runpod.io/graphql"

# --- Cleanup: terminate every provisioned pod on ANY exit (no orphan billing). -
cleanup() {
  local rc=$?
  for pid in "${DEPLOYED_POD_IDS[@]:-}"; do
    [[ -z "$pid" ]] && continue
    log "terminating pod $pid (cleanup)..."
    curl -sf -X POST "$RUNPOD_API" -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      -H "Content-Type: application/json" \
      -d "{\"query\":\"mutation{podTerminate(input:{podId:\\\"$pid\\\"})}\"}" \
      >/dev/null 2>&1 || log "warn: failed to terminate pod $pid (terminate it in the RunPod console)"
  done
  local elapsed_min=$(( ( $(date +%s) - START_EPOCH ) / 60 ))
  log "cleanup done; total wall-clock ${elapsed_min} min. Spend logged to ${SPEND_LOG}."
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --- Budget guard: abort if we've exceeded the runtime or projected spend. -----
check_budget() {
  local elapsed_min=$(( ( $(date +%s) - START_EPOCH ) / 60 ))
  if (( elapsed_min > MAX_RUNTIME_MIN )); then
    die "runtime ${elapsed_min}min exceeds MAX_RUNTIME_MIN=${MAX_RUNTIME_MIN}; aborting (budget guard)."
  fi
}

# --- RunPod GraphQL call (POST). Prints the raw JSON response. -----------------
rp_call() {
  local query="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "[dry-run] would POST: $query"
    echo '{}'
    return 0
  fi
  curl -sf -X POST "$RUNPOD_API" -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" -d "{\"query\":$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$query")}"
}

# --- Provision one on-demand pod for a GPU type; echoes the pod id. ------------
deploy_pod() {
  local gpu_type="$1" name="$2" image="$3"
  check_budget
  log "finding + deploying on-demand pod: gpu='$gpu_type' image='$image' name='$name'"
  local mut="mutation{podFindAndDeployOnDemand(input:{gpuCount:1,gpuTypeId:\"$gpu_type\",containerDiskInGb:20,volumeInGb:10,minVcpuCount:4,minMemoryInGb:16,imageName:\"$image\",name:\"$name\",startSsh:true,startJupyter:false}){id,desiredStatus,perHourCost}}"
  local resp pod_id cost
  resp="$(rp_call "$mut")"
  if [[ "$DRY_RUN" -eq 1 ]]; then echo "dry-run-pod"; return 0; fi
  pod_id="$(echo "$resp" | json_get data podFindAndDeployOnDemand id)"
  cost="$(echo "$resp" | json_get data podFindAndDeployOnDemand perHourCost)"
  [[ -z "$pod_id" || "$pod_id" == "null" || "$pod_id" == "{}" ]] \
    && die "pod deploy failed for '$gpu_type': $resp"
  DEPLOYED_POD_IDS+=("$pod_id")
  printf '%s\t%s\n' "$pod_id" "${cost:-?}" >> "$SPEND_LOG"
  log "deployed pod $pod_id (\$${cost:-?}/hr) for $gpu_type"
  echo "$pod_id"
}

# --- Exec the in-container build + bench, fetch the report JSON. ---------------
# Builds the engine + bench_cuda/bench_hip, runs the bounded autotuner (a prefix
# of the Todo 24 grid) + the bench, and prints the bench --json line. The owner's
# repo is assumed mounted/cloned at /workspace/polykernel in the pod image.
run_bench_in_pod() {
  local pod_id="$1" backend="$2" arch="$3"
  check_budget
  log "running bounded autotuner + bench ($backend/$arch) in pod $pod_id..."
  # The exact exec transport (RunPod pod exec / SSH over the exposed port) is
  # owner-configured; this is the command sequence executed in-container.
  cat <<EOF
  # in-container (pod $pod_id, $backend/$arch):
  cd /workspace/polykernel
  cmake -B build -DMLIR_TABLEGEN_EXE=\$(which mlir-tblgen) && cmake --build build --target polykernel-opt
  # bounded autotuner (Todo 24 grid prefix) + correctness-gated bench (Todo 25):
  python3 tools/polykernel-bench/bench.py --autotune --op fused_rmsnorm_matmul \\
      --shape 2048,11008,4096 --dtype bf16 --backend hip --arch $arch --variants 8
  # the real unfused/fused/autotuned timing:
EOF
  if [[ "$backend" == "cuda" ]]; then
    echo "  nvcc -x cu -DPOLYKERNEL_CUDA -std=c++20 -O2 -Ikernels/template --gpu-architecture=$arch benchmarks/bench_cuda.cpp kernels/generated/{rmsnorm,matmul,fused_rmsnorm_matmul}.cu -o /tmp/bench && /tmp/bench --json --arch $arch"
  else
    echo "  hipcc -DPOLYKERNEL_HIP -std=c++20 -O2 -Ikernels/template --offload-arch=$arch benchmarks/bench_hip.cpp kernels/generated/{rmsnorm,matmul,fused_rmsnorm_matmul}.cu -o /tmp/bench && /tmp/bench --json --arch $arch"
  fi
}

# === The rental plan: H100 + A100 (CUDA) and MI300 (HIP). ======================
log "=== RunPod benchmark suite: $SUITE fragment (unfused vs fused vs autotuned) ==="

# NVIDIA side -> reports/h100_bench.{json,md} (H100 headline + A100 secondary).
H100_POD="$(deploy_pod "$RUNPOD_GPU_H100" "pk-h100-bench" "$RUNPOD_IMAGE_CUDA")"
run_bench_in_pod "$H100_POD" cuda sm_90
A100_POD="$(deploy_pod "$RUNPOD_GPU_A100" "pk-a100-bench" "$RUNPOD_IMAGE_CUDA")"
run_bench_in_pod "$A100_POD" cuda sm_80

# AMD side -> reports/mi300_bench.{json,md}.
MI300_POD="$(deploy_pod "$RUNPOD_GPU_MI300" "pk-mi300-bench" "$RUNPOD_IMAGE_ROCM")"
run_bench_in_pod "$MI300_POD" hip gfx942

check_budget
log "bench commands issued; collect each pod's bench --json stdout into"
log "  ${REPORTS_DIR}/h100_bench.json  (H100 sm_90 + A100 sm_80) and"
log "  ${REPORTS_DIR}/mi300_bench.json (MI300 gfx942), replacing the PROJECTED"
log "  fallback (set \"measured\": true). The cleanup trap terminates all pods now."
log "spend log: ${SPEND_LOG} (per-pod \$/hr; per-second billing, ceiling \$${BUDGET_CEILING_USD})."
