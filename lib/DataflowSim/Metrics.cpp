//===- Metrics.cpp - Dataflow simulator metrics -----------------*- C++ -*-===//
//
// PolyKernel dataflow simulator metrics + fusion traffic reduction (Todo 38 /
// Wave 7). See Metrics.h.
//
// The metrics are an analytic/cycle model derived from the SUMMA plan + the
// RunSumma simulation result - the simulator analog of `cslc --out-routes`
// (route + hop accounting) and `calculate_cycles` (cycle model). The model is
// cross-validated against the cycle-accurate simulation in metrics_test.cpp.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Metrics.h"

#include <cassert>

namespace polykernel::dataflow {

const char *ToString(CommState s) {
  switch (s) {
  case CommState::Active:
    return "active";
  case CommState::Delayed:
    return "delayed";
  case CommState::Backpressure:
    return "backpressure";
  case CommState::Idle:
    return "idle";
  }
  return "idle"; // unreachable; silences non-exhaustive fallthrough warnings.
}

int64_t ResidentBytes(const SummaPlan &plan) {
  const int64_t e = Metrics::kElemBytes;
  // Output-stationary C tile (tileM x tileN), always resident in local SRAM.
  const int64_t cTile = int64_t{plan.tileM} * plan.tileN * e;
  // Ping-pong A/B double-buffers: 2 colors per axis -> 2 buffers each. The A
  // panel is tileM x tileK, the B panel is tileK x tileN.
  const int64_t aDbuf = 2 * int64_t{plan.tileM} * plan.tileK * e;
  const int64_t bDbuf = 2 * int64_t{plan.tileK} * plan.tileN * e;
  return cTile + aDbuf + bDbuf;
}

CommState ClassifyBottleneck(const Metrics &m) {
  // SRAM over-subscription stalls the fabric: the PE cannot buffer another
  // panel, so the lossless backpressure propagates (the negative case).
  if (m.sramPressure > 1.0)
    return CommState::Backpressure;
  // A wavelet dropped off-grid is a lossy fabric (pathological for SUMMA, whose
  // edges tap only) - treat as backpressure/loss.
  if (m.droppedOffGrid > 0)
    return CommState::Backpressure;
  // Nothing in flight and nothing computing: the fabric is idle.
  if (m.messagesSent == 0 && m.panelComputes == 0)
    return CommState::Idle;
  // Fabric-bound: the simulated cycles run far beyond the analytic critical path
  // (wavelets queueing in transit dominate over the compute).
  if (m.criticalPathCycles > 0 && m.cycles > 4 * m.criticalPathCycles)
    return CommState::Delayed;
  return CommState::Active;
}

Metrics ComputeMetrics(const SummaPlan &plan, const SummaResult &result,
                       int64_t intermediateElements) {
  assert(plan.gridP > 0 && "plan must be lowered onto a positive grid");

  Metrics m;

  // --- Grid utilization: a SUMMA mapping assigns every PE an output-stationary
  // C tile and fires its panel computes, so the whole P x P grid is active. ---
  m.totalPes = plan.gridP * plan.gridP;
  m.activePes = (result.panelComputes > 0) ? m.totalPes : 0;
  m.utilization =
      m.totalPes > 0 ? double(m.activePes) / double(m.totalPes) : 0.0;

  // --- Local SRAM pressure (resident tiles + double-buffers vs 48 KB). ---
  m.residentBytes = ResidentBytes(plan);
  m.sramBytes = Sram::kTotalBytes;
  m.sramPressure = double(m.residentBytes) / double(m.sramBytes);

  // --- Messages sent (CE fabin seeds) + average hop distance. ---
  m.aWavelets = result.aWavelets;
  m.bWavelets = result.bWavelets;
  m.messagesSent = result.aWavelets + result.bWavelets;
  // A broadcast along the width P (A east) and the height P (B south) each span
  // P-1 hops at 1 cycle/hop; equal A/B wavelet counts weight the mean to P-1.
  m.avgHopDistance = plan.gridP > 0 ? double(plan.gridP - 1) : 0.0;

  // --- Critical-path cycles: broadcast fill (P-1 hops) + one pipelined compute
  // per SUMMA step. The simulated cycles cannot beat this lower bound. ---
  m.criticalPathCycles =
      static_cast<uint64_t>((plan.gridP - 1) + plan.steps);
  m.cycles = result.cycles;

  // --- Fusion traffic reduction: the fused op eliminates an intermediate's
  // fabric round-trip (write + read = 2 wavelets/element); the matmul broadcast
  // wavelets are invariant under fusion. ---
  m.intermediateElements = intermediateElements;
  m.eliminatedWavelets =
      intermediateElements > 0 ? static_cast<uint64_t>(2) *
                                     static_cast<uint64_t>(intermediateElements)
                               : 0;
  m.fusedWavelets = m.messagesSent;
  m.unfusedWavelets = m.fusedWavelets + m.eliminatedWavelets;
  m.trafficReductionPct =
      m.unfusedWavelets > 0
          ? 100.0 * double(m.eliminatedWavelets) / double(m.unfusedWavelets)
          : 0.0;

  // --- Execution counters + bottleneck classification. ---
  m.panelComputes = result.panelComputes;
  m.droppedOffGrid = result.droppedOffGrid;
  m.commState = ClassifyBottleneck(m);
  m.bottleneck = ToString(m.commState);
  m.bottleneckCause =
      (m.commState == CommState::Backpressure)
          ? (m.sramPressure > 1.0 ? "sram_over_subscription" : "fabric_drop")
          : (m.commState == CommState::Delayed ? "fabric_latency" : "none");

  return m;
}

} // namespace polykernel::dataflow
