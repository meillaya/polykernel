//===- Metrics.h - Dataflow simulator metrics -------------------*- C++ -*-===//
//
// PolyKernel dataflow simulator metrics + fusion traffic reduction (Todo 38 /
// Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware). The metrics layer OBSERVES the Todo 35 dataflow core (Grid/Pe/
// Router: 48 KB SRAM, single-cycle hop, lossless backpressure) and the Todo 37
// SUMMA engine (SummaPlan + RunSumma/SummaResult: cycles, A/B wavelet counts,
// panel computes) and derives the Cerebras-style performance metrics:
//
//   * grid utilization      - active PEs / total PEs (a SUMMA mapping lights up
//                             the whole P x P grid it is lowered onto),
//   * local SRAM pressure   - resident output tile + ping-pong A/B double-buffers
//                             vs the 48 KB per-PE SRAM (> 100% = over-subscribed),
//   * messages sent         - total source wavelets the CE injects (fabin seeds),
//   * average hop distance  - 1 cycle/hop; a broadcast along width P spans P-1,
//   * communication bottleneck - Active / Delayed / Backpressure / Idle,
//   * critical-path cycles  - analytic broadcast-fill + pipelined-compute bound,
//   * fusion traffic reduction - wavelet count unfused vs fused: the fused MLP
//                             keeps the eliminated intermediate in local SRAM so
//                             it crosses the fabric as ZERO wavelets.
//
// The real-SDK analogs being MODELLED here are `cslc --out-routes` (the route +
// hop accounting) and `calculate_cycles` (the cycle model). The analytic model
// is cross-validated against the cycle-accurate RunSumma simulation in
// lib/DataflowSim/tests/metrics_test.cpp (hand-computed 4x4 values).
//
// Terminology guardrail (F4): CE + FMAC (no FMU/PMU); `fabin_dsd`/`fabout_dsd`;
// `@set_color_config` rx/tx. This is a simulator, not a cslc compiler and not
// Cerebras hardware.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_METRICS_H
#define POLYKERNEL_DATAFLOW_METRICS_H

#include "PolyKernel/Dataflow/Pe.h"
#include "PolyKernel/Dataflow/Summa.h"

#include <cstdint>

namespace polykernel::dataflow {

//===----------------------------------------------------------------------===//
// CommState: the communication bottleneck classification.
//===----------------------------------------------------------------------===//

enum class CommState {
  Active,       // healthy: the fabric keeps up with the compute (normal op).
  Delayed,      // fabric-bound: wavelets queue in transit (cycles >> critical path).
  Backpressure, // stalled: SRAM over-subscription (> 100%) or fabric loss.
  Idle,         // no wavelets in flight and no computes firing.
};

// Stable lowercase name (for JSON / text reports).
const char *ToString(CommState s);

//===----------------------------------------------------------------------===//
// Metrics: the dataflow simulator metrics for one SUMMA execution.
//===----------------------------------------------------------------------===//

struct Metrics {
  // Element width the SRAM-residency model counts (fp32: the simulator's PeState
  // buffers are std::vector<float>; the on-wire wavelet payload is 16-bit fp16).
  static constexpr int kElemBytes = 4;

  // --- Grid utilization (active PEs / total PEs). ---
  int activePes = 0;
  int totalPes = 0;
  double utilization = 0.0; // activePes / totalPes (0..1); 1.0 for a full SUMMA.

  // --- Local SRAM pressure (resident tiles + double-buffers vs 48 KB). ---
  int64_t residentBytes = 0;                 // C tile + ping-pong A/B buffers.
  int64_t sramBytes = Sram::kTotalBytes;     // 48 KB per PE.
  double sramPressure = 0.0;                 // residentBytes / sramBytes (>1 = over).

  // --- Messages sent (total source wavelets) + hop distance. ---
  uint64_t aWavelets = 0;   // A broadcast wavelets (EAST seeds).
  uint64_t bWavelets = 0;   // B broadcast wavelets (SOUTH seeds).
  uint64_t messagesSent = 0; // aWavelets + bWavelets (CE fabin injections).
  double avgHopDistance = 0.0; // broadcast span = gridP - 1 (1 cycle/hop).

  // --- Communication bottleneck. ---
  CommState commState = CommState::Idle;
  const char *bottleneck = "idle";       // ToString(commState).
  const char *bottleneckCause = "none";  // why (sram_over_subscription / ...).

  // --- Critical-path cycles. ---
  uint64_t cycles = 0;             // simulated elapsed cycles (from SummaResult).
  uint64_t criticalPathCycles = 0; // analytic lower bound (broadcast + compute).

  // --- Fusion traffic reduction (wavelet count unfused vs fused). ---
  int64_t intermediateElements = 0;   // numel of the fused-away intermediate.
  uint64_t eliminatedWavelets = 0;    // 2 * intermediateElements (round-trip).
  uint64_t fusedWavelets = 0;         // matmul broadcast wavelets (fusion-invariant).
  uint64_t unfusedWavelets = 0;       // fusedWavelets + eliminatedWavelets.
  double trafficReductionPct = 0.0;   // 100 * eliminated / unfused.

  // --- Execution counters echoed from the simulation (for traceability). ---
  uint64_t panelComputes = 0;
  uint64_t droppedOffGrid = 0;
};

//===----------------------------------------------------------------------===//
// Metric computation (analytic model over the plan + simulation result).
//===----------------------------------------------------------------------===//

// Local SRAM footprint (bytes) a SUMMA plan resident on one PE occupies: the
// output-stationary C tile (tileM x tileN) plus the ping-pong A/B double-buffers
// (2 colors each). Pure function of the plan tiles.
int64_t ResidentBytes(const SummaPlan &plan);

// Classify the communication bottleneck from the metric signals already gathered
// in `m` (SRAM pressure, dropped wavelets, messages, computes, cycles vs the
// critical path). SRAM over-subscription (pressure > 1.0) and fabric loss
// (droppedOffGrid > 0) are Backpressure; an idle fabric is Idle; cycles far
// beyond the critical path are Delayed; otherwise Active.
CommState ClassifyBottleneck(const Metrics &m);

// Compute the full metrics for one SUMMA execution. `intermediateElements` is the
// numel of the intermediate the fused op eliminates (0 for an unfused matmul):
// the fused MLP keeps it in local SRAM, so it crosses the fabric as zero
// wavelets (the fusion traffic reduction).
Metrics ComputeMetrics(const SummaPlan &plan, const SummaResult &result,
                       int64_t intermediateElements = 0);

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_METRICS_H
