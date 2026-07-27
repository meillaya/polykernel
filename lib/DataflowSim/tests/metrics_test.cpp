//===- metrics_test.cpp - Dataflow simulator metrics -------------*- C++ -*-===//
//
// PolyKernel dataflow simulator metrics + fusion traffic reduction (Todo 38 /
// Wave 7). Suite `metrics` so `ctest -R metrics` discovers these.
//
// Asserts the metrics layer (Metrics.cpp) against HAND-COMPUTED values for a
// small 4x4 grid running an 8x8x8 matmul via the Todo 37 SUMMA engine, plus the
// SRAM-over-subscription negative (pressure > 100% flagged as the bottleneck).
//
// HAND-COMPUTATION (8x8x8 matmul, gridP=4 -> tileM=tileN=2, tileK=8, steps=1,
// panelSize = tileM*tileK = 16; RunSumma gives panelComputes = gridP^2*steps =
// 16, aWavelets = bWavelets = gridP*panelSize*steps = 64, droppedOffGrid = 0):
//   * utilization   = activePes/totalPes = 16/16 = 1.0 (SUMMA maps the full grid)
//   * residentBytes = C tile (2*2*4 = 16) + A dbuf (2*16*4 = 128) + B dbuf
//                     (2*16*4 = 128) = 272 B;  SRAM pressure = 272/49152
//   * messagesSent  = aWavelets + bWavelets = 64 + 64 = 128
//   * avgHopDistance= gridP - 1 = 3 (broadcast along width 4 = 3 hops, 1 cyc/hop)
//   * criticalPath  = (gridP-1) + steps = 3 + 1 = 4 (simulated cycles >= 4)
//   * bottleneck    = Active (pressure < 100%, nothing dropped, fabric keeps up)
//   * fusion        = eliminated intermediate 64 elems -> round-trip 128 wavelets;
//                     fused = 128 (matmul broadcasts), unfused = 128+128 = 256;
//                     reduction = 128/256 = 50%
//
// This is a GPU-free functional/cycle SIMULATOR (the cslc --out-routes /
// calculate_cycles analog), not real CSL and not Cerebras hardware.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Metrics.h"
#include "PolyKernel/Dataflow/Summa.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using polykernel::dataflow::CommState;
using polykernel::dataflow::ComputeMetrics;
using polykernel::dataflow::LowerMatmulToSumma;
using polykernel::dataflow::Metrics;
using polykernel::dataflow::ResidentBytes;
using polykernel::dataflow::RunSumma;
using polykernel::dataflow::SummaPlan;
using polykernel::dataflow::SummaResult;
using polykernel::dataflow::Sram;

constexpr int64_t kSram = Sram::kTotalBytes; // 48 KB = 49152 B.

std::vector<float> RandPM1(int n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed ? seed : 0x12345678u;
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
  }
  return v;
}

// The small 4x4 scenario: 8x8x8 matmul on gridP=4.
SummaPlan Plan4x4() { return LowerMatmulToSumma(8, 8, 8, /*gridP=*/4); }

SummaResult Run4x4(const SummaPlan &plan) {
  const auto a = RandPM1(plan.m * plan.k, 1);
  const auto b = RandPM1(plan.k * plan.n, 2);
  return RunSumma(plan, a, b);
}

// Grid utilization, SRAM pressure, messages sent, and average hop distance match
// the hand-computed 4x4 values.
TEST(metrics, HandComputed4x4UtilizationSramMessagesHops) {
  const SummaPlan plan = Plan4x4();
  ASSERT_EQ(plan.gridP, 4);
  ASSERT_EQ(plan.tileM, 2);
  ASSERT_EQ(plan.tileK, 8);
  ASSERT_EQ(plan.steps, 1);

  const Metrics m = ComputeMetrics(plan, Run4x4(plan));

  // Utilization: all 16 PEs active (SUMMA maps the full 4x4 grid).
  EXPECT_EQ(m.totalPes, 16);
  EXPECT_EQ(m.activePes, 16);
  EXPECT_DOUBLE_EQ(m.utilization, 1.0);

  // SRAM pressure: resident 272 B vs 48 KB.
  EXPECT_EQ(m.residentBytes, 272); // 16 (C) + 128 (A dbuf) + 128 (B dbuf).
  EXPECT_EQ(m.sramBytes, kSram);
  EXPECT_DOUBLE_EQ(m.sramPressure, 272.0 / double(kSram));
  EXPECT_LT(m.sramPressure, 1.0); // fits comfortably.

  // Messages sent: 64 A + 64 B source wavelets.
  EXPECT_EQ(m.aWavelets, 64u);
  EXPECT_EQ(m.bWavelets, 64u);
  EXPECT_EQ(m.messagesSent, 128u);

  // Average hop distance: broadcast along width 4 = 3 hops.
  EXPECT_DOUBLE_EQ(m.avgHopDistance, 3.0);
}

// Critical-path cycles (analytic lower bound) and the communication bottleneck
// classification match the hand-computed 4x4 values.
TEST(metrics, HandComputed4x4CriticalPathAndBottleneck) {
  const SummaPlan plan = Plan4x4();
  const SummaResult res = Run4x4(plan);
  const Metrics m = ComputeMetrics(plan, res);

  // Critical path = (gridP-1) + steps = 3 + 1 = 4; the simulation cannot beat it.
  EXPECT_EQ(m.criticalPathCycles, 4u);
  EXPECT_GE(m.cycles, m.criticalPathCycles);
  EXPECT_EQ(m.cycles, res.cycles);

  // Healthy 4x4 run: fabric keeps up -> Active, no over-subscription / no drops.
  EXPECT_EQ(m.panelComputes, 16u);
  EXPECT_EQ(m.droppedOffGrid, 0u);
  EXPECT_EQ(m.commState, CommState::Active);
  EXPECT_STREQ(m.bottleneck, "active");
  EXPECT_STREQ(m.bottleneckCause, "none");
}

// Fusion traffic reduction: a fused intermediate of 64 elements removes a 128-
// wavelet round-trip -> 50% fewer fabric wavelets (hand-computed).
TEST(metrics, HandComputed4x4FusionTrafficReduction) {
  const SummaPlan plan = Plan4x4();
  const Metrics m = ComputeMetrics(plan, Run4x4(plan), /*intermediateElements=*/64);

  EXPECT_EQ(m.intermediateElements, 64);
  EXPECT_EQ(m.eliminatedWavelets, 128u); // 2 * 64 (write + read round-trip).
  EXPECT_EQ(m.fusedWavelets, 128u);      // matmul broadcasts (fusion-invariant).
  EXPECT_EQ(m.unfusedWavelets, 256u);    // 128 + 128.
  EXPECT_DOUBLE_EQ(m.trafficReductionPct, 50.0);

  // With NO eliminated intermediate (an unfused matmul) the reduction is zero.
  const Metrics plain = ComputeMetrics(plan, Run4x4(plan), /*intermediateElements=*/0);
  EXPECT_EQ(plain.eliminatedWavelets, 0u);
  EXPECT_DOUBLE_EQ(plain.trafficReductionPct, 0.0);
}

// ResidentBytes is a pure function of the plan tiles (no simulation needed).
TEST(metrics, ResidentBytesIsPureFunctionOfPlan) {
  const SummaPlan plan = Plan4x4();
  EXPECT_EQ(ResidentBytes(plan), 272); // 2*2*4 + 2*16*4 + 2*16*4.
  EXPECT_EQ(ComputeMetrics(plan, SummaResult{}).residentBytes, 272);
}

// NEGATIVE: a config whose tiles exceed the 48 KB local SRAM reports pressure >
// 100% and is flagged as the communication bottleneck (Backpressure). The C tile
// alone (128*128*4 = 64 KB) over-subscribes the 48 KB SRAM.
//
// Hand-computation (512x512x1 matmul, gridP=4 -> tileM=tileN=128, tileK=1,
// steps=1, panelSize=128): residentBytes = 128*128*4 (C = 65536) + 2*128*1*4
// (A dbuf = 1024) + 2*1*128*4 (B dbuf = 1024) = 67584 B; pressure = 67584/49152
// = 1.375 (137.5%) > 100% -> Backpressure / sram_over_subscription.
TEST(metrics, SramOverSubscriptionFlaggedAsBottleneck) {
  const SummaPlan plan = LowerMatmulToSumma(512, 512, 1, /*gridP=*/4, /*tileK=*/1);
  ASSERT_EQ(plan.tileM, 128);
  ASSERT_EQ(plan.tileN, 128);
  ASSERT_EQ(plan.tileK, 1);

  const SummaResult res = Run4x4(plan); // simulate the over-subscribed config.
  const Metrics m = ComputeMetrics(plan, res);

  std::printf("[negative] SRAM over-subscription: resident=%lld B vs %d B SRAM "
              "-> pressure=%.1f%% (> 100%%) -> bottleneck=%s (cause=%s)\n",
              static_cast<long long>(m.residentBytes), static_cast<int>(m.sramBytes),
              m.sramPressure * 100.0, m.bottleneck, m.bottleneckCause);

  EXPECT_EQ(m.residentBytes, 67584);
  EXPECT_DOUBLE_EQ(m.sramPressure, 67584.0 / double(kSram)); // 1.375.
  EXPECT_GT(m.sramPressure, 1.0); // over-subscribed (> 100%).

  // The over-subscription is flagged as THE bottleneck.
  EXPECT_EQ(m.commState, CommState::Backpressure);
  EXPECT_STREQ(m.bottleneck, "backpressure");
  EXPECT_STREQ(m.bottleneckCause, "sram_over_subscription");
}

} // namespace
