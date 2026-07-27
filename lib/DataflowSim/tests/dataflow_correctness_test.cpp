//===- dataflow_correctness_test.cpp - SUMMA golden correctness -*- C++ -*-===//
//
// PolyKernel SUMMA matmul mapping + lowering (Todo 37 / Wave 7). Suite
// `dataflow_correctness` so `ctest -R 'summa|dataflow_correctness'` discovers
// these.
//
// The simulator FUNCTIONALLY EXECUTES the SUMMA tile program for a representative
// matmul (32x32x32 on a 4x4 grid, 8x8 output tiles, 8-wide K-slab, 4 steps): it
// routes the A panels EAST and B panels SOUTH as wavelets, runs the per-PE
// dataflow rendezvous, and accumulates the output-stationary C tiles via the CE
// FMAC. The resulting C is validated against an fp32 reference matmul under the
// project's golden contract C (cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99;
// same thresholds as tests/golden/metrics.py).
//
// NEGATIVE: breaking the rendezvous (fire the compute on the A panel ALONE)
// produces WRONG output that FAILS the golden contract - proving the correctness
// test catches dataflow bugs.
//
// This is a GPU-free functional SIMULATOR, not real CSL and not Cerebras HW.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Summa.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using polykernel::dataflow::LowerMatmulToSumma;
using polykernel::dataflow::RunSumma;
using polykernel::dataflow::SummaPlan;

//===----------------------------------------------------------------------===//
// Golden contract C (mirrors tests/golden/metrics.py thresholds exactly).
//===----------------------------------------------------------------------===//

constexpr double kCosineThreshold = 0.999;
constexpr double kMaxRelErrThreshold = 1e-2;
constexpr double kPccThreshold = 0.99;
constexpr double kRelErrEps = 1e-6;

double Cosine(const std::vector<float> &a, const std::vector<float> &ref) {
  double dot = 0, na = 0, nr = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += double(a[i]) * ref[i];
    na += double(a[i]) * a[i];
    nr += double(ref[i]) * ref[i];
  }
  const double denom = std::sqrt(na) * std::sqrt(nr);
  if (denom == 0.0)
    return (na == 0.0 && nr == 0.0) ? 1.0 : 0.0;
  return dot / denom;
}

double MaxRelErr(const std::vector<float> &a, const std::vector<float> &ref) {
  double mx = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    mx = std::max(mx, std::abs(double(a[i]) - ref[i]) /
                          (std::abs(double(ref[i])) + kRelErrEps));
  return mx;
}

double Pcc(const std::vector<float> &a, const std::vector<float> &ref) {
  const std::size_t n = a.size();
  double ma = 0, mr = 0;
  for (std::size_t i = 0; i < n; ++i) {
    ma += a[i];
    mr += ref[i];
  }
  ma /= n;
  mr /= n;
  double sxy = 0, sxx = 0, syy = 0;
  bool equal = true;
  for (std::size_t i = 0; i < n; ++i) {
    const double xc = double(a[i]) - ma;
    const double yc = double(ref[i]) - mr;
    sxy += xc * yc;
    sxx += xc * xc;
    syy += yc * yc;
    if (a[i] != ref[i])
      equal = false;
  }
  const double denom = std::sqrt(sxx) * std::sqrt(syy);
  if (denom == 0.0)
    return equal ? 1.0 : 0.0;
  return sxy / denom;
}

struct ContractC {
  double cosine, rel, pcc;
  bool pass;
};

ContractC CheckContractC(const std::vector<float> &a,
                         const std::vector<float> &ref) {
  ContractC c;
  c.cosine = Cosine(a, ref);
  c.rel = MaxRelErr(a, ref);
  c.pcc = Pcc(a, ref);
  c.pass = (c.cosine >= kCosineThreshold) && (c.rel <= kMaxRelErrThreshold) &&
           (c.pcc >= kPccThreshold);
  return c;
}

//===----------------------------------------------------------------------===//
// Inputs + reference.
//===----------------------------------------------------------------------===//

std::vector<float> RandPM1(int n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed ? seed : 0x9E3779B9u;
  for (int i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = ((s >> 16) & 1u) ? 1.0f : -1.0f;
  }
  return v;
}

std::vector<float> RefMatmul(const std::vector<float> &a,
                             const std::vector<float> &b, int m, int k, int n) {
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.f);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float s = 0.f;
      for (int kk = 0; kk < k; ++kk)
        s += a[static_cast<std::size_t>(i) * k + kk] *
             b[static_cast<std::size_t>(kk) * n + j];
      c[static_cast<std::size_t>(i) * n + j] = s;
    }
  return c;
}

//===----------------------------------------------------------------------===//
// Tests.
//===----------------------------------------------------------------------===//

// KEY: the simulator-executed SUMMA matmul matches the golden reference under
// contract C, and the schedule really executed (routing + rendezvous + FMAC).
TEST(dataflow_correctness, SimulatorExecutedMatmulMatchesGolden) {
  // Representative matmul: 32x32x32 on a 4x4 grid (8x8 output tiles, 8-wide
  // K-slab, 4 SUMMA steps). Inputs are +/-1 (exactly representable in fp16), so
  // the CE FMAC accumulate is bit-exact vs the fp32 reference.
  const int m = 32, n = 32, k = 32;
  const SummaPlan plan = LowerMatmulToSumma(m, n, k, /*gridP=*/4);
  ASSERT_EQ(plan.gridP, 4);
  ASSERT_EQ(plan.tileM, 8);
  ASSERT_EQ(plan.steps * plan.tileK, k);

  const auto a = RandPM1(m * k, 11);
  const auto b = RandPM1(k * n, 22);
  const auto ref = RefMatmul(a, b, m, k, n);

  const auto res = RunSumma(plan, a, b);
  ASSERT_EQ(res.c.size(), ref.size());

  const ContractC cc = CheckContractC(res.c, ref);
  std::printf("[golden] 32x32x32 SUMMA: cosine=%.6f max_rel_err=%.6e pcc=%.6f "
              "-> %s\n",
              cc.cosine, cc.rel, cc.pcc, cc.pass ? "PASS" : "FAIL");
  EXPECT_GE(cc.cosine, kCosineThreshold);
  EXPECT_LE(cc.rel, kMaxRelErrThreshold);
  EXPECT_GE(cc.pcc, kPccThreshold);
  EXPECT_TRUE(cc.pass);

  // The simulator FUNCTIONALLY EXECUTED the tile program: one rendezvous
  // panel-compute per PE per step, A broadcast east + B broadcast south, nothing
  // dropped off-grid.
  const int panelSize = plan.tileM * plan.tileK;
  EXPECT_EQ(res.panelComputes,
            static_cast<uint64_t>(plan.gridP) * plan.gridP * plan.steps);
  EXPECT_EQ(res.aWavelets,
            static_cast<uint64_t>(plan.gridP) * panelSize * plan.steps);
  EXPECT_EQ(res.bWavelets,
            static_cast<uint64_t>(plan.gridP) * panelSize * plan.steps);
  EXPECT_EQ(res.droppedOffGrid, 0u);
  EXPECT_GT(res.cycles, 0u);
}

// The output-stationary FMAC accumulate is bit-exact for integer inputs (the
// routing + rendezvous lose nothing): sim C == fp32 reference, element for
// element, at a second size.
TEST(dataflow_correctness, OutputStationaryAccumulateIsExact) {
  const int m = 16, n = 16, k = 16;
  const SummaPlan plan = LowerMatmulToSumma(m, n, k, /*gridP=*/4);
  const auto a = RandPM1(m * k, 7);
  const auto b = RandPM1(k * n, 8);
  const auto ref = RefMatmul(a, b, m, k, n);

  const auto res = RunSumma(plan, a, b);
  ASSERT_EQ(res.c.size(), ref.size());
  for (std::size_t i = 0; i < ref.size(); ++i)
    EXPECT_FLOAT_EQ(res.c[i], ref[i]) << "element " << i;
}

// NEGATIVE: break the rendezvous (fire the compute on the A panel ALONE, skipping
// the B broadcast) -> the simulator output is WRONG and FAILS the golden contract
// C. Proves the correctness test catches dataflow bugs.
TEST(dataflow_correctness, BrokenRendezvousFailsGolden) {
  const int m = 32, n = 32, k = 32;
  const SummaPlan plan = LowerMatmulToSumma(m, n, k, /*gridP=*/4);
  const auto a = RandPM1(m * k, 11);
  const auto b = RandPM1(k * n, 22);
  const auto ref = RefMatmul(a, b, m, k, n);

  const auto res = RunSumma(plan, a, b, /*breakRendezvous=*/true);
  ASSERT_EQ(res.c.size(), ref.size());

  // Firing compute on the A panel alone accumulates against an EMPTY B panel ->
  // C is all zeros (the B broadcast never happened).
  EXPECT_EQ(res.bWavelets, 0u); // no B broadcast: compute did not wait for B.
  double cmax = 0;
  for (const float v : res.c)
    cmax = std::max(cmax, std::abs(double(v)));
  EXPECT_EQ(cmax, 0.0); // output is (wrongly) all zeros.

  const ContractC cc = CheckContractC(res.c, ref);
  std::printf("[negative] broken rendezvous: cosine=%.6f max_rel_err=%.6e "
              "pcc=%.6f -> %s (expected FAIL)\n",
              cc.cosine, cc.rel, cc.pcc, cc.pass ? "PASS" : "FAIL");
  // The broken output must FAIL the golden contract on every metric.
  EXPECT_FALSE(cc.pass);
  EXPECT_LT(cc.cosine, kCosineThreshold);
  EXPECT_GT(cc.rel, kMaxRelErrThreshold);
  EXPECT_LT(cc.pcc, kPccThreshold);
}

} // namespace
