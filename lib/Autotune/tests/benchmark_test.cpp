//===- benchmark_test.cpp - Correctness-gate + selection tests --*- C++ -*-===//
//
// Unit tests for the GPU-free core of the correctness-gated benchmarking harness
// (Todo 25 / Wave 5): PassesCorrectnessGate (the contract-C gate decision) and
// SelectBestValidated (pick the FASTEST variant among those that PASSED the gate;
// an unvalidated variant is NEVER selected even if it would have been fastest).
// These prove the core invariant - the gate, not the timing, decides the winner -
// independent of any GPU; the end-to-end HIP path is exercised by
// tests/autotune/test_bench_gate.py.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/Benchmark.h"

#include "gtest/gtest.h"

#include <optional>
#include <vector>

namespace {

using polykernel::autotune::Correctness;
using polykernel::autotune::kGateMaxRelErrLargeK;
using polykernel::autotune::kGateMaxRelErrStrict;
using polykernel::autotune::PassesCorrectnessGate;
using polykernel::autotune::SelectBestValidated;
using polykernel::autotune::VariantResult;

// A correct variant: cosine == pcc == 1.0, tiny rel err (a real kernel on a
// small shape passes the full single-op contract).
Correctness Good() { return Correctness{1.0, 1e-4, 1.0}; }

// --- PassesCorrectnessGate --------------------------------------------------

TEST(benchmark, GatePassesCorrectVariant) {
  EXPECT_TRUE(PassesCorrectnessGate(Good()));
}

TEST(benchmark, GateRejectsLowCosine) {
  // cosine just below the pinned 0.999 threshold (pcc + rel fine).
  EXPECT_FALSE(PassesCorrectnessGate(Correctness{0.9989, 1e-4, 1.0}));
}

TEST(benchmark, GateRejectsLowPcc) {
  EXPECT_FALSE(PassesCorrectnessGate(Correctness{1.0, 1e-4, 0.989}));
}

TEST(benchmark, GateRejectsHighRelErrStrict) {
  // rel err above the strict 1e-2 ceiling fails even with perfect cosine + pcc.
  EXPECT_FALSE(PassesCorrectnessGate(Correctness{1.0, 2e-2, 1.0}));
}

// The scaled-wrong class: a missing-normalization bug yields out = k*ref, which
// keeps cosine == pcc == 1.0 (both scale-invariant) but blows up max_rel_err.
// ONLY the max_rel_err term catches it - proving that term is essential to the
// gate (a cosine/pcc-only gate would wrongly pass a fast-but-wrong variant).
TEST(benchmark, GateRejectsScaledWrongViaRelErr) {
  EXPECT_FALSE(PassesCorrectnessGate(Correctness{1.0, 1.0, 1.0}));
}

// The documented large-K tiled-GEMM reduction-order class: an isolated element
// lands ~3e-2 off (cosine == pcc == 1.0). It FAILS the strict 1e-2 ceiling but
// PASSES the calibrated 5e-2 ceiling (mirrors tests/kernels/test_hip_run.py).
TEST(benchmark, GateLargeKCalibratedCeiling) {
  const Correctness large_k{1.0, 3e-2, 1.0};
  EXPECT_FALSE(PassesCorrectnessGate(large_k, kGateMaxRelErrStrict));
  EXPECT_TRUE(PassesCorrectnessGate(large_k, kGateMaxRelErrLargeK));
}

// --- SelectBestValidated ----------------------------------------------------

VariantResult Variant(const char *name, bool validated,
                      std::optional<double> time_ms) {
  VariantResult v;
  v.name = name;
  v.correctness = validated ? Good() : Correctness{0.5, 1.0, 0.0};
  v.validated = validated;
  v.time_ms = time_ms; // nullopt <=> never timed (gate failed).
  return v;
}

TEST(benchmark, SelectEmptyHasNoBest) {
  EXPECT_FALSE(SelectBestValidated({}).has_value());
}

TEST(benchmark, SelectAllRejectedHasNoBest) {
  // Every variant failed the gate (validated:false, no time) -> no winner.
  std::vector<VariantResult> r = {
      Variant("broken_a", false, std::nullopt),
      Variant("broken_b", false, std::nullopt),
  };
  EXPECT_FALSE(SelectBestValidated(r).has_value());
}

TEST(benchmark, SelectsFastestAmongValidated) {
  std::vector<VariantResult> r = {
      Variant("slow", true, 0.50),
      Variant("fast", true, 0.10),
      Variant("mid", true, 0.30),
  };
  const auto best = SelectBestValidated(r);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(*best, 1u);
  EXPECT_EQ(r[*best].name, "fast");
}

// THE CORE INVARIANT: a fast-but-wrong variant (here it WOULD be the fastest by
// far) is unvalidated, so it is NEVER selected; a slower validated variant wins.
// The gate, not the timing, decides.
TEST(benchmark, NeverSelectsFasterButUnvalidated) {
  std::vector<VariantResult> r = {
      Variant("broken_fast", false, std::nullopt), // fastest, but gate-failed.
      Variant("valid_slow", true, 0.80),
      Variant("valid_mid", true, 0.40),
  };
  const auto best = SelectBestValidated(r);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(r[*best].name, "valid_mid"); // fastest VALIDATED, not broken_fast.
  EXPECT_TRUE(r[*best].validated);
  ASSERT_TRUE(r[*best].time_ms.has_value());
}

TEST(benchmark, DeterministicLowestIndexTieBreak) {
  std::vector<VariantResult> r = {
      Variant("a", true, 0.20),
      Variant("b", true, 0.20), // equal time -> lowest index wins.
  };
  const auto best = SelectBestValidated(r);
  ASSERT_TRUE(best.has_value());
  EXPECT_EQ(*best, 0u);
}

} // namespace
