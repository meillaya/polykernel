//===- roofline_test.cpp - Roofline model unit tests -----------*- C++ -*-===//
//
// GEMM roofline. bytes = (M*K + K*N + M*N)*dtype_bytes, flops = 2*M*N*K,
// AI = flops/bytes; ridge(sm_90) = 989e12/3.35e9 ~= 295.2 FLOP/byte.
//   large GEMM (M=N=K=4096, fp16): AI = 2*4096/6 = 1365.33 >= 295 -> compute.
//   small GEMM (M=N=128, K=4096, fp16): AI ~= 63.0 < 295 -> memory.
//
//===----------------------------------------------------------------------===//

#include "Roofline.h"

#include "gtest/gtest.h"

namespace {

using polykernel::analysis::Arch;
using polykernel::analysis::ComputeRoofline;
using polykernel::analysis::GemmShape;
using polykernel::analysis::RooflineBound;

// Large square GEMM, fp16: compute-bound. Exact integer byte/flop counts lock
// the integer math; AI = 2*4096^3 / (3*4096^2*2) = 8192/6 = 1365.333.
TEST(Roofline, LargeGemmIsComputeBound) {
  const auto r = ComputeRoofline(GemmShape{4096, 4096, 4096, /*dtype=*/2},
                                 Arch::sm_90);
  EXPECT_EQ(r.bytes, (4096LL * 4096 + 4096LL * 4096 + 4096LL * 4096) * 2);
  EXPECT_EQ(r.bytes, 100663296LL);
  EXPECT_EQ(r.flops, 2LL * 4096 * 4096 * 4096);
  EXPECT_EQ(r.flops, 137438953472LL);
  EXPECT_NEAR(r.arithmetic_intensity_flop_per_byte, 1365.3333, 1e-3);
  EXPECT_EQ(r.bound, RooflineBound::compute_bound);
}

// Small M=N, large K, fp16: memory-bound. AI = 2*128*128*4096 / 2129920 ~= 63.0.
TEST(Roofline, SmallGemmIsMemoryBound) {
  const auto r = ComputeRoofline(GemmShape{128, 128, 4096, /*dtype=*/2},
                                 Arch::sm_90);
  EXPECT_EQ(r.bytes, (128LL * 4096 + 4096LL * 128 + 128LL * 128) * 2);
  EXPECT_EQ(r.bytes, 2129920LL);
  EXPECT_EQ(r.flops, 2LL * 128 * 128 * 4096);
  EXPECT_NEAR(r.arithmetic_intensity_flop_per_byte, 63.0144, 1e-3);
  EXPECT_EQ(r.bound, RooflineBound::memory_bound);
}

// The actual analyze.py demo shape (128^3, fp16) on sm_90: AI = 42.67 < 295.
TEST(Roofline, Matmul128CubeIsMemoryBound) {
  const auto r = ComputeRoofline(GemmShape{128, 128, 128, /*dtype=*/2},
                                 Arch::sm_90);
  EXPECT_EQ(r.bytes, (128LL * 128 + 128LL * 128 + 128LL * 128) * 2);
  EXPECT_EQ(r.bytes, 98304LL);
  EXPECT_EQ(r.flops, 2LL * 128 * 128 * 128);
  EXPECT_NEAR(r.arithmetic_intensity_flop_per_byte, 42.6667, 1e-3);
  EXPECT_EQ(r.bound, RooflineBound::memory_bound);
}

// A100 ridge ~= 312e12/1.555e9 ~= 200.6; the 128^3 GEMM (AI 42.67) is still
// memory-bound there, and the large GEMM (AI 1365) still compute-bound.
TEST(Roofline, Sm80RidgeClassification) {
  const auto small = ComputeRoofline(GemmShape{128, 128, 128, 2}, Arch::sm_80);
  EXPECT_EQ(small.bound, RooflineBound::memory_bound);
  const auto large = ComputeRoofline(GemmShape{4096, 4096, 4096, 2}, Arch::sm_80);
  EXPECT_EQ(large.bound, RooflineBound::compute_bound);
}

TEST(Roofline, BoundNames) {
  EXPECT_EQ(polykernel::analysis::RooflineName(RooflineBound::compute_bound),
            "compute-bound");
  EXPECT_EQ(polykernel::analysis::RooflineName(RooflineBound::memory_bound),
            "memory-bound");
}

} // namespace
