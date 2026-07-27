//===- config_space_test.cpp - ConfigSpace unit tests ----------*- C++ -*-===//
//
// Asserts the bounded grid size, the pruned candidate count (~100-150 band),
// deterministic ordering, and that the pruner excludes the physically
// inadmissible configs (small tile + 8 warps; smem-over-budget stages; register
// hogs). Mirrors the lib/Analysis gtest style.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/ConfigSpace.h"

#include "gtest/gtest.h"

#include <set>

namespace {

using polykernel::autotune::ArchLimits;
using polykernel::autotune::Config;
using polykernel::autotune::ConfigSpace;

// The bounded Cartesian grid is exactly 4*4*3*2*4*4*3 = 4608 configs.
TEST(config_space, FullGridIsBounded4608) {
  const auto grid = ConfigSpace::FullGrid();
  EXPECT_EQ(grid.size(), 4608u);
}

// The pruner leaves the candidate list inside the ~100-150 acceptance band
// (141 for the default sm_90 limits). Locked exactly to guarantee determinism.
TEST(config_space, PrunedCountInBand) {
  const auto pruned = ConfigSpace::Enumerate();
  EXPECT_GE(pruned.size(), 100u);
  EXPECT_LE(pruned.size(), 150u);
  EXPECT_EQ(pruned.size(), 141u); // deterministic lock for the sm_90 defaults.
}

// Enumeration is reproducible: two calls yield identical sequences, with no
// duplicate configs.
TEST(config_space, DeterministicAndUnique) {
  const auto a = ConfigSpace::Enumerate();
  const auto b = ConfigSpace::Enumerate();
  EXPECT_EQ(a, b);
  const std::set<Config> unique(a.begin(), a.end());
  EXPECT_EQ(unique.size(), a.size());
}

// Every surviving config is admissible (empty rejection reason).
TEST(config_space, AllSurvivorsAdmissible) {
  for (const Config &c : ConfigSpace::Enumerate()) {
    EXPECT_TRUE(ConfigSpace::IsAdmissible(c))
        << "config kept but rejected: " << ConfigSpace::RejectionReason(c);
    EXPECT_TRUE(ConfigSpace::RejectionReason(c).empty());
  }
}

// R1: no survivor has BLOCK_M*BLOCK_N < 4096 with num_warps == 8.
TEST(config_space, ExcludesSmallTileWith8Warps) {
  for (const Config &c : ConfigSpace::Enumerate()) {
    const bool small = 1LL * c.block_m * c.block_n < 4096;
    EXPECT_FALSE(small && c.num_warps == 8)
        << "small tile + 8 warps survived: " << polykernel::autotune::ToString(c);
  }
}

// R2: no survivor's staged shared memory blows the per-SM budget once the
// occupancy floor (min_blocks_per_sm resident blocks) is accounted for.
TEST(config_space, ExcludesSmemOverBudgetStages) {
  const ArchLimits limits;
  for (const Config &c : ConfigSpace::Enumerate()) {
    const long long operands =
        1LL * c.block_m * c.block_k + 1LL * c.block_k * c.block_n;
    const long long smem =
        1LL * c.shared_memory_stages * operands * limits.elem_bytes;
    EXPECT_LE(smem * limits.min_blocks_per_sm, limits.smem_per_sm_bytes)
        << "smem-over-budget stages survived: "
        << polykernel::autotune::ToString(c);
  }
}

// R6: every survivor's per-thread tile equals vector_width*unroll, and R5:
// vector_width divides BLOCK_K.
TEST(config_space, VectorizationClosure) {
  for (const Config &c : ConfigSpace::Enumerate()) {
    const int threads = c.num_warps * 32;
    const long long ept = 1LL * c.block_m * c.block_n / threads;
    EXPECT_EQ(ept, 1LL * c.vector_width * c.unroll);
    EXPECT_EQ(c.block_k % c.vector_width, 0);
  }
}

// Specific inadmissible configs are rejected with a non-empty reason.
TEST(config_space, SpecificInvalidConfigsRejected) {
  // Small 16x16 tile (256 < 4096) with 8 warps -> R1.
  const Config small8{16, 16, 32, 8, 1, 1, 2};
  EXPECT_FALSE(ConfigSpace::IsAdmissible(small8));
  EXPECT_FALSE(ConfigSpace::RejectionReason(small8).empty());

  // Huge tile + deep pipeline + max unroll -> smem / register blow-up.
  const Config hog{128, 128, 128, 8, 8, 8, 4};
  EXPECT_FALSE(ConfigSpace::IsAdmissible(hog));
  EXPECT_FALSE(ConfigSpace::RejectionReason(hog).empty());

  // vector_width that does not divide BLOCK_K (3 % 2) -> R5.
  const Config badvec{64, 64, 32, 4, 3, 4, 2};
  EXPECT_FALSE(ConfigSpace::IsAdmissible(badvec));
}

// A known-good config is admitted.
TEST(config_space, KnownGoodConfigAdmitted) {
  // 64x64 tile, 8 warps (256 threads): ept = 4096/256 = 16 = 4*4 (vw*unroll),
  // BLOCK_K=32 % 4 == 0; regs = acc(64) + staging(64) = 128 (== ceiling);
  // smem = 2 stages * (64*32+32*64) * 2 B = 16384, *2 blocks = 32768 <= 233472.
  const Config good{64, 64, 32, 8, 4, 4, 2};
  EXPECT_TRUE(ConfigSpace::IsAdmissible(good))
      << ConfigSpace::RejectionReason(good);
}

} // namespace
