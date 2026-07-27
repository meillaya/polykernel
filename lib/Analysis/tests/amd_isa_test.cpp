//===- amd_isa_test.cpp - AmdIsaAnalyzer unit tests ------------*- C++ -*-===//
//
// Covers: real captured matmul fixture (VGPR/SGPR/LDS/scratch), crafted spill
// fixture (scratch_store/load + private_segment > 0), hand-computed gfx1101
// occupancy (wave-limited, VGPR-limited, LDS-limited), spill detection, and
// the explicit parse-error on garbage input. Suite named `amd_isa` so
// `ctest -R amd_isa` discovers it (mirrors Todo 24's naming convention).
//
// gfx1101 per-CU limits (verified via rocminfo on RX 7800 XT):
//   VGPR file = 3072 (1536/SIMD × 2), SGPR file = 3200 (1600/SIMD × 2),
//   LDS = 64 KiB = 65536 B (GROUP pool), max waves = 32, wave32.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Analysis/AmdIsaAnalyzer.h"

#include "gtest/gtest.h"

#include <fstream>
#include <sstream>
#include <string>

#ifndef AMD_FIXTURE_DIR
#define AMD_FIXTURE_DIR "."
#endif

namespace {

using polykernel::analysis::AmdLimiter;
using polykernel::analysis::ComputeGfx1101Occupancy;
using polykernel::analysis::HasSpills;
using polykernel::analysis::ParseAmdIsa;

std::string ReadFixture(const char *name) {
  std::ifstream f(std::string(AMD_FIXTURE_DIR) + "/" + name);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// --- Real captured fixture: matmul_gfx1101.s (hipcc --save-temps) ---

TEST(amd_isa, MatmulFixtureParses) {
  const auto r = ParseAmdIsa(ReadFixture("matmul_gfx1101.s"));
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->vgpr, 31);
  EXPECT_EQ(r.stats->sgpr, 16);
  EXPECT_EQ(r.stats->lds_bytes, 1024);
  EXPECT_EQ(r.stats->scratch_bytes, 0);
  EXPECT_EQ(r.stats->scratch_store_count, 0);
  EXPECT_EQ(r.stats->scratch_load_count, 0);
}

TEST(amd_isa, MatmulFixtureNoSpills) {
  const auto r = ParseAmdIsa(ReadFixture("matmul_gfx1101.s"));
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_FALSE(HasSpills(*r.stats));
}

// --- Crafted spill fixture: spill_gfx1101.s ---

TEST(amd_isa, SpillFixtureParses) {
  const auto r = ParseAmdIsa(ReadFixture("spill_gfx1101.s"));
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->vgpr, 20);
  EXPECT_EQ(r.stats->sgpr, 16);
  EXPECT_EQ(r.stats->lds_bytes, 2048);
  EXPECT_EQ(r.stats->scratch_bytes, 16);
  EXPECT_EQ(r.stats->scratch_store_count, 4);
  EXPECT_EQ(r.stats->scratch_load_count, 2);
}

TEST(amd_isa, SpillFixtureDetected) {
  const auto r = ParseAmdIsa(ReadFixture("spill_gfx1101.s"));
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_TRUE(HasSpills(*r.stats));
}

// --- Inline minimal assembly (no file dependency) ---

TEST(amd_isa, InlineMinimal) {
  const auto r = ParseAmdIsa(
      "\t.amdhsa_kernel _Z4testv\n"
      "\t\t.amdhsa_group_segment_fixed_size 512\n"
      "\t\t.amdhsa_private_segment_fixed_size 0\n"
      "\t\t.amdhsa_next_free_vgpr 24\n"
      "\t\t.amdhsa_next_free_sgpr 12\n"
      "\t.end_amdhsa_kernel\n");
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->vgpr, 24);
  EXPECT_EQ(r.stats->sgpr, 12);
  EXPECT_EQ(r.stats->lds_bytes, 512);
  EXPECT_EQ(r.stats->scratch_bytes, 0);
}

// --- Garbage / empty input -> explicit error ---

TEST(amd_isa, GarbageInputIsError) {
  const auto r = ParseAmdIsa("this is not AMDGPU assembly\nrandom noise\n");
  EXPECT_FALSE(r.ok());
  EXPECT_FALSE(r.stats.has_value());
  EXPECT_FALSE(r.error.empty());
}

TEST(amd_isa, EmptyInputIsError) {
  const auto r = ParseAmdIsa("");
  EXPECT_FALSE(r.ok());
  EXPECT_FALSE(r.error.empty());
}

// --- Hand-computed gfx1101 occupancy ---

// Matmul (vgpr=31, sgpr=16, lds=1024, threads=256 -> waves/block=8):
//   wg_by_vgpr  = 3072 / (31*8)  = 12
//   wg_by_sgpr  = 3200 / (16*8)  = 25
//   wg_by_lds   = 65536 / 1024   = 64
//   wg_by_waves = 32 / 8         = 4
//   min = 4 (waves); active = 4*8 = 32; occ = 32/32 = 100%.
TEST(amd_isa, OccupancyWaveLimited) {
  const auto occ = ComputeGfx1101Occupancy(/*vgpr=*/31, /*sgpr=*/16,
                                           /*lds=*/1024, /*threads=*/256);
  EXPECT_EQ(occ.active_waves_per_cu, 32);
  EXPECT_EQ(occ.max_waves_per_cu, 32);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 100.0);
  EXPECT_EQ(occ.limiter, AmdLimiter::waves);
}

// VGPR-limited (vgpr=128, sgpr=16, lds=1024, threads=256 -> waves/block=8):
//   wg_by_vgpr  = 3072 / (128*8) = 3
//   wg_by_sgpr  = 25 ; wg_by_lds = 64 ; wg_by_waves = 4
//   min = 3 (vgpr); active = 3*8 = 24; occ = 24/32 = 75%.
TEST(amd_isa, OccupancyVgprLimited) {
  const auto occ = ComputeGfx1101Occupancy(128, 16, 1024, 256);
  EXPECT_EQ(occ.active_waves_per_cu, 24);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 75.0);
  EXPECT_EQ(occ.limiter, AmdLimiter::vgpr);
}

// LDS-limited (vgpr=16, sgpr=16, lds=32768, threads=256 -> waves/block=8):
//   wg_by_vgpr = 24 ; wg_by_sgpr = 25
//   wg_by_lds  = 65536 / 32768 = 2 ; wg_by_waves = 4
//   min = 2 (lds); active = 2*8 = 16; occ = 16/32 = 50%.
TEST(amd_isa, OccupancyLdsLimited) {
  const auto occ = ComputeGfx1101Occupancy(16, 16, 32768, 256);
  EXPECT_EQ(occ.active_waves_per_cu, 16);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 50.0);
  EXPECT_EQ(occ.limiter, AmdLimiter::lds);
}

// SGPR-limited (vgpr=16, sgpr=200, lds=0, threads=256 -> waves/block=8):
//   wg_by_vgpr = 24 ; wg_by_sgpr = 3200/(200*8) = 2
//   wg_by_lds = unlimited ; wg_by_waves = 4
//   min = 2 (sgpr); active = 16; occ = 50%.
TEST(amd_isa, OccupancySgprLimited) {
  const auto occ = ComputeGfx1101Occupancy(16, 200, 0, 256);
  EXPECT_EQ(occ.active_waves_per_cu, 16);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 50.0);
  EXPECT_EQ(occ.limiter, AmdLimiter::sgpr);
}

// --- Limiter name + contract-H mapping ---

TEST(amd_isa, LimiterNames) {
  EXPECT_EQ(polykernel::analysis::AmdLimiterName(AmdLimiter::vgpr), "vgpr");
  EXPECT_EQ(polykernel::analysis::AmdLimiterName(AmdLimiter::sgpr), "sgpr");
  EXPECT_EQ(polykernel::analysis::AmdLimiterName(AmdLimiter::lds), "lds");
  EXPECT_EQ(polykernel::analysis::AmdLimiterName(AmdLimiter::waves), "waves");
}

TEST(amd_isa, LimiterContractHMapping) {
  using polykernel::analysis::AmdLimiterToContractH;
  EXPECT_EQ(AmdLimiterToContractH(AmdLimiter::vgpr), "registers");
  EXPECT_EQ(AmdLimiterToContractH(AmdLimiter::sgpr), "blocks");
  EXPECT_EQ(AmdLimiterToContractH(AmdLimiter::lds), "smem");
  EXPECT_EQ(AmdLimiterToContractH(AmdLimiter::waves), "warps");
}

} // namespace
