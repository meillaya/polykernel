//===- occupancy_test.cpp - Occupancy model unit tests ---------*- C++ -*-===//
//
// HAND-COMPUTED cases. For each, the expected resident-block count is derived
// by hand from the four limits (regs, smem, warps, blocks) and asserted, along
// with the active-warp count, occupancy %, and the binding limiter.
//
// sm_90 (H100): smem/SM = 228 KB = 233472 B; regs/SM = 65536; warps/SM = 64;
//               blocks/SM = 32; warp = 32.
//
//===----------------------------------------------------------------------===//

#include "Occupancy.h"

#include "gtest/gtest.h"

namespace {

using polykernel::analysis::Arch;
using polykernel::analysis::ComputeOccupancy;
using polykernel::analysis::Limiter;

// Hand computation (sm_90): warps/block = 256/32 = 8.
//   blocks_by_regs  = 65536 / (128*256) = 65536/32768 = 2
//   blocks_by_smem  = 233472 / 32768    = 7
//   blocks_by_warps = 64 / 8            = 8
//   blocks_by_limit = 32
//   blocks = min = 2 (registers); active_warps = 2*8 = 16; occ = 16/64 = 25%.
TEST(Occupancy, HandComputedRegisterLimitedSm90) {
  const auto occ = ComputeOccupancy(/*regs=*/128, /*smem=*/32 * 1024,
                                    /*threads=*/256, Arch::sm_90);
  EXPECT_EQ(occ.active_warps_per_sm, 16);
  EXPECT_EQ(occ.max_warps_per_sm, 64);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 25.0);
  EXPECT_EQ(occ.limiter, Limiter::registers);
}

// Same register-limited shape on sm_80 (A100, smem/SM = 164 KB = 167936 B):
//   blocks_by_regs  = 2 ; blocks_by_smem = 167936/32768 = 5 ; warps = 8 ; cap 32
//   blocks = 2 (registers); active_warps = 16; occ = 25%.
TEST(Occupancy, RegisterLimitedSm80) {
  const auto occ = ComputeOccupancy(128, 32 * 1024, 256, Arch::sm_80);
  EXPECT_EQ(occ.active_warps_per_sm, 16);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 25.0);
  EXPECT_EQ(occ.limiter, Limiter::registers);
}

// Smem-limited (sm_90): warps/block = 8.
//   blocks_by_regs  = 65536 / (32*256) = 8
//   blocks_by_smem  = 233472 / 102400  = 2   (100 KB = 102400 B)
//   blocks_by_warps = 8 ; cap 32
//   blocks = min = 2 (smem); active_warps = 16; occ = 25%.
TEST(Occupancy, SmemLimitedSm90) {
  const auto occ = ComputeOccupancy(/*regs=*/32, /*smem=*/100 * 1024,
                                    /*threads=*/256, Arch::sm_90);
  EXPECT_EQ(occ.active_warps_per_sm, 16);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 25.0);
  EXPECT_EQ(occ.limiter, Limiter::smem);
}

// Warp-limited (sm_90): threads=1024 -> warps/block = 32.
//   blocks_by_regs  = 65536 / (16*1024) = 4
//   blocks_by_smem  = unlimited (0 smem)
//   blocks_by_warps = 64 / 32           = 2
//   blocks = min = 2 (warps); active_warps = 2*32 = 64; occ = 100%.
TEST(Occupancy, WarpLimitedSm90) {
  const auto occ = ComputeOccupancy(/*regs=*/16, /*smem=*/0,
                                    /*threads=*/1024, Arch::sm_90);
  EXPECT_EQ(occ.active_warps_per_sm, 64);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 100.0);
  EXPECT_EQ(occ.limiter, Limiter::warps);
}

// Block-limited (sm_90): threads=32 -> warps/block = 1.
//   blocks_by_regs  = 65536 / (16*32) = 128
//   blocks_by_smem  = unlimited ; blocks_by_warps = 64 ; cap = 32
//   blocks = min = 32 (blocks); active_warps = 32*1 = 32; occ = 50%.
TEST(Occupancy, BlockLimitedSm90) {
  const auto occ = ComputeOccupancy(/*regs=*/16, /*smem=*/0,
                                    /*threads=*/32, Arch::sm_90);
  EXPECT_EQ(occ.active_warps_per_sm, 32);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 50.0);
  EXPECT_EQ(occ.limiter, Limiter::blocks);
}

// The matmul kernel (31 regs, 1024 B smem, 256 threads) on sm_90:
//   blocks_by_regs  = 65536 / (31*256) = 65536/7936 = 8
//   blocks_by_smem  = 233472 / 1024    = 228
//   blocks_by_warps = 64 / 8           = 8 ; cap 32
//   blocks = 8 (registers tie-broken before warps); active_warps = 64; occ 100%.
TEST(Occupancy, MatmulKernelSm90) {
  const auto occ = ComputeOccupancy(31, 1024, 256, Arch::sm_90);
  EXPECT_EQ(occ.active_warps_per_sm, 64);
  EXPECT_DOUBLE_EQ(occ.occupancy_pct, 100.0);
  EXPECT_EQ(occ.limiter, Limiter::registers);
}

TEST(Occupancy, ArchParseAndNames) {
  EXPECT_EQ(polykernel::analysis::ParseArch("sm_80"), Arch::sm_80);
  EXPECT_EQ(polykernel::analysis::ParseArch("sm_90"), Arch::sm_90);
  EXPECT_FALSE(polykernel::analysis::ParseArch("gfx1101").has_value());
  EXPECT_EQ(polykernel::analysis::ArchName(Arch::sm_80), "sm_80");
  EXPECT_EQ(polykernel::analysis::LimiterName(Limiter::smem), "smem");
}

} // namespace
