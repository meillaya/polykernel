//===- ptxas_parser_test.cpp - PtxasParser unit tests ----------*- C++ -*-===//
//
// Covers BOTH real ptxas -v formats (OLD + NEW), the spill/stack line, missing
// fields (gmem/cmem absent), and the explicit parse-error on garbage input.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Analysis/PtxasParser.h"

#include "gtest/gtest.h"

namespace {

using polykernel::analysis::ParsePtxasLog;

// NEW format (sm_90+): registers + barriers + smem on one line. Captured shape
// matches the matmul kernel on sm_80/sm_90 (31 regs, 1024 B smem).
TEST(PtxasParser, NewFormatMatmul) {
  const auto r = ParsePtxasLog(
      "ptxas info    : Compiling entry function '_Z12matmul_kernel...' for "
      "'sm_90'\n"
      "ptxas info    : Used 31 registers, used 1 barriers, 1024 bytes smem\n");
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->registers, 31);
  EXPECT_EQ(r.stats->smem_bytes, 1024);
  EXPECT_EQ(r.stats->spill_stores_bytes, 0);
  EXPECT_EQ(r.stats->spill_loads_bytes, 0);
}

// OLD format: registers + smem + cmem[0] on one line, plus a stack/spill line.
TEST(PtxasParser, OldFormatWithSpills) {
  const auto r = ParsePtxasLog(
      "ptxas info    : Used 64 registers, 16384 bytes smem, 32 bytes cmem[0]\n"
      "ptxas info    : Used 8 bytes stack frame, 256 bytes spill stores, "
      "128 bytes spill loads\n");
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->registers, 64);
  EXPECT_EQ(r.stats->smem_bytes, 16384);
  EXPECT_EQ(r.stats->cmem_bytes, 32);
  EXPECT_EQ(r.stats->stack_bytes, 8);
  EXPECT_EQ(r.stats->spill_stores_bytes, 256);
  EXPECT_EQ(r.stats->spill_loads_bytes, 128);
}

// Softmax-shaped NEW format: registers + smem, no spills, no gmem/cmem.
TEST(PtxasParser, NewFormatSoftmaxMissingFields) {
  const auto r = ParsePtxasLog(
      "ptxas info    : Used 28 registers, 40 bytes smem\n");
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->registers, 28);
  EXPECT_EQ(r.stats->smem_bytes, 40);
  // Missing fields are tolerated as 0 (NOT a parse error).
  EXPECT_EQ(r.stats->gmem_bytes, 0);
  EXPECT_EQ(r.stats->cmem_bytes, 0);
  EXPECT_EQ(r.stats->spill_stores_bytes, 0);
}

// gmem appears only on some archs; parse it when present.
TEST(PtxasParser, GmemField) {
  const auto r = ParsePtxasLog(
      "ptxas info    : Used 32 registers, 2048 bytes smem, 512 bytes gmem\n");
  ASSERT_TRUE(r.ok()) << r.error;
  EXPECT_EQ(r.stats->gmem_bytes, 512);
}

// Garbage (no register line) -> explicit parse error, not silent zeros.
TEST(PtxasParser, GarbageInputIsError) {
  const auto r = ParsePtxasLog("this is not a ptxas log at all\nrandom noise\n");
  EXPECT_FALSE(r.ok());
  EXPECT_FALSE(r.stats.has_value());
  EXPECT_FALSE(r.error.empty());
}

// Empty input is also garbage.
TEST(PtxasParser, EmptyInputIsError) {
  const auto r = ParsePtxasLog("");
  EXPECT_FALSE(r.ok());
  EXPECT_FALSE(r.error.empty());
}

} // namespace
