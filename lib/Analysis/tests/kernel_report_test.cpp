//===- kernel_report_test.cpp - Per-kernel report + fix rules ---*- C++ -*-===//
//
// PolyKernel full per-kernel report (Todo 27 / Wave 5). Locks the contract-H
// report assembly (BuildReport) and, critically, the SUGGESTED-FIX RULES — the
// heart of this task. Each rule is exercised with a HAND-CRAFTED ReportInputs
// whose occupancy / roofline outcome is derived by hand from the sm_90 models
// (see occupancy_test.cpp / roofline_test.cpp for the constant tables), so the
// asserted fix is provably the one the rule table fires, not an accident.
//
// Rule table (KernelReport.cpp DeriveFixes, in order):
//   spills>0                       -> "reduce register pressure (smaller tile/...)"
//   limiter==registers && occ<100% -> "reduce unroll/block size to lower register pressure"
//   occupancy_pct<50%              -> "increase occupancy (smaller tile / fewer registers / less smem)"
//   roofline==memory-bound         -> "wider vectorized loads / fuse to reduce traffic"
//
// Suite is named `kernel_report` so `ctest -R kernel_report` discovers it.
//
//===----------------------------------------------------------------------===//

#include "KernelReport.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <string_view>

namespace {

using polykernel::analysis::Arch;
using polykernel::analysis::Backend;
using polykernel::analysis::BuildReport;
using polykernel::analysis::Bottleneck;
using polykernel::analysis::GemmShape;
using polykernel::analysis::KernelReport;
using polykernel::analysis::Limiter;
using polykernel::analysis::Path;
using polykernel::analysis::PtxasStats;
using polykernel::analysis::ReportInputs;
using polykernel::analysis::RooflineBound;
using polykernel::analysis::ToJson;

/// True iff some suggested fix contains `needle` as a substring. The fix strings
/// are the rule engine's machine-consumable output; matching a stable token
/// asserts WHICH rule fired (not prose formatting).
bool HasFix(const KernelReport &r, std::string_view needle) {
  return std::any_of(r.suggested_fixes.begin(), r.suggested_fixes.end(),
                     [&](const std::string &f) {
                       return f.find(needle) != std::string::npos;
                     });
}

/// Build a report from (regs, smem, spills) + a GEMM shape on sm_90, 256 threads.
KernelReport MakeReport(int regs, int smem, int spill_stores, int spill_loads,
                        const GemmShape &shape) {
  ReportInputs in;
  in.kernel = "crafted";
  in.backend = Backend::cuda;
  in.arch = Arch::sm_90;
  in.ptxas = PtxasStats{.registers = regs,
                        .smem_bytes = smem,
                        .spill_stores_bytes = spill_stores,
                        .spill_loads_bytes = spill_loads};
  in.threads_per_block = 256;
  in.shape = shape;
  in.path = Path::scalar;
  return BuildReport(in);
}

// A large cube GEMM is compute-bound on sm_90 (AI ~= 1365 > ridge ~= 295), so it
// never trips the memory-bound rule — used to isolate the register/occupancy rules.
const GemmShape kComputeBound = {.m = 4096, .n = 4096, .k = 4096, .dtype_bytes = 2};
// A thin GEMM (small K) is memory-bound on sm_90 (AI ~= 15.9 < ridge ~= 295).
const GemmShape kMemoryBound = {.m = 4096, .n = 4096, .k = 16, .dtype_bytes = 2};

// --- Rule 1: high registers (register-limited, occ<100%) -> reduce unroll -----
// Hand (sm_90): warps/block = 256/32 = 8.
//   blocks_by_regs  = 65536 / (128*256) = 2   (binding)
//   blocks_by_smem  = 233472 / 32768    = 7
//   blocks_by_warps = 64 / 8            = 8 ; cap 32
//   -> blocks = 2 (registers); active_warps = 16; occ = 25%; limiter = registers.
// No spills + compute-bound shape => the ONLY register rule that fires is the
// "reduce unroll/block size" one (occupancy<50% also fires, asserted separately).
TEST(kernel_report, HighRegisterFiresReduceUnroll) {
  const auto r = MakeReport(/*regs=*/128, /*smem=*/32 * 1024, /*spill_stores=*/0,
                            /*spill_loads=*/0, kComputeBound);
  ASSERT_EQ(r.occupancy.limiter, Limiter::registers);
  ASSERT_DOUBLE_EQ(r.occupancy.occupancy_pct, 25.0);
  EXPECT_TRUE(HasFix(r, "reduce unroll"))
      << "high-register kernel must suggest reducing unroll/block size";
  EXPECT_EQ(r.bottleneck, Bottleneck::latency); // occ<50% => latency-bound.
}

// --- Rule 2: spills>0 -> reduce register pressure ----------------------------
// spills>0 short-circuits the bottleneck to latency and MUST emit the
// register-pressure remedy regardless of the occupancy/roofline outcome.
TEST(kernel_report, SpillsFireReduceRegisterPressure) {
  const auto r = MakeReport(/*regs=*/64, /*smem=*/16 * 1024, /*spill_stores=*/64,
                            /*spill_loads=*/32, kComputeBound);
  EXPECT_EQ(r.spill_stores_bytes, 64);
  EXPECT_EQ(r.spill_loads_bytes, 32);
  EXPECT_TRUE(HasFix(r, "reduce register pressure"))
      << "a spilling kernel must suggest reducing register pressure";
  EXPECT_EQ(r.bottleneck, Bottleneck::latency); // spills stall on memory latency.
}

// --- Rule 3: low occupancy (NON-register-limited) -> increase occupancy ------
// Hand (sm_90): warps/block = 8.
//   blocks_by_regs  = 65536 / (32*256) = 8
//   blocks_by_smem  = 233472 / 102400  = 2   (binding; 100 KB = 102400 B)
//   blocks_by_warps = 8 ; cap 32
//   -> blocks = 2 (smem); active_warps = 16; occ = 25%; limiter = smem.
// Because the limiter is smem (NOT registers), the "reduce unroll" rule must NOT
// fire — isolating the occupancy rule. Compute-bound shape => no memory fix.
TEST(kernel_report, LowOccupancyFiresIncreaseOccupancy) {
  const auto r = MakeReport(/*regs=*/32, /*smem=*/100 * 1024, /*spill_stores=*/0,
                            /*spill_loads=*/0, kComputeBound);
  ASSERT_EQ(r.occupancy.limiter, Limiter::smem);
  ASSERT_DOUBLE_EQ(r.occupancy.occupancy_pct, 25.0);
  EXPECT_TRUE(HasFix(r, "increase occupancy"))
      << "a low-occupancy kernel must suggest increasing occupancy";
  EXPECT_FALSE(HasFix(r, "reduce unroll"))
      << "smem-limited kernel must NOT trip the register reduce-unroll rule";
}

// --- Rule 4: memory-bound roofline -> wider vectorized loads / fusion --------
// Hand (sm_90): regs=31, smem=1024, 256 threads => occ = 100% (registers
// tie-broken before warps; see Occupancy.MatmulKernelSm90). At 100% occupancy
// neither the reduce-unroll nor the increase-occupancy rule fires, and with no
// spills the ONLY remaining rule is the memory-bound one (thin GEMM, AI ~= 15.9).
TEST(kernel_report, MemoryBoundFiresVectorize) {
  const auto r = MakeReport(/*regs=*/31, /*smem=*/1024, /*spill_stores=*/0,
                            /*spill_loads=*/0, kMemoryBound);
  ASSERT_EQ(r.roofline, RooflineBound::memory_bound);
  ASSERT_DOUBLE_EQ(r.occupancy.occupancy_pct, 100.0);
  EXPECT_TRUE(HasFix(r, "wider vectorized loads"))
      << "a memory-bound kernel must suggest wider vectorized loads / fusion";
  EXPECT_EQ(r.bottleneck, Bottleneck::memory);
  EXPECT_EQ(r.suggested_fixes.size(), 1u)
      << "at 100% occupancy with no spills, ONLY the memory rule fires";
}

// --- Clean kernel: 100% occ, compute-bound, no spills -> NO fixes ------------
// The negative case proves the rules are precise (they do not fire spuriously).
TEST(kernel_report, CleanKernelHasNoFixes) {
  const auto r = MakeReport(/*regs=*/31, /*smem=*/1024, /*spill_stores=*/0,
                            /*spill_loads=*/0, kComputeBound);
  ASSERT_DOUBLE_EQ(r.occupancy.occupancy_pct, 100.0);
  ASSERT_EQ(r.roofline, RooflineBound::compute_bound);
  EXPECT_TRUE(r.suggested_fixes.empty());
  EXPECT_EQ(r.bottleneck, Bottleneck::compute);
}

// --- Schema lock: ToJson emits the EXACT contract-H field names + enums ------
// Guards the pinned schema: nested occupancy{} + traffic{}, split spill bytes,
// `roofline` (NOT `bound_type`), and the limiter/roofline/bottleneck/path enums.
TEST(kernel_report, ToJsonEmitsContractHSchema) {
  const auto r = MakeReport(/*regs=*/31, /*smem=*/1024, /*spill_stores=*/0,
                            /*spill_loads=*/0, kMemoryBound);
  const std::string j = ToJson(r);

  // Top-level scalar fields.
  for (const char *key : {"\"kernel\":", "\"backend\":", "\"arch\":",
                          "\"registers_per_thread\":", "\"smem_per_block_bytes\":",
                          "\"spill_stores_bytes\":", "\"spill_loads_bytes\":",
                          "\"bottleneck\":", "\"path\":", "\"suggested_fixes\":"})
    EXPECT_NE(j.find(key), std::string::npos) << "missing top-level key " << key;

  // Nested occupancy{} object.
  EXPECT_NE(j.find("\"occupancy\": {"), std::string::npos);
  for (const char *key : {"\"active_warps_per_sm\":", "\"max_warps_per_sm\":",
                          "\"occupancy_pct\":", "\"limiter\":"})
    EXPECT_NE(j.find(key), std::string::npos) << "missing occupancy key " << key;

  // Nested traffic{} object.
  EXPECT_NE(j.find("\"traffic\": {"), std::string::npos);
  for (const char *key : {"\"global_read_bytes\":", "\"global_write_bytes\":",
                          "\"arithmetic_intensity_flop_per_byte\":", "\"roofline\":"})
    EXPECT_NE(j.find(key), std::string::npos) << "missing traffic key " << key;

  // Enum values are the pinned contract-H strings.
  EXPECT_NE(j.find("\"roofline\": \"memory-bound\""), std::string::npos);
  EXPECT_NE(j.find("\"limiter\": \"registers\""), std::string::npos);
  EXPECT_NE(j.find("\"bottleneck\": \"memory\""), std::string::npos);
  EXPECT_NE(j.find("\"path\": \"scalar\""), std::string::npos);

  // The schema field is `roofline`, NEVER the legacy `bound_type`.
  EXPECT_EQ(j.find("bound_type"), std::string::npos);
}

} // namespace
