//===- KernelReport.h - Assemble the per-kernel report ----------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 11 / Wave 2). Assembles the AUTHORITATIVE
// per-kernel report (Pinned contract H) from the parsed ptxas stats + occupancy
// model + roofline model, derives the bottleneck + suggested fixes, and emits
// JSON + human-readable text. Field names + enum string values match contract H
// EXACTLY (see docs: limiter/roofline/bottleneck/path enums).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_ANALYSIS_KERNELREPORT_H
#define POLYKERNEL_ANALYSIS_KERNELREPORT_H

#include "Occupancy.h"
#include "Roofline.h"
#include "PolyKernel/Analysis/PtxasParser.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace polykernel::analysis {

/// Target backend. Enum string values pinned by contract H.
enum class Backend { cuda, hip };
[[nodiscard]] std::string_view BackendName(Backend backend);

/// Performance bottleneck classification. Enum values pinned by contract H.
enum class Bottleneck { compute, memory, latency };
[[nodiscard]] std::string_view BottleneckName(Bottleneck bottleneck);

/// Kernel implementation path. Enum values pinned by contract H.
enum class Path { scalar, wmma, mma };
[[nodiscard]] std::string_view PathName(Path path);

/// The full per-kernel report (contract H schema). `occupancy` and the four
/// `traffic` fields are the nested objects in the schema; they are flattened
/// here and re-nested on JSON emission.
struct KernelReport {
  std::string kernel;
  Backend backend = Backend::cuda;
  Arch arch = Arch::sm_90;
  int registers_per_thread = 0;
  int smem_per_block_bytes = 0;
  int spill_stores_bytes = 0;
  int spill_loads_bytes = 0;
  Occupancy occupancy;
  std::int64_t global_read_bytes = 0;
  std::int64_t global_write_bytes = 0;
  double arithmetic_intensity_flop_per_byte = 0.0;
  RooflineBound roofline = RooflineBound::memory_bound;
  Bottleneck bottleneck = Bottleneck::latency;
  std::vector<std::string> suggested_fixes;
  Path path = Path::scalar;
};

/// Inputs to BuildReport: identity (kernel/backend/arch/path), the parsed ptxas
/// resource usage, the launch block size (for occupancy), and the GEMM shape
/// (for the traffic model).
struct ReportInputs {
  std::string kernel;
  Backend backend = Backend::cuda;
  Arch arch = Arch::sm_90;
  PtxasStats ptxas;
  int threads_per_block = 256;
  GemmShape shape;
  Path path = Path::scalar;
};

/// Assemble the report: occupancy from (regs, smem, threads, arch); traffic +
/// roofline from the GEMM shape; bottleneck + suggested fixes from rule tables.
[[nodiscard]] KernelReport BuildReport(const ReportInputs &inputs);

/// Serialize to JSON (contract H field names + enum values, 2-space indent).
[[nodiscard]] std::string ToJson(const KernelReport &report);

/// Serialize to a human-readable multi-line text summary.
[[nodiscard]] std::string ToText(const KernelReport &report);

} // namespace polykernel::analysis

#endif // POLYKERNEL_ANALYSIS_KERNELREPORT_H
