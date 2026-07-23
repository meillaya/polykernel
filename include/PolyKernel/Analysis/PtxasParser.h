//===- PtxasParser.h - Parse `ptxas -v` resource usage ----------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 11 / Wave 2). GPU-FREE: ptxas runs
// without a device. Parses the verbose stderr of `ptxas -v` into a typed
// PtxasStats. Handles BOTH ptxas output formats and tolerates fields that some
// archs omit (gmem/cmem/spills). Garbage input (no recognizable resource line)
// yields an explicit parse error - never a crash, never silent zeros.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_ANALYSIS_PTXASPARSER_H
#define POLYKERNEL_ANALYSIS_PTXASPARSER_H

#include <optional>
#include <string>
#include <string_view>

namespace polykernel::analysis {

/// Resource usage reported by `ptxas -v`. A field is 0 when ptxas did not
/// report it for this arch/kernel (e.g. gmem/cmem/spills are absent on many
/// targets) - 0 is a legitimate "not reported / none" value, NOT a parse error.
struct PtxasStats {
  int registers = 0;
  int smem_bytes = 0;
  int spill_stores_bytes = 0;
  int spill_loads_bytes = 0;
  int stack_bytes = 0;
  int gmem_bytes = 0;
  int cmem_bytes = 0;
};

/// Result of parsing a ptxas -v log. Exactly one of {stats, error} is
/// meaningful: ok() == true => stats is populated and error is empty;
/// ok() == false => stats is empty and error describes the failure.
struct PtxasParseResult {
  std::optional<PtxasStats> stats;
  std::string error;

  [[nodiscard]] bool ok() const { return stats.has_value(); }
};

/// Parse a full `ptxas -v` log (may contain many lines; non-ptxas noise is
/// ignored). Both formats are accepted:
///   OLD: "Used N registers, N bytes smem, N bytes cmem[0]"
///        "N bytes stack frame, N bytes spill stores, N bytes spill loads"
///   NEW: "Used N registers, used N barriers, N bytes smem"   (sm_90+)
/// The register line is present in BOTH formats and is the validity anchor:
/// a log without it is treated as garbage (explicit error).
[[nodiscard]] PtxasParseResult ParsePtxasLog(std::string_view log);

} // namespace polykernel::analysis

#endif // POLYKERNEL_ANALYSIS_PTXASPARSER_H
