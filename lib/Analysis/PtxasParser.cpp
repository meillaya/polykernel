//===- PtxasParser.cpp - Parse `ptxas -v` resource usage --------*- C++ -*-===//
//
// Implementation of ParsePtxasLog. Each resource field is extracted with an
// independent regex (mirroring tools/polykernel-bench/nvcc_driver.py), so a
// field missing on some arch simply stays 0. The register line anchors
// validity: it appears in every real ptxas -v log (OLD + NEW format), so its
// absence means the input is not a ptxas log -> explicit error.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Analysis/PtxasParser.h"

#include <regex>
#include <string>

namespace polykernel::analysis {

namespace {

/// Search `log` for `pattern` (one capture group) and return the captured int,
/// or `fallback` if the pattern is absent. Used for every optional field.
int GrabField(const std::string &log, const char *pattern, int fallback = 0) {
  const std::regex re(pattern);
  std::smatch m;
  if (std::regex_search(log, m, re))
    return std::stoi(m[1].str());
  return fallback;
}

} // namespace

PtxasParseResult ParsePtxasLog(std::string_view log) {
  const std::string text(log);

  // Registers are the validity anchor: present in BOTH the OLD format
  // ("Used N registers, ...") and the NEW format ("Used N registers, used
  // N barriers, ..."). No register line => not a ptxas -v log => garbage.
  const std::regex registers_re(R"((\d+) registers)");
  std::smatch m;
  if (!std::regex_search(text, m, registers_re))
    return PtxasParseResult{
        std::nullopt,
        "parse error: no 'N registers' line found; input is not a ptxas -v log"};

  PtxasStats stats;
  stats.registers = std::stoi(m[1].str());
  stats.smem_bytes = GrabField(text, R"((\d+) bytes smem)");
  stats.spill_stores_bytes = GrabField(text, R"((\d+) bytes spill stores)");
  stats.spill_loads_bytes = GrabField(text, R"((\d+) bytes spill loads)");
  stats.stack_bytes = GrabField(text, R"((\d+) bytes stack frame)");
  stats.gmem_bytes = GrabField(text, R"((\d+) bytes gmem)");
  // cmem matches "N bytes cmem[0]" (the [0] suffix follows the captured token).
  stats.cmem_bytes = GrabField(text, R"((\d+) bytes cmem)");

  return PtxasParseResult{stats, ""};
}

} // namespace polykernel::analysis
