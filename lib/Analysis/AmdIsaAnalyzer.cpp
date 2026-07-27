//===- AmdIsaAnalyzer.cpp - Parse AMDGPU ISA resource usage -----*- C++ -*-===//
//
// Implementation of ParseAmdIsa + ComputeGfx1101Occupancy. Each resource field
// is extracted with an independent regex from the `.amdhsa_kernel` metadata
// block (mirroring PtxasParser.cpp's field-per-regex approach). The
// .amdhsa_kernel directive anchors validity: it appears in every real
// `hipcc --save-temps` .s, so its absence means the input is not AMDGPU
// assembly -> explicit error. Scratch store/load instructions are counted
// across the full assembly text (they appear in the .text section).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Analysis/AmdIsaAnalyzer.h"

#include <array>
#include <limits>
#include <regex>
#include <string>

namespace polykernel::analysis {

namespace {

/// Search `text` for `pattern` (one capture group) and return the captured int,
/// or `fallback` if the pattern is absent. Used for every optional field.
int GrabField(const std::string &text, const char *pattern, int fallback = 0) {
  const std::regex re(pattern);
  std::smatch m;
  if (std::regex_search(text, m, re))
    return std::stoi(m[1].str());
  return fallback;
}

/// Count non-overlapping matches of `pattern` in `text`.
int CountMatches(const std::string &text, const char *pattern) {
  const std::regex re(pattern);
  auto begin = std::sregex_iterator(text.begin(), text.end(), re);
  auto end = std::sregex_iterator();
  return static_cast<int>(std::distance(begin, end));
}

constexpr int kUnlimited = std::numeric_limits<int>::max();

} // namespace

AmdIsaParseResult ParseAmdIsa(std::string_view assembly) {
  const std::string text(assembly);

  // The .amdhsa_kernel directive is the validity anchor: present in every real
  // hipcc --save-temps .s. Its absence means the input is not AMDGPU assembly.
  const std::regex kernel_re(R"(\.amdhsa_kernel\s)");
  if (!std::regex_search(text, kernel_re))
    return AmdIsaParseResult{
        std::nullopt,
        "parse error: no '.amdhsa_kernel' directive found; "
        "input is not an AMDGPU assembly (.s) file"};

  AmdIsaStats stats;
  stats.vgpr = GrabField(text, R"(\.amdhsa_next_free_vgpr\s+(\d+))");
  stats.sgpr = GrabField(text, R"(\.amdhsa_next_free_sgpr\s+(\d+))");
  stats.lds_bytes =
      GrabField(text, R"(\.amdhsa_group_segment_fixed_size\s+(\d+))");
  stats.scratch_bytes =
      GrabField(text, R"(\.amdhsa_private_segment_fixed_size\s+(\d+))");
  stats.scratch_store_count = CountMatches(text, R"(scratch_store_\w+)");
  stats.scratch_load_count = CountMatches(text, R"(scratch_load_\w+)");

  return AmdIsaParseResult{stats, ""};
}

bool HasSpills(const AmdIsaStats &stats) {
  return stats.scratch_store_count > 0 || stats.scratch_load_count > 0 ||
         stats.scratch_bytes > 0;
}

std::string_view AmdLimiterName(AmdLimiter limiter) {
  switch (limiter) {
  case AmdLimiter::vgpr:
    return "vgpr";
  case AmdLimiter::sgpr:
    return "sgpr";
  case AmdLimiter::lds:
    return "lds";
  case AmdLimiter::waves:
    return "waves";
  }
  return "unknown"; // unreachable: AmdLimiter is exhaustively matched above.
}

std::string_view AmdLimiterToContractH(AmdLimiter limiter) {
  switch (limiter) {
  case AmdLimiter::vgpr:
    return "registers"; // VGPRs are the AMD equivalent of CUDA registers.
  case AmdLimiter::sgpr:
    return "blocks"; // SGPR pressure maps to the hardware-cap bucket.
  case AmdLimiter::lds:
    return "smem"; // LDS is the AMD equivalent of CUDA shared memory.
  case AmdLimiter::waves:
    return "warps"; // Waves are the AMD equivalent of CUDA warps.
  }
  return "unknown"; // unreachable: AmdLimiter is exhaustively matched above.
}

AmdOccupancy ComputeGfx1101Occupancy(int vgpr_per_wave, int sgpr_per_wave,
                                      int lds_per_block_bytes,
                                      int threads_per_block) {
  const Gfx1101Limits l;

  const int waves_per_block =
      (threads_per_block + l.wave_size - 1) / l.wave_size;

  // Workgroup-level limits: a zero footprint means the resource cannot limit.
  const int vgpr_per_block = vgpr_per_wave * waves_per_block;
  const int sgpr_per_block = sgpr_per_wave * waves_per_block;
  const int wg_by_vgpr =
      vgpr_per_block > 0 ? l.vgpr_file_per_cu / vgpr_per_block : kUnlimited;
  const int wg_by_sgpr =
      sgpr_per_block > 0 ? l.sgpr_file_per_cu / sgpr_per_block : kUnlimited;
  const int wg_by_lds =
      lds_per_block_bytes > 0 ? l.lds_per_cu_bytes / lds_per_block_bytes
                              : kUnlimited;
  const int wg_by_waves = l.max_waves_per_cu / waves_per_block;

  // Deterministic tie-break: vgpr -> sgpr -> lds -> waves (strict '<' so the
  // FIRST constraint to reach the running minimum is named).
  struct Candidate {
    AmdLimiter limiter;
    int workgroups;
  };
  const std::array<Candidate, 4> candidates = {
      {{AmdLimiter::vgpr, wg_by_vgpr},
       {AmdLimiter::sgpr, wg_by_sgpr},
       {AmdLimiter::lds, wg_by_lds},
       {AmdLimiter::waves, wg_by_waves}}};

  Candidate best = candidates.front();
  for (const Candidate &c : candidates)
    if (c.workgroups < best.workgroups)
      best = c;

  AmdOccupancy occ;
  occ.active_waves_per_cu = best.workgroups * waves_per_block;
  occ.max_waves_per_cu = l.max_waves_per_cu;
  occ.occupancy_pct =
      100.0 * static_cast<double>(occ.active_waves_per_cu) / l.max_waves_per_cu;
  occ.limiter = best.limiter;
  return occ;
}

} // namespace polykernel::analysis
