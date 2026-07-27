//===- AmdIsaAnalyzer.h - Parse AMDGPU ISA resource usage -------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 21 / Wave 4). GPU-FREE: parses the
// assembly emitted by `hipcc --save-temps` (AMDGPU `.s`) into a typed
// AmdIsaStats. Extracts .amdhsa_next_free_vgpr/sgpr, LDS (group segment),
// per-thread scratch (private segment = spills), and scratch_store/load
// instruction counts. Computes gfx1101 occupancy from per-CU resource limits
// (verified against rocminfo). Garbage input (no .amdhsa_kernel directive)
// yields an explicit parse error - never a crash, never silent zeros.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_ANALYSIS_AMDISAANALYZER_H
#define POLYKERNEL_ANALYSIS_AMDISAANALYZER_H

#include <optional>
#include <string>
#include <string_view>

namespace polykernel::analysis {

/// Resource usage extracted from an AMDGPU `.s` (hipcc --save-temps output).
/// A field is 0 when the assembly does not report it (legitimate "none").
struct AmdIsaStats {
  int vgpr = 0;              // .amdhsa_next_free_vgpr
  int sgpr = 0;              // .amdhsa_next_free_sgpr
  int lds_bytes = 0;         // .amdhsa_group_segment_fixed_size
  int scratch_bytes = 0;     // .amdhsa_private_segment_fixed_size (per-thread)
  int scratch_store_count = 0; // scratch_store_* instruction count
  int scratch_load_count = 0;  // scratch_load_* instruction count
};

/// Result of parsing an AMDGPU `.s`. Exactly one of {stats, error} is
/// meaningful: ok() == true => stats populated, error empty; ok() == false =>
/// stats empty, error describes the failure.
struct AmdIsaParseResult {
  std::optional<AmdIsaStats> stats;
  std::string error;

  [[nodiscard]] bool ok() const { return stats.has_value(); }
};

/// Parse an AMDGPU `.s` (from `hipcc --save-temps`). The .amdhsa_kernel
/// directive is the validity anchor: its absence means the input is not an
/// AMDGPU assembly file -> explicit error.
[[nodiscard]] AmdIsaParseResult ParseAmdIsa(std::string_view assembly);

/// Whether the kernel spills: scratch instruction count > 0 OR
/// private_segment_fixed_size > 0.
[[nodiscard]] bool HasSpills(const AmdIsaStats &stats);

// --- gfx1101 occupancy model ---

/// Per-CU hardware limits for gfx1101 (RDNA3, RX 7800 XT).
/// Verified against `rocminfo` (GROUP pool = 64 KiB LDS/CU, Max Waves = 32).
/// VGPR/SGPR file sizes: 1536 VGPRs + 1600 SGPRs per SIMD × 2 SIMDs/CU.
struct Gfx1101Limits {
  int vgpr_file_per_cu = 3072;   // 1536 per SIMD × 2 SIMDs.
  int sgpr_file_per_cu = 3200;   // 1600 per SIMD × 2 SIMDs.
  int lds_per_cu_bytes = 65536;  // 64 KiB (rocminfo GROUP pool).
  int max_waves_per_cu = 32;     // rocminfo "Max Waves Per CU".
  int wave_size = 32;            // wave32 (rocminfo "Wavefront Size").
};

/// Which hardware constraint capped the resident workgroup count.
/// Maps to contract-H limiter strings: vgpr->"registers", sgpr->"blocks",
/// lds->"smem", waves->"warps".
enum class AmdLimiter { vgpr, sgpr, lds, waves };

[[nodiscard]] std::string_view AmdLimiterName(AmdLimiter limiter);

/// Contract-H limiter string (maps AmdLimiter to the pinned enum values).
[[nodiscard]] std::string_view AmdLimiterToContractH(AmdLimiter limiter);

/// Occupancy outcome for gfx1101. Mirrors the CUDA Occupancy struct shape.
struct AmdOccupancy {
  int active_waves_per_cu = 0;
  int max_waves_per_cu = 0;
  double occupancy_pct = 0.0;
  AmdLimiter limiter = AmdLimiter::waves;
};

/// Compute gfx1101 occupancy from parsed ISA stats + launch config.
/// `threads_per_block` is rounded up to whole waves. Deterministic tie-break
/// order: vgpr, sgpr, lds, waves (first constraint to reach the min wins).
[[nodiscard]] AmdOccupancy ComputeGfx1101Occupancy(int vgpr_per_wave,
                                                    int sgpr_per_wave,
                                                    int lds_per_block_bytes,
                                                    int threads_per_block);

} // namespace polykernel::analysis

#endif // POLYKERNEL_ANALYSIS_AMDISAANALYZER_H
