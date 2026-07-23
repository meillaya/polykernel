//===- Occupancy.h - SM occupancy model -------------------------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 11 / Wave 2). GPU-FREE occupancy
// model: from (registers_per_thread, smem_per_block, threads_per_block, arch)
// compute the achievable active warps / SM, the occupancy percentage, and WHICH
// hardware constraint is the binding limiter. Per-arch constants are hardcoded
// from the NVIDIA programming-guide limits (sm_80 A100, sm_90 H100).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_ANALYSIS_OCCUPANCY_H
#define POLYKERNEL_ANALYSIS_OCCUPANCY_H

#include <optional>
#include <string_view>

namespace polykernel::analysis {

/// Supported target architectures (the analyzer's per-arch constant tables).
enum class Arch { sm_80, sm_90 };

/// Parse an arch string ("sm_80" / "sm_90"). Returns nullopt on unknown arch.
[[nodiscard]] std::optional<Arch> ParseArch(std::string_view text);

/// Canonical arch string ("sm_80" / "sm_90") for report emission.
[[nodiscard]] std::string_view ArchName(Arch arch);

/// Hardware limits for one arch. smem differs per arch; the register / warp /
/// block / thread limits are shared by sm_80 and sm_90.
struct ArchLimits {
  int smem_per_sm_bytes = 0;      // shared memory pool per SM.
  int smem_per_block_max_bytes = 0; // max opt-in smem a single block may use.
  int regs_per_sm = 65536;
  int regs_per_thread_max = 255;
  int warps_per_sm = 64;
  int blocks_per_sm = 32;
  int threads_per_sm = 2048;
  int warp_size = 32;
};

[[nodiscard]] ArchLimits LimitsFor(Arch arch);

/// Which hardware constraint capped the resident block count (the min of the
/// four block-count limits). Enum values pinned by contract H.
enum class Limiter { registers, smem, warps, blocks };

[[nodiscard]] std::string_view LimiterName(Limiter limiter);

/// Occupancy outcome. occupancy_pct is a percentage in [0, 100].
struct Occupancy {
  int active_warps_per_sm = 0;
  int max_warps_per_sm = 0;
  double occupancy_pct = 0.0;
  Limiter limiter = Limiter::blocks;
};

/// Compute occupancy. `threads_per_block` is rounded up to whole warps. A
/// zero smem/register footprint means that resource does not limit (it can
/// never be the binding limiter). Deterministic tie-break order when several
/// constraints reach the same minimum: registers, smem, warps, blocks.
[[nodiscard]] Occupancy ComputeOccupancy(int registers_per_thread,
                                         int smem_per_block_bytes,
                                         int threads_per_block, Arch arch);

} // namespace polykernel::analysis

#endif // POLYKERNEL_ANALYSIS_OCCUPANCY_H
