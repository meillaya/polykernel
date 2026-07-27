//===- ConfigSpace.h - Autotuner config space + enumerator ------*- C++ -*-===//
//
// PolyKernel autotuner (Todo 24 / Wave 5). GPU-FREE pure logic: enumerates the
// bounded Triton/CUTLASS-style matmul config grid and applies a deterministic
// pruner that drops physically-inadmissible configs, leaving ~100-150 candidates
// for the (later) measurement / compile-time-model scorer.
//
// Grid axes (research bg_fbed4570):
//   BLOCK_M {16,32,64,128}   BLOCK_N {16,32,64,128}   BLOCK_K {32,64,128}
//   num_warps {4,8}          vector_width {1,2,4,8}   unroll {1,2,4,8}
//   shared_memory_stages {2,3,4}
// Full Cartesian grid = 4*4*3*2*4*4*3 = 4608 configs.
//
// The pruner is a fixed, documented rule set (see ConfigSpace.cpp); with the
// sm_90 (H100) resource limits it leaves 141 configs - inside the ~100-150
// acceptance band. Enumeration order is deterministic (lexicographic over the
// axis order above) so the emitted list is reproducible run-to-run.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_AUTOTUNE_CONFIGSPACE_H
#define POLYKERNEL_AUTOTUNE_CONFIGSPACE_H

#include <compare>
#include <string>
#include <string_view>
#include <vector>

namespace polykernel::autotune {

/// One point in the autotuner search space: a matmul tile + schedule. Field
/// names match the tuning-cache `best` sub-object EXACTLY (contract H / Todo 24).
struct Config {
  int block_m = 0;              ///< Tile rows (M).
  int block_n = 0;              ///< Tile cols (N).
  int block_k = 0;              ///< Reduction tile (K).
  int num_warps = 0;            ///< Warps per block (threads = num_warps*32).
  int vector_width = 0;         ///< Vectorized load/store width (elements).
  int unroll = 0;               ///< Inner-loop unroll factor.
  int shared_memory_stages = 0; ///< Software-pipeline depth (double/triple buf).

  // Value semantics + a total order so configs are set-comparable and sortable
  // into a deterministic sequence (lexicographic over the field order above).
  [[nodiscard]] bool operator==(const Config &) const = default;
  [[nodiscard]] std::strong_ordering operator<=>(const Config &) const = default;
};

/// Target resource limits driving the pruner. Defaults are sm_90 (H100):
/// 228 KB opt-in shared memory per SM, 65536 registers per SM.
struct ArchLimits {
  long long smem_per_sm_bytes = 233472; ///< 228 KB (sm_90 opt-in max).
  long long regs_per_sm = 65536;        ///< Architectural register file / SM.
  int min_blocks_per_sm = 2;            ///< Occupancy floor the pruner enforces.
  int max_regs_per_thread = 128;        ///< Register-pressure ceiling / thread.
  int elem_bytes = 2;                   ///< Operand element size (bf16).
};

/// The bounded config space: the axis values + the pruner. All methods are
/// deterministic and side-effect free.
class ConfigSpace {
public:
  /// The full bounded Cartesian grid (4608 configs) in deterministic
  /// lexicographic order. No pruning applied.
  [[nodiscard]] static std::vector<Config> FullGrid();

  /// True iff `config` survives every pruning rule under `limits`.
  [[nodiscard]] static bool IsAdmissible(const Config &config,
                                         const ArchLimits &limits = {});

  /// If `config` is inadmissible, a short human-readable reason naming the
  /// violated rule; empty string when admissible. Used by tests + tooling to
  /// explain WHY a config was dropped (never a silent pass).
  [[nodiscard]] static std::string RejectionReason(const Config &config,
                                                   const ArchLimits &limits = {});

  /// Apply the pruner to an arbitrary config list, preserving input order.
  [[nodiscard]] static std::vector<Config>
  Prune(const std::vector<Config> &configs, const ArchLimits &limits = {});

  /// Convenience: Prune(FullGrid()) - the bounded, pruned candidate list
  /// (~100-150 configs for the default sm_90 limits).
  [[nodiscard]] static std::vector<Config>
  Enumerate(const ArchLimits &limits = {});
};

/// Render a config as a compact one-line string, e.g.
/// "block_m=128 block_n=128 block_k=64 num_warps=8 vector_width=4 unroll=4
///  shared_memory_stages=3". For logs + test diagnostics.
[[nodiscard]] std::string ToString(const Config &config);

} // namespace polykernel::autotune

#endif // POLYKERNEL_AUTOTUNE_CONFIGSPACE_H
