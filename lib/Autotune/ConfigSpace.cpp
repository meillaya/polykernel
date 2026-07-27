//===- ConfigSpace.cpp - Autotuner config space + pruner --------*- C++ -*-===//
//
// Enumerates the bounded Triton/CUTLASS-style grid and applies a fixed,
// documented pruner. With the default sm_90 (H100) ArchLimits the pruner leaves
// 141 of the 4608 grid points - inside the ~100-150 acceptance band.
//
// Pruning rules (evaluated in order; first violation names the rejection):
//   STRUCT  BLOCK_M*BLOCK_N must be divisible by the thread count
//           (threads = num_warps*32): the output tile must map evenly onto the
//           warps. (Also rejects degenerate configs with num_warps == 0.)
//   R1      Drop BLOCK_M*BLOCK_N < 4096 AND num_warps == 8: a sub-4096 tile
//           cannot keep 8 warps (256 threads) busy. [plan-pinned rule]
//   R2      Drop stages whose shared-memory budget exceeds what leaves
//           min_blocks_per_sm resident blocks/SM:
//             stages*(BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*elem_bytes
//               * min_blocks_per_sm  <=  smem_per_sm_bytes.
//           (double-buffered A[BLOCK_M,BLOCK_K] + B[BLOCK_K,BLOCK_N] tiles per
//           pipeline stage). [plan-pinned "smem-over-budget stages" rule]
//   R3      Drop configs whose per-thread register estimate exceeds
//           max_regs_per_thread. Estimate = accumulator + staged operands:
//             acc     = BLOCK_M*BLOCK_N*unroll / threads
//             staging = (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N)*unroll / threads
//   R4      Drop configs whose register file cannot host min_blocks_per_sm
//           blocks/SM: regs * threads * min_blocks_per_sm <= regs_per_sm.
//   R5      Drop BLOCK_K % vector_width != 0: contiguous K-dim loads must
//           vectorize cleanly.
//   R6      Drop elems_per_thread != vector_width * unroll: the per-thread
//           output micro-tile must be exactly one vectorized group repeated by
//           the unroll factor (no partial vectors, no leftover scalars).
//   R7      Drop elems_per_thread < 4: below minimum per-thread productivity.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/ConfigSpace.h"

#include <string>

namespace polykernel::autotune {

namespace {

// The bounded axis values (research bg_fbed4570). Order here fixes the
// deterministic enumeration order.
constexpr int kBlockM[] = {16, 32, 64, 128};
constexpr int kBlockN[] = {16, 32, 64, 128};
constexpr int kBlockK[] = {32, 64, 128};
constexpr int kNumWarps[] = {4, 8};
constexpr int kVectorWidth[] = {1, 2, 4, 8};
constexpr int kUnroll[] = {1, 2, 4, 8};
constexpr int kStages[] = {2, 3, 4};

constexpr int kMinTileFor8Warps = 4096; ///< R1 threshold (BLOCK_M*BLOCK_N).
constexpr int kMinElemsPerThread = 4;   ///< R7 floor.

/// Per-thread register estimate (accumulator + staged operand fragments).
long long RegisterEstimate(const Config &c, int threads) {
  const long long tile = 1LL * c.block_m * c.block_n;
  const long long operands = 1LL * c.block_m * c.block_k +
                             1LL * c.block_k * c.block_n;
  const long long acc = tile * c.unroll / threads;
  const long long staging = operands * c.unroll / threads;
  return acc + staging;
}

} // namespace

std::vector<Config> ConfigSpace::FullGrid() {
  std::vector<Config> grid;
  grid.reserve(4608); // 4*4*3*2*4*4*3
  for (int bm : kBlockM)
    for (int bn : kBlockN)
      for (int bk : kBlockK)
        for (int nw : kNumWarps)
          for (int vw : kVectorWidth)
            for (int un : kUnroll)
              for (int st : kStages)
                grid.push_back(Config{bm, bn, bk, nw, vw, un, st});
  return grid;
}

std::string ConfigSpace::RejectionReason(const Config &c,
                                         const ArchLimits &limits) {
  const int threads = c.num_warps * 32;
  const long long tile = 1LL * c.block_m * c.block_n;

  // STRUCT: tile maps evenly onto warps (and warps exist).
  if (threads <= 0 || tile % threads != 0)
    return "BLOCK_M*BLOCK_N must be divisible by num_warps*32";

  const long long elems_per_thread = tile / threads;

  // R1: small tile cannot feed 8 warps.
  if (tile < kMinTileFor8Warps && c.num_warps == 8)
    return "BLOCK_M*BLOCK_N < 4096 cannot feed num_warps=8";

  // R2: shared-memory budget (leave room for min_blocks_per_sm blocks/SM).
  const long long operands = 1LL * c.block_m * c.block_k +
                             1LL * c.block_k * c.block_n;
  const long long smem =
      1LL * c.shared_memory_stages * operands * limits.elem_bytes;
  if (smem * limits.min_blocks_per_sm > limits.smem_per_sm_bytes)
    return "shared_memory_stages exceed the per-SM smem budget";

  // R3: per-thread register-pressure ceiling.
  const long long regs = RegisterEstimate(c, threads);
  if (regs > limits.max_regs_per_thread)
    return "per-thread register pressure exceeds max_regs_per_thread";

  // R4: register file must host min_blocks_per_sm blocks/SM.
  if (regs * threads * limits.min_blocks_per_sm > limits.regs_per_sm)
    return "register file cannot host min_blocks_per_sm blocks/SM";

  // R5: vector width divides the contiguous K tile.
  if (c.vector_width <= 0 || c.block_k % c.vector_width != 0)
    return "vector_width must divide BLOCK_K";

  // R6: per-thread micro-tile == vector_width * unroll (no partial vectors).
  if (elems_per_thread != 1LL * c.vector_width * c.unroll)
    return "elems_per_thread must equal vector_width*unroll";

  // R7: minimum per-thread productivity.
  if (elems_per_thread < kMinElemsPerThread)
    return "elems_per_thread below minimum productivity";

  return {}; // admissible
}

bool ConfigSpace::IsAdmissible(const Config &c, const ArchLimits &limits) {
  return RejectionReason(c, limits).empty();
}

std::vector<Config> ConfigSpace::Prune(const std::vector<Config> &configs,
                                       const ArchLimits &limits) {
  std::vector<Config> kept;
  kept.reserve(configs.size());
  for (const Config &c : configs)
    if (IsAdmissible(c, limits))
      kept.push_back(c);
  return kept;
}

std::vector<Config> ConfigSpace::Enumerate(const ArchLimits &limits) {
  return Prune(FullGrid(), limits);
}

std::string ToString(const Config &c) {
  std::string s;
  s += "block_m=" + std::to_string(c.block_m);
  s += " block_n=" + std::to_string(c.block_n);
  s += " block_k=" + std::to_string(c.block_k);
  s += " num_warps=" + std::to_string(c.num_warps);
  s += " vector_width=" + std::to_string(c.vector_width);
  s += " unroll=" + std::to_string(c.unroll);
  s += " shared_memory_stages=" + std::to_string(c.shared_memory_stages);
  return s;
}

} // namespace polykernel::autotune
