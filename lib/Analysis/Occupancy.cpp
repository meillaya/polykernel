//===- Occupancy.cpp - SM occupancy model -----------------------*- C++ -*-===//
//
// Occupancy = min over four resident-block limits, then active warps + pct.
//   blocks_by_regs   = regs_per_sm  / (regs_per_thread * threads_per_block)
//   blocks_by_smem   = smem_per_sm  / smem_per_block        (0 smem => no limit)
//   blocks_by_warps  = warps_per_sm / warps_per_block
//   blocks_by_limit  = blocks_per_sm                         (hardware cap)
// active_warps = blocks * warps_per_block; pct = active_warps / warps_per_sm.
// The limiter is whichever constraint produced the min (deterministic tie-break
// order: registers, smem, warps, blocks).
//
//===----------------------------------------------------------------------===//

#include "Occupancy.h"

#include <array>
#include <limits>

namespace polykernel::analysis {

std::optional<Arch> ParseArch(std::string_view text) {
  if (text == "sm_80")
    return Arch::sm_80;
  if (text == "sm_90")
    return Arch::sm_90;
  return std::nullopt;
}

std::string_view ArchName(Arch arch) {
  switch (arch) {
  case Arch::sm_80:
    return "sm_80";
  case Arch::sm_90:
    return "sm_90";
  }
  return "unknown"; // unreachable: Arch is exhaustively matched above.
}

ArchLimits LimitsFor(Arch arch) {
  ArchLimits l; // register/warp/block/thread defaults are arch-shared.
  switch (arch) {
  case Arch::sm_80: // A100: 164 KB/SM, 163 KB/block max.
    l.smem_per_sm_bytes = 164 * 1024;
    l.smem_per_block_max_bytes = 163 * 1024;
    return l;
  case Arch::sm_90: // H100: 228 KB/SM, 227 KB/block max.
    l.smem_per_sm_bytes = 228 * 1024;
    l.smem_per_block_max_bytes = 227 * 1024;
    return l;
  }
  return l; // unreachable: Arch is exhaustively matched above.
}

std::string_view LimiterName(Limiter limiter) {
  switch (limiter) {
  case Limiter::registers:
    return "registers";
  case Limiter::smem:
    return "smem";
  case Limiter::warps:
    return "warps";
  case Limiter::blocks:
    return "blocks";
  }
  return "unknown"; // unreachable: Limiter is exhaustively matched above.
}

namespace {
constexpr int kUnlimited = std::numeric_limits<int>::max();
} // namespace

Occupancy ComputeOccupancy(int registers_per_thread, int smem_per_block_bytes,
                           int threads_per_block, Arch arch) {
  const ArchLimits l = LimitsFor(arch);

  const int warps_per_block =
      (threads_per_block + l.warp_size - 1) / l.warp_size;

  // A zero footprint means the resource cannot limit -> sentinel kUnlimited so
  // it never wins the min (the hardware block cap still bounds the result).
  const int regs_per_block = registers_per_thread * threads_per_block;
  const int blocks_by_regs =
      regs_per_block > 0 ? l.regs_per_sm / regs_per_block : kUnlimited;
  const int blocks_by_smem =
      smem_per_block_bytes > 0 ? l.smem_per_sm_bytes / smem_per_block_bytes
                               : kUnlimited;
  const int blocks_by_warps = l.warps_per_sm / warps_per_block;
  const int blocks_by_limit = l.blocks_per_sm;

  // Deterministic tie-break: walk registers -> smem -> warps -> blocks with a
  // strict '<' so the FIRST constraint to reach the running minimum is named.
  struct Candidate {
    Limiter limiter;
    int blocks;
  };
  const std::array<Candidate, 4> candidates = {
      {{Limiter::registers, blocks_by_regs},
       {Limiter::smem, blocks_by_smem},
       {Limiter::warps, blocks_by_warps},
       {Limiter::blocks, blocks_by_limit}}};

  Candidate best = candidates.front();
  for (const Candidate &c : candidates)
    if (c.blocks < best.blocks)
      best = c;

  Occupancy occ;
  occ.active_warps_per_sm = best.blocks * warps_per_block;
  occ.max_warps_per_sm = l.warps_per_sm;
  occ.occupancy_pct =
      100.0 * static_cast<double>(occ.active_warps_per_sm) / l.warps_per_sm;
  occ.limiter = best.limiter;
  return occ;
}

} // namespace polykernel::analysis
