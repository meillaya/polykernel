//===- Grid.h - PE mesh (default 64x64) + coordinate routing ---*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7).
//
// THIS IS A SIMULATOR (a functional/cycle model, not real CSL and not Cerebras
// hardware).
//
// The Grid is a width x height mesh of PEs (default 64x64). It owns the PEs and
// advances the fabric one cycle at a time. Each cycle has two phases:
//   1. routing  - every PE applies its per-color `@set_color_config` (forwarding
//                 input VCs into output VCs),
//   2. hop      - every wavelet sitting in an output VC crosses exactly ONE link
//                 to the neighbour's input VC (single-cycle hop). A wavelet
//                 whose next hop would leave the mesh is DROPPED and counted
//                 (flagged) - never an out-of-bounds access, never a crash.
//
// Because phase 2 only drains the output VCs that existed at the start of the
// cycle (newly-arrived wavelets land in input VCs), a wavelet advances exactly
// one hop per cycle - never zero, never two.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DATAFLOW_GRID_H
#define POLYKERNEL_DATAFLOW_GRID_H

#include "PolyKernel/Dataflow/Pe.h"

#include <cstdint>
#include <vector>

namespace polykernel::dataflow {

class Grid {
public:
  static constexpr int kDefaultWidth = 64;
  static constexpr int kDefaultHeight = 64;

  explicit Grid(int width = kDefaultWidth, int height = kDefaultHeight);

  int Width() const { return width_; }
  int Height() const { return height_; }
  std::size_t Size() const { return pes_.size(); }

  bool InBounds(int x, int y) const {
    return x >= 0 && x < width_ && y >= 0 && y < height_;
  }

  // PE access. Callers must ensure InBounds(x, y) (checked in the .cpp).
  Pe &At(int x, int y);
  const Pe &At(int x, int y) const;

  // Grid neighbour for a direction. RAMP has no neighbour (returns false).
  // Returns false (without touching nx/ny) when the neighbour is off-grid.
  bool Neighbor(int x, int y, Dir d, int &nx, int &ny) const;

  // Inject a wavelet at PE (x,y): transmit it on `color` to the direction set
  // `dirs` (multicast when > 1 bit). Returns false on backpressure / bad coord.
  bool Inject(int x, int y, Color color, DirMask dirs, const Wavelet &w);

  // Configure forwarding ("the route") on PE (x,y): bind `color` to receive
  // direction `rx` and transmit direction set `tx` (the PE's @set_color_config).
  // Wavelets that arrive on `color` are re-emitted to `tx` during Step phase 1.
  bool SetRoute(int x, int y, Color color, Dir rx, DirMask tx);

  // Advance the fabric one cycle (routing phase, then single-cycle hop phase).
  void Step();

  // Number of wavelets dropped because a hop would leave the mesh.
  uint64_t DroppedOffGrid() const { return dropped_offgrid_; }
  // Number of cycles elapsed (Step calls).
  uint64_t Cycle() const { return cycle_; }

private:
  std::size_t Index(int x, int y) const {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
           static_cast<std::size_t>(x);
  }

  int width_;
  int height_;
  std::vector<Pe> pes_;
  uint64_t dropped_offgrid_ = 0;
  uint64_t cycle_ = 0;
};

} // namespace polykernel::dataflow

#endif // POLYKERNEL_DATAFLOW_GRID_H
