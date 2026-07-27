//===- Grid.cpp - PE mesh + coordinate routing -----------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). See Grid.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Grid.h"

#include <cassert>
#include <vector>

namespace polykernel::dataflow {

Grid::Grid(int width, int height) : width_(width), height_(height) {
  assert(width > 0 && height > 0 && "grid dimensions must be positive");
  pes_.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
}

Pe &Grid::At(int x, int y) {
  assert(InBounds(x, y) && "Grid::At out of bounds");
  return pes_[Index(x, y)];
}

const Pe &Grid::At(int x, int y) const {
  assert(InBounds(x, y) && "Grid::At out of bounds");
  return pes_[Index(x, y)];
}

bool Grid::Neighbor(int x, int y, Dir d, int &nx, int &ny) const {
  int cx = x, cy = y;
  switch (d) {
  case Dir::Ramp:
    return false; // local port: no grid neighbour.
  case Dir::East:
    ++cx;
    break;
  case Dir::West:
    --cx;
    break;
  case Dir::North:
    ++cy;
    break;
  case Dir::South:
    --cy;
    break;
  case Dir::Count:
    return false;
  }
  if (!InBounds(cx, cy))
    return false; // off-grid: leave nx/ny untouched.
  nx = cx;
  ny = cy;
  return true;
}

bool Grid::Inject(int x, int y, Color color, DirMask dirs, const Wavelet &w) {
  if (!InBounds(x, y))
    return false;
  return At(x, y).GetRouter().Tx(color, dirs, w);
}

bool Grid::SetRoute(int x, int y, Color color, Dir rx, DirMask tx) {
  if (!InBounds(x, y))
    return false;
  return At(x, y).GetRouter().SetColorConfig(ColorConfig{color, rx, tx});
}

void Grid::Step() {
  // Phase 1 - routing: each PE applies its per-color @set_color_config, moving
  // wavelets from input VCs into output VCs (a combinational decision, 0 cycles).
  for (Pe &pe : pes_)
    pe.GetRouter().ApplyForwarding();

  // Phase 2 - hop: snapshot the output VCs, then cross exactly one link each.
  // Snapshotting guarantees a wavelet advances exactly one hop this cycle: it
  // lands in a neighbour INPUT VC, which is not re-hopped until a later cycle.
  struct Move {
    int x, y;
    Dir d;
    uint8_t color;
    std::size_t count;
  };
  std::vector<Move> moves;
  for (int y = 0; y < height_; ++y)
    for (int x = 0; x < width_; ++x) {
      Router &r = At(x, y).GetRouter();
      for (unsigned d = 0; d < static_cast<unsigned>(kNumDirs); ++d)
        for (unsigned id = 0; id < Color::kNumRoutable; ++id) {
          const Color c = *Color::Routable(id);
          const std::size_t n = r.OutputSize(static_cast<Dir>(d), c);
          if (n > 0)
            moves.push_back({x, y, static_cast<Dir>(d), c.Id(), n});
        }
    }

  for (const Move &m : moves) {
    Router &src = At(m.x, m.y).GetRouter();
    const Color c = *Color::Routable(m.color);
    for (std::size_t i = 0; i < m.count; ++i) {
      const auto w = src.PeekOutput(m.d, c);
      if (!w)
        break;
      if (m.d == Dir::Ramp) {
        // RAMP "hop" delivers locally to this PE's input VC.
        if (!src.Rx(Dir::Ramp, c, *w))
          break; // input VC full: lossless, retry next cycle.
        src.PopOutput(m.d, c);
        continue;
      }
      int nx, ny;
      if (!Neighbor(m.x, m.y, m.d, nx, ny)) {
        // Off-grid hop: drop + flag. NEVER an out-of-bounds access.
        src.PopOutput(m.d, c);
        ++dropped_offgrid_;
        continue;
      }
      Router &dst = At(nx, ny).GetRouter();
      if (!dst.Rx(m.d, c, *w))
        break; // neighbour input VC full: lossless backpressure, retry later.
      src.PopOutput(m.d, c);
    }
  }

  ++cycle_;
}

} // namespace polykernel::dataflow
