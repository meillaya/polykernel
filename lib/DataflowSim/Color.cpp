//===- Color.cpp - Dataflow fabric color helpers ----------------*- C++ -*-===//
//
// PolyKernel dataflow simulator core (Todo 35 / Wave 7). See Color.h.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Dataflow/Color.h"

namespace polykernel::dataflow {

int PopCount(DirMask mask) {
  int n = 0;
  ForEachDir(mask, [&n](Dir) { ++n; });
  return n;
}

const char *DirName(Dir d) {
  switch (d) {
  case Dir::Ramp:
    return "ramp";
  case Dir::East:
    return "east";
  case Dir::West:
    return "west";
  case Dir::North:
    return "north";
  case Dir::South:
    return "south";
  case Dir::Count:
    break;
  }
  return "unknown"; // unreachable: Dir is exhaustively matched above.
}

} // namespace polykernel::dataflow
