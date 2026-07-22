//===- PolyKernelDialect.cpp - PolyKernel dialect ---------------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect (W1 spike: definition only).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/IR/PolyKernelDialect.h"

using namespace mlir;
using namespace mlir::polykernel;

// Generated dialect definition (-gen-dialect-defs): emits the dialect
// constructor, which calls initialize().
#include "PolyKernel/IR/PolyKernelDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PolyKernel dialect.
//===----------------------------------------------------------------------===//

void PolyKernelDialect::initialize() {
  // W1 spike: no operations / types / attributes registered yet.
  // Operations are added in Todo 3 (addOperations<...>(), registerTypes(), ...).
}
