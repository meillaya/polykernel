//===- PolyKernelDialect.cpp - PolyKernel dialect ---------------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect (W1 spike: definition only).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/IR/PolyKernelDialect.h"

#include "PolyKernel/IR/PolyKernelOps.h"

using namespace mlir;
using namespace mlir::polykernel;

// Generated dialect definition (-gen-dialect-defs): emits the dialect
// constructor, which calls initialize().
#include "PolyKernel/IR/PolyKernelDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// PolyKernel dialect.
//===----------------------------------------------------------------------===//

void PolyKernelDialect::initialize() {
  // Todo 3 (Wave 1): register EXACTLY the named transformer-inference op set.
  // `FuncOp`/`ReturnOp` are the function container + terminator (not compute
  // ops); the rest are the base + fused compute ops. No other ops exist.
  addOperations<
#define GET_OP_LIST
#include "PolyKernel/IR/PolyKernelOps.cpp.inc"
      >();
}
