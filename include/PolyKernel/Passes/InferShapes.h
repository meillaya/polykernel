//===- InferShapes.h - PolyKernel shape inference pass ---------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--infer-shapes` pass declaration
// (Todo 4 / Wave 1).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_INFER_SHAPES_H
#define POLYKERNEL_PASSES_INFER_SHAPES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--infer-shapes` pass: computes every PolyKernel compute op's
/// result type(s) from its operands (via InferTypeOpInterface) and refines the
/// IR, signalling pass failure on any shape inconsistency.
std::unique_ptr<::mlir::Pass> createInferShapesPass();

} // namespace mlir::polykernel

// Generated pass declaration for `InferShapes` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::InferShapesBase) is emitted by GEN_PASS_DEF_INFERSHAPES in the pass's
// .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_INFERSHAPES
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_INFER_SHAPES_H
