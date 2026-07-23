//===- Canonicalize.h - PolyKernel canonicalization pass -------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--canonicalize` pass declaration
// (Todo 5 / Wave 1).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_CANONICALIZE_H
#define POLYKERNEL_PASSES_CANONICALIZE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--canonicalize` pass: applies the PolyKernel canonicalization
/// patterns — identity residual-add elimination (`add(x, 0) -> x`), constant
/// folding of pure elementwise ops (gelu/silu of a dense constant) and the
/// softmax-of-splat no-op fold — with the greedy pattern driver, which also
/// folds ops, DCEs ops left dead by the rewrites and simplifies regions.
std::unique_ptr<::mlir::Pass> createCanonicalizePass();

} // namespace mlir::polykernel

// Generated pass declaration for `Canonicalize` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::CanonicalizeBase) is emitted by GEN_PASS_DEF_CANONICALIZE in the
// pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_CANONICALIZE
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_CANONICALIZE_H
