//===- FuseResidualRmsnorm.h - PolyKernel residual-add+rmsnorm --*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-residual-rmsnorm` pass
// declaration (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_FUSE_RESIDUAL_RMSNORM_H
#define POLYKERNEL_PASSES_FUSE_RESIDUAL_RMSNORM_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--fuse-residual-rmsnorm` pass: fuses
/// `add(residual, input); rmsnorm(sum, eps)` into one
/// `polykernel.fused_residual_rmsnorm(residual, input, eps)` when the add result
/// is single-use (feeds exactly one rmsnorm). The add's `lhs` operand maps to the
/// fused op's `residual` and its `rhs` operand maps to `input` (positional, order-
/// preserving convention — documented in FuseResidualRmsnorm.cpp). Carries the
/// rmsnorm's optional `epsilon` attribute and produces the rmsnorm's result type.
/// An add whose result has more than one use is NOT fused (fusing would duplicate
/// the residual sum). Each fused op is tagged for the Todo 17 traffic report with
/// `polykernel.fused_from = "residual_rmsnorm"` and
/// `polykernel.eliminated_type = <the eliminated add result's tensor type>`.
std::unique_ptr<::mlir::Pass> createFuseResidualRmsnormPass();

} // namespace mlir::polykernel

// Generated pass declaration for `FuseResidualRmsnorm` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::FuseResidualRmsnormBase) is emitted by GEN_PASS_DEF_FUSERESIDUALRMSNORM
// in the pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_FUSERESIDUALRMSNORM
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_FUSE_RESIDUAL_RMSNORM_H
