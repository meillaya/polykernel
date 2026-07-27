//===- FuseRmsnormMatmul.h - PolyKernel rmsnorm+matmul fusion --*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-rmsnorm-matmul` pass
// declaration (Todo 12 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_FUSE_RMSNORM_MATMUL_H
#define POLYKERNEL_PASSES_FUSE_RMSNORM_MATMUL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--fuse-rmsnorm-matmul` pass: fuses a `polykernel.matmul` whose
/// input (A) operand is produced by a SINGLE-USE `polykernel.rmsnorm` into one
/// `polykernel.fused_rmsnorm_matmul`, carrying the rmsnorm's optional `epsilon`
/// attribute and producing the matmul's (MxN) result type. A rmsnorm whose
/// result has more than one use is NOT fused (fusing would duplicate the
/// normalization). Each fused op is tagged for the Todo 17 traffic report with
/// `polykernel.fused_from = "rmsnorm_matmul"` and
/// `polykernel.eliminated_type = <the eliminated intermediate's tensor type>`.
std::unique_ptr<::mlir::Pass> createFuseRmsnormMatmulPass();

} // namespace mlir::polykernel

// Generated pass declaration for `FuseRmsnormMatmul` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::FuseRmsnormMatmulBase) is emitted by GEN_PASS_DEF_FUSERMSNORMMATMUL in
// the pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_FUSERMSNORMMATMUL
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_FUSE_RMSNORM_MATMUL_H
