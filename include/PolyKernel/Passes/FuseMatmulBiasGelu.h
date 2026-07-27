//===- FuseMatmulBiasGelu.h - PolyKernel matmul+bias+gelu ------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-matmul-bias-gelu` pass
// declaration (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_FUSE_MATMUL_BIAS_GELU_H
#define POLYKERNEL_PASSES_FUSE_MATMUL_BIAS_GELU_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--fuse-matmul-bias-gelu` pass: fuses the chain
/// `matmul(a, b); bias(mm, bias); gelu(b)` into one
/// `polykernel.fused_matmul_bias_gelu(a, b, bias)` when BOTH absorbed
/// intermediates are single-use — the matmul result feeds exactly one bias and
/// the bias result feeds exactly one gelu. Produces the gelu's result type. A
/// chain whose matmul OR bias result has more than one use is NOT fused (fusing
/// would duplicate the shared producer). Each fused op is tagged for the Todo 17
/// traffic report with `polykernel.fused_from = "matmul_bias_gelu"` and, because
/// this fusion eliminates TWO intermediates, the indexed attributes
/// `polykernel.eliminated_type_0 = <matmul result type>` and
/// `polykernel.eliminated_type_1 = <bias result type>` (see the multi-intermediate
/// tracking scheme documented in FuseMatmulBiasGelu.cpp).
std::unique_ptr<::mlir::Pass> createFuseMatmulBiasGeluPass();

} // namespace mlir::polykernel

// Generated pass declaration for `FuseMatmulBiasGelu` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::FuseMatmulBiasGeluBase) is emitted by GEN_PASS_DEF_FUSEMATMULBIASGELU in
// the pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_FUSEMATMULBIASGELU
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_FUSE_MATMUL_BIAS_GELU_H
