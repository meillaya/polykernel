//===- FuseSoftmaxMask.h - PolyKernel mask-add+softmax ---------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-softmax-mask` pass
// declaration (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_FUSE_SOFTMAX_MASK_H
#define POLYKERNEL_PASSES_FUSE_SOFTMAX_MASK_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--fuse-softmax-mask` pass: fuses
/// `add(input, mask); softmax(sum)` into one
/// `polykernel.fused_softmax_mask(input, mask)` when the add result is single-use
/// (feeds exactly one softmax). The add's `lhs` operand maps to the fused op's
/// `input` and its `rhs` operand maps to `mask` (positional, order-preserving
/// convention — documented in FuseSoftmaxMask.cpp). Produces the softmax's result
/// type. The fused op declares NO `axis` attribute (PolyKernelOps.td), so the
/// softmax's optional `axis` is intentionally NOT carried. An add whose result has
/// more than one use is NOT fused (fusing would duplicate the masked sum). Each
/// fused op is tagged for the Todo 17 traffic report with
/// `polykernel.fused_from = "softmax_mask"` and
/// `polykernel.eliminated_type = <the eliminated add result's tensor type>`.
std::unique_ptr<::mlir::Pass> createFuseSoftmaxMaskPass();

} // namespace mlir::polykernel

// Generated pass declaration for `FuseSoftmaxMask` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::FuseSoftmaxMaskBase) is emitted by GEN_PASS_DEF_FUSESOFTMAXMASK in the
// pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_FUSESOFTMAXMASK
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_FUSE_SOFTMAX_MASK_H
