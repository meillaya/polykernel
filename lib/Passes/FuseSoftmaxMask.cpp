//===- FuseSoftmaxMask.cpp - PolyKernel mask-add+softmax fusion -*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-softmax-mask` pass
// (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// Fusion pass extending the Todo 12 pattern (FuseRmsnormMatmul.cpp): an
// OpRewritePattern on the CONSUMER op (polykernel.softmax) that inspects its
// producer, guards the absorbed intermediate with a single-use check, replaces
// the consumer with the fused op (carrying the producer's operands + the
// consumer's result type), erases the dead producer, and runs under the greedy
// pattern driver.
//
// Pattern (FuseSoftmaxMaskPattern), on polykernel.softmax:
//   %sum = polykernel.add %input, %mask : tensor<...>
//   %out = polykernel.softmax %sum      : tensor<...>
//     ==>
//   %out = polykernel.fused_softmax_mask %input, %mask
//       {polykernel.fused_from = "softmax_mask",
//        polykernel.eliminated_type = tensor<...>}
//       : tensor<...>, tensor<...> -> tensor<...>
//
// Guardrails:
//   - The softmax's input must be DEFINED by a polykernel.add (the mask add).
//   - SINGLE-USE CHECK (essential): the add result must have EXACTLY ONE use —
//     this softmax. An add feeding two or more ops is NOT fused, or the masked
//     sum would be duplicated.
//   - OPERAND MAPPING (positional, order-preserving convention): the fused op's
//     operands are (input, mask) per PolyKernelOps.td. The add's `lhs` maps to
//     `input` and its `rhs` maps to `mask`. `add` is mathematically commutative
//     (input + mask == mask + input), so either mapping is numerically equivalent;
//     we fix the positional mapping so the rewrite is deterministic and
//     FileCheck-able.
//   - The fused op uses the EXACT operand order from PolyKernelOps.td
//     (input, mask) and produces the softmax's result type. fused_softmax_mask
//     declares NO `axis` attribute (PolyKernelOps.td), so the softmax's optional
//     `axis` is intentionally NOT carried.
//   - The dead add is erased explicitly (its only use was the softmax).
//
// Eliminated-intermediate tracking (for the Todo 17 traffic report): this fusion
// eliminates ONE intermediate (the add result), so it uses the UNINDEXED
// discardable attributes (consistent with rmsnorm_matmul / residual_rmsnorm) —
//   - `polykernel.fused_from = "softmax_mask"` (the fusion kind), and
//   - `polykernel.eliminated_type = <the eliminated add result's tensor type>`.
// See FuseMatmulBiasGelu.cpp for the full single-vs-indexed scheme note. Todo 17
// walks the fused ops, reads the eliminated type(s), and computes the saved
// global round-trip as numElements(type) * sizeof(element).
//
// Applied greedily via applyPatternsGreedily (the same driver --canonicalize
// uses), which also folds + DCEs + simplifies regions.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/FuseSoftmaxMask.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_FUSESOFTMAXMASK
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable-attribute names attached to each fused op so the Todo 17 traffic
/// report can identify the fusion and the eliminated intermediate.
constexpr llvm::StringLiteral kFusedFromAttr = "polykernel.fused_from";
constexpr llvm::StringLiteral kEliminatedTypeAttr =
    "polykernel.eliminated_type";
/// Value of `polykernel.fused_from` for this fusion kind.
constexpr llvm::StringLiteral kFusedFromSoftmaxMask = "softmax_mask";

//===----------------------------------------------------------------------===//
// add(input, mask); softmax(sum)  ->  fused_softmax_mask(input, mask)
//===----------------------------------------------------------------------===//

struct FuseSoftmaxMaskPattern
    : public OpRewritePattern<polykernel::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::SoftmaxOp softmax,
                                PatternRewriter &rewriter) const override {
    // The softmax's input must be produced by a polykernel.add (the mask add).
    auto add = softmax.getInput().getDefiningOp<polykernel::AddOp>();
    if (!add)
      return failure();

    // SINGLE-USE CHECK (essential): fuse only when the add result has EXACTLY ONE
    // use — this softmax. A multi-use add must NOT be fused, or the masked sum
    // would be duplicated.
    if (!add.getResult().hasOneUse())
      return failure();

    // Replace the softmax with fused_softmax_mask(input, mask), producing the
    // softmax's result type. Operand order matches PolyKernelOps.td exactly:
    // (input, mask). Positional operand mapping: the add's lhs -> input, rhs ->
    // mask (see the OPERAND MAPPING note above). The fused op declares no `axis`
    // attribute, so the softmax's optional axis is intentionally NOT carried.
    auto fused = rewriter.replaceOpWithNewOp<polykernel::FusedSoftmaxMaskOp>(
        softmax, softmax.getResult().getType(), /*input=*/add.getLhs(),
        /*mask=*/add.getRhs());

    // Eliminated-intermediate tracking (Todo 17): tag the fusion kind + record the
    // eliminated intermediate's tensor type (the add result).
    fused->setAttr(kFusedFromAttr,
                   rewriter.getStringAttr(kFusedFromSoftmaxMask));
    fused->setAttr(kEliminatedTypeAttr,
                   TypeAttr::get(add.getResult().getType()));

    // Erase the now-dead add (its only use was the softmax we replaced).
    rewriter.eraseOp(add);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class FuseSoftmaxMaskPass
    : public impl::FuseSoftmaxMaskBase<FuseSoftmaxMaskPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseSoftmaxMaskPattern>(&getContext());

    // Greedy driver (mirrors --canonicalize): applies the fusion, then folds +
    // DCEs ops left dead by the rewrites + simplifies regions (defaults).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseSoftmaxMaskPass() {
  return std::make_unique<FuseSoftmaxMaskPass>();
}

} // namespace mlir::polykernel
