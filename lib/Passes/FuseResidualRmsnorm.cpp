//===- FuseResidualRmsnorm.cpp - PolyKernel residual-add+rmsnorm -*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-residual-rmsnorm` pass
// (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// Fusion pass extending the Todo 12 pattern (FuseRmsnormMatmul.cpp): an
// OpRewritePattern on the CONSUMER op (polykernel.rmsnorm) that inspects its
// producer, guards the absorbed intermediate with a single-use check, replaces
// the consumer with the fused op (carrying the producer's operands + the
// rmsnorm's epsilon + the consumer's result type), erases the dead producer, and
// runs under the greedy pattern driver.
//
// Pattern (FuseResidualRmsnormPattern), on polykernel.rmsnorm:
//   %sum = polykernel.add %residual, %input : tensor<...>
//   %out = polykernel.rmsnorm %sum {epsilon = e} : tensor<...>
//     ==>
//   %out = polykernel.fused_residual_rmsnorm %residual, %input
//       {epsilon = e, polykernel.fused_from = "residual_rmsnorm",
//        polykernel.eliminated_type = tensor<...>}
//       : tensor<...>, tensor<...> -> tensor<...>
//
// Guardrails:
//   - The rmsnorm's input must be DEFINED by a polykernel.add (the residual add).
//   - SINGLE-USE CHECK (essential): the add result must have EXACTLY ONE use —
//     this rmsnorm. An add feeding two or more ops is NOT fused, or the residual
//     sum would be duplicated.
//   - OPERAND MAPPING (positional, order-preserving convention): the fused op's
//     operands are (residual, input) per PolyKernelOps.td. The add's `lhs` maps
//     to `residual` and its `rhs` maps to `input`. `add` is mathematically
//     commutative (residual + input == input + residual), so either mapping is
//     numerically equivalent; we fix the positional mapping so the rewrite is
//     deterministic and FileCheck-able.
//   - The fused op uses the EXACT operand order from PolyKernelOps.td
//     (residual, input, epsilon?) and produces the rmsnorm's result type. The
//     rmsnorm's optional `epsilon` is carried verbatim (a null FloatAttr when the
//     rmsnorm had none -> the fused op has none).
//   - The dead add is erased explicitly (its only use was the rmsnorm).
//
// Eliminated-intermediate tracking (for the Todo 17 traffic report): this fusion
// eliminates ONE intermediate (the add result), so it uses the UNINDEXED
// discardable attributes (consistent with rmsnorm_matmul / softmax_mask) —
//   - `polykernel.fused_from = "residual_rmsnorm"` (the fusion kind), and
//   - `polykernel.eliminated_type = <the eliminated add result's tensor type>`.
// See FuseMatmulBiasGelu.cpp for the full single-vs-indexed scheme note. Todo 17
// walks the fused ops, reads the eliminated type(s), and computes the saved
// global round-trip as numElements(type) * sizeof(element).
//
// Applied greedily via applyPatternsGreedily (the same driver --canonicalize
// uses), which also folds + DCEs + simplifies regions.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/FuseResidualRmsnorm.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_FUSERESIDUALRMSNORM
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
constexpr llvm::StringLiteral kFusedFromResidualRmsnorm = "residual_rmsnorm";

//===----------------------------------------------------------------------===//
// add(residual, input); rmsnorm(sum, eps)  ->
//     fused_residual_rmsnorm(residual, input, eps)
//===----------------------------------------------------------------------===//

struct FuseResidualRmsnormPattern
    : public OpRewritePattern<polykernel::RmsNormOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::RmsNormOp rmsnorm,
                                PatternRewriter &rewriter) const override {
    // The rmsnorm's input must be produced by a polykernel.add (residual add).
    auto add = rmsnorm.getInput().getDefiningOp<polykernel::AddOp>();
    if (!add)
      return failure();

    // SINGLE-USE CHECK (essential): fuse only when the add result has EXACTLY ONE
    // use — this rmsnorm. A multi-use add must NOT be fused, or the residual sum
    // would be duplicated.
    if (!add.getResult().hasOneUse())
      return failure();

    // Replace the rmsnorm with fused_residual_rmsnorm(residual, input, eps),
    // producing the rmsnorm's result type. Operand order matches PolyKernelOps.td
    // exactly: (residual, input, epsilon?). Positional operand mapping: the add's
    // lhs -> residual, rhs -> input (see the OPERAND MAPPING note above). Carry
    // the rmsnorm's optional epsilon attribute (a null FloatAttr when the rmsnorm
    // had none -> the fused op has none).
    auto fused = rewriter.replaceOpWithNewOp<polykernel::FusedResidualRmsNormOp>(
        rmsnorm, rmsnorm.getResult().getType(), /*residual=*/add.getLhs(),
        /*input=*/add.getRhs(), /*epsilon=*/rmsnorm.getEpsilonAttr());

    // Eliminated-intermediate tracking (Todo 17): tag the fusion kind + record the
    // eliminated intermediate's tensor type (the add result).
    fused->setAttr(kFusedFromAttr,
                   rewriter.getStringAttr(kFusedFromResidualRmsnorm));
    fused->setAttr(kEliminatedTypeAttr,
                   TypeAttr::get(add.getResult().getType()));

    // Erase the now-dead add (its only use was the rmsnorm we replaced).
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

class FuseResidualRmsnormPass
    : public impl::FuseResidualRmsnormBase<FuseResidualRmsnormPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseResidualRmsnormPattern>(&getContext());

    // Greedy driver (mirrors --canonicalize): applies the fusion, then folds +
    // DCEs ops left dead by the rewrites + simplifies regions (defaults).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseResidualRmsnormPass() {
  return std::make_unique<FuseResidualRmsnormPass>();
}

} // namespace mlir::polykernel
