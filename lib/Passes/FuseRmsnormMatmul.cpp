//===- FuseRmsnormMatmul.cpp - PolyKernel rmsnorm+matmul fusion -*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-rmsnorm-matmul` pass
// (Todo 12 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// First PolyKernel FUSION pass; establishes the fusion-pass pattern the later
// Wave 3 fusion passes (Todo 13) extend: an OpRewritePattern on the consumer op
// that inspects its producer, guards on a single-use check, replaces the
// consumer with the fused op (carrying the producer's operands/attrs + the
// consumer's result type), erases the dead producer, and runs under the greedy
// pattern driver.
//
// Pattern (FuseRmsnormMatmulPattern), on polykernel.matmul:
//   %rms = polykernel.rmsnorm %x {epsilon = e} : tensor<...xK>
//   %out = polykernel.matmul %rms, %w
//       : tensor<...xK>, tensor<KxN> -> tensor<...xN>
//     ==>
//   %out = polykernel.fused_rmsnorm_matmul %x, %w
//       {epsilon = e, polykernel.fused_from = "rmsnorm_matmul",
//        polykernel.eliminated_type = tensor<...xK>}
//       : tensor<...xK>, tensor<KxN> -> tensor<...xN>
//
// Guardrails:
//   - The matmul's A (input) operand must be DEFINED by a polykernel.rmsnorm
//     (a rmsnorm feeding the matmul's B operand is NOT fused).
//   - SINGLE-USE CHECK (essential): the rmsnorm result must have EXACTLY ONE use
//     — this matmul. A rmsnorm feeding two or more ops (e.g. the matmul AND a
//     residual add) is NOT fused, or the normalization would be duplicated.
//   - The fused op uses the EXACT operand order from PolyKernelOps.td
//     (input, weight, epsilon?) and produces the matmul's (MxN) result type.
//     The rmsnorm's optional `epsilon` is carried verbatim (a null FloatAttr
//     when the rmsnorm had none -> the fused op has none).
//   - The dead rmsnorm is erased explicitly (its only use was the matmul).
//
// Eliminated-intermediate tracking (for the Todo 17 traffic report): each fused
// op is tagged with two discardable attributes —
//   - `polykernel.fused_from = "rmsnorm_matmul"` (the fusion kind), and
//   - `polykernel.eliminated_type = <tensor type>` (the eliminated rmsnorm
//     result's tensor type/shape — the intermediate the unfused pipeline
//     materialized to global memory and read back).
// Todo 17 walks the fused ops, reads `polykernel.eliminated_type`, and computes
// the saved global round-trip as numElements(type) * sizeof(element). Attributes
// (not a pass statistic) are the mechanism: they are exact under the greedy
// driver (one tag per committed fusion — no speculative-retry miscounting),
// IR-observable, and FileCheck-able.
//
// Applied greedily via applyPatternsGreedily (the same driver --canonicalize
// uses), which also folds + DCEs + simplifies regions.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/FuseRmsnormMatmul.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_FUSERMSNORMMATMUL
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
constexpr llvm::StringLiteral kFusedFromRmsnormMatmul = "rmsnorm_matmul";

//===----------------------------------------------------------------------===//
// rmsnorm(x, eps); matmul(rms, w)  ->  fused_rmsnorm_matmul(x, w, eps)
//===----------------------------------------------------------------------===//

struct FuseRmsnormMatmulPattern
    : public OpRewritePattern<polykernel::MatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::MatMulOp matmul,
                                PatternRewriter &rewriter) const override {
    // The matmul's A (input) operand must be produced by a polykernel.rmsnorm.
    auto rmsnorm = matmul.getA().getDefiningOp<polykernel::RmsNormOp>();
    if (!rmsnorm)
      return failure();

    // SINGLE-USE CHECK (essential): fuse only when the rmsnorm result has
    // EXACTLY ONE use — this matmul. A multi-use rmsnorm (e.g. also feeding a
    // residual add) must NOT be fused, or the normalization would be duplicated.
    if (!rmsnorm.getResult().hasOneUse())
      return failure();

    // Replace the matmul with fused_rmsnorm_matmul(x, w, eps), producing the
    // matmul's (MxN) result type. Operand order matches PolyKernelOps.td exactly:
    // (input, weight, epsilon?). Carry the rmsnorm's optional epsilon attribute
    // (a null FloatAttr when the rmsnorm had none -> the fused op has none).
    auto fused = rewriter.replaceOpWithNewOp<polykernel::FusedRmsNormMatMulOp>(
        matmul, matmul.getResult().getType(), /*input=*/rmsnorm.getInput(),
        /*weight=*/matmul.getB(), /*epsilon=*/rmsnorm.getEpsilonAttr());

    // Eliminated-intermediate tracking (Todo 17): tag the fusion kind + record
    // the eliminated intermediate's tensor type (the rmsnorm result).
    fused->setAttr(kFusedFromAttr,
                   rewriter.getStringAttr(kFusedFromRmsnormMatmul));
    fused->setAttr(kEliminatedTypeAttr,
                   TypeAttr::get(rmsnorm.getResult().getType()));

    // Erase the now-dead rmsnorm (its only use was the matmul we replaced).
    rewriter.eraseOp(rmsnorm);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class FuseRmsnormMatmulPass
    : public impl::FuseRmsnormMatmulBase<FuseRmsnormMatmulPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseRmsnormMatmulPattern>(&getContext());

    // Greedy driver (mirrors --canonicalize): applies the fusion, then folds +
    // DCEs ops left dead by the rewrites + simplifies regions (defaults).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseRmsnormMatmulPass() {
  return std::make_unique<FuseRmsnormMatmulPass>();
}

} // namespace mlir::polykernel
