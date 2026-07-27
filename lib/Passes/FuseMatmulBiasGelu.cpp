//===- FuseMatmulBiasGelu.cpp - PolyKernel matmul+bias+gelu fusion -*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-matmul-bias-gelu` pass
// (Todo 13 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// Fusion pass extending the Todo 12 pattern (FuseRmsnormMatmul.cpp): an
// OpRewritePattern on the CONSUMER op (polykernel.gelu) that walks its producer
// chain, guards EVERY absorbed intermediate with a single-use check, replaces
// the consumer with the fused op (carrying the producers' operands + the
// consumer's result type), erases the dead producers, and runs under the greedy
// pattern driver.
//
// Pattern (FuseMatmulBiasGeluPattern), on polykernel.gelu:
//   %mm  = polykernel.matmul %a, %b     : tensor<..xK>, tensor<KxN> -> tensor<..xN>
//   %bs  = polykernel.bias   %mm, %bias : tensor<..xN>, tensor<N>   -> tensor<..xN>
//   %out = polykernel.gelu   %bs        : tensor<..xN>
//     ==>
//   %out = polykernel.fused_matmul_bias_gelu %a, %b, %bias
//       {polykernel.fused_from = "matmul_bias_gelu",
//        polykernel.eliminated_type_0 = tensor<..xN>,   // matmul result
//        polykernel.eliminated_type_1 = tensor<..xN>}   // bias result
//       : tensor<..xK>, tensor<KxN>, tensor<N> -> tensor<..xN>
//
// Guardrails:
//   - The gelu's input must be DEFINED by a polykernel.bias, and THAT bias's
//     input must be DEFINED by a polykernel.matmul (the exact 3-op chain).
//   - SINGLE-USE CHECK (essential, per absorbed intermediate): the matmul result
//     must have EXACTLY ONE use (the bias) AND the bias result must have EXACTLY
//     ONE use (the gelu). A matmul feeding the bias AND another op (e.g. a
//     residual add), or a bias feeding the gelu AND another op, is NOT fused —
//     fusing would duplicate the shared producer.
//   - The fused op uses the EXACT operand order from PolyKernelOps.td
//     (a, b, bias) and produces the gelu's result type. fused_matmul_bias_gelu
//     declares NO attributes, so none are carried.
//   - The dead bias + matmul are erased explicitly (bias first — its only use was
//     the gelu we replaced — then matmul, whose only use was the erased bias).
//
// Multi-intermediate eliminated-type tracking (for the Todo 17 traffic report):
// this fusion eliminates TWO intermediates (the matmul result AND the bias
// result), so it uses INDEXED discardable attributes —
//   - `polykernel.fused_from = "matmul_bias_gelu"` (the fusion kind),
//   - `polykernel.eliminated_type_0 = <matmul result tensor type>`,
//   - `polykernel.eliminated_type_1 = <bias result tensor type>`.
// SCHEME (consistent across all Wave 3 fusion passes, keeps the Todo 17 walker
// simple): a fusion that eliminates ONE intermediate uses the unindexed
// `polykernel.eliminated_type` (rmsnorm_matmul, residual_rmsnorm, softmax_mask);
// a fusion that eliminates N>1 uses `polykernel.eliminated_type_0` .. `_N-1` in
// producer order (matmul first, then bias). The Todo 17 walker reads
// `polykernel.eliminated_type` if present, else iterates
// `polykernel.eliminated_type_N` for N=0,1,... while present, summing
// numElements(type) * sizeof(element) per eliminated intermediate. Attributes
// (not a pass statistic) are the mechanism: exact under the greedy driver (one
// tag set per committed fusion), IR-observable, and FileCheck-able.
//
// Applied greedily via applyPatternsGreedily (the same driver --canonicalize
// uses), which also folds + DCEs + simplifies regions.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/FuseMatmulBiasGelu.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_FUSEMATMULBIASGELU
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable-attribute names attached to each fused op so the Todo 17 traffic
/// report can identify the fusion and the eliminated intermediates. This fusion
/// eliminates TWO intermediates, so the eliminated-type attribute is indexed
/// (`_0` = matmul result, `_1` = bias result) — see the scheme note above.
constexpr llvm::StringLiteral kFusedFromAttr = "polykernel.fused_from";
constexpr llvm::StringLiteral kEliminatedTypeAttr0 =
    "polykernel.eliminated_type_0";
constexpr llvm::StringLiteral kEliminatedTypeAttr1 =
    "polykernel.eliminated_type_1";
/// Value of `polykernel.fused_from` for this fusion kind.
constexpr llvm::StringLiteral kFusedFromMatmulBiasGelu = "matmul_bias_gelu";

//===----------------------------------------------------------------------===//
// matmul(a, b); bias(mm, bias); gelu(bs)  ->  fused_matmul_bias_gelu(a, b, bias)
//===----------------------------------------------------------------------===//

struct FuseMatmulBiasGeluPattern
    : public OpRewritePattern<polykernel::GeluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::GeluOp gelu,
                                PatternRewriter &rewriter) const override {
    // The gelu's input must be produced by a polykernel.bias ...
    auto bias = gelu.getInput().getDefiningOp<polykernel::BiasOp>();
    if (!bias)
      return failure();

    // ... whose input must be produced by a polykernel.matmul (the 3-op chain).
    auto matmul = bias.getInput().getDefiningOp<polykernel::MatMulOp>();
    if (!matmul)
      return failure();

    // SINGLE-USE CHECK (essential, per absorbed intermediate): fuse only when the
    // matmul result has EXACTLY ONE use (the bias) AND the bias result has EXACTLY
    // ONE use (the gelu). A multi-use matmul or bias must NOT be fused, or the
    // shared producer would be duplicated.
    if (!matmul.getResult().hasOneUse())
      return failure();
    if (!bias.getResult().hasOneUse())
      return failure();

    // Replace the gelu with fused_matmul_bias_gelu(a, b, bias), producing the
    // gelu's result type. Operand order matches PolyKernelOps.td exactly:
    // (a, b, bias). The fused op declares no attributes, so none are carried.
    auto fused = rewriter.replaceOpWithNewOp<polykernel::FusedMatMulBiasGeluOp>(
        gelu, gelu.getResult().getType(), /*a=*/matmul.getA(),
        /*b=*/matmul.getB(), /*bias=*/bias.getBias());

    // Multi-intermediate eliminated-type tracking (Todo 17): tag the fusion kind +
    // record BOTH eliminated intermediates' tensor types, in producer order
    // (matmul result = _0, bias result = _1).
    fused->setAttr(kFusedFromAttr,
                   rewriter.getStringAttr(kFusedFromMatmulBiasGelu));
    fused->setAttr(kEliminatedTypeAttr0,
                   TypeAttr::get(matmul.getResult().getType()));
    fused->setAttr(kEliminatedTypeAttr1,
                   TypeAttr::get(bias.getResult().getType()));

    // Erase the dead producers in reverse topological order: the bias first (its
    // only use was the gelu we replaced), then the matmul (whose only use was the
    // now-erased bias).
    rewriter.eraseOp(bias);
    rewriter.eraseOp(matmul);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class FuseMatmulBiasGeluPass
    : public impl::FuseMatmulBiasGeluBase<FuseMatmulBiasGeluPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseMatmulBiasGeluPattern>(&getContext());

    // Greedy driver (mirrors --canonicalize): applies the fusion, then folds +
    // DCEs ops left dead by the rewrites + simplifies regions (defaults).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseMatmulBiasGeluPass() {
  return std::make_unique<FuseMatmulBiasGeluPass>();
}

} // namespace mlir::polykernel
