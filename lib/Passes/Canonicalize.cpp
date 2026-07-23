//===- Canonicalize.cpp - PolyKernel canonicalization pass ------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--canonicalize` pass (Todo 5 /
// Wave 1).
//
//===----------------------------------------------------------------------===//
//
// Design A (self-contained): the pass builds a RewritePatternSet with the
// PolyKernel canonicalization patterns and runs `applyPatternsGreedily`, whose
// default GreedyRewriteConfig also folds ops (fold = true), removes ops left
// dead by the rewrites (DCE) and simplifies regions. The patterns therefore
// get dead-code cleanup for free: e.g. eliminating `add(x, 0)` leaves the zero
// constant unused, and the driver erases it.
//
// Patterns:
//   1. AddZeroElimination — `polykernel.add(x, zero) -> x` (and the commuted
//      `add(zero, x) -> x`), where `zero` is a dense-zero `arith.constant`.
//      `add` has SameTypeOperands and its result type == operand type, so
//      replacing the op with the non-zero operand keeps types consistent (no
//      inferReturnTypes call needed; the refineReturnTypes verify hook is a
//      global no-op anyway — see docs/compiler_pipeline.md, Todo 4).
//   2. GeluConstantFold / SiluConstantFold — gelu/silu of a dense FP constant
//      fold to a computed `arith.constant`: every element is mapped through
//      the activation, evaluated in double precision and rounded to the
//      element type (exact for splats; a documented approximation for general
//      dense constants of f16/bf16). GELU is the exact erf-based definition
//      `0.5 * x * (1 + erf(x / sqrt(2)))`; SiLU is `x / (1 + exp(-x))`.
//   3. SoftmaxSplatFold (no-op fold) — `softmax(splat(c)) -> splat(1/N)`,
//      N = size of the reduction axis (`axis` attr, default last dim): softmax
//      of equal values is the uniform distribution because exp(c) cancels in
//      the normalization, so no transcendental math is needed.
//
// Scope: elementwise folding covers gelu/silu (the pure elementwise
// activations); non-splat softmax is deliberately NOT folded (it needs a
// reduction + exp over the axis — heavier, out of Todo 5 scope). Non-foldable
// ops (e.g. add of two non-constant operands) are left unchanged.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/Canonicalize.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>

namespace mlir::polykernel {
#define GEN_PASS_DEF_CANONICALIZE
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Return the DenseElementsAttr if `value` is an `arith.constant` holding one;
/// null otherwise.
DenseElementsAttr getDenseConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return {};
  return dyn_cast<DenseElementsAttr>(constant.getValue());
}

/// True iff every element of `attr` is zero (float or integer constants).
bool isDenseZero(DenseElementsAttr attr) {
  if (isa<FloatType>(attr.getElementType())) {
    if (attr.isSplat())
      return attr.getSplatValue<APFloat>().isZero();
    return llvm::all_of(attr.getValues<APFloat>(),
                        [](const APFloat &element) { return element.isZero(); });
  }
  if (isa<IntegerType>(attr.getElementType())) {
    if (attr.isSplat())
      return attr.getSplatValue<APInt>().isZero();
    return llvm::all_of(attr.getValues<APInt>(),
                        [](const APInt &element) { return element.isZero(); });
  }
  return false;
}

//===----------------------------------------------------------------------===//
// 1. Identity residual-add elimination: add(x, 0) -> x
//===----------------------------------------------------------------------===//

struct AddZeroElimination : public OpRewritePattern<polykernel::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::AddOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr lhs = getDenseConstant(op.getLhs());
    DenseElementsAttr rhs = getDenseConstant(op.getRhs());
    bool lhsZero = lhs && isDenseZero(lhs);
    bool rhsZero = rhs && isDenseZero(rhs);
    if (!lhsZero && !rhsZero)
      return failure();
    // add(x, 0) -> x; add(0, x) -> x; add(0, 0) -> the rhs zero constant.
    rewriter.replaceOp(op, lhsZero ? op.getRhs() : op.getLhs());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 2. Constant folding of pure elementwise ops (gelu/silu)
//===----------------------------------------------------------------------===//

/// Map every element of a dense floating-point constant through `fn` (evaluated
/// in double precision, rounded to the element type) and return the folded
/// constant; null for non-ranked / non-float constants.
DenseElementsAttr mapConstantElements(DenseElementsAttr input,
                                      llvm::function_ref<double(double)> fn) {
  auto type = dyn_cast<RankedTensorType>(input.getType());
  if (!type)
    return {};
  auto elementType = dyn_cast<FloatType>(type.getElementType());
  if (!elementType)
    return {};
  const llvm::fltSemantics &semantics = elementType.getFloatSemantics();
  return input.mapValues(elementType, [&](const APFloat &value) {
    // APFloat(semantics, double) is deleted in LLVM 21: build in IEEE double,
    // then convert (round-to-nearest) to the element type's semantics.
    APFloat folded(fn(value.convertToDouble()));
    bool losesInfo;
    folded.convert(semantics, APFloat::rmNearestTiesToEven, &losesInfo);
    return folded.bitcastToAPInt();
  });
}

struct GeluConstantFold : public OpRewritePattern<polykernel::GeluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::GeluOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr input = getDenseConstant(op.getInput());
    if (!input)
      return failure();
    // Exact (erf-based) GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    DenseElementsAttr folded = mapConstantElements(input, [](double x) {
      return 0.5 * x * (1.0 + std::erf(x * 0.70710678118654752440));
    });
    if (!folded)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, folded);
    return success();
  }
};

struct SiluConstantFold : public OpRewritePattern<polykernel::SiluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::SiluOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr input = getDenseConstant(op.getInput());
    if (!input)
      return failure();
    // SiLU / swish: x * sigmoid(x) = x / (1 + exp(-x)).
    DenseElementsAttr folded = mapConstantElements(
        input, [](double x) { return x / (1.0 + std::exp(-x)); });
    if (!folded)
      return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, folded);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// 3. No-op fold: softmax(splat(c)) -> splat(1/N)
//===----------------------------------------------------------------------===//

struct SoftmaxSplatFold : public OpRewritePattern<polykernel::SoftmaxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    DenseElementsAttr input = getDenseConstant(op.getInput());
    if (!input || !input.isSplat())
      return failure();
    auto type = dyn_cast<RankedTensorType>(input.getType());
    if (!type || type.getRank() == 0)
      return failure();
    auto elementType = dyn_cast<FloatType>(type.getElementType());
    if (!elementType)
      return failure();
    int64_t axis = op.getAxis().value_or(type.getRank() - 1);
    if (axis < 0 || axis >= type.getRank())
      return failure();
    int64_t n = type.getShape()[axis];
    if (n <= 0)
      return failure();
    // Softmax of equal values over the axis is uniform: exp(c) / (N * exp(c)).
    APFloat uniform(1.0 / static_cast<double>(n));
    bool losesInfo;
    uniform.convert(elementType.getFloatSemantics(),
                    APFloat::rmNearestTiesToEven, &losesInfo);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, DenseElementsAttr::get(type, uniform));
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class CanonicalizePass : public impl::CanonicalizeBase<CanonicalizePass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddZeroElimination, GeluConstantFold, SiluConstantFold,
                 SoftmaxSplatFold>(&getContext());

    // The greedy driver folds ops, DCEs ops left dead by the rewrites and
    // simplifies regions (GreedyRewriteConfig defaults: fold = true,
    // aggressive region simplification).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createCanonicalizePass() {
  return std::make_unique<CanonicalizePass>();
}

} // namespace mlir::polykernel
