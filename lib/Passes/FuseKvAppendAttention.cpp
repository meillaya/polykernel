//===- FuseKvAppendAttention.cpp - KV-append+attention fusion ---*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--fuse-kv-append-attention` pass
// (Todo 40 / Wave 8).
//
//===----------------------------------------------------------------------===//
//
// Fusion pass extending the Wave-3 pattern (FuseSoftmaxMask.cpp): an
// OpRewritePattern on the CONSUMER op (polykernel.attention) that finds the
// PRODUCER-SIBLING (polykernel.kv_cache_update) sharing the same new K/V SSA
// values, guards the shared operands with a use-count check, replaces BOTH ops
// with the single fused op (carrying the query + cache + new K/V), and runs under
// the greedy pattern driver.
//
// Pattern (FuseKvAppendAttentionPattern), on polykernel.attention:
//   %updated = polykernel.kv_cache_update %cache, %nk, %nv
//   %out     = polykernel.attention %q, %nk, %nv
//     ==>
//   %out, %updated = polykernel.fused_kv_append_attention %q, %cache, %nk, %nv
//       {polykernel.fused_from = "kv_append_attention",
//        polykernel.eliminated_type_0 = <update result>,
//        polykernel.eliminated_type_1 = <attention result>}
//
// The new K/V are both APPENDED to the cache (kv_cache_update) and USED as the
// attention key/value (attention) - the fused op does both in one kernel
// (kernels/generated/{kv_cache_update,attention_prefill}.cu). The fused op's
// operand order matches PolyKernelOps.td exactly: (query, cache, new_keys,
// new_values); its results are (output, updated_cache), inferred from the operands
// via InferTypeOpInterface (output == query type, updated_cache == cache type).
//
// Guardrails:
//   - The attention's `key`/`value` operands must be the EXACT same SSA values as a
//     kv_cache_update's `new_keys`/`new_values`.
//   - SINGLE-USE CHECK (essential, per shared operand): `new_keys` and `new_values`
//     must each have EXACTLY TWO uses - the attention and the kv_cache_update. A
//     new K/V feeding any third op is NOT fused, or the appended tensor would be
//     duplicated.
//
// Eliminated-intermediate tracking (for the Todo 17 traffic report): this fusion
// eliminates TWO intermediates (the standalone updated-cache + the standalone
// attention output the fused op replaces), so it uses the INDEXED discardable
// attributes `polykernel.eliminated_type_0` (update result) +
// `polykernel.eliminated_type_1` (attention result), consistent with
// FuseMatmulBiasGelu.cpp's indexed scheme.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/Passes.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_FUSEKVAPPENDATTENTION
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable-attribute names attached to each fused op so the Todo 17 traffic
/// report can identify the fusion + the eliminated intermediates (indexed scheme:
/// this fusion eliminates two intermediates).
constexpr llvm::StringLiteral kFusedFromAttr = "polykernel.fused_from";
constexpr llvm::StringLiteral kEliminatedType0Attr =
    "polykernel.eliminated_type_0";
constexpr llvm::StringLiteral kEliminatedType1Attr =
    "polykernel.eliminated_type_1";
/// Value of `polykernel.fused_from` for this fusion kind.
constexpr llvm::StringLiteral kFusedFromKvAppendAttention = "kv_append_attention";

//===----------------------------------------------------------------------===//
// kv_cache_update(cache, nk, nv); attention(q, nk, nv)
//   ->  fused_kv_append_attention(q, cache, nk, nv)
//===----------------------------------------------------------------------===//

struct FuseKvAppendAttentionPattern
    : public OpRewritePattern<polykernel::AttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(polykernel::AttentionOp attn,
                                PatternRewriter &rewriter) const override {
    Value key = attn.getKey();
    Value value = attn.getValue();

    // Find the kv_cache_update that appends these SAME new K/V (the attention's
    // key/value operands are the freshly-appended tokens).
    polykernel::KVCacheUpdateOp update;
    for (OpOperand &use : key.getUses()) {
      auto candidate = dyn_cast<polykernel::KVCacheUpdateOp>(use.getOwner());
      if (candidate && candidate.getNewKeys() == key &&
          candidate.getNewValues() == value) {
        update = candidate;
        break;
      }
    }
    if (!update)
      return failure();

    // SINGLE-USE CHECK (per shared operand): the new K/V must be consumed by
    // EXACTLY the attention + this update. A third use must NOT be fused, or the
    // appended tensor would be duplicated.
    if (!key.hasNUses(2) || !value.hasNUses(2))
      return failure();

    // Replace both ops with one fused op. Result types are inferred from the
    // operands (output == query type, updated_cache == cache type), matching the
    // attention + update result types exactly. Operand order matches
    // PolyKernelOps.td: (query, cache, new_keys, new_values).
    auto fused = rewriter.create<polykernel::FusedKVAppendAttentionOp>(
        attn.getLoc(), attn.getQuery(), update.getCache(), key, value);

    // Eliminated-intermediate tracking (Todo 17, indexed): the standalone update
    // result (type 0) + the standalone attention result (type 1) are gone.
    fused->setAttr(kFusedFromAttr,
                   rewriter.getStringAttr(kFusedFromKvAppendAttention));
    fused->setAttr(kEliminatedType0Attr,
                   TypeAttr::get(update.getResult().getType()));
    fused->setAttr(kEliminatedType1Attr,
                   TypeAttr::get(attn.getResult().getType()));

    rewriter.replaceAllUsesWith(attn.getResult(), fused.getOutput());
    rewriter.replaceAllUsesWith(update.getResult(), fused.getUpdatedCache());
    rewriter.eraseOp(attn);
    rewriter.eraseOp(update);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class FuseKvAppendAttentionPass
    : public impl::FuseKvAppendAttentionBase<FuseKvAppendAttentionPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseKvAppendAttentionPattern>(&getContext());

    // Greedy driver (mirrors --canonicalize): applies the fusion, then folds +
    // DCEs ops left dead by the rewrites + simplifies regions (defaults).
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createFuseKvAppendAttentionPass() {
  return std::make_unique<FuseKvAppendAttentionPass>();
}

} // namespace mlir::polykernel
