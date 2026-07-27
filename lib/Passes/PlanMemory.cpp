//===- PlanMemory.cpp - PolyKernel memory-plan pass -------------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--plan-memory` pass (Todo 16 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// Computes a PER-KERNEL MEMORY PLAN for each matmul-structured op and records it
// as DISCARDABLE attributes (consistent with the tile attrs of Todo 15 and the
// fusion-tracking attrs of Todos 12/13 — the op set stays CLOSED; no new declared
// op attributes are added to PolyKernelOps.td). These are PLANNING / REPORTING
// estimates for the autotuner + traffic report (Todo 17/24), NOT cycle-accurate.
//
// Scope: the three PRIMARY matmul ops — `matmul`, `fused_rmsnorm_matmul`,
// `fused_matmul_bias_gelu` (the same scope as --infer-tile-layout; qkv_projection
// / fused_kv_append_attention have a head-split / cache structure and are out of
// scope for this initial plan).
//
// THE MEMORY MODEL (simple, deterministic, documented):
//
//   Shared-memory budget — `polykernel.smem_bytes` (I64Attr):
//
//     smem_bytes = (tile_m*tile_k + tile_k*tile_n) * dtype_bytes * kPipelineStages
//
//     The tiled GEMM stages one A-tile (tile_m x tile_k) and one B-tile
//     (tile_k x tile_n) into shared memory per pipeline stage.
//       - tile_m / tile_n / tile_k: read from the `polykernel.tile_m/n/k` attrs
//         attached by --infer-tile-layout. ABSENT -> FALLBACK 16x16x16 (the
//         baseline CUDA GEMM tile, kernels/generated/matmul.cu). The pass does NOT
//         assume the attrs are present (Todo 15 only sets them on concrete shapes).
//       - dtype_bytes: operand-0 element type width / 8 (bf16/fp16 = 2, fp32 = 4).
//       - kPipelineStages = 2: a documented default DOUBLE-BUFFER (one tile loads
//         while the previous computes). Single-buffer would be 1; we plan for the
//         common double-buffered case so the budget is conservative.
//     `polykernel.layout` is READ (fallback "row_major") rather than hardcoded; the
//     baseline footprint is layout-agnostic (a tile's element count is independent
//     of row/col-major order), so the layout only surfaces in the over-budget
//     diagnostic today — a future model may branch on it.
//
//   Workspace — `polykernel.workspace_bytes` (I64Attr):
//
//     Global scratch for intermediates that are NOT fused away.
//       - FUSED op (carries `polykernel.fused_from`): its eliminated intermediates
//         (the `polykernel.eliminated_type*` attrs Todo 12/13 attach) are fused away
//         — they never materialize as separate global buffers — so workspace = 0.
//         (Those attrs record WHAT was eliminated for the Todo 17 traffic report;
//         this pass only needs to know the op IS fused, hence 0.)
//       - UNFUSED (standalone) op: workspace = result-tensor size =
//         numElements(result) * result_dtype_bytes. Rationale: absent fusion, the
//         op's output is a global-memory buffer the planning layer accounts as
//         working set; fusion is precisely what drives such buffers to 0.
//
//   Over-budget guard — `polykernel.smem_over_budget` (BoolAttr=true):
//
//     kSmemLimitBytes = 163 * 1024 = 166912: the sm_80 per-block shared-memory
//     limit, used as a single CONSERVATIVE default (sm_90 allows 227KB, but a plan
//     that fits the smaller sm_80 budget fits on both). When smem_bytes exceeds the
//     limit, the flag is attached AND a warning is emitted (the tile config may not
//     fit and the autotuner must shrink it).
//
// Ops with any dynamic operand or result dim are SKIPPED (consistent with Todo 15):
// the plan runs on typed IR after --infer-shapes, where shapes are concrete, and a
// dynamic dim has no computable footprint.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/PlanMemory.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::polykernel {
#define GEN_PASS_DEF_PLANMEMORY
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable attributes READ (attached by earlier Wave 3 passes).
constexpr llvm::StringLiteral kTileMAttr = "polykernel.tile_m";
constexpr llvm::StringLiteral kTileNAttr = "polykernel.tile_n";
constexpr llvm::StringLiteral kTileKAttr = "polykernel.tile_k";
constexpr llvm::StringLiteral kLayoutAttr = "polykernel.layout";
constexpr llvm::StringLiteral kFusedFromAttr = "polykernel.fused_from";

/// Discardable attributes ATTACHED by this pass (the memory plan).
constexpr llvm::StringLiteral kSmemBytesAttr = "polykernel.smem_bytes";
constexpr llvm::StringLiteral kWorkspaceBytesAttr = "polykernel.workspace_bytes";
constexpr llvm::StringLiteral kSmemOverBudgetAttr = "polykernel.smem_over_budget";

/// Fallback tile when the --infer-tile-layout attrs are absent: the baseline CUDA
/// GEMM tile (kernels/generated/matmul.cu).
constexpr int64_t kFallbackTile = 16;
/// Fallback operand layout when `polykernel.layout` is absent.
constexpr llvm::StringLiteral kLayoutRowMajor = "row_major";
/// Default pipeline depth: a double-buffer (one tile loads while the previous
/// computes). Documented in the file header.
constexpr int64_t kPipelineStages = 2;
/// Per-block shared-memory limit (bytes): the sm_80 budget, a single conservative
/// default (sm_90 allows 227KB; a plan fitting sm_80 fits both). See header.
constexpr int64_t kSmemLimitBytes = 163 * 1024;

/// True when `type` is a ranked tensor with NO dynamic dim (so its footprint is
/// computable). Unranked / dynamic -> false (the op is skipped).
bool isStaticRankedTensor(Type type) {
  auto shaped = dyn_cast<RankedTensorType>(type);
  return shaped && shaped.hasStaticShape();
}

/// Product of a static ranked tensor's dims (numElements). Caller guarantees
/// isStaticRankedTensor(type).
int64_t numElements(Type type) {
  int64_t count = 1;
  for (int64_t dim : cast<RankedTensorType>(type).getShape())
    count *= dim;
  return count;
}

/// Element width in bytes (bf16/fp16 = 2, fp32 = 4, ...). Caller guarantees a
/// ranked tensor with an int/float element type.
int64_t elementBytes(Type type) {
  return cast<RankedTensorType>(type).getElementType().getIntOrFloatBitWidth() / 8;
}

/// Read an I64 tile attr, falling back to `fallback` when absent (Todo 15 sets the
/// tile attrs only on concrete shapes, so absence is a normal, handled case).
int64_t readTileAttr(Operation *op, llvm::StringLiteral name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class PlanMemoryPass : public impl::PlanMemoryBase<PlanMemoryPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto i64Type = IntegerType::get(ctx, 64);

    getOperation().walk([&](Operation *op) {
      // Scope: the three primary matmul ops (see the file header for why the
      // matmul-like qkv_projection / fused_kv_append_attention are excluded).
      if (!isa<polykernel::MatMulOp, polykernel::FusedRmsNormMatMulOp,
               polykernel::FusedMatMulBiasGeluOp>(op))
        return;

      // Skip ops with a dynamic operand/result dim (consistent with Todo 15): no
      // computable footprint. Operand 0/1 are the [..,M,K] lhs / [K,N] rhs; result
      // 0 is the [..,M,N] output (all three ops have exactly one result).
      Type aType = op->getOperand(0).getType();
      Type bType = op->getOperand(1).getType();
      Type resultType = op->getResult(0).getType();
      if (!isStaticRankedTensor(aType) || !isStaticRankedTensor(bType) ||
          !isStaticRankedTensor(resultType))
        return;

      // Tile shape: read the --infer-tile-layout attrs, falling back to the
      // baseline 16x16x16 when absent. Layout is READ (not hardcoded); the baseline
      // footprint is layout-agnostic, so it only surfaces in the diagnostic below.
      int64_t tileM = readTileAttr(op, kTileMAttr, kFallbackTile);
      int64_t tileN = readTileAttr(op, kTileNAttr, kFallbackTile);
      int64_t tileK = readTileAttr(op, kTileKAttr, kFallbackTile);
      auto layoutAttr = op->getAttrOfType<StringAttr>(kLayoutAttr);
      StringRef layout =
          layoutAttr ? layoutAttr.getValue() : StringRef(kLayoutRowMajor);

      // Shared-memory budget (see header formula): A-tile + B-tile, times dtype
      // bytes (from operand 0), times the double-buffer pipeline depth.
      int64_t dtypeBytes = elementBytes(aType);
      int64_t smemBytes =
          (tileM * tileK + tileK * tileN) * dtypeBytes * kPipelineStages;

      // Workspace: 0 for a fused op (eliminated intermediates are fused away), else
      // the standalone result-tensor size (see header for the model + rationale).
      bool fused = op->hasAttr(kFusedFromAttr);
      int64_t workspaceBytes =
          fused ? 0 : numElements(resultType) * elementBytes(resultType);

      op->setAttr(kSmemBytesAttr, IntegerAttr::get(i64Type, smemBytes));
      op->setAttr(kWorkspaceBytesAttr, IntegerAttr::get(i64Type, workspaceBytes));

      // Over-budget guard: flag + warn when the shared-memory budget exceeds the
      // per-block limit (the autotuner must shrink the tile).
      if (smemBytes > kSmemLimitBytes) {
        op->setAttr(kSmemOverBudgetAttr, BoolAttr::get(ctx, true));
        op->emitWarning()
            << "shared-memory budget " << smemBytes
            << " bytes exceeds the " << kSmemLimitBytes
            << "-byte per-block limit (sm_80); tile config (layout=" << layout
            << ") may not fit";
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> createPlanMemoryPass() {
  return std::make_unique<PlanMemoryPass>();
}

} // namespace mlir::polykernel
