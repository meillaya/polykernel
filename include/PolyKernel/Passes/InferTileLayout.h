//===- InferTileLayout.h - PolyKernel tile-layout pass ---------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--infer-tile-layout` pass
// declaration (Todo 15 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_INFER_TILE_LAYOUT_H
#define POLYKERNEL_PASSES_INFER_TILE_LAYOUT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--infer-tile-layout` pass: walks the matmul-structured ops
/// (matmul, fused_rmsnorm_matmul, fused_matmul_bias_gelu) and assigns each an
/// initial tile shape + operand layout as DISCARDABLE attributes —
/// `polykernel.tile_m` / `polykernel.tile_n` / `polykernel.tile_k` (I64Attr) and
/// `polykernel.layout` (StringAttr, "row_major"). The tiles are shape-heuristic
/// DEFAULTS (the autotuner, Todo 24, refines them later); each tile dim is
/// clamped to its problem dim so small problems get small tiles. Runs on typed
/// IR (after `--infer-shapes`).
std::unique_ptr<::mlir::Pass> createInferTileLayoutPass();

} // namespace mlir::polykernel

// Generated pass declaration for `InferTileLayout` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::InferTileLayoutBase) is emitted by GEN_PASS_DEF_INFERTILELAYOUT in the
// pass's .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_INFERTILELAYOUT
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_INFER_TILE_LAYOUT_H
