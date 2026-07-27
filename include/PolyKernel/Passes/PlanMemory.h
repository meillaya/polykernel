//===- PlanMemory.h - PolyKernel memory-plan pass --------------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--plan-memory` pass declaration
// (Todo 16 / Wave 3).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_PLAN_MEMORY_H
#define POLYKERNEL_PASSES_PLAN_MEMORY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--plan-memory` pass: walks the matmul-structured ops (matmul,
/// fused_rmsnorm_matmul, fused_matmul_bias_gelu) and computes a per-kernel memory
/// plan, attaching DISCARDABLE attributes (the op set stays closed — no new
/// declared op attributes are added to PolyKernelOps.td):
///
///   - `polykernel.smem_bytes` (I64Attr): the tiled shared-memory budget,
///     `(tile_m*tile_k + tile_k*tile_n) * dtype_bytes * stages` (see PlanMemory.cpp
///     for the documented formula; tiles fall back to 16x16x16 when the
///     `--infer-tile-layout` attrs are absent, stages = 2 double-buffer).
///   - `polykernel.workspace_bytes` (I64Attr): global scratch for intermediates NOT
///     fused away — 0 for a fused op (its eliminated intermediates are gone), the
///     result-tensor size for a standalone op.
///   - `polykernel.smem_over_budget` (BoolAttr=true): set only when `smem_bytes`
///     exceeds the documented per-block limit (163KB, the sm_80 budget).
///
/// Runs on typed IR (after `--infer-shapes`, and after `--infer-tile-layout` when
/// tile-accurate budgets are wanted); ops with a dynamic operand/result dim are
/// skipped.
std::unique_ptr<::mlir::Pass> createPlanMemoryPass();

} // namespace mlir::polykernel

// Generated pass declaration for `PlanMemory` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::PlanMemoryBase) is emitted by GEN_PASS_DEF_PLANMEMORY in the pass's .cpp;
// this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_PLANMEMORY
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_PLAN_MEMORY_H
