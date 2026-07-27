//===- Passes.h - PolyKernel pass registration -----------------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: umbrella header for PolyKernel passes +
// their EXPLICIT registration (Todo 4 / Wave 1).
//
//===----------------------------------------------------------------------===//
//
// Include this from a tool (e.g. polykernel-opt) and call
// `mlir::polykernel::registerPolyKernelPasses()` before `MlirOptMain` to register
// every PolyKernel pass. Registration is EXPLICIT via the TableGen-generated
// GEN_PASS_REGISTRATION block — we deliberately do NOT use registerAllPasses /
// MLIRRegisterAll* (nixpkgs MLIR does not ship the aggregate object archives;
// see docs/compiler_pipeline.md).
//
// Passes:
//   - `--infer-shapes`   (Todo 4): shape + dtype inference (InferShapes.h).
//   - `--canonicalize`   (Todo 5): canonicalization patterns + DCE
//     (Canonicalize.h).
//   - `--lower-to-cuda`  (Todo 8): emit portable CUDA/HIP kernel source
//     (LowerToCuda.h).
//   - `--lower-to-hip`   (Todo 19): HIP sibling of --lower-to-cuda; emits the
//     SAME portable kernel source (shared kernel_common.h template), compiled by
//     the hipcc driver with -DPOLYKERNEL_HIP (LowerToHip.h).
//   - `--fuse-rmsnorm-matmul` (Todo 12): fuse single-use rmsnorm + matmul into
//     fused_rmsnorm_matmul (FuseRmsnormMatmul.h).
//   - `--fuse-matmul-bias-gelu` (Todo 13): fuse single-use matmul + bias + gelu
//     into fused_matmul_bias_gelu (FuseMatmulBiasGelu.h).
//   - `--fuse-residual-rmsnorm` (Todo 13): fuse single-use residual add + rmsnorm
//     into fused_residual_rmsnorm (FuseResidualRmsnorm.h).
//   - `--fuse-softmax-mask` (Todo 13): fuse single-use mask add + softmax into
//     fused_softmax_mask (FuseSoftmaxMask.h).
//   - `--infer-tile-layout` (Todo 15): assign tile shapes (BLOCK_M/N/K) + operand
//     layout to the matmul ops via shape heuristics (InferTileLayout.h).
//   - `--plan-memory` (Todo 16): compute a per-kernel memory plan (smem budget +
//     non-fused workspace + over-budget guard) for the matmul ops (PlanMemory.h).
//   - `--emit-kernel-report` (Todo 27): emit/attach the contract-H per-kernel
//     report skeleton (kernel/backend/arch/path/shape/tile/smem/fusion) for the
//     matmul ops; the compile-time fields are filled by polykernel-report.
// All are picked up by registerPolyKernelPasses() automatically.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_PASSES_H
#define POLYKERNEL_PASSES_PASSES_H

#include "PolyKernel/Passes/Canonicalize.h"
#include "PolyKernel/Passes/FuseMatmulBiasGelu.h"
#include "PolyKernel/Passes/FuseResidualRmsnorm.h"
#include "PolyKernel/Passes/FuseRmsnormMatmul.h"
#include "PolyKernel/Passes/FuseSoftmaxMask.h"
#include "PolyKernel/Passes/InferShapes.h"
#include "PolyKernel/Passes/InferTileLayout.h"
#include "PolyKernel/Passes/LowerToCuda.h"
#include "PolyKernel/Passes/LowerToHip.h"
#include "PolyKernel/Passes/PlanMemory.h"

namespace mlir::polykernel {

/// Create the `--emit-kernel-report` pass (Todo 27 / Wave 5): walks the
/// matmul-structured ops (matmul, fused_rmsnorm_matmul, fused_matmul_bias_gelu)
/// and, for each generated kernel, attaches the IR-derivable contract-H report
/// fields as DISCARDABLE attributes (`polykernel.report_kernel/backend/arch/path`)
/// and emits a per-kernel report manifest (`kernel_report.json`) into
/// `--output-dir`. The compile-time fields (registers/spills/occupancy/traffic/
/// roofline/bottleneck/suggested-fixes) are filled by the `polykernel-report`
/// CLI, which merges this skeleton with the CUDA + AMD analyzers + autotuner.
/// (Declared here rather than a per-pass header to keep the pass surface in one
/// place; the CRTP base is emitted by GEN_PASS_DEF_EMITKERNELREPORT in the .cpp.)
std::unique_ptr<::mlir::Pass> createEmitKernelReportPass();

// Generated registration helpers: registerInferShapesPass() /
// registerCanonicalizePass() (per pass) and registerPolyKernelPasses() (all
// PolyKernel passes), from Passes.td via `-gen-pass-decls -name PolyKernel`.
#define GEN_PASS_REGISTRATION
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

// Generated pass declaration for `EmitKernelReport` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). Mirrors the per-pass headers' decl hook;
// the CRTP base (impl::EmitKernelReportBase) is emitted by
// GEN_PASS_DEF_EMITKERNELREPORT in lib/Passes/EmitKernelReport.cpp.
#define GEN_PASS_DECL_EMITKERNELREPORT
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_PASSES_H
