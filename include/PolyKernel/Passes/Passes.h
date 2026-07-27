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

namespace mlir::polykernel {
// Generated registration helpers: registerInferShapesPass() /
// registerCanonicalizePass() (per pass) and registerPolyKernelPasses() (all
// PolyKernel passes), from Passes.td via `-gen-pass-decls -name PolyKernel`.
#define GEN_PASS_REGISTRATION
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

#endif // POLYKERNEL_PASSES_PASSES_H
