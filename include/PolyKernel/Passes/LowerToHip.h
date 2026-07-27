//===- LowerToHip.h - PolyKernel --lower-to-hip pass -----------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--lower-to-hip` source-emitter pass
// declaration (Todo 19 / Wave 4).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_LOWER_TO_HIP_H
#define POLYKERNEL_PASSES_LOWER_TO_HIP_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--lower-to-hip` pass: the HIP sibling of `--lower-to-cuda`. Walks
/// the PolyKernel IR and, for each recognized compute op kind, EMITS a portable
/// CUDA/HIP kernel source file (.cu) into the output directory (string emission
/// against the template in kernels/template/kernel_common.h). It SHARES the
/// codegen template with `--lower-to-cuda`: the emitted .cu is byte-identical
/// portable compute — the two passes differ ONLY in that the HIP variant is
/// compiled by the hipcc driver with `-DPOLYKERNEL_HIP` (the POLYKERNEL_HIP
/// define is a DRIVER concern, not an emitter concern). The IR itself is left
/// unchanged — this is a source emitter, not an IR-to-IR transform (Pinned
/// contract F: it does NOT use gpu-to-llvm / convert-to-llvm).
std::unique_ptr<::mlir::Pass> createLowerToHipPass();

} // namespace mlir::polykernel

// Generated pass declaration for `LowerToHip` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::LowerToHipBase) is emitted by GEN_PASS_DEF_LOWERTOHIP in the pass's
// .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_LOWERTOHIP
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_LOWER_TO_HIP_H
