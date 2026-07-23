//===- LowerToCuda.h - PolyKernel --lower-to-cuda pass ---------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--lower-to-cuda` source-emitter pass
// declaration (Todo 8 / Wave 2).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_LOWER_TO_CUDA_H
#define POLYKERNEL_PASSES_LOWER_TO_CUDA_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::polykernel {

/// Create the `--lower-to-cuda` pass: walks the PolyKernel IR and, for each
/// recognized compute op kind, EMITS a portable CUDA/HIP kernel source file
/// (.cu) into the output directory (string emission against the template in
/// kernels/template/kernel_common.h). Todo 8 emits RMSNorm; Todo 9/10 extend the
/// dispatch to gelu/silu/matmul/softmax. The IR itself is left unchanged — this
/// is a source emitter, not an IR-to-IR transform (Pinned contract F: it does
/// NOT use gpu-to-llvm / convert-to-llvm).
std::unique_ptr<::mlir::Pass> createLowerToCudaPass();

} // namespace mlir::polykernel

// Generated pass declaration for `LowerToCuda` (from Passes.td via
// `-gen-pass-decls -name PolyKernel`). The CRTP base class itself
// (impl::LowerToCudaBase) is emitted by GEN_PASS_DEF_LOWERTOCUDA in the pass's
// .cpp; this decl include is the pass's public declaration hook.
#define GEN_PASS_DECL_LOWERTOCUDA
#include "PolyKernel/Passes/Passes.h.inc"

#endif // POLYKERNEL_PASSES_LOWER_TO_CUDA_H
