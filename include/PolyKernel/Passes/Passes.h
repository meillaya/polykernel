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
// Todo 5 adds the `--canonicalize` pass: declare it in Passes.td, add its
// create helper header include below, and it is picked up by
// registerPolyKernelPasses() automatically.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES_PASSES_H
#define POLYKERNEL_PASSES_PASSES_H

#include "PolyKernel/Passes/InferShapes.h"
// Todo 5: #include "PolyKernel/Passes/Canonicalize.h"

namespace mlir::polykernel {
// Generated registration helpers: registerInferShapesPass() (per pass) and
// registerPolyKernelPasses() (all PolyKernel passes), from Passes.td via
// `-gen-pass-decls -name PolyKernel`.
#define GEN_PASS_REGISTRATION
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

#endif // POLYKERNEL_PASSES_PASSES_H
