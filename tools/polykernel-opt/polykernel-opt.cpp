//===- polykernel-opt.cpp - PolyKernel opt driver ---------------*- C++ -*-===//
//
// polykernel-opt: MLIR optimizer driver with the PolyKernel dialect registered
// (W1 spike). Modeled on mlir/examples/standalone/standalone-opt.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "PolyKernel/IR/PolyKernelDialect.h"
#include "PolyKernel/Passes/Passes.h"

int main(int argc, char **argv) {
  // Register the PolyKernel dialect and hand off to MlirOptMain.
  //
  // NOTE (nix out-of-tree constraint): the upstream standalone example also
  // calls mlir::registerAllDialects(registry) + mlir::registerAllPasses() and
  // links MLIRRegisterAllDialects / MLIRRegisterAllPasses. nixpkgs builds MLIR
  // with MLIR_INSTALL_AGGREGATE_OBJECTS=1 and does NOT install
  // libMLIRRegisterAll{Dialects,Passes}.a, so those symbols are not linkable
  // here (verified: no such archives in the mlir-21.1.8 store outputs). We
  // therefore register the PolyKernel dialect + the PolyKernel passes EXPLICITLY
  // (the TableGen-generated registration in PolyKernel/Passes/Passes.h):
  // registerPolyKernelPasses() registers every PolyKernel pass.
  mlir::DialectRegistry registry;
  registry.insert<mlir::polykernel::PolyKernelDialect>();

  // Explicit PolyKernel pass registration (no registerAllPasses / MLIRRegisterAll*).
  mlir::polykernel::registerPolyKernelPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "polykernel-opt\n", registry));
}
