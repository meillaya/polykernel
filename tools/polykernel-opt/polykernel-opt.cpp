//===- polykernel-opt.cpp - PolyKernel opt driver ---------------*- C++ -*-===//
//
// polykernel-opt: MLIR optimizer driver with the PolyKernel dialect registered
// (W1 spike). Modeled on mlir/examples/standalone/standalone-opt.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "PolyKernel/IR/PolyKernelDialect.h"

int main(int argc, char **argv) {
  // Register the PolyKernel dialect and hand off to MlirOptMain.
  //
  // NOTE (nix out-of-tree constraint): the upstream standalone example also
  // calls mlir::registerAllDialects(registry) + mlir::registerAllPasses() and
  // links MLIRRegisterAllDialects / MLIRRegisterAllPasses. nixpkgs builds MLIR
  // with MLIR_INSTALL_AGGREGATE_OBJECTS=1 and does NOT install
  // libMLIRRegisterAll{Dialects,Passes}.a, so those symbols are not linkable
  // here (verified: no such archives in the mlir-21.1.8 store outputs). The W1
  // spike only needs the PolyKernel dialect registered + a working MlirOptMain
  // (the smoke test parses a builtin `module {}`), so we register PolyKernel
  // directly. registerAll* + upstream passes/dialects can be revisited once we
  // need them (Todo 3+), e.g. by linking the individual dialect/pass libs that
  // nix DOES ship.
  mlir::DialectRegistry registry;
  registry.insert<mlir::polykernel::PolyKernelDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "polykernel-opt\n", registry));
}
