//===- PolyKernelDialect.td - PolyKernel dialect -----------*- tablegen -*-===//
//
// PolyKernel out-of-tree MLIR dialect (W1 spike).
//
//===----------------------------------------------------------------------===//
//
// W1 spike: this is a MINIMAL dialect declaration only - NO operations, types,
// or attributes yet. It exists to prove the out-of-tree MLIR-21 build +
// dialect registration + polykernel-opt tool + lit pipeline. Operations are
// added in Todo 3.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_DIALECT
#define POLYKERNEL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PolyKernel dialect definition.
//===----------------------------------------------------------------------===//

def PolyKernel_Dialect : Dialect {
  let name = "polykernel";
  let summary = "PolyKernel out-of-tree dialect (W1 spike: decl only).";
  let description = [{
    The PolyKernel dialect. This W1 spike carries only the dialect declaration
    to validate the out-of-tree MLIR toolchain (build + registration + opt tool
    + lit). PolyKernel operations are introduced in Todo 3.
  }];
  let cppNamespace = "::mlir::polykernel";
}

#endif // POLYKERNEL_DIALECT
