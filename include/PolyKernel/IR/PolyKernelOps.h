//===- PolyKernelOps.h - PolyKernel dialect operations ---------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: operation declarations (Todo 3 / Wave 1).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_IR_POLYKERNELOPS_H
#define POLYKERNEL_IR_POLYKERNELOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "PolyKernel/IR/PolyKernelDialect.h"

// Generated operation declarations (-gen-op-decls).
#define GET_OP_CLASSES
#include "PolyKernel/IR/PolyKernelOps.h.inc"

#endif // POLYKERNEL_IR_POLYKERNELOPS_H
