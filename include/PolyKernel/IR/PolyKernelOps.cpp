//===- PolyKernelOps.cpp - PolyKernel dialect operations -------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: operation definitions (Todo 3 / Wave 1).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::polykernel;

// Generated operation definitions (-gen-op-defs).
#define GET_OP_CLASSES
#include "PolyKernel/IR/PolyKernelOps.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

bool FuncOp::isExternal() { return getBody().empty(); }

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

LogicalResult FuncOp::verify() {
  // Light verifier (Todo 3): a non-external function's entry block arguments
  // must match the declared signature. Full body/result type checking and shape
  // inference land in Todo 4.
  if (isExternal())
    return success();
  Region &body = getBody();
  if (body.empty())
    return success();
  Block &entry = body.front();
  ArrayRef<Type> inputs = getArgumentTypes();
  if (entry.getNumArguments() != inputs.size())
    return emitOpError("entry block must have ")
           << inputs.size() << " arguments to match the function signature";
  for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
    Type argType = entry.getArgument(i).getType();
    if (argType != inputs[i])
      return emitOpError("type of entry block argument #")
             << i << " (" << argType
             << ") must match the function signature (" << inputs[i] << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Light verifier: operands must match the enclosing polykernel.func results.
  auto func = (*this)->getParentOfType<FuncOp>();
  if (!func)
    return emitOpError("must be nested inside a polykernel.func");
  ArrayRef<Type> expected = func.getResultTypes();
  if (getNumOperands() != expected.size())
    return emitOpError("expected ")
           << expected.size()
           << " operands to match the function results, got "
           << getNumOperands();
  for (unsigned i = 0, e = expected.size(); i != e; ++i) {
    Type operandType = getOperand(i).getType();
    if (operandType != expected[i])
      return emitOpError("operand #")
             << i << " type " << operandType
             << " does not match function result type " << expected[i];
  }
  return success();
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  // Light verifier (cheap rank sanity only; full shape inference is Todo 4):
  // both operands must be ranked tensors of rank >= 2.
  auto aType = llvm::dyn_cast<RankedTensorType>(getA().getType());
  auto bType = llvm::dyn_cast<RankedTensorType>(getB().getType());
  if (!aType || !bType)
    return emitOpError("operands must be ranked tensors");
  if (aType.getRank() < 2 || bType.getRank() < 2)
    return emitOpError("operands must have rank >= 2");
  return success();
}

//===----------------------------------------------------------------------===//
// AttentionOp
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  // Light verifier (cheap rank sanity only; full shape inference is Todo 4):
  // Q, K, V must be ranked tensors of rank >= 2.
  auto qType = llvm::dyn_cast<RankedTensorType>(getQuery().getType());
  auto kType = llvm::dyn_cast<RankedTensorType>(getKey().getType());
  auto vType = llvm::dyn_cast<RankedTensorType>(getValue().getType());
  if (!qType || !kType || !vType)
    return emitOpError("Q/K/V operands must be ranked tensors");
  if (qType.getRank() < 2 || kType.getRank() < 2 || vType.getRank() < 2)
    return emitOpError("Q/K/V operands must have rank >= 2");
  return success();
}
