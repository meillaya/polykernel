//===- PolyKernelOps.cpp - PolyKernel dialect operations -------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: operation definitions (Todo 3 / Wave 1).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
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

//===----------------------------------------------------------------------===//
// Shape + dtype inference (Todo 4 / Wave 1).
//===----------------------------------------------------------------------===//
//
// Each compute op implements InferTypeOpInterface::inferReturnTypes (adaptor
// form), computing its result type(s) from its operands per the PolyKernel shape
// rules. On any inconsistency these emit a diagnostic (via the optional location)
// and return failure; the --infer-shapes pass turns that into signalPassFailure().
//
// NOTE: the InferTypeOpInterface verification hook (refineReturnTypes) is a no-op
// for these ops (see PolyKernel_InferTypeOp in PolyKernelOps.td), so this
// inference is enforced by the --infer-shapes pass rather than at IR parse time.

namespace {

/// Return the operand's type as a RankedTensorType, or nullptr if it is not a
/// ranked tensor.
RankedTensorType getRankedTensorType(Value value) {
  return llvm::dyn_cast<RankedTensorType>(value.getType());
}

/// Matmul result type: A<...xMxK> x B<...xKxN> -> <...xMxN>. The contraction
/// dimension K (A's last dim, B's second-to-last dim) must match and the element
/// types must agree. The result takes A's leading dims (batch + M) and replaces
/// A's last dim with N (B's last dim). Shared by matmul + the matmul-based fused
/// ops.
LogicalResult inferMatmulResultType(std::optional<Location> location,
                                    RankedTensorType aType,
                                    RankedTensorType bType,
                                    SmallVectorImpl<Type> &inferredReturnTypes) {
  if (aType.getRank() < 2 || bType.getRank() < 2)
    return emitOptionalError(location,
                             "matmul operands must have rank >= 2, got ranks ",
                             aType.getRank(), " and ", bType.getRank());
  int64_t aK = aType.getShape().back();                // A: ...xMxK
  int64_t bK = bType.getShape()[bType.getRank() - 2];  // B: ...xKxN
  if (aK != bK)
    return emitOptionalError(
        location,
        "matmul shape mismatch: contraction dimension K differs (lhs K = ",
        aK, " vs rhs K = ", bK, ")");
  if (aType.getElementType() != bType.getElementType())
    return emitOptionalError(location,
                             "matmul element type mismatch: lhs ",
                             aType.getElementType(), " vs rhs ",
                             bType.getElementType());
  SmallVector<int64_t> resultShape(aType.getShape());
  resultShape.back() = bType.getShape().back();
  inferredReturnTypes.push_back(
      RankedTensorType::get(resultShape, aType.getElementType()));
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Shape-preserving ops (result type == input type).
//===----------------------------------------------------------------------===//

LogicalResult RmsNormOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult GeluOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult SiluOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult SoftmaxOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult RopeOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

LogicalResult FusedSoftmaxMaskOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Elementwise add (equal operand shapes).
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto lhsType = getRankedTensorType(adaptor.getLhs());
  auto rhsType = getRankedTensorType(adaptor.getRhs());
  if (lhsType && rhsType &&
      (lhsType.getShape() != rhsType.getShape() ||
       lhsType.getElementType() != rhsType.getElementType()))
    return emitOptionalError(location,
                             "add operand shapes must be equal, got ",
                             lhsType, " and ", rhsType);
  inferredReturnTypes.push_back(adaptor.getLhs().getType());
  return success();
}

LogicalResult FusedResidualRmsNormOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto residualType = getRankedTensorType(adaptor.getResidual());
  auto inputType = getRankedTensorType(adaptor.getInput());
  if (residualType && inputType &&
      (residualType.getShape() != inputType.getShape() ||
       residualType.getElementType() != inputType.getElementType()))
    return emitOptionalError(
        location,
        "fused_residual_rmsnorm residual and input shapes must be equal, got ",
        residualType, " and ", inputType);
  inferredReturnTypes.push_back(adaptor.getResidual().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Matmul + matmul-based fused ops.
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto aType = getRankedTensorType(adaptor.getA());
  auto bType = getRankedTensorType(adaptor.getB());
  if (!aType || !bType)
    return emitOptionalError(location,
                             "matmul operands must be ranked tensors");
  return inferMatmulResultType(location, aType, bType, inferredReturnTypes);
}

LogicalResult FusedRmsNormMatMulOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // rmsnorm preserves the input shape, then matmul(input, weight) -> MxN.
  auto inputType = getRankedTensorType(adaptor.getInput());
  auto weightType = getRankedTensorType(adaptor.getWeight());
  if (!inputType || !weightType)
    return emitOptionalError(
        location, "fused_rmsnorm_matmul operands must be ranked tensors");
  return inferMatmulResultType(location, inputType, weightType,
                               inferredReturnTypes);
}

LogicalResult FusedMatMulBiasGeluOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto aType = getRankedTensorType(adaptor.getA());
  auto bType = getRankedTensorType(adaptor.getB());
  auto biasType = getRankedTensorType(adaptor.getBias());
  if (!aType || !bType)
    return emitOptionalError(
        location, "fused_matmul_bias_gelu operands must be ranked tensors");
  if (failed(inferMatmulResultType(location, aType, bType, inferredReturnTypes)))
    return failure();
  // The bias broadcasts over the last dim (N) of the matmul result.
  if (biasType && biasType.getRank() >= 1) {
    int64_t n =
        cast<RankedTensorType>(inferredReturnTypes.back()).getShape().back();
    if (biasType.getShape().back() != n)
      return emitOptionalError(
          location,
          "fused_matmul_bias_gelu bias last dimension (",
          biasType.getShape().back(),
          ") must match the matmul output last dimension (", n, ")");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Bias (broadcast over the last dim).
//===----------------------------------------------------------------------===//

LogicalResult BiasOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = getRankedTensorType(adaptor.getInput());
  auto biasType = getRankedTensorType(adaptor.getBias());
  if (!inputType)
    return emitOptionalError(location, "bias input must be a ranked tensor");
  if (biasType && biasType.getRank() >= 1 && inputType.getRank() >= 1 &&
      biasType.getShape().back() != inputType.getShape().back())
    return emitOptionalError(
        location, "bias last dimension (", biasType.getShape().back(),
        ") must match the input last dimension (",
        inputType.getShape().back(), ")");
  // Result follows the input shape (bias broadcasts over the last dim).
  inferredReturnTypes.push_back(adaptor.getInput().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// Attention + KV-cache ops.
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // Attention output preserves the query shape/type.
  inferredReturnTypes.push_back(adaptor.getQuery().getType());
  return success();
}

LogicalResult KVCacheUpdateOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // The updated cache has the cache's shape/type.
  inferredReturnTypes.push_back(adaptor.getCache().getType());
  return success();
}

LogicalResult FusedKVAppendAttentionOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // output preserves the query shape; updated_cache preserves the cache shape.
  inferredReturnTypes.push_back(adaptor.getQuery().getType());
  inferredReturnTypes.push_back(adaptor.getCache().getType());
  return success();
}

//===----------------------------------------------------------------------===//
// QKV projection (head split via num_heads).
//===----------------------------------------------------------------------===//

LogicalResult QKVProjectionOp::inferReturnTypes(
    MLIRContext *, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputType = getRankedTensorType(adaptor.getInput());
  auto weightType = getRankedTensorType(adaptor.getWeight());
  if (!inputType || !weightType)
    return emitOptionalError(location,
                             "qkv_projection operands must be ranked tensors");
  if (inputType.getRank() < 2)
    return emitOptionalError(
        location, "qkv_projection input must have rank >= 2, got rank ",
        inputType.getRank());

  std::optional<uint64_t> numHeads = adaptor.getNumHeads();
  if (!numHeads)
    return emitOptionalError(
        location,
        "qkv_projection requires the 'num_heads' attribute for shape inference");
  int64_t heads = static_cast<int64_t>(*numHeads);

  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  int64_t hidden = inputShape.back();
  if (heads <= 0)
    return emitOptionalError(location,
                             "qkv_projection 'num_heads' must be positive, got ",
                             heads);
  if (hidden % heads != 0)
    return emitOptionalError(
        location, "qkv_projection 'num_heads' (", heads,
        ") must divide the hidden dimension (", hidden, ")");

  // weight: [hidden, 3*hidden].
  if (weightType.getRank() >= 2) {
    int64_t wIn = weightType.getShape()[weightType.getRank() - 2];
    int64_t wOut = weightType.getShape().back();
    if (wIn != hidden)
      return emitOptionalError(
          location, "qkv_projection weight input dimension (", wIn,
          ") must match the input hidden dimension (", hidden, ")");
    if (wOut != 3 * hidden)
      return emitOptionalError(
          location, "qkv_projection weight output dimension (", wOut,
          ") must be 3x the hidden dimension (", 3 * hidden, ")");
  }

  // Q = K = V = [batch..., num_heads, seq, head_dim], where input is
  // [batch..., seq, hidden] and head_dim = hidden / num_heads.
  int64_t headDim = hidden / heads;
  SmallVector<int64_t> qkvShape;
  for (int64_t i = 0; i + 2 < rank; ++i)
    qkvShape.push_back(inputShape[i]);
  qkvShape.push_back(heads);
  qkvShape.push_back(inputShape[rank - 2]);
  qkvShape.push_back(headDim);
  auto qkvType = RankedTensorType::get(qkvShape, inputType.getElementType());
  inferredReturnTypes.push_back(qkvType);
  inferredReturnTypes.push_back(qkvType);
  inferredReturnTypes.push_back(qkvType);
  return success();
}

//===----------------------------------------------------------------------===//
// InferTypeOpInterface plumbing (shared by every compute op).
//===----------------------------------------------------------------------===//
//
// For each compute op this defines:
//   - the full-signature InferTypeOpInterface::inferReturnTypes, which builds the
//     op Adaptor and delegates to the per-op adaptor-form inferReturnTypes above;
//   - a no-op refineReturnTypes, which disables the InferTypeOpInterface
//     verification hook so shape consistency is enforced by the --infer-shapes
//     pass (signalPassFailure on mismatch) rather than rejected at IR parse time.
#define POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(OpClass)                        \
  ::llvm::LogicalResult OpClass::inferReturnTypes(                             \
      ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,  \
      ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,          \
      ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,        \
      ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes) {            \
    Adaptor adaptor(operands, attributes, properties, regions);                \
    return inferReturnTypes(context, location, adaptor, inferredReturnTypes);  \
  }                                                                            \
  ::llvm::LogicalResult OpClass::refineReturnTypes(                            \
      ::mlir::MLIRContext *, std::optional<::mlir::Location>,                  \
      ::mlir::ValueRange, ::mlir::DictionaryAttr, ::mlir::OpaqueProperties,    \
      ::mlir::RegionRange, ::llvm::SmallVectorImpl<::mlir::Type> &) {          \
    return ::llvm::success();                                                  \
  }

POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(RmsNormOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(GeluOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(SiluOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(SoftmaxOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(AddOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(MatMulOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(BiasOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(RopeOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(AttentionOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(KVCacheUpdateOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(FusedRmsNormMatMulOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(FusedMatMulBiasGeluOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(FusedResidualRmsNormOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(FusedSoftmaxMaskOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(QKVProjectionOp)
POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE(FusedKVAppendAttentionOp)

#undef POLYKERNEL_DEFINE_INFER_TYPE_INTERFACE
