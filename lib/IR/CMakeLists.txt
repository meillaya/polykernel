# ===- MLIRPolyKernel: the PolyKernel dialect library (dialect + ops) -===
#
# The dialect/ops .cpp are co-located with their .td/.h under
# include/PolyKernel/IR/ (per the deliverable layout); we compile them here into
# the dialect lib.
#
# Upstream libs (nix ships individual libs only; NO MLIRRegisterAll*):
#   - MLIRFunctionInterfaces: FunctionOpInterface (used by polykernel.func) +
#     the function_interface_impl::parse/printFunctionOp helpers. polykernel.func
#     is our OWN op (not func.func), so MLIRFuncDialect is NOT required.
#   - MLIRInferTypeOpInterface: InferTypeOpInterface (used by every compute op for
#     shape inference, Todo 4) — provides detail::verifyInferredResultTypes + the
#     interface's default method scaffolding.

add_mlir_dialect_library(MLIRPolyKernel
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel/IR/PolyKernelDialect.cpp
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel/IR/PolyKernelOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel

  DEPENDS
  MLIRPolyKernelDialectIncGen
  MLIRPolyKernelOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFunctionInterfaces
  MLIRInferTypeOpInterface
)
