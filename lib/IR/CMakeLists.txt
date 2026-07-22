# ===- MLIRPolyKernel: the PolyKernel dialect library (W1 spike) -===
#
# The dialect .cpp is co-located with its .td/.h under include/PolyKernel/IR/
# (per the W1 deliverable layout); we compile it here into the dialect lib.

add_mlir_dialect_library(MLIRPolyKernel
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel/IR/PolyKernelDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel

  DEPENDS
  MLIRPolyKernelDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
