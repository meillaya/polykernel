# ===- MLIRPolyKernelPasses: PolyKernel transformation passes -===
#
# The --infer-shapes pass (Todo 4) + the --canonicalize pass (Todo 5).
# registerPolyKernelPasses() (Passes.h) picks both up automatically.
#
# Upstream libs (nix ships individual libs only; NO MLIRRegisterAll*):
#   - MLIRPolyKernel:           the dialect + ops (InferTypeOpInterface impls).
#   - MLIRArithDialect:         arith.constant (matched/created by the
#                               canonicalization patterns).
#   - MLIRInferTypeOpInterface: InferTypeOpInterface used by --infer-shapes.
#   - MLIRTransforms:           applyPatternsGreedily (greedy driver: patterns
#                               + folding + DCE + region simplification).
#   - MLIRPass / MLIRIR / MLIRSupport: pass + IR infrastructure.

add_mlir_library(MLIRPolyKernelPasses
  Canonicalize.cpp
  InferShapes.cpp

  ADDITIONAL_HEADER_DIRS
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel

  DEPENDS
  MLIRPolyKernelPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPolyKernel
  MLIRArithDialect
  MLIRInferTypeOpInterface
  MLIRTransforms
  MLIRPass
  MLIRIR
  MLIRSupport
)
