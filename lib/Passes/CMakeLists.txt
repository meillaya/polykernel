# ===- MLIRPolyKernelPasses: PolyKernel transformation passes -===
#
# The --infer-shapes pass (Todo 4), the --canonicalize pass (Todo 5), and the
# --lower-to-cuda source-emitter pass (Todo 8). registerPolyKernelPasses()
# (Passes.h) picks all of them up automatically.
#
# Upstream libs (nix ships individual libs only; NO MLIRRegisterAll*):
#   - MLIRPolyKernel:           the dialect + ops (InferTypeOpInterface impls;
#                               RmsNormOp etc. matched by --lower-to-cuda).
#   - MLIRArithDialect:         arith.constant (matched/created by the
#                               canonicalization patterns).
#   - MLIRInferTypeOpInterface: InferTypeOpInterface used by --infer-shapes.
#   - MLIRTransforms:           applyPatternsGreedily (greedy driver: patterns
#                               + folding + DCE + region simplification).
#   - MLIRPass / MLIRIR / MLIRSupport: pass + IR infrastructure (MLIRSupport
#                               pulls LLVMSupport: sys::fs / sys::path /
#                               raw_fd_ostream used by --lower-to-cuda).

add_mlir_library(MLIRPolyKernelPasses
  Canonicalize.cpp
  InferShapes.cpp
  LowerToCuda.cpp

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
