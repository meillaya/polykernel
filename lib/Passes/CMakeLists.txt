# ===- MLIRPolyKernelPasses: PolyKernel transformation passes -===
#
# The --infer-shapes pass (Todo 4). Todo 5 adds Canonicalize.cpp here and lists
# it below; registerPolyKernelPasses() (Passes.h) picks it up automatically.
#
# Upstream libs (nix ships individual libs only; NO MLIRRegisterAll*):
#   - MLIRPolyKernel:           the dialect + ops (InferTypeOpInterface impls).
#   - MLIRInferTypeOpInterface: InferTypeOpInterface used by the pass to infer.
#   - MLIRPass / MLIRIR / MLIRSupport: pass + IR infrastructure.

add_mlir_library(MLIRPolyKernelPasses
  InferShapes.cpp

  ADDITIONAL_HEADER_DIRS
  ${POLYKERNEL_SOURCE_DIR}/include/PolyKernel

  DEPENDS
  MLIRPolyKernelPassesIncGen

  LINK_LIBS PUBLIC
  MLIRPolyKernel
  MLIRInferTypeOpInterface
  MLIRPass
  MLIRIR
  MLIRSupport
)
