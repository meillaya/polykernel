# ===- MLIRPolyKernelPasses: PolyKernel transformation passes -===
#
# The --infer-shapes pass (Todo 4), the --canonicalize pass (Todo 5), the
# --lower-to-cuda source-emitter pass (Todo 8), the --fuse-rmsnorm-matmul fusion
# pass (Todo 12), and the three remaining Wave 3 fusion passes (Todo 13):
# --fuse-matmul-bias-gelu, --fuse-residual-rmsnorm, --fuse-softmax-mask.
# The --infer-tile-layout pass (Todo 15) assigns tile shapes + operand layout to
# the matmul ops. registerPolyKernelPasses() (Passes.h) picks all of them up
# automatically.
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
  FuseMatmulBiasGelu.cpp
  FuseResidualRmsnorm.cpp
  FuseRmsnormMatmul.cpp
  FuseSoftmaxMask.cpp
  InferShapes.cpp
  InferTileLayout.cpp
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
