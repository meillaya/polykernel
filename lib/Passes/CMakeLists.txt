# ===- MLIRPolyKernelPasses: PolyKernel transformation passes -===
#
# The --infer-shapes pass (Todo 4), the --canonicalize pass (Todo 5), the
# --lower-to-cuda source-emitter pass (Todo 8), its HIP sibling --lower-to-hip
# (Todo 19; emits the same portable kernel source, compiled by the hipcc driver
# with -DPOLYKERNEL_HIP), the --fuse-rmsnorm-matmul fusion pass (Todo 12), and
# the three remaining Wave 3 fusion passes (Todo 13):
# --fuse-matmul-bias-gelu, --fuse-residual-rmsnorm, --fuse-softmax-mask.
# The --infer-tile-layout pass (Todo 15) assigns tile shapes + operand layout to
# the matmul ops. The --plan-memory pass (Todo 16) computes a per-kernel memory
# plan (smem budget + non-fused workspace + over-budget guard) for the matmul ops.
# registerPolyKernelPasses() (Passes.h) picks all of them up automatically.
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
  EmitKernelReport.cpp
  FuseKvAppendAttention.cpp
  FuseMatmulBiasGelu.cpp
  FuseResidualRmsnorm.cpp
  FuseRmsnormMatmul.cpp
  FuseSoftmaxMask.cpp
  InferShapes.cpp
  InferTileLayout.cpp
  LowerToCuda.cpp
  LowerToHip.cpp
  PlanMemory.cpp

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
