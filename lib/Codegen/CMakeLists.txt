# ===- PolyKernel quantization codegen lowering (Todo 41 / Wave 8) -===
#
# GPU-FREE quantization source emitter (Pinned contract F, the same approach as
# lib/Passes/LowerToCuda.cpp): a custom string emitter for the quantized matmul
# variants of the closed `polykernel.matmul` op. Pure C++ + llvm::raw_ostream
# (LLVM Support) — no GPU, no MLIR tablegen, no kernel compilation here.
#
# Targets:
#   - PolyKernelCodegen: static lib (Quantize.cpp). emitMatMulInt8Cuda emits the
#                        int8 weight-only kernel (byte-identical to the kernel
#                        portion of kernels/generated/matmul_int8.cu); the fp8
#                        e4m3/e5m2 paths emit a clearly-labeled SIMULATION note
#                        (fp8 is simulated in the golden + CPU ref, NOT hardware).
#   - tests/: quantize_test gtest executable, registered with CTest via
#             gtest_discover_tests (mirrors lib/Autotune). `ctest -R quantize`
#             runs it.

add_library(PolyKernelCodegen STATIC
  Quantize.cpp
)
# Public header lives under include/PolyKernel/Codegen/ (project-wide include path,
# re-added for standalone use); raw_ostream comes from LLVMSupport.
target_include_directories(PolyKernelCodegen PUBLIC
  ${POLYKERNEL_SOURCE_DIR}/include
)
target_link_libraries(PolyKernelCodegen PUBLIC LLVMSupport)

add_subdirectory(tests)
