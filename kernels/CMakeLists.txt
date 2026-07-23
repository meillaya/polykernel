# ===- PolyKernel host CPU reference kernels (Todo 8 / Wave 2) -===
#
# Plain host C++ (NO CUDA, NO MLIR): the CPU reference is the local executable
# spec for kernel correctness, validated against the Python golden by
# tests/kernels/test_cpu_ref.py. Built with the same clang as the rest of the
# project; the only external dependency is libm (sqrt).
#
# Targets:
#   - cpu_reference:    static lib (cpu_reference.cpp + npy_io.cpp). Reused by
#                       every future CPU-ref driver (gelu/silu/matmul/softmax in
#                       Todo 9/10) — those add sources here, not new libs.
#   - cpu_ref_rmsnorm:  driver exe (reads input.npy[+weight], writes output.npy).
#                       Lands at build/kernels/cpu_ref_rmsnorm (no
#                       LLVM_RUNTIME_OUTPUT_INTDIR is set project-wide).

add_library(cpu_reference STATIC
  cpu/cpu_reference.cpp
  cpu/npy_io.cpp
)
target_include_directories(cpu_reference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpu)
target_link_libraries(cpu_reference PUBLIC m)

add_executable(cpu_ref_rmsnorm cpu/cpu_ref_rmsnorm.cpp)
target_link_libraries(cpu_ref_rmsnorm PRIVATE cpu_reference)
