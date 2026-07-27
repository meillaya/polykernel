# ===- PolyKernel autotuner (Todo 24 / Wave 5) -===
#
# GPU-FREE autotuner infrastructure: the bounded matmul config space + pruner
# (ConfigSpace) and the contract-H JSON tuning cache (TuningCache). Pure C++20
# logic + llvm::json (LLVM Support) - no GPU, no kernel compilation.
#
# Targets:
#   - PolyKernelAutotune: static lib (ConfigSpace + TuningCache + AmdTuningDb +
#                         Benchmark). Foundation for the bench (Todo 25) and the
#                         report (Todo 27). AmdTuningDb (Todo 23) partitions
#                         contract-H CacheEntry objects into per-gpu namespaces
#                         (gfx1101 local / gfx942 rental) over the one pinned
#                         schema. Benchmark (Todo 25) adds the GPU-free
#                         correctness-gate + fastest-validated selection logic;
#                         its HIP-event timing + launch driver are preprocessed
#                         out here (no -DPOLYKERNEL_HIP) so the lib stays GPU-free
#                         - the Python bench compiles that driver with hipcc.
#   - tests/: config_space_test + tuning_cache_test + amd_tuning_db_test gtest
#             executables, registered with CTest via gtest_discover_tests
#             (mirrors lib/Analysis).
#             `ctest -R 'config_space|tuning_cache|amd_tuning_db'` runs them.
#   - benchmark_test: the Todo 25 correctness-gate gtest (wired here, not in
#             tests/, to keep the bench wiring in this file). Registered with
#             CTest; `ctest -R benchmark` runs it.

add_library(PolyKernelAutotune STATIC
  ConfigSpace.cpp
  TuningCache.cpp
  AmdTuningDb.cpp
  Benchmark.cpp
)
# Public headers live under include/PolyKernel/Autotune/ (project-wide include
# path, re-added for standalone use); llvm::json comes from LLVMSupport.
target_include_directories(PolyKernelAutotune PUBLIC
  ${POLYKERNEL_SOURCE_DIR}/include
)
target_link_libraries(PolyKernelAutotune PUBLIC LLVMSupport)

add_subdirectory(tests)

# Todo 25: correctness-gate + fastest-validated selection unit test. Built here
# (the bench wiring lives in this file); the pure gate logic in Benchmark.cpp is
# GPU-free, so this needs neither hipcc nor a device.
find_package(GTest REQUIRED)
include(GoogleTest)
add_executable(benchmark_test tests/benchmark_test.cpp)
target_link_libraries(benchmark_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(benchmark_test)
