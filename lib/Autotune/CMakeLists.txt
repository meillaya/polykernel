# ===- PolyKernel autotuner (Todo 24 / Wave 5) -===
#
# GPU-FREE autotuner infrastructure: the bounded matmul config space + pruner
# (ConfigSpace) and the contract-H JSON tuning cache (TuningCache). Pure C++20
# logic + llvm::json (LLVM Support) - no GPU, no kernel compilation.
#
# Targets:
#   - PolyKernelAutotune: static lib (ConfigSpace + TuningCache + AmdTuningDb).
#                         Foundation for the bench (Todo 25) and the report
#                         (Todo 27). AmdTuningDb (Todo 23) partitions contract-H
#                         CacheEntry objects into per-gpu namespaces (gfx1101
#                         local / gfx942 rental) over the one pinned schema.
#   - tests/: config_space_test + tuning_cache_test + amd_tuning_db_test gtest
#             executables, registered with CTest via gtest_discover_tests
#             (mirrors lib/Analysis).
#             `ctest -R 'config_space|tuning_cache|amd_tuning_db'` runs them.

add_library(PolyKernelAutotune STATIC
  ConfigSpace.cpp
  TuningCache.cpp
  AmdTuningDb.cpp
)
# Public headers live under include/PolyKernel/Autotune/ (project-wide include
# path, re-added for standalone use); llvm::json comes from LLVMSupport.
target_include_directories(PolyKernelAutotune PUBLIC
  ${POLYKERNEL_SOURCE_DIR}/include
)
target_link_libraries(PolyKernelAutotune PUBLIC LLVMSupport)

add_subdirectory(tests)
