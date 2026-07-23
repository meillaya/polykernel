# ===- PolyKernel compile-time analyzer (Todo 11 / Wave 2) -===
#
# GPU-FREE analysis library: parses `ptxas -v` output and models occupancy +
# roofline + the contract H per-kernel report. Plain C++20 (no MLIR, no CUDA);
# nvcc/ptxas run without a device, so nothing here needs a GPU.
#
# Targets:
#   - PolyKernelAnalysis: static lib (PtxasParser + Occupancy + Roofline +
#                         KernelReport). Wrapped by tools/polykernel-analyze and
#                         (later) polykernel-bench - built once, reused.
#   - ptxas_parser_test / occupancy_test / roofline_test: gtest executables,
#                         registered with CTest via gtest_discover_tests.

add_library(PolyKernelAnalysis STATIC
  PtxasParser.cpp
  Occupancy.cpp
  Roofline.cpp
  KernelReport.cpp
)
# Occupancy/Roofline/KernelReport headers are colocated here; PtxasParser.h is
# public under include/ (project-wide include path, re-added for standalone use).
target_include_directories(PolyKernelAnalysis PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${POLYKERNEL_SOURCE_DIR}/include
)

# --- gtest unit tests (gtest is provided by the dev shell, pkgs.gtest) ---
enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(ptxas_parser_test tests/ptxas_parser_test.cpp)
target_link_libraries(ptxas_parser_test PRIVATE PolyKernelAnalysis GTest::gtest_main)
gtest_discover_tests(ptxas_parser_test)

add_executable(occupancy_test tests/occupancy_test.cpp)
target_link_libraries(occupancy_test PRIVATE PolyKernelAnalysis GTest::gtest_main)
gtest_discover_tests(occupancy_test)

add_executable(roofline_test tests/roofline_test.cpp)
target_link_libraries(roofline_test PRIVATE PolyKernelAnalysis GTest::gtest_main)
gtest_discover_tests(roofline_test)
