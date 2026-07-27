# ===- PolyKernelCodegen gtest unit tests (Todo 41 / Wave 8) -===
#
# Mirrors lib/Autotune: gtest executable registered with CTest via
# gtest_discover_tests. The gtest SUITE name is `quantize` so
# `ctest -R quantize` discovers it.

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(quantize_test quantize_test.cpp)
target_link_libraries(quantize_test PRIVATE PolyKernelCodegen GTest::gtest_main)
gtest_discover_tests(quantize_test)
