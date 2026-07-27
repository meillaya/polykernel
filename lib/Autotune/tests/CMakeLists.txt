# ===- PolyKernelAutotune gtest unit tests (Todo 24 / Wave 5) -===
#
# Mirrors lib/Analysis: gtest executables registered with CTest via
# gtest_discover_tests. The gtest SUITE names are `config_space` and
# `tuning_cache` so `ctest -R 'config_space|tuning_cache'` discovers them.

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(config_space_test config_space_test.cpp)
target_link_libraries(config_space_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(config_space_test)

add_executable(tuning_cache_test tuning_cache_test.cpp)
target_link_libraries(tuning_cache_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(tuning_cache_test)
