# ===- PolyKernelAutotune gtest unit tests (Todo 24 / Wave 5) -===
#
# Mirrors lib/Analysis: gtest executables registered with CTest via
# gtest_discover_tests. The gtest SUITE names are `config_space`,
# `tuning_cache` and `amd_tuning_db` so
# `ctest -R 'config_space|tuning_cache|amd_tuning_db'` discovers them.

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(config_space_test config_space_test.cpp)
target_link_libraries(config_space_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(config_space_test)

add_executable(tuning_cache_test tuning_cache_test.cpp)
target_link_libraries(tuning_cache_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(tuning_cache_test)

# Todo 23: AMD tuning DB (per-gpu gfx1101/gfx942 namespaces over contract H).
add_executable(amd_tuning_db_test amd_tuning_db_test.cpp)
target_link_libraries(amd_tuning_db_test PRIVATE PolyKernelAutotune GTest::gtest_main)
gtest_discover_tests(amd_tuning_db_test)
