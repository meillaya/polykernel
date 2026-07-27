# ===- PolyKernelDataflowSim gtest unit tests (Todo 35 / Wave 7) -===
#
# Mirrors lib/Autotune: gtest executables registered with CTest via
# gtest_discover_tests. The gtest SUITE names are `router`, `color` and `grid`
# so `ctest -R 'router|color|grid'` discovers them.

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(router_test router_test.cpp)
target_link_libraries(router_test PRIVATE PolyKernelDataflowSim GTest::gtest_main)
gtest_discover_tests(router_test)

add_executable(color_test color_test.cpp)
target_link_libraries(color_test PRIVATE PolyKernelDataflowSim GTest::gtest_main)
gtest_discover_tests(color_test)

add_executable(grid_test grid_test.cpp)
target_link_libraries(grid_test PRIVATE PolyKernelDataflowSim GTest::gtest_main)
gtest_discover_tests(grid_test)
