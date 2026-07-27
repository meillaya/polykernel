# ===- PolyKernelDataflowSim gtest unit tests (Todo 35 + 36 / Wave 7) -===
#
# Mirrors lib/Autotune: gtest executables registered with CTest via
# gtest_discover_tests. The gtest SUITE names are `router`, `color`, `grid`
# (Todo 35) and `scheduler` (Todo 36) so `ctest -R 'router|color|grid'` and
# `ctest -R scheduler` discover them.

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

add_executable(scheduler_test scheduler_test.cpp)
target_link_libraries(scheduler_test PRIVATE PolyKernelDataflowSim GTest::gtest_main)
gtest_discover_tests(scheduler_test)
