# ===- PolyKernelRuntime gtest unit tests (Todo 26 / Wave 5) -===
#
# GPU-free tests of the production selection runtime: device-detect arch parsing +
# the mock detector (device_detect suite), and the detect -> select -> load ->
# serve flow with a mocked device + recording launcher (runtime suite), including
# the graceful cache-miss error. Mirrors lib/Autotune/tests: gtest executables
# registered with CTest via gtest_discover_tests. The gtest SUITE names are
# `device_detect` and `runtime` so `ctest -R 'runtime|device_detect'` discovers
# them. Both link PolyKernelRuntime (which transitively provides PolyKernelAutotune
# + LLVMSupport + the HIP host lib); neither test CALLS a HIP device function, so
# they run with no GPU present.

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(device_detect_test device_detect_test.cpp)
target_link_libraries(device_detect_test PRIVATE PolyKernelRuntime GTest::gtest_main)
gtest_discover_tests(device_detect_test)

add_executable(runtime_test runtime_test.cpp)
target_link_libraries(runtime_test PRIVATE PolyKernelRuntime GTest::gtest_main)
gtest_discover_tests(runtime_test)
