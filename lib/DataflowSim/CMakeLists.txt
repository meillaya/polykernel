# ===- PolyKernel dataflow simulator core (Todo 35 / Wave 7) -===
#
# GPU-FREE, cycle-level functional model of a Cerebras-style dataflow fabric:
# a 64x64 PE mesh where each PE = CE (4x FP16 FMAC, 64-bit datapath) + 48 KB
# SRAM (8 banks x 6 KB, 2 reads + 1 write/cycle) + a 5-port router (RAMP +
# E/W/N/S, 32b/link, single-cycle hop, lossless backpressure). Wavelets are
# 32-bit (16b data + 16b control); colors are 5-bit tags (24 routable, IDs
# 0-23) acting as virtual channels over a shared physical link.
#
# This is a SIMULATOR: it does NOT compile real CSL and does NOT run on Cerebras
# hardware. Pure portable C++20 - no GPU, no LLVM dependency, no new third-party
# libraries. Foundation for the rest of Wave 7 (T36 task model + scheduler,
# T37 SUMMA mapping + golden, T38 metrics + fusion traffic, T39 HTML viz).
#
# Targets:
#   - PolyKernelDataflowSim: static lib (Color + Wavelet + Router + Pe + Grid
#     from Todo 35, the Todo 36 task model + scheduler: Task + Scheduler, where
#     Scheduler.cpp also hosts the TaskPicker selector, plus the Todo 37 SUMMA
#     matmul mapping: Summa.cpp runs the SUMMA tile program on the Grid and
#     LowerToDataflow.cpp is the simulator lowering of polykernel.matmul /
#     fused-matmul onto the SUMMA schedule).
#   - tests/: router_test + color_test + grid_test (Todo 35), scheduler_test
#             (Todo 36), and summa_test + dataflow_correctness_test (Todo 37)
#             gtest executables, registered with CTest via gtest_discover_tests
#             (mirrors lib/Autotune). The gtest SUITE names are `router`, `color`,
#             `grid`, `scheduler`, `summa`, `dataflow_correctness` so
#             `ctest -R 'router|color|grid'`, `ctest -R scheduler`, and
#             `ctest -R 'summa|dataflow_correctness'` discover them.

add_library(PolyKernelDataflowSim STATIC
  Color.cpp
  Wavelet.cpp
  Router.cpp
  Pe.cpp
  Grid.cpp
  Task.cpp
  Scheduler.cpp
  Summa.cpp
  LowerToDataflow.cpp
)
# Public headers live under include/PolyKernel/Dataflow/ (project-wide include
# path, re-added for standalone use).
target_include_directories(PolyKernelDataflowSim PUBLIC
  ${POLYKERNEL_SOURCE_DIR}/include
)

add_subdirectory(tests)
