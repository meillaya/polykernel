# ===- PolyKernel: out-of-tree MLIR toolchain (W1 spike) -===
#
# Out-of-tree MLIR-21 build, modeled on llvm-project
# mlir/examples/standalone. Build model (Pinned contract A): configure + build
# INSIDE `nix develop` with manual cmake+ninja against the prebuilt
# llvmPackages_21 MLIR from the nix cache. NOT a sandboxed `nix build`.
#
# nix gotcha (llvm/llvm-project#150986): nixpkgs patches MLIRConfig.cmake, so
# MLIR_TABLEGEN_EXE MUST be set (via -DMLIR_TABLEGEN_EXE=...) BEFORE
# find_package(MLIR). We enforce that here.

cmake_minimum_required(VERSION 3.20.0)

# CXX + C: LLVM/MLIR CMake modules (HandleLLVMOptions/AddLLVM) probe the C
# compiler too; the upstream standalone example likewise enables `CXX C`.
project(PolyKernel LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

# C++20 first (W1 spike). If MLIR-21 headers reject C++20, drop to 17 and
# record the result in docs/compiler_pipeline.md + reports/w1_cppstd.log.
set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- nix gotcha guard: mlir-tblgen must be pinned before find_package(MLIR) ---
if(NOT MLIR_TABLEGEN_EXE)
  message(FATAL_ERROR
    "MLIR_TABLEGEN_EXE is not set. nixpkgs patches MLIRConfig.cmake "
    "(llvm/llvm-project#150986), so you MUST pass "
    "-DMLIR_TABLEGEN_EXE=$(which mlir-tblgen) BEFORE find_package(MLIR).")
endif()

# We are always the top-level project in the dev-shell build model.
find_package(MLIR REQUIRED CONFIG)

# Default lit args (verbose, show output).
set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using mlir-tblgen: ${MLIR_TABLEGEN_EXE}")

# Output dirs + MLIR_BINARY_DIR (used by MLIR CMake functions + lit site cfg).
# NOTE: we deliberately do NOT set LLVM_RUNTIME_OUTPUT_INTDIR, so executables
# land in their natural subdir (build/tools/polykernel-opt/polykernel-opt) -
# the path the W1 gate verification command expects.
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Pull in the MLIR/LLVM CMake modules from the installed (nix) config.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(POLYKERNEL_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(POLYKERNEL_BINARY_DIR ${PROJECT_BINARY_DIR})

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${POLYKERNEL_SOURCE_DIR}/include)
include_directories(${POLYKERNEL_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/PolyKernel)
add_subdirectory(lib)
add_subdirectory(tools/polykernel-opt)
add_subdirectory(test)
