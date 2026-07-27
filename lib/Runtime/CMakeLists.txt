# ===- PolyKernel HIP runtime launch layer (Todo 20 / Wave 4) -===
#
# A thin host-side wrapper over the HIP runtime API: error-checked hipMalloc /
# hipMemcpy (H2D + D2H) / hipDeviceSynchronize / hipFree (see HipRuntime.h). This
# is the reusable launch layer the Todo 20 golden-validation harness
# (lib/Runtime/hip_run_main.cpp, built by tests/kernels/test_hip_run.py with
# hipcc) and the later production selection/cache runtime (Todo 26) both build on.
#
# Target:
#   - PolyKernelRuntime: static lib (HipRuntime.cpp). Host-ONLY HIP API code, so
#     it compiles with the project's clang++ against the ROCm host headers (found
#     via hipcc's install prefix) - NO device codegen, so it needs neither hipcc
#     nor --offload-arch. Guarded: if hipcc / the ROCm headers are absent (e.g. a
#     non-ROCm checkout), the target is skipped with a status message rather than
#     failing the build. The full GPU launcher (which links device code) is built
#     by the test with hipcc, not here.

find_program(PK_HIPCC_EXECUTABLE hipcc)

if(PK_HIPCC_EXECUTABLE)
  # hipcc lives at <clr>/bin/hipcc; the ROCm host headers + lib are at
  # <clr>/include and <clr>/lib (rocmPackages.clr in the nix dev shell).
  get_filename_component(_PK_CLR_BIN ${PK_HIPCC_EXECUTABLE} DIRECTORY)
  get_filename_component(_PK_CLR_ROOT ${_PK_CLR_BIN} DIRECTORY)
  set(_PK_HIP_INCLUDE ${_PK_CLR_ROOT}/include)
  set(_PK_HIP_LIBDIR ${_PK_CLR_ROOT}/lib)
endif()

if(PK_HIPCC_EXECUTABLE AND EXISTS ${_PK_HIP_INCLUDE}/hip/hip_runtime_api.h)
  add_library(PolyKernelRuntime STATIC HipRuntime.cpp)
  # Public headers live under include/PolyKernel/Runtime/ (project-wide include
  # path, re-added for standalone use; mirrors lib/Autotune).
  target_include_directories(PolyKernelRuntime PUBLIC
    ${POLYKERNEL_SOURCE_DIR}/include
  )
  # HIP host API: plain clang++ compiles it once __HIP_PLATFORM_AMD__ selects the
  # AMD platform (hipcc defines this implicitly; we set it for the host compiler).
  target_compile_definitions(PolyKernelRuntime PRIVATE __HIP_PLATFORM_AMD__)
  target_include_directories(PolyKernelRuntime PRIVATE ${_PK_HIP_INCLUDE})
  # Recorded for downstream linkers (a static lib does not itself link): the HIP
  # host runtime that resolves hipMalloc / hipMemcpy / hipDeviceSynchronize.
  target_link_directories(PolyKernelRuntime INTERFACE ${_PK_HIP_LIBDIR})
  target_link_libraries(PolyKernelRuntime INTERFACE amdhip64)
  message(STATUS
    "PolyKernelRuntime: HIP host launch layer enabled (hipcc=${PK_HIPCC_EXECUTABLE})")
else()
  message(STATUS
    "PolyKernelRuntime: SKIPPED - hipcc / ROCm host headers not found. Build "
    "inside `nix develop` (rocmPackages.clr) to enable the HIP runtime layer.")
endif()
