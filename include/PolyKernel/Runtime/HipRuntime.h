//===- HipRuntime.h - Thin error-checked HIP launch layer -------*- C++ -*-===//
//
// PolyKernel HIP runtime launch layer (Todo 20 / Wave 4).
//
//===----------------------------------------------------------------------===//
//
// A thin host-side wrapper over the HIP runtime API that every generated
// PolyKernel kernel launch goes through on the AMD backend. It is the GPU
// sibling of the CPU-reference driver bridge (kernels/cpu): read bf16 .npy ->
// run -> write bf16 .npy, except the run step is a REAL kernel launch on the
// local gfx1101 GPU (RX 7800 XT). This is where the SHARED portable template's
// runtime correctness is validated (the same kernels/generated/*.cu the CUDA
// backend compiles with -DPOLYKERNEL_CUDA are here executed under hipcc).
//
// ERROR DISCIPLINE (binding; the F2 quality review checks this): EVERY HIP call
// (hipMalloc / hipMemcpy H2D + D2H / hipDeviceSynchronize / hipFree) is checked.
// On failure the offending call + file:line + the HIP error string are printed
// and the launch is aborted cleanly (non-zero exit) - never a silent continue
// that would corrupt results. An asynchronous kernel-launch error (e.g. an
// out-of-bounds grid / too-large block) surfaces at hipDeviceSynchronize and is
// caught the same way.
//
// NOTE on the error stringifier: CUDA exposes cudaGetLastErrorString; ROCm ships
// NO hipGetLastErrorString, so the HIP equivalent is hipGetErrorString(err) fed
// the captured code (hipGetLastError() for a launch probe). This matches the
// portable template's mapping in kernels/template/kernel_common.h
// (pk_get_error_string(pk_get_last_error())).
//
// This header includes ONLY <hip/hip_runtime_api.h> (the pure HOST runtime API -
// no device builtins), so it compiles under the project's plain clang++ for the
// CMake PolyKernelRuntime static lib (with -D__HIP_PLATFORM_AMD__), AND under
// hipcc when the launcher driver (lib/Runtime/hip_run_main.cpp) is built with
// the generated kernels.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_RUNTIME_HIPRUNTIME_H
#define POLYKERNEL_RUNTIME_HIPRUNTIME_H

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace polykernel::runtime {

/// Surface the HIP error string for a code (hipGetErrorString - HIP's equivalent
/// of CUDA's *GetLastErrorString when fed hipGetLastError()).
const char *HipErrorString(hipError_t err);

/// Checked device allocation: *ptr <- `bytes` of device memory. Aborts cleanly
/// (message to stderr, non-zero exit) on failure - a failed alloc cannot proceed.
void HipMalloc(void **ptr, std::size_t bytes);

/// Checked host->device copy (`bytes` bytes). Aborts on failure.
void HipCopyH2D(void *dst, const void *src, std::size_t bytes);

/// Checked device->host copy (`bytes` bytes). Aborts on failure.
void HipCopyD2H(void *dst, const void *src, std::size_t bytes);

/// Checked device synchronize. This is WHERE an asynchronous kernel-launch error
/// (e.g. an out-of-bounds grid / too-large block) is surfaced + caught; aborts
/// on failure (no silent corruption).
void HipSync();

/// Checked device free. Aborts on failure.
void HipFree(void *ptr);

/// Non-fatal launch-error probe: returns hipGetLastError() and, if it is an
/// error, prints `what` + the HIP error string WITHOUT aborting. The negative QA
/// path (an intentionally invalid launch) uses this to prove the error is CAUGHT
/// + reported, then exits non-zero itself.
hipError_t ProbeLaunchError(const char *what);

} // namespace polykernel::runtime

/// Error-check a HIP runtime call. On failure: print the call, file:line and the
/// HIP error string, then abort cleanly (non-zero exit). The HIP-specific,
/// abort-on-failure sibling of the portable template's PK_CHECK (which only
/// prints). `call` must yield a hipError_t.
#define PK_HIP_CHECK(call)                                                      \
  do {                                                                          \
    hipError_t pk_hip_err__ = (call);                                           \
    if (pk_hip_err__ != hipSuccess) {                                           \
      std::fprintf(stderr, "[polykernel-hip] %s failed at %s:%d: %s\n", #call,  \
                   __FILE__, __LINE__,                                          \
                   polykernel::runtime::HipErrorString(pk_hip_err__));          \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#endif // POLYKERNEL_RUNTIME_HIPRUNTIME_H
