//===- HipRuntime.cpp - Thin error-checked HIP launch layer -----*- C++ -*-===//
//
// PolyKernel HIP runtime launch layer (Todo 20 / Wave 4). See HipRuntime.h.
//
//===----------------------------------------------------------------------===//
//
// Every HIP runtime call is funnelled through PK_HIP_CHECK: on failure it prints
// the call + file:line + hipGetErrorString and aborts cleanly (non-zero exit),
// so a failed alloc / copy / sync can never silently corrupt a result. The async
// launch error (invalid grid/block) surfaces at HipSync (hipDeviceSynchronize)
// and is caught identically. ProbeLaunchError is the non-fatal variant the
// negative QA path uses to prove a bad launch is CAUGHT, not silent.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/HipRuntime.h"

namespace polykernel::runtime {

const char *HipErrorString(hipError_t err) { return hipGetErrorString(err); }

void HipMalloc(void **ptr, std::size_t bytes) {
  PK_HIP_CHECK(hipMalloc(ptr, bytes));
}

void HipCopyH2D(void *dst, const void *src, std::size_t bytes) {
  PK_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

void HipCopyD2H(void *dst, const void *src, std::size_t bytes) {
  PK_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

void HipSync() { PK_HIP_CHECK(hipDeviceSynchronize()); }

void HipFree(void *ptr) { PK_HIP_CHECK(hipFree(ptr)); }

hipError_t ProbeLaunchError(const char *what) {
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    std::fprintf(stderr, "[polykernel-hip] %s: caught launch error: %s\n", what,
                 hipGetErrorString(err));
  return err;
}

} // namespace polykernel::runtime
