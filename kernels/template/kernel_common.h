//===- kernel_common.h - Portable CUDA/HIP kernel template ------*- C++ -*-===//
//
// PolyKernel portable GPU kernel template (Todo 8 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// One header that every generated PolyKernel kernel is written against. It maps
// a small set of portable macros + types onto the CUDA or HIP runtime, selected
// by exactly one of:
//
//     -DPOLYKERNEL_CUDA   (nvcc,  Wave 2)
//     -DPOLYKERNEL_HIP    (hipcc, Wave 4)
//
// PORTABILITY RULE (binding on every kernel + on the --lower-to-cuda emitter):
// kernel code uses ONLY the pk_* / PK_* names below. No raw `__nv_*` / `__hip_*`
// type or intrinsic may appear in a kernel; anything backend-specific lives here
// behind the #ifdef. This is what lets the SAME generated .cu compile unchanged
// under `nvcc -DPOLYKERNEL_CUDA` (now) and `hipcc -DPOLYKERNEL_HIP` (Wave 4).
//
// The bf16x2 vectorized load/store is implemented with a 4-byte memcpy (NOT a
// vendor bf16x2 intrinsic such as __nv_bfloat162 / __hip_bfloat162) so the
// vector path is byte-identical across backends; the compiler lowers the
// constant-size memcpy to a single 32-bit global load/store on both.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_KERNELS_TEMPLATE_KERNEL_COMMON_H
#define POLYKERNEL_KERNELS_TEMPLATE_KERNEL_COMMON_H

#if !defined(POLYKERNEL_CUDA) && !defined(POLYKERNEL_HIP)
#error "kernel_common.h: define exactly one of -DPOLYKERNEL_CUDA or -DPOLYKERNEL_HIP"
#endif

#if defined(POLYKERNEL_CUDA) && defined(POLYKERNEL_HIP)
#error "kernel_common.h: define only ONE of -DPOLYKERNEL_CUDA / -DPOLYKERNEL_HIP"
#endif

//===----------------------------------------------------------------------===//
// Backend headers + the portable bf16 type alias.
//===----------------------------------------------------------------------===//

#if defined(POLYKERNEL_CUDA)
#include <cuda_runtime.h>
#include <cuda_bf16.h>
using pk_bf16 = __nv_bfloat16;
using pk_error_t = cudaError_t;
using pk_stream_t = cudaStream_t;
#define PK_SUCCESS cudaSuccess
#define pk_get_last_error() cudaGetLastError()
#define pk_get_error_string(e) cudaGetErrorString(e)
#else // POLYKERNEL_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
using pk_bf16 = __hip_bfloat16;
using pk_error_t = hipError_t;
using pk_stream_t = hipStream_t;
#define PK_SUCCESS hipSuccess
#define pk_get_last_error() hipGetLastError()
#define pk_get_error_string(e) hipGetErrorString(e)
#endif

//===----------------------------------------------------------------------===//
// Function-qualifier + storage macros (identical spelling on CUDA and HIP).
//===----------------------------------------------------------------------===//

#define PK_GLOBAL __global__
#define PK_DEVICE __device__
#define PK_HOST_DEVICE __host__ __device__
#define PK_SHARED __shared__
#define PK_DEVICE_INLINE __device__ __forceinline__

//===----------------------------------------------------------------------===//
// bf16 <-> float conversions. Both CUDA (<cuda_bf16.h>) and HIP (<hip/hip_bf16.h>)
// provide __float2bfloat16 / __bfloat162float with the same signature, so these
// are NOT backend-specific and need no #ifdef beyond the header selection above.
// Rounding is round-to-nearest-even, matching the golden contract.
//===----------------------------------------------------------------------===//

PK_DEVICE_INLINE pk_bf16 pk_float2bf16(float x) { return __float2bfloat16(x); }
PK_DEVICE_INLINE float pk_bf162float(pk_bf16 x) { return __bfloat162float(x); }

//===----------------------------------------------------------------------===//
// Portable bf16x2 (2 x bf16 = 4 bytes) for vectorized global loads/stores.
//===----------------------------------------------------------------------===//

struct pk_bf16x2 {
  pk_bf16 x;
  pk_bf16 y;
};
static_assert(sizeof(pk_bf16x2) == 4, "pk_bf16x2 must pack to 4 bytes");

PK_DEVICE_INLINE pk_bf16x2 pk_make_bf16x2(pk_bf16 a, pk_bf16 b) {
  return pk_bf16x2{a, b};
}

// 4-byte vectorized load/store via memcpy (aliasing-safe; lowered to a single
// 32-bit LDG.32 / global_load_dword when the pointer is 4-byte aligned).
PK_DEVICE_INLINE pk_bf16x2 pk_load_bf16x2(const pk_bf16 *p) {
  pk_bf16x2 r;
  memcpy(&r, p, sizeof r);
  return r;
}
PK_DEVICE_INLINE void pk_store_bf16x2(pk_bf16 *p, pk_bf16x2 v) {
  memcpy(p, &v, sizeof v);
}

//===----------------------------------------------------------------------===//
// Warp + block synchronization primitives. __syncthreads and the *_sync warp
// shuffles map directly across backends (warpSize == 32 on sm_80/sm_90 and
// gfx1101), with ONE exception: the shuffle MASK WIDTH differs. CUDA's
// __shfl_*_sync take a 32-bit mask (one bit per lane, warpSize == 32); HIP's
// __shfl_*_sync are templates that static_assert sizeof(mask) == 8 — HIP supports
// wave32 AND wave64, so the mask is a 64-bit integer. PK_FULL_MASK is therefore
// backend-guarded (Todo 19 portability fix): 32-bit on CUDA, 64-bit on HIP. This
// is a lane-participation mask only (all lanes enabled either way), so the
// reduction numerics are identical across backends.
//===----------------------------------------------------------------------===//

#define PK_WARP_SIZE 32
#if defined(POLYKERNEL_CUDA)
#define PK_FULL_MASK 0xffffffffu // CUDA: 32-bit mask (warpSize == 32).
#else // POLYKERNEL_HIP
#define PK_FULL_MASK 0xffffffffffffffffull // HIP: 64-bit mask (sizeof == 8).
#endif

PK_DEVICE_INLINE float pk_shfl_xor_sync(float v, int laneMask) {
  return __shfl_xor_sync(PK_FULL_MASK, v, laneMask);
}
PK_DEVICE_INLINE float pk_shfl_down_sync(float v, int delta) {
  return __shfl_down_sync(PK_FULL_MASK, v, delta);
}
PK_DEVICE_INLINE void pk_syncthreads() { __syncthreads(); }

//===----------------------------------------------------------------------===//
// Portable kernel launch. Both CUDA and HIP accept the triple-chevron syntax
// kernel<<<grid, block, smem, stream>>>(args).
//===----------------------------------------------------------------------===//

#define PK_LAUNCH(kernel, grid, block, smem, stream, ...)                       \
  kernel<<<(grid), (block), (smem), (stream)>>>(__VA_ARGS__)

//===----------------------------------------------------------------------===//
// Runtime-error check for host-side launch / API calls (maps to the backend's
// error type + stringifier selected above).
//===----------------------------------------------------------------------===//

#define PK_CHECK(call)                                                          \
  do {                                                                          \
    pk_error_t pk__err = (call);                                                \
    if (pk__err != PK_SUCCESS) {                                                \
      fprintf(stderr, "[polykernel] %s failed: %s\n", #call,                    \
              pk_get_error_string(pk__err));                                    \
    }                                                                           \
  } while (0)

#endif // POLYKERNEL_KERNELS_TEMPLATE_KERNEL_COMMON_H
