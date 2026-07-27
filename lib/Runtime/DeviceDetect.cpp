//===- DeviceDetect.cpp - Mockable GPU device/arch detection ----*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). See DeviceDetect.h.
//
//===----------------------------------------------------------------------===//
//
// The arch-string parsing (ParseGcnArchName / SmArchFromComputeCapability) and the
// FixedArchDetector mock are pure + GPU-free - the gtest exercises them with no
// device. The real HipDeviceDetector calls hipGetDeviceProperties and maps the
// reported gcnArchName ("gfx1101:sramecc+:xnack-") to the bare cache key
// ("gfx1101"); CudaDeviceDetector (POLYKERNEL_CUDA only) calls
// cudaGetDeviceProperties and maps the compute capability to "sm_XY". The backend
// headers are included HERE (not in the header), so the GPU-free gtest never needs
// the ROCm/CUDA headers on its include path.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/DeviceDetect.h"

#if defined(POLYKERNEL_CUDA)
#include <cuda_runtime_api.h>
#elif defined(__HIP_PLATFORM_AMD__) || defined(POLYKERNEL_HIP)
#include <hip/hip_runtime_api.h>
#endif

namespace polykernel::runtime {

std::string ParseGcnArchName(std::string_view gcnArchName) {
  // The arch token is the leading "gfx<digits>[<letter>]"; ROCm appends feature
  // flags after a ':' (e.g. "gfx1101:sramecc+:xnack-"), and some archs carry a
  // trailing letter (e.g. "gfx90a"). Find "gfx", require a digit immediately after
  // it (arch names always start gfx<digit>), then take the run of lowercase
  // alphanumerics up to the first non-alphanumeric (the ':' or end). "gfx" with no
  // leading digit is not a valid arch -> empty.
  const std::size_t pos = gcnArchName.find("gfx");
  if (pos == std::string_view::npos)
    return {};
  std::size_t end = pos + 3;
  if (end >= gcnArchName.size() || gcnArchName[end] < '0' ||
      gcnArchName[end] > '9')
    return {}; // "gfx" not followed by a digit.
  while (end < gcnArchName.size()) {
    const char c = gcnArchName[end];
    const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z');
    if (!alnum)
      break;
    ++end;
  }
  return std::string(gcnArchName.substr(pos, end - pos));
}

std::string SmArchFromComputeCapability(int major, int minor) {
  return "sm_" + std::to_string(major) + std::to_string(minor);
}

std::optional<DeviceInfo> FixedArchDetector::Detect(int) const {
  return DeviceInfo{arch_, name_};
}

#if defined(POLYKERNEL_CUDA)

std::optional<DeviceInfo> CudaDeviceDetector::Detect(int ordinal) const {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess)
    return std::nullopt;
  return DeviceInfo{SmArchFromComputeCapability(prop.major, prop.minor),
                    prop.name};
}

// The AMD detector is unavailable in a CUDA build (and vice versa); a translation
// unit that needs both backends would select one at configure time. Provide a
// stub so the symbol always exists for the Runtime to hold a reference to.
std::optional<DeviceInfo> HipDeviceDetector::Detect(int) const {
  return std::nullopt;
}

#elif defined(__HIP_PLATFORM_AMD__) || defined(POLYKERNEL_HIP)

std::optional<DeviceInfo> HipDeviceDetector::Detect(int ordinal) const {
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, ordinal) != hipSuccess)
    return std::nullopt;
  return DeviceInfo{ParseGcnArchName(prop.gcnArchName), prop.name};
}

#else

// No GPU backend compiled (a non-ROCm/CUDA checkout): the real detector reports no
// device. The mock (FixedArchDetector) still works, so the selection logic + gtest
// are unaffected.
std::optional<DeviceInfo> HipDeviceDetector::Detect(int) const {
  return std::nullopt;
}

#endif

} // namespace polykernel::runtime
