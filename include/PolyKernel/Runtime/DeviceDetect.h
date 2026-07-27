//===- DeviceDetect.h - Mockable GPU device/arch detection ------*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). First stage of the runtime
// flow: detect GPU -> select best cached kernel -> load -> serve. This header is
// the detection stage: it resolves the device ARCHITECTURE string the kernel cache
// keys on (gfx1101 / gfx942 for AMD; sm_80 / sm_90 for CUDA).
//
// MOCKABLE BY DESIGN (binding; the gtest runs the selection logic with NO GPU):
// the Runtime takes a `const DeviceDetector &`, so a test injects a fake arch via
// FixedArchDetector and exercises detect -> select -> serve deterministically on a
// machine with no device. The on-GPU path uses HipDeviceDetector, which calls
// hipGetDeviceProperties (HIP) / CudaDeviceDetector calls cudaGetDeviceProperties
// (CUDA) to read the real arch. The arch-string parsing (ParseGcnArchName for AMD,
// SmArchFromComputeCapability for CUDA) is pure + GPU-free, so the detection LOGIC
// is unit-testable independent of any device.
//
// This header includes NO HIP/CUDA headers (DeviceInfo + the detector interface are
// plain C++), so it compiles under the project's clang++ for the GPU-free gtest;
// the backend headers are included only in DeviceDetect.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_RUNTIME_DEVICEDETECT_H
#define POLYKERNEL_RUNTIME_DEVICEDETECT_H

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace polykernel::runtime {

/// A detected device's identity: the arch string the kernel cache keys on
/// (gfx1101 / gfx942 for AMD; sm_80 / sm_90 for CUDA) plus a human-readable name.
struct DeviceInfo {
  std::string arch; ///< e.g. "gfx1101", "gfx942", "sm_80", "sm_90".
  std::string name; ///< e.g. "AMD Radeon RX 7800 XT".

  [[nodiscard]] bool operator==(const DeviceInfo &) const = default;
};

/// Mockable device detector. The Runtime holds a `const DeviceDetector &`; the
/// gtest injects FixedArchDetector (no GPU), the production path injects
/// HipDeviceDetector / CudaDeviceDetector (real device).
class DeviceDetector {
public:
  virtual ~DeviceDetector() = default;

  /// Detect device `ordinal` (default 0). std::nullopt on failure (no device /
  /// backend error); the reason is backend-specific and not surfaced here (the
  /// Runtime reports detection failure as a clear typed error).
  [[nodiscard]] virtual std::optional<DeviceInfo>
  Detect(int ordinal = 0) const = 0;
};

/// Pure: extract the bare gfx arch token from a raw HIP gcnArchName string. ROCm
/// reports e.g. "gfx1101:sramecc+:xnack-"; the cache key is the leading "gfx1101".
/// Returns the leading "gfx<digits>" token, or an empty string if no gfx token is
/// present. GPU-free + unit-tested - the mockable core of AMD detection.
[[nodiscard]] std::string ParseGcnArchName(std::string_view gcnArchName);

/// Pure: map a CUDA compute capability (major, minor) to its sm arch string, e.g.
/// (8,0) -> "sm_80", (9,0) -> "sm_90". GPU-free + unit-tested - the CUDA sibling
/// of ParseGcnArchName (CudaDeviceDetector uses it).
[[nodiscard]] std::string SmArchFromComputeCapability(int major, int minor);

/// Real AMD detector: hipGetDeviceProperties(ordinal) -> gcnArchName ->
/// ParseGcnArchName. Compiled with the HIP host API (__HIP_PLATFORM_AMD__); needs
/// a real GPU only at CALL time (construction is trivial + GPU-free, so the gtest
/// can construct it without a device).
class HipDeviceDetector final : public DeviceDetector {
public:
  [[nodiscard]] std::optional<DeviceInfo>
  Detect(int ordinal = 0) const override;
};

/// Real NVIDIA detector: cudaGetDeviceProperties(ordinal) -> major/minor ->
/// SmArchFromComputeCapability. Compiled only under POLYKERNEL_CUDA.
#if defined(POLYKERNEL_CUDA)
class CudaDeviceDetector final : public DeviceDetector {
public:
  [[nodiscard]] std::optional<DeviceInfo>
  Detect(int ordinal = 0) const override;
};
#endif

/// Mock detector: returns a fixed injected arch + name with NO GPU. The gtest uses
/// this to drive the detect -> select -> serve logic deterministically.
class FixedArchDetector final : public DeviceDetector {
public:
  explicit FixedArchDetector(std::string arch,
                             std::string name = "mock-device")
      : arch_(std::move(arch)), name_(std::move(name)) {}

  [[nodiscard]] std::optional<DeviceInfo>
  Detect(int ordinal = 0) const override;

private:
  std::string arch_;
  std::string name_;
};

} // namespace polykernel::runtime

#endif // POLYKERNEL_RUNTIME_DEVICEDETECT_H
