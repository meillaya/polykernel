//===- device_detect_test.cpp - DeviceDetect unit tests ---------*- C++ -*-===//
//
// GPU-free tests of the runtime's device-detection stage (Todo 26). The detection
// LOGIC is pure + mockable, so it is asserted here with NO GPU:
//   - ParseGcnArchName: the AMD arch-token extraction (hipGetDeviceProperties
//     reports "gfx1101:sramecc+:xnack-"; the cache key is the bare "gfx1101").
//   - SmArchFromComputeCapability: the CUDA compute-capability -> "sm_XY" map.
//   - FixedArchDetector: the mock the runtime test injects (returns the injected
//     arch with no device).
//   - HipDeviceDetector is constructible without a GPU (construction is trivial;
//     only Detect() touches a device, which this GPU-free suite never calls).
//
// Suite name is `device_detect` so `ctest -R device_detect` discovers it.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/DeviceDetect.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>

namespace {

using polykernel::runtime::DeviceInfo;
using polykernel::runtime::FixedArchDetector;
using polykernel::runtime::HipDeviceDetector;
using polykernel::runtime::ParseGcnArchName;
using polykernel::runtime::SmArchFromComputeCapability;

// The AMD arch token is the leading "gfx<digits>"; ROCm appends feature flags.
TEST(device_detect, ParseGcnArchNameStripsFeatureFlags) {
  EXPECT_EQ(ParseGcnArchName("gfx1101:sramecc+:xnack-"), "gfx1101");
  EXPECT_EQ(ParseGcnArchName("gfx942:sramecc+:xnack-"), "gfx942");
  EXPECT_EQ(ParseGcnArchName("gfx90a"), "gfx90a");
}

// A bare arch string (no feature suffix) parses to itself.
TEST(device_detect, ParseGcnArchNameBareToken) {
  EXPECT_EQ(ParseGcnArchName("gfx1101"), "gfx1101");
  EXPECT_EQ(ParseGcnArchName("gfx942"), "gfx942");
}

// No gfx token, or "gfx" with no digits, yields an empty string (never a guess).
TEST(device_detect, ParseGcnArchNameAbsentIsEmpty) {
  EXPECT_EQ(ParseGcnArchName(""), "");
  EXPECT_EQ(ParseGcnArchName("no-arch-here"), "");
  EXPECT_EQ(ParseGcnArchName("gfx"), "");       // no digits.
  EXPECT_EQ(ParseGcnArchName("gfx:sramecc+"), ""); // no digits before ':'.
}

// CUDA compute capability -> sm arch string.
TEST(device_detect, SmArchFromComputeCapability) {
  EXPECT_EQ(SmArchFromComputeCapability(8, 0), "sm_80");
  EXPECT_EQ(SmArchFromComputeCapability(9, 0), "sm_90");
  EXPECT_EQ(SmArchFromComputeCapability(8, 6), "sm_86");
}

// The mock detector returns the injected arch + name with NO GPU. This is what the
// runtime test uses to drive the selection logic deterministically.
TEST(device_detect, FixedArchDetectorReturnsInjectedArch) {
  const FixedArchDetector det("gfx1101", "mock-7800-xt");
  const std::optional<DeviceInfo> info = det.Detect();
  ASSERT_TRUE(info.has_value());
  EXPECT_EQ(info->arch, "gfx1101");
  EXPECT_EQ(info->name, "mock-7800-xt");
}

// The mock ignores the ordinal (it always returns the fixed arch).
TEST(device_detect, FixedArchDetectorOrdinalIndependent) {
  const FixedArchDetector det("gfx942");
  ASSERT_TRUE(det.Detect(0).has_value());
  ASSERT_TRUE(det.Detect(3).has_value());
  EXPECT_EQ(det.Detect(0)->arch, "gfx942");
  EXPECT_EQ(det.Detect(3)->arch, "gfx942");
  EXPECT_EQ(det.Detect(0)->name, "mock-device"); // default name.
}

// The real HIP detector is constructible without a GPU (construction is trivial;
// only Detect() touches a device). This GPU-free suite does NOT call Detect().
TEST(device_detect, HipDeviceDetectorConstructibleWithoutGpu) {
  const HipDeviceDetector det; // no device call here.
  (void)det;
  SUCCEED();
}

} // namespace
