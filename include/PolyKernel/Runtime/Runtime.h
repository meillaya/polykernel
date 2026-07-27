//===- Runtime.h - detect -> select -> load -> serve ------------*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). The top-level flow:
//
//     detect GPU arch -> select best cached kernel -> load -> serve (execute)
//
// for a (gpu, op, shape) request. Composition of the two stages beside it:
//   - DeviceDetect.h : detect the arch (mockable; FixedArchDetector for the
//                      GPU-free gtest, HipDeviceDetector on the real GPU).
//   - KernelCache.h  : select the best VALIDATED cached variant for the key
//                      (contract-H tuning DB; graceful miss).
// This class wires them together and adds the LOAD + SERVE step: a per-op launcher
// registry stands in for "load the compiled variant" (a registered host symbol is
// the loaded compiled kernel - the dlopen-a-cached-.so / select-a-compiled-symbol
// abstraction). On a cache miss, a detection failure, a missing launcher, or a
// launch failure, Run returns a clear typed error - NEVER a wrong kernel, NEVER a
// crash.
//
// GPU-FREE TESTABLE (binding): the detector AND the launchers are injected, so
// lib/Runtime/tests/runtime_test.cpp drives the full detect -> select -> load ->
// serve flow with a fake arch + a recording launcher and no device. The on-GPU
// path injects HipDeviceDetector + registers the generated launch_<op>.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_RUNTIME_RUNTIME_H
#define POLYKERNEL_RUNTIME_RUNTIME_H

#include "PolyKernel/Runtime/DeviceDetect.h"
#include "PolyKernel/Runtime/KernelCache.h"

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace polykernel::runtime {

/// A loaded compiled variant's host launch entry. Given the selected contract-H
/// CacheEntry (its `best` is the tuned config that identifies the variant) plus an
/// opaque I/O bundle the launcher understands, execute the kernel. Returns false +
/// a clear error on failure. The gtest injects a recording fake; the on-GPU path
/// registers the generated launch_<op>. This is the "load the compiled variant"
/// abstraction (a registered symbol stands in for a dlopen'd cached .so).
using LaunchFn = std::function<bool(const CacheEntry &entry, void *io,
                                    std::string &error)>;

/// Result of detect -> select (the resolve step before load + serve). ok() =>
/// `device` is the detected device and `entry` the best validated cached variant
/// for (device.arch, op, shape). !ok() => `error` is the clear detection-failure
/// or cache-miss error (and `entry` is empty).
struct ResolveResult {
  std::optional<DeviceInfo> device;
  std::optional<CacheEntry> entry;
  std::string error;

  [[nodiscard]] bool ok() const { return entry.has_value(); }
};

/// The production runtime. Holds a reference to an injected device detector + an
/// injected kernel cache, and owns a per-op launcher registry. Resolve does detect
/// -> select (no launch); Run does detect -> select -> load -> serve.
class Runtime {
public:
  Runtime(const DeviceDetector &detector, const KernelCache &cache)
      : detector_(detector), cache_(cache) {}

  /// Register the compiled variant's launcher for `op` (last-writer-wins). This is
  /// the "load the compiled variant" step: the registered symbol IS the loaded
  /// kernel the Runtime will serve for that op.
  void RegisterLauncher(std::string op, LaunchFn fn) {
    launchers_[std::move(op)] = std::move(fn);
  }

  /// Detect the device arch + select the best validated cached kernel for
  /// (detected arch, op, shape). No launch. Graceful: a detection failure or a
  /// cache miss yields !ok() + a clear error, never a wrong kernel, never a crash.
  [[nodiscard]] ResolveResult Resolve(std::string_view op,
                                      const Shape &shape) const;

  /// Detect -> select -> load -> serve: Resolve, then invoke the launcher
  /// registered for `op` with the selected entry + `io`. Returns false + a clear
  /// error on a detection failure, a cache miss, a missing launcher, or a launch
  /// failure - never a wrong kernel, never a crash.
  bool Run(std::string_view op, const Shape &shape, void *io,
           std::string &error);

private:
  const DeviceDetector &detector_;
  const KernelCache &cache_;
  // Transparent comparator so Run can look up by string_view without a copy.
  std::map<std::string, LaunchFn, std::less<>> launchers_;
};

} // namespace polykernel::runtime

#endif // POLYKERNEL_RUNTIME_RUNTIME_H
