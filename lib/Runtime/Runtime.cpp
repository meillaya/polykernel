//===- Runtime.cpp - detect -> select -> load -> serve ----------*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). See Runtime.h.
//
//===----------------------------------------------------------------------===//
//
// Resolve = detect (DeviceDetector) + select (KernelCache). Run = Resolve + load
// (look up the registered launcher) + serve (invoke it). Every failure mode is a
// clear typed error returned to the caller - a detection failure, a cache miss
// ("no tuned kernel ...; run the autotuner"), a missing launcher, or a launch
// failure - so a (gpu, op, shape) with no tuned entry can never yield a wrong
// kernel or a crash.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/Runtime.h"

namespace polykernel::runtime {

ResolveResult Runtime::Resolve(std::string_view op, const Shape &shape) const {
  ResolveResult result;

  const std::optional<DeviceInfo> device = detector_.Detect();
  if (!device.has_value()) {
    result.error = "device detection failed; no usable GPU to serve op " +
                   std::string(op);
    return result;
  }
  result.device = device;

  SelectionResult selected = cache_.Select(device->arch, op, shape);
  if (!selected.ok()) {
    result.error = std::move(selected.error);
    return result;
  }
  result.entry = std::move(selected.entry);
  return result;
}

bool Runtime::Run(std::string_view op, const Shape &shape, void *io,
                  std::string &error) {
  ResolveResult resolved = Resolve(op, shape);
  if (!resolved.ok()) {
    error = std::move(resolved.error);
    return false;
  }

  const auto it = launchers_.find(op);
  if (it == launchers_.end()) {
    error = "no compiled launcher registered for op " + std::string(op) +
            "; load the compiled variant (register its launch entry) first";
    return false;
  }

  // Serve: invoke the loaded compiled variant with the selected tuned entry.
  return it->second(*resolved.entry, io, error);
}

} // namespace polykernel::runtime
