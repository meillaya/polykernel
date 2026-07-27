//===- KernelCache.cpp - (gpu,op,shape) -> best cached kernel ---*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). See KernelCache.h.
//
//===----------------------------------------------------------------------===//
//
// The cache is a thin selection layer over the contract-H AmdTuningDb (Todo 23):
// LoadTuningCache parses a contract-H JSON envelope (reusing ParseTuningCache, so
// any schema error propagates with its field path) and partitions the entries by
// gpu; Select looks up (gpu, op, shape) WITHIN the detected gpu namespace and
// gates on `validated`. The two graceful-miss errors (absent entry; present-but-
// unvalidated) are the ONLY outcomes besides a served validated entry - a wrong
// kernel is never returned and a miss never crashes.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/KernelCache.h"

namespace polykernel::runtime {

std::string FormatKey(std::string_view gpu, std::string_view op,
                      const Shape &shape) {
  return std::string(gpu) + "/" + std::string(op) + "/M=" +
         std::to_string(shape.m) + ",N=" + std::to_string(shape.n) + ",K=" +
         std::to_string(shape.k) + "," + shape.dtype;
}

bool KernelCache::LoadTuningCache(std::string_view json, std::string &error) {
  autotune::AmdTuningDbParseResult parsed = autotune::AmdTuningDb::Parse(json);
  if (!parsed.ok()) {
    error = std::move(parsed.error);
    return false;
  }
  db_ = std::move(*parsed.db);
  return true;
}

bool KernelCache::Put(CacheEntry entry) { return db_.Put(std::move(entry)); }

SelectionResult KernelCache::Select(std::string_view gpu, std::string_view op,
                                    const Shape &shape) const {
  SelectionResult result;
  const std::optional<CacheEntry> entry = db_.Get(gpu, op, shape);
  if (!entry.has_value()) {
    // The pinned negative path: NO cache entry for this (gpu, op, shape). Return a
    // clear, actionable error - never a wrong kernel, never a crash.
    result.error =
        "no tuned kernel for " + FormatKey(gpu, op, shape) +
        "; run the autotuner (polykernel-bench --autotune) to populate the cache";
    return result;
  }
  if (!entry->validated) {
    // An entry exists but never passed the correctness gate (validated:false). Do
    // NOT serve it - a bench-rejected variant is treated as a miss.
    result.error = "cached entry for " + FormatKey(gpu, op, shape) +
                   " is not validated; run the autotuner to produce a "
                   "correctness-gated variant";
    return result;
  }
  result.entry = entry;
  return result;
}

} // namespace polykernel::runtime
