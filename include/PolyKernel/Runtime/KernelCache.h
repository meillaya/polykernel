//===- KernelCache.h - (gpu,op,shape) -> best cached kernel -----*- C++ -*-===//
//
// PolyKernel production runtime (Todo 26 / Wave 5). Second stage of the runtime
// flow: detect GPU -> SELECT BEST CACHED KERNEL -> load -> serve. This header is
// the selection stage: it maps a (gpu, op, shape) request to the best compiled
// variant recorded in the tuning DB.
//
// NO NEW SCHEMA (binding): the tuning DB is the contract-H TuningCache (Todo 24)
// partitioned into per-gpu namespaces by the AmdTuningDb (Todo 23). KernelCache
// REUSES those types (CacheEntry / Shape / AmdTuningDb / ParseTuningCache) - it
// adds no schema of its own. The gpu component of the key IS the detected arch
// (gfx1101 / gfx942), so a gfx1101 request only ever selects a gfx1101 entry - a
// gfx942 best-config can never shadow it (namespace isolation, inherited from the
// AmdTuningDb).
//
// GRACEFUL MISS (binding; the negative QA path checks this): a request with NO
// cache entry - or a present-but-unvalidated entry - yields a clear typed error
// ("no tuned kernel for <gpu>/<op>/<shape>; run the autotuner"). Selection ONLY
// ever returns a validated:true entry; it NEVER returns a wrong kernel and NEVER
// crashes on a miss.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_RUNTIME_KERNELCACHE_H
#define POLYKERNEL_RUNTIME_KERNELCACHE_H

#include "PolyKernel/Autotune/AmdTuningDb.h"
#include "PolyKernel/Autotune/TuningCache.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace polykernel::runtime {

// Reuse the contract-H types directly (no redefinition).
using autotune::CacheEntry;
using autotune::Shape;

/// Render a (gpu, op, shape) request as a stable key string for diagnostics + the
/// miss error, e.g. "gfx1101/fused_matmul_bias_gelu/M=2048,N=4096,K=11008,bf16".
[[nodiscard]] std::string FormatKey(std::string_view gpu, std::string_view op,
                                    const Shape &shape);

/// Result of selecting a cached kernel for a (gpu, op, shape) request. Exactly one
/// of {entry, error} is meaningful: ok() => `entry` is the best VALIDATED cached
/// variant (its `best` is the winning config); !ok() => `error` is a clear typed
/// miss (absent entry, or present-but-unvalidated) - NEVER a wrong kernel.
struct SelectionResult {
  std::optional<CacheEntry> entry;
  std::string error;

  [[nodiscard]] bool ok() const { return entry.has_value(); }
};

/// Maps (gpu, op, shape) -> the best compiled variant from the tuning DB. Reuses
/// the contract-H schema (Todo 24 TuningCache / Todo 23 AmdTuningDb). Selection
/// gates on `validated`: only a validated:true entry is served; an absent or
/// unvalidated entry is a graceful miss with a clear "run the autotuner" error.
class KernelCache {
public:
  /// Load a contract-H TuningCache JSON document (reuses ParseTuningCache via the
  /// AmdTuningDb), partitioning entries by their gpu field. Returns false + a
  /// schema error (with the offending field path) on a malformed document; the
  /// cache is left UNCHANGED on failure (parse-don't-validate, no partial load).
  bool LoadTuningCache(std::string_view json, std::string &error);

  /// Add one entry programmatically (tests + the on-GPU driver). The entry's gpu
  /// field is its namespace. Returns false (no-op) if entry.gpu is empty.
  bool Put(CacheEntry entry);

  /// Select the best VALIDATED cached kernel for (gpu, op, shape). On a hit,
  /// `entry` is the contract-H CacheEntry (its `best` is the winning config). On a
  /// miss (no entry) or an unvalidated entry, a clear error - never a wrong kernel.
  [[nodiscard]] SelectionResult Select(std::string_view gpu, std::string_view op,
                                       const Shape &shape) const;

  /// Total cached entries across all gpu namespaces.
  [[nodiscard]] std::size_t Size() const { return db_.Size(); }

private:
  autotune::AmdTuningDb db_;
};

} // namespace polykernel::runtime

#endif // POLYKERNEL_RUNTIME_KERNELCACHE_H
