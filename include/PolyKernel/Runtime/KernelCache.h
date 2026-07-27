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
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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

/// Hardening metadata for a cached COMPILED BINARY (Todo 42 / Wave 8): where the
/// compiled .so lives on disk + the kernel-hash (a hash of the binary's bytes)
/// recorded when it was cached. This is the invalidation layer ON TOP of the
/// contract-H tuning entry (which carries the tuned config, NOT the binary): the
/// invalidation key is (gpu, op, shape, dtype, kernel-hash). Invalidation recomputes
/// the hash of the on-disk .so and compares; a mismatch (or a missing/unreadable .so)
/// means the cached binary is STALE and the entry is dropped so selection falls back
/// to re-autotuning - a stale/wrong binary is never served. NOT part of contract H
/// (the tuning schema is unchanged; modal/app.py mirrors only the tuning entry).
struct CachedBinary {
  std::string so_path;     ///< Path to the compiled .so on disk.
  std::string kernel_hash; ///< Hash of the binary's bytes at cache time.

  [[nodiscard]] bool operator==(const CachedBinary &) const = default;
};

/// Compute a stable hex kernel-hash (FNV-1a 64-bit) of the bytes of the file at
/// `so_path`. GPU-free + deterministic: the gtest hashes temp files it controls, the
/// on-GPU path hashes the real compiled .so. Returns false + a clear error if the
/// file cannot be opened/read (a missing/unreadable binary is treated as stale).
[[nodiscard]] bool HashFile(std::string_view so_path, std::string &hash_out,
                            std::string &error);

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

  // ---- Hardening (Todo 42 / Wave 8): persist + invalidate + multi-GPU ----

  /// Record the compiled binary (.so path + kernel-hash) backing the entry for
  /// (gpu, op, shape). Returns false (no-op) if gpu is empty OR there is no tuning
  /// entry for that key - a binary must back a real, present tuning entry.
  bool PutBinary(std::string_view gpu, std::string_view op, const Shape &shape,
                 CachedBinary binary);

  /// The cached binary metadata for (gpu, op, shape), if one was recorded.
  [[nodiscard]] std::optional<CachedBinary>
  GetBinary(std::string_view gpu, std::string_view op, const Shape &shape) const;

  /// Persist the tuning DB + binary metadata to `cache_dir` (created if absent) as
  /// two documented files: tuning_db.json (the contract-H envelope, via the AmdTuningDb
  /// serializer) + binaries.json (the hardening metadata). Error-checked I/O: returns
  /// false + a clear error on any directory/file failure. Inverse of LoadPersisted.
  [[nodiscard]] bool Persist(std::string_view cache_dir, std::string &error) const;

  /// Load a cache persisted by Persist (inverse). Parse-don't-validate: a malformed
  /// or unreadable cache yields false + a field-path error and leaves the cache
  /// UNCHANGED (no partial load, no silent default). A missing cache dir/file is an
  /// error. The tuning part reuses the contract-H parser (field-path errors intact);
  /// a binary record with no matching tuning entry is rejected as inconsistent.
  bool LoadPersisted(std::string_view cache_dir, std::string &error);

  /// Drop every entry whose cached binary is STALE: recompute the on-disk .so hash
  /// (HashFile) and compare to the recorded kernel-hash. A hash mismatch OR a
  /// missing/unreadable .so invalidates the entry - it is removed from the tuning DB
  /// AND the binary map - so a later Select falls back to the graceful "run the
  /// autotuner" miss. A stale/wrong binary is never served. Entries with no recorded
  /// binary are left untouched. Returns the number of entries invalidated.
  std::size_t InvalidateStale();

  /// Multi-GPU selection: for EACH gpu in `gpus` (in input order), Select the best
  /// VALIDATED cached kernel for (gpu, op, shape). Per-gpu graceful miss: a gpu with
  /// no/unvalidated entry yields the clear "no tuned kernel" error for THAT gpu (the
  /// byte-identical miss), never a wrong kernel. Returns one (gpu, result) pair per
  /// input gpu - the best cached kernel PER GPU.
  [[nodiscard]] std::vector<std::pair<std::string, SelectionResult>>
  SelectPerGpu(const std::vector<std::string> &gpus, std::string_view op,
               const Shape &shape) const;

private:
  autotune::AmdTuningDb db_;
  /// Hardening metadata keyed by FormatKey(gpu, op, shape) - parallel to db_, holding
  /// the compiled-binary location + kernel-hash the tuning entry itself does not carry.
  std::map<std::string, CachedBinary> binaries_;
};

} // namespace polykernel::runtime

#endif // POLYKERNEL_RUNTIME_KERNELCACHE_H
