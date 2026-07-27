//===- AmdTuningDb.h - Per-GPU-namespaced AMD tuning DB ---------*- C++ -*-===//
//
// PolyKernel autotuner (Todo 23 / Wave 4). Finalizes the AMD tuning database on
// top of the contract-H TuningCache (Todo 24): it stores CacheEntry objects
// keyed by (gpu, op, shape) and PARTITIONS them into per-GPU namespaces. The
// gpu field of each CacheEntry IS its namespace, so the two report targets -
// gfx1101 (the local RX 7800 XT) and gfx942 (the MI300 / CDNA3 headline part,
// compile-only locally in Wave 4, runtime on the Wave 5 rental) - are stored
// and retrieved completely separately: a gfx1101 best-config can never shadow
// or collide with a gfx942 best-config for the same (op, shape).
//
// NO new schema: persistence reuses SerializeTuningCache / ParseTuningCache
// (llvm::json, contract H). The whole DB serializes as one contract-H envelope
// (each entry carries its gpu = its namespace); a single namespace serializes
// as a standalone contract-H envelope holding only that gpu's entries, so the
// gfx1101 local DB and the gfx942 rental DB can live in separate files while
// staying on the one pinned schema.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_AUTOTUNE_AMDTUNINGDB_H
#define POLYKERNEL_AUTOTUNE_AMDTUNINGDB_H

#include "PolyKernel/Autotune/TuningCache.h"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace polykernel::autotune {

/// The two report GPU namespaces. gfx1101 is the local RX 7800 XT (runtime
/// now); gfx942 is the MI300 / CDNA3 headline part (compile-only locally in
/// Wave 4, runtime on the Wave 5 rental). The DB is general - any non-empty gpu
/// string is a valid namespace - but these are the two the report partitions.
inline constexpr std::string_view kAmdGfx1101 = "gfx1101";
inline constexpr std::string_view kAmdGfx942 = "gfx942";

struct AmdTuningDbParseResult; // defined after AmdTuningDb (holds it by value).

/// Per-GPU-namespaced AMD tuning database. Stores contract-H CacheEntry objects
/// keyed by (gpu, op, shape); the gpu component is the namespace, so gfx1101
/// and gfx942 entries are isolated from each other. Value type; deterministic
/// iteration (std::map order over the key, gpu leading so a namespace is a
/// contiguous key range).
class AmdTuningDb {
public:
  /// Store `entry` in the namespace named by entry.gpu. Replaces any existing
  /// entry with the same (gpu, op, shape) key (last-writer-wins). Returns false
  /// (no-op) if entry.gpu is empty - an entry must belong to a namespace.
  bool Put(CacheEntry entry);

  /// The best config for (gpu, op, shape) looked up WITHIN the `gpu` namespace
  /// only. std::nullopt if that namespace has no entry for (op, shape). A
  /// gfx1101 lookup never returns a gfx942 entry, and vice versa.
  [[nodiscard]] std::optional<CacheEntry>
  Get(std::string_view gpu, std::string_view op, const Shape &shape) const;

  /// All entries in one gpu namespace, in deterministic (op, shape) key order.
  [[nodiscard]] std::vector<CacheEntry> Entries(std::string_view gpu) const;

  /// The gpu namespaces present (sorted, deterministic).
  [[nodiscard]] std::vector<std::string> Namespaces() const;

  /// Total entries across all namespaces.
  [[nodiscard]] std::size_t Size() const { return entries_.size(); }

  /// Serialize the WHOLE DB (every namespace) as one contract-H TuningCache JSON
  /// envelope; each entry carries its gpu (= its namespace). Inverse of Parse:
  /// Parse(Serialize()) reproduces the same entries.
  [[nodiscard]] std::string Serialize() const;

  /// Serialize ONE gpu namespace as a standalone contract-H TuningCache JSON
  /// envelope holding only that gpu's entries (empty entries[] if the namespace
  /// is absent). Lets the gfx1101 local DB and the gfx942 rental DB persist to
  /// separate files while staying on the one contract-H schema.
  [[nodiscard]] std::string SerializeNamespace(std::string_view gpu) const;

  /// Parse a contract-H TuningCache JSON document into a DB, partitioning the
  /// entries by their gpu field into namespaces. Strict: reuses
  /// ParseTuningCache, so any contract-H schema error propagates with its field
  /// path; additionally rejects an entry with an empty gpu (a namespace-less
  /// entry is invalid for a partitioned DB).
  [[nodiscard]] static AmdTuningDbParseResult Parse(std::string_view json);

private:
  /// (gpu, op, shape) identity key with a total order for std::map. gpu is the
  /// leading component so one namespace is a contiguous key range.
  struct Key {
    std::string gpu;
    std::string op;
    Shape shape;

    [[nodiscard]] friend bool operator<(const Key &a, const Key &b) {
      if (a.gpu != b.gpu)
        return a.gpu < b.gpu;
      if (a.op != b.op)
        return a.op < b.op;
      if (a.shape.m != b.shape.m)
        return a.shape.m < b.shape.m;
      if (a.shape.n != b.shape.n)
        return a.shape.n < b.shape.n;
      if (a.shape.k != b.shape.k)
        return a.shape.k < b.shape.k;
      return a.shape.dtype < b.shape.dtype;
    }
  };

  std::map<Key, CacheEntry> entries_;
};

/// Result of parsing an AMD tuning DB document. Exactly one of {db, error} is
/// meaningful (mirrors TuningCacheParseResult): ok() => db populated + error
/// empty; !ok() => db empty + error names the offending field.
struct AmdTuningDbParseResult {
  std::optional<AmdTuningDb> db;
  std::string error;

  [[nodiscard]] bool ok() const { return db.has_value(); }
};

} // namespace polykernel::autotune

#endif // POLYKERNEL_AUTOTUNE_AMDTUNINGDB_H
