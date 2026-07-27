//===- TuningCache.h - JSON tuning-cache (contract H) ----------*- C++ -*-===//
//
// PolyKernel autotuner (Todo 24 / Wave 5). Reads + writes the AUTHORITATIVE
// tuning-cache schema (Pinned contract H) as a versioned JSON envelope:
//
//   { "version": 1,
//     "entries": [
//       { "gpu": "sm_90", "op": "matmul",
//         "shape": { "M": 4096, "N": 4096, "K": 4096, "dtype": "bf16" },
//         "best":  { "block_m": 128, "block_n": 128, "block_k": 64,
//                    "num_warps": 8, "vector_width": 4, "unroll": 4,
//                    "shared_memory_stages": 3 },
//         "scored_by": "measure",          // or "compile_time_model"
//         "time_ms": 0.123,                // null when scored_by compile-time
//         "validated": true,
//         "correctness": { "cosine": 0.9999, "max_rel_err": 1e-4,
//                          "pcc": 0.999 } } ] }
//
// PARSE-DON'T-VALIDATE: ParseTuningCache returns a typed result. A missing or
// wrong-typed required field (e.g. absent `scored_by`, absent
// `correctness.cosine`) yields an explicit schema error naming the field -
// NEVER a silent default, NEVER a partially-filled entry. `time_ms` may be JSON
// null (legitimate when scored by a compile-time model rather than measured),
// but the KEY must be present.
//
// Built on llvm::json (LLVM Support, already a project dependency) - NO new
// third-party JSON library.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_AUTOTUNE_TUNINGCACHE_H
#define POLYKERNEL_AUTOTUNE_TUNINGCACHE_H

#include "PolyKernel/Autotune/ConfigSpace.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace polykernel::autotune {

/// Current tuning-cache envelope version. Bumped only on a breaking schema
/// change; ParseTuningCache rejects any other version with a schema error.
inline constexpr int kTuningCacheVersion = 1;

/// The (gpu, op, shape) identity key's shape component. dtype is the operand
/// element type string (e.g. "bf16", "fp16", "fp32").
struct Shape {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::string dtype;

  [[nodiscard]] bool operator==(const Shape &) const = default;
};

/// Contract-H correctness metrics recorded per tuned variant so the bench
/// (Todo 25) can correctness-gate against the golden thresholds (contract C:
/// cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99).
struct Correctness {
  double cosine = 0.0;
  double max_rel_err = 0.0;
  double pcc = 0.0;

  [[nodiscard]] bool operator==(const Correctness &) const = default;
};

/// One tuning-cache entry: the winning `best` config for (gpu, op, shape), how
/// it was scored, its (optional) measured time, a validated flag, and the
/// correctness metrics. `time_ms` is std::nullopt <=> JSON null (compile-time
/// model); a measured entry carries a concrete number.
struct CacheEntry {
  std::string gpu;
  std::string op;
  Shape shape;
  Config best;
  std::string scored_by;          ///< "measure" | "compile_time_model".
  std::optional<double> time_ms;  ///< nullopt => JSON null.
  bool validated = false;
  Correctness correctness;

  [[nodiscard]] bool operator==(const CacheEntry &) const = default;
};

/// The full tuning cache: a versioned envelope over a list of entries.
struct TuningCache {
  int version = kTuningCacheVersion;
  std::vector<CacheEntry> entries;

  [[nodiscard]] bool operator==(const TuningCache &) const = default;
};

/// Result of parsing a tuning-cache JSON document. Exactly one of {cache,
/// error} is meaningful: ok() => cache populated + error empty; !ok() => cache
/// empty + error describes the offending field (schema error).
struct TuningCacheParseResult {
  std::optional<TuningCache> cache;
  std::string error;

  [[nodiscard]] bool ok() const { return cache.has_value(); }
};

/// Parse + validate a tuning-cache JSON document (contract H). Rejects:
/// malformed JSON, a missing/wrong `version`, and ANY entry missing a required
/// field or carrying a wrong-typed field - with an error naming the field path
/// (e.g. "entries[0].scored_by: missing required string field").
[[nodiscard]] TuningCacheParseResult ParseTuningCache(std::string_view json);

/// Serialize a tuning cache to JSON (contract H field names, compact form).
/// `time_ms == nullopt` is emitted as JSON null. Inverse of ParseTuningCache:
/// ParseTuningCache(SerializeTuningCache(c)) == c.
[[nodiscard]] std::string SerializeTuningCache(const TuningCache &cache);

} // namespace polykernel::autotune

#endif // POLYKERNEL_AUTOTUNE_TUNINGCACHE_H
