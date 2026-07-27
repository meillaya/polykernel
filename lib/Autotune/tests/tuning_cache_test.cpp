//===- tuning_cache_test.cpp - TuningCache unit tests ----------*- C++ -*-===//
//
// Asserts the contract-H JSON tuning cache round-trips (serialize -> parse ->
// equal), that `time_ms` null is preserved, and - parse-don't-validate - that a
// document missing ANY required field (e.g. `scored_by`, `correctness.cosine`,
// the `time_ms` key) or carrying a wrong type / wrong version is REJECTED with
// a schema error naming the field, never silently defaulted.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/TuningCache.h"

#include "gtest/gtest.h"

#include <string>

namespace {

using polykernel::autotune::CacheEntry;
using polykernel::autotune::Config;
using polykernel::autotune::Correctness;
using polykernel::autotune::ParseTuningCache;
using polykernel::autotune::SerializeTuningCache;
using polykernel::autotune::Shape;
using polykernel::autotune::TuningCache;

CacheEntry MeasuredEntry() {
  CacheEntry e;
  e.gpu = "sm_90";
  e.op = "matmul";
  e.shape = Shape{4096, 4096, 4096, "bf16"};
  e.best = Config{64, 64, 32, 8, 4, 4, 2};
  e.scored_by = "measure";
  e.time_ms = 0.123; // measured -> concrete number.
  e.validated = true;
  e.correctness = Correctness{0.9999, 1e-4, 0.999};
  return e;
}

CacheEntry ModeledEntry() {
  CacheEntry e;
  e.gpu = "gfx942";
  e.op = "matmul";
  e.shape = Shape{128, 128, 128, "fp16"};
  e.best = Config{16, 32, 32, 4, 1, 4, 2};
  e.scored_by = "compile_time_model";
  e.time_ms = std::nullopt; // compile-time model -> JSON null.
  e.validated = false;
  e.correctness = Correctness{0.9995, 5e-3, 0.995};
  return e;
}

// A measured entry and a compile-time-model entry both round-trip exactly.
TEST(tuning_cache, RoundTrip) {
  TuningCache cache;
  cache.entries.push_back(MeasuredEntry());
  cache.entries.push_back(ModeledEntry());

  const std::string json = SerializeTuningCache(cache);
  const auto parsed = ParseTuningCache(json);
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  EXPECT_EQ(*parsed.cache, cache);
}

// time_ms == null survives the round-trip as std::nullopt (and the serialized
// form actually contains a JSON null).
TEST(tuning_cache, NullTimeMsPreserved) {
  TuningCache cache;
  cache.entries.push_back(ModeledEntry());
  const std::string json = SerializeTuningCache(cache);
  EXPECT_NE(json.find("\"time_ms\":null"), std::string::npos) << json;

  const auto parsed = ParseTuningCache(json);
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  ASSERT_EQ(parsed.cache->entries.size(), 1u);
  EXPECT_FALSE(parsed.cache->entries[0].time_ms.has_value());
}

// The serialized measured entry carries its concrete time (not null).
TEST(tuning_cache, MeasuredTimeIsNumber) {
  TuningCache cache;
  cache.entries.push_back(MeasuredEntry());
  const std::string json = SerializeTuningCache(cache);
  EXPECT_NE(json.find("\"time_ms\":0.123"), std::string::npos) << json;
}

// --- negative cases: parse-don't-validate, each names the offending field ---

// A fully-valid document used as the base for field-removal negatives.
constexpr const char *kValid = R"({
  "version": 1,
  "entries": [{
    "gpu": "sm_90", "op": "matmul",
    "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
    "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
             "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
    "scored_by": "measure", "time_ms": 0.123, "validated": true,
    "correctness": {"cosine": 0.9999, "max_rel_err": 1e-4, "pcc": 0.999}
  }]
})";

TEST(tuning_cache, ValidDocumentParses) {
  const auto parsed = ParseTuningCache(kValid);
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  EXPECT_EQ(parsed.cache->entries.size(), 1u);
  EXPECT_EQ(parsed.cache->entries[0].gpu, "sm_90");
  EXPECT_EQ(parsed.cache->entries[0].best.block_m, 64);
}

// The plan's pinned negative: an entry missing `scored_by` is rejected with a
// schema error naming the field - never a silent default.
TEST(tuning_cache, MissingScoredByRejected) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "sm_90", "op": "matmul",
      "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
      "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
               "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
      "time_ms": 0.123, "validated": true,
      "correctness": {"cosine": 0.9999, "max_rel_err": 1e-4, "pcc": 0.999}
    }]
  })";
  const auto parsed = ParseTuningCache(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_FALSE(parsed.cache.has_value());
  EXPECT_NE(parsed.error.find("scored_by"), std::string::npos) << parsed.error;
}

// A missing nested correctness field (cosine) is rejected and named.
TEST(tuning_cache, MissingCorrectnessCosineRejected) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "sm_90", "op": "matmul",
      "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
      "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
               "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
      "scored_by": "measure", "time_ms": 0.123, "validated": true,
      "correctness": {"max_rel_err": 1e-4, "pcc": 0.999}
    }]
  })";
  const auto parsed = ParseTuningCache(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("cosine"), std::string::npos) << parsed.error;
}

// The time_ms KEY must be present (null is fine, absent is not).
TEST(tuning_cache, MissingTimeMsKeyRejected) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "sm_90", "op": "matmul",
      "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
      "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
               "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
      "scored_by": "measure", "validated": true,
      "correctness": {"cosine": 0.9999, "max_rel_err": 1e-4, "pcc": 0.999}
    }]
  })";
  const auto parsed = ParseTuningCache(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("time_ms"), std::string::npos) << parsed.error;
}

// A wrong-typed field (validated as a string) is rejected.
TEST(tuning_cache, WrongTypeRejected) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "sm_90", "op": "matmul",
      "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
      "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
               "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
      "scored_by": "measure", "time_ms": 0.123, "validated": "yes",
      "correctness": {"cosine": 0.9999, "max_rel_err": 1e-4, "pcc": 0.999}
    }]
  })";
  const auto parsed = ParseTuningCache(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("validated"), std::string::npos) << parsed.error;
}

// An unsupported envelope version is rejected.
TEST(tuning_cache, WrongVersionRejected) {
  const auto parsed = ParseTuningCache(R"({"version": 99, "entries": []})");
  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("version"), std::string::npos) << parsed.error;
}

// Malformed JSON is rejected (not a crash).
TEST(tuning_cache, MalformedJsonRejected) {
  const auto parsed = ParseTuningCache("{ this is not json ");
  EXPECT_FALSE(parsed.ok());
  EXPECT_FALSE(parsed.error.empty());
}

} // namespace
