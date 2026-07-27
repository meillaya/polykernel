//===- amd_tuning_db_test.cpp - AmdTuningDb unit tests ----------*- C++ -*-===//
//
// Asserts the AMD tuning DB (Todo 23) stores contract-H CacheEntry objects
// keyed by (gpu, op, shape) with PER-GPU NAMESPACE ISOLATION: a gfx1101 (local)
// best-config and a gfx942 (MI300 rental) best-config for the SAME (op, shape)
// coexist and are retrieved separately, never colliding. Also asserts the
// write -> read round-trip through the contract-H Serialize/Parse (no new
// schema), per-namespace serialization, last-writer-wins, and parse-don't-
// validate rejection of an empty gpu namespace + propagation of contract-H
// schema errors.
//
// Suite name is `amd_tuning_db` so `ctest -R amd_tuning_db` discovers it.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/AmdTuningDb.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

using polykernel::autotune::AmdTuningDb;
using polykernel::autotune::CacheEntry;
using polykernel::autotune::Config;
using polykernel::autotune::Correctness;
using polykernel::autotune::Shape;

const Shape kShape{4096, 4096, 4096, "bf16"};

// gfx1101 (local) and gfx942 (rental) winners DIFFER so a cross-namespace leak
// or a key collision is observable.
const Config kGfx1101Best{64, 64, 32, 8, 4, 4, 2};
const Config kGfx942Best{128, 128, 64, 8, 8, 4, 3};

// A measured matmul entry for one gpu namespace.
CacheEntry Entry(const std::string &gpu, const Config &best) {
  CacheEntry e;
  e.gpu = gpu;
  e.op = "matmul";
  e.shape = kShape;
  e.best = best;
  e.scored_by = "measure";
  e.time_ms = 0.123;
  e.validated = true;
  e.correctness = Correctness{0.9999, 1e-4, 0.999};
  return e;
}

// write -> read round-trip through the contract-H Serialize/Parse.
TEST(amd_tuning_db, RoundTrip) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(db.Put(Entry("gfx942", kGfx942Best)));

  const auto parsed = AmdTuningDb::Parse(db.Serialize());
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  EXPECT_EQ(parsed.db->Size(), 2u);

  const auto local = parsed.db->Get("gfx1101", "matmul", kShape);
  const auto rental = parsed.db->Get("gfx942", "matmul", kShape);
  ASSERT_TRUE(local.has_value());
  ASSERT_TRUE(rental.has_value());
  EXPECT_EQ(local->best, kGfx1101Best);
  EXPECT_EQ(rental->best, kGfx942Best);
}

// The pinned acceptance: gfx1101 and gfx942 entries for the SAME (op, shape)
// are stored + retrieved SEPARATELY (namespace isolation, no collision).
TEST(amd_tuning_db, NamespaceIsolation) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(db.Put(Entry("gfx942", kGfx942Best)));
  EXPECT_EQ(db.Size(), 2u); // same (op, shape), different gpu -> two entries.

  const auto local = db.Get("gfx1101", "matmul", kShape);
  const auto rental = db.Get("gfx942", "matmul", kShape);
  ASSERT_TRUE(local.has_value());
  ASSERT_TRUE(rental.has_value());
  EXPECT_EQ(local->best, kGfx1101Best);
  EXPECT_EQ(rental->best, kGfx942Best);
  EXPECT_NE(local->best, rental->best); // the two namespaces never merge.
}

// A lookup in a namespace that has no entry for (op, shape) yields nullopt - it
// never leaks an entry from a sibling namespace.
TEST(amd_tuning_db, GetAbsentIsNullopt) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  EXPECT_FALSE(db.Get("gfx942", "matmul", kShape).has_value()); // other gpu.
  EXPECT_FALSE(db.Get("gfx1101", "softmax", kShape).has_value()); // other op.
}

// SerializeNamespace emits ONLY that gpu's entries as a standalone contract-H
// envelope; re-parsing it yields a single-namespace DB.
TEST(amd_tuning_db, SerializeNamespaceIsPartitioned) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(db.Put(Entry("gfx942", kGfx942Best)));

  const auto parsed = AmdTuningDb::Parse(db.SerializeNamespace("gfx942"));
  ASSERT_TRUE(parsed.ok()) << parsed.error;
  EXPECT_EQ(parsed.db->Size(), 1u);
  EXPECT_EQ(parsed.db->Namespaces(), (std::vector<std::string>{"gfx942"}));
  EXPECT_TRUE(parsed.db->Get("gfx942", "matmul", kShape).has_value());
  EXPECT_FALSE(parsed.db->Get("gfx1101", "matmul", kShape).has_value());
}

// Namespaces() returns the sorted set of gpu partitions present (insertion
// order is irrelevant).
TEST(amd_tuning_db, NamespacesSorted) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx942", kGfx942Best))); // inserted out of order
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  EXPECT_EQ(db.Namespaces(), (std::vector<std::string>{"gfx1101", "gfx942"}));
}

// Last-writer-wins on an identical (gpu, op, shape) key.
TEST(amd_tuning_db, LastWriterWins) {
  AmdTuningDb db;
  ASSERT_TRUE(db.Put(Entry("gfx1101", kGfx1101Best)));
  const Config replacement{32, 32, 32, 4, 2, 2, 2};
  ASSERT_TRUE(db.Put(Entry("gfx1101", replacement)));
  EXPECT_EQ(db.Size(), 1u);
  const auto got = db.Get("gfx1101", "matmul", kShape);
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(got->best, replacement);
}

// An entry with an empty gpu has no namespace -> Put rejects it (no-op).
TEST(amd_tuning_db, PutEmptyGpuRejected) {
  AmdTuningDb db;
  EXPECT_FALSE(db.Put(Entry("", kGfx1101Best)));
  EXPECT_EQ(db.Size(), 0u);
}

// Parse rejects a namespace-less (empty gpu) entry, naming the field.
TEST(amd_tuning_db, ParseEmptyGpuRejected) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "", "op": "matmul",
      "shape": {"M": 4096, "N": 4096, "K": 4096, "dtype": "bf16"},
      "best": {"block_m": 64, "block_n": 64, "block_k": 32, "num_warps": 8,
               "vector_width": 4, "unroll": 4, "shared_memory_stages": 2},
      "scored_by": "measure", "time_ms": 0.123, "validated": true,
      "correctness": {"cosine": 0.9999, "max_rel_err": 1e-4, "pcc": 0.999}
    }]
  })";
  const auto parsed = AmdTuningDb::Parse(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_FALSE(parsed.db.has_value());
  EXPECT_NE(parsed.error.find("gpu"), std::string::npos) << parsed.error;
}

// A contract-H schema error (missing scored_by) propagates through Parse with
// its field path intact - the DB adds no schema of its own.
TEST(amd_tuning_db, ParseSchemaErrorPropagates) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "gfx942", "op": "matmul",
      "shape": {"M": 128, "N": 128, "K": 128, "dtype": "fp16"},
      "best": {"block_m": 16, "block_n": 32, "block_k": 32, "num_warps": 4,
               "vector_width": 1, "unroll": 4, "shared_memory_stages": 2},
      "time_ms": null, "validated": false,
      "correctness": {"cosine": 0.9995, "max_rel_err": 5e-3, "pcc": 0.995}
    }]
  })";
  const auto parsed = AmdTuningDb::Parse(json);
  EXPECT_FALSE(parsed.ok());
  EXPECT_NE(parsed.error.find("scored_by"), std::string::npos) << parsed.error;
}

} // namespace
