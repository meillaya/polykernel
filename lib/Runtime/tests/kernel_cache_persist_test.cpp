//===- kernel_cache_persist_test.cpp - hardened cache tests -----*- C++ -*-===//
//
// GPU-free tests of the Todo-42 kernel-cache hardening: persistent on-disk cache
// (tuning DB + compiled .so paths), hash-keyed invalidation, multi-GPU per-GPU
// selection, and graceful fallback to re-autotuning on a stale/missing binary. The
// compiled binaries are MOCKED as temp files the test writes + corrupts (HashFile
// hashes their real bytes), so the whole flow runs with NO GPU - mirroring the
// FixedArchDetector / recording-launcher style of the existing runtime gtests.
//
// Suite name is `kernel_cache_persist` so `ctest -R kernel_cache_persist` discovers it.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/KernelCache.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

using polykernel::runtime::CachedBinary;
using polykernel::runtime::CacheEntry;
using polykernel::runtime::HashFile;
using polykernel::runtime::KernelCache;
using polykernel::runtime::SelectionResult;
using polykernel::runtime::Shape;
using polykernel::autotune::Config;
using polykernel::autotune::Correctness;

const Shape kShape{2048, 4096, 11008, "bf16"};
const char *const kOp = "fused_matmul_bias_gelu";

// gfx1101 (local) and gfx942 (rental) winners DIFFER so a cross-GPU leak or a
// wrong-arch selection is observable in the multi-GPU test.
const Config kGfx1101Best{16, 32, 32, 4, 2, 2, 2};
const Config kGfx942Best{128, 128, 64, 8, 8, 4, 3};

CacheEntry Entry(const std::string &gpu, const Config &best, bool validated = true) {
  CacheEntry e;
  e.gpu = gpu;
  e.op = kOp;
  e.shape = kShape;
  e.best = best;
  e.scored_by = "measure";
  e.time_ms = 0.123;
  e.validated = validated;
  e.correctness = Correctness{0.9999, 1e-4, 0.999};
  return e;
}

// A unique temp dir for one test (cleaned up by the OS temp reaper; tests are
// isolated by a unique path so concurrent ctest runs never collide).
std::string MakeTempDir() {
  llvm::SmallString<128> dir;
  const std::error_code ec =
      llvm::sys::fs::createUniqueDirectory("kernel_cache_persist", dir);
  EXPECT_FALSE(ec) << ec.message();
  return std::string(dir.str());
}

void WriteBytes(const std::string &path, const std::string &content) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_None);
  ASSERT_FALSE(ec) << ec.message();
  os << content;
}

// Persist + reload round-trips the tuning DB AND the binary metadata: a fresh cache
// (a new "run") reloads what the first persisted and selects the same entry.
TEST(kernel_cache_persist, PersistThenReloadAcrossRuns) {
  const std::string dir = MakeTempDir();
  {
    KernelCache writer;
    ASSERT_TRUE(writer.Put(Entry("gfx1101", kGfx1101Best)));
    const std::string so = dir + "/gfx1101.so";
    WriteBytes(so, "binary-v1");
    std::string hash, hash_err;
    ASSERT_TRUE(HashFile(so, hash, hash_err)) << hash_err;
    ASSERT_TRUE(writer.PutBinary("gfx1101", kOp, kShape, CachedBinary{so, hash}));

    std::string error;
    ASSERT_TRUE(writer.Persist(dir, error)) << error;
  }

  KernelCache reader; // a fresh process/run.
  std::string error;
  ASSERT_TRUE(reader.LoadPersisted(dir, error)) << error;
  EXPECT_EQ(reader.Size(), 1u);

  const SelectionResult res = reader.Select("gfx1101", kOp, kShape);
  ASSERT_TRUE(res.ok()) << res.error;
  ASSERT_TRUE(res.entry.has_value());
  EXPECT_EQ(res.entry->gpu, "gfx1101");
  EXPECT_EQ(res.entry->best, kGfx1101Best);

  const std::optional<CachedBinary> bin = reader.GetBinary("gfx1101", kOp, kShape);
  ASSERT_TRUE(bin.has_value());
  EXPECT_EQ(bin->so_path, dir + "/gfx1101.so");
  EXPECT_FALSE(bin->kernel_hash.empty());
}

// Multi-GPU selection picks the best cached kernel PER GPU: each detected gpu gets
// its OWN namespace's winner, and a gpu with no entry gets the graceful miss.
TEST(kernel_cache_persist, MultiGpuSelectsBestPerGpu) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(cache.Put(Entry("gfx942", kGfx942Best)));

  const std::vector<std::string> gpus = {"gfx1101", "gfx942", "gfx900"};
  const auto results = cache.SelectPerGpu(gpus, kOp, kShape);
  ASSERT_EQ(results.size(), 3u);

  EXPECT_EQ(results[0].first, "gfx1101");
  ASSERT_TRUE(results[0].second.ok()) << results[0].second.error;
  EXPECT_EQ(results[0].second.entry->best, kGfx1101Best); // the gfx1101 winner.

  EXPECT_EQ(results[1].first, "gfx942");
  ASSERT_TRUE(results[1].second.ok()) << results[1].second.error;
  EXPECT_EQ(results[1].second.entry->best, kGfx942Best); // the gfx942 winner.

  // gfx900 has no entry: the byte-identical graceful miss, never a wrong kernel.
  EXPECT_EQ(results[2].first, "gfx900");
  EXPECT_FALSE(results[2].second.ok());
  EXPECT_EQ(results[2].second.error,
            "no tuned kernel for gfx900/fused_matmul_bias_gelu/"
            "M=2048,N=4096,K=11008,bf16; run the autotuner "
            "(polykernel-bench --autotune) to populate the cache");
}

// Persisted multi-GPU cache reloads and STILL selects per-GPU (persist + multi-GPU
// compose): the reloaded cache serves each gpu its own winner.
TEST(kernel_cache_persist, ReloadedCacheSelectsPerGpu) {
  const std::string dir = MakeTempDir();
  KernelCache writer;
  ASSERT_TRUE(writer.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(writer.Put(Entry("gfx942", kGfx942Best)));
  std::string error;
  ASSERT_TRUE(writer.Persist(dir, error)) << error;

  KernelCache reader;
  ASSERT_TRUE(reader.LoadPersisted(dir, error)) << error;
  const auto results =
      reader.SelectPerGpu({"gfx1101", "gfx942"}, kOp, kShape);
  ASSERT_EQ(results.size(), 2u);
  ASSERT_TRUE(results[0].second.ok()) << results[0].second.error;
  ASSERT_TRUE(results[1].second.ok()) << results[1].second.error;
  EXPECT_EQ(results[0].second.entry->best, kGfx1101Best);
  EXPECT_EQ(results[1].second.entry->best, kGfx942Best);
}

// THE pinned negative path: corrupt a cached binary (hash mismatch) -> the runtime
// INVALIDATES it and falls back to re-autotuning (the graceful miss) - it NEVER
// serves the stale/wrong binary.
TEST(kernel_cache_persist, StaleBinaryInvalidatedFallsBackToMiss) {
  const std::string dir = MakeTempDir();
  const std::string so = dir + "/gfx1101.so";
  WriteBytes(so, "binary-v1");
  std::string hash, hash_err;
  ASSERT_TRUE(HashFile(so, hash, hash_err)) << hash_err;

  KernelCache writer;
  ASSERT_TRUE(writer.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(writer.PutBinary("gfx1101", kOp, kShape, CachedBinary{so, hash}));
  std::string error;
  ASSERT_TRUE(writer.Persist(dir, error)) << error;

  KernelCache cache; // a fresh run reloading the persisted cache.
  ASSERT_TRUE(cache.LoadPersisted(dir, error)) << error;
  ASSERT_TRUE(cache.Select("gfx1101", kOp, kShape).ok()); // sane before corruption.

  WriteBytes(so, "binary-v2-CORRUPTED"); // the on-disk binary changed under us.
  EXPECT_EQ(cache.InvalidateStale(), 1u); // stale (hash mismatch) -> invalidated.

  // The entry is GONE: selection falls back to the byte-identical re-autotune miss.
  const SelectionResult res = cache.Select("gfx1101", kOp, kShape);
  EXPECT_FALSE(res.ok());
  EXPECT_FALSE(res.entry.has_value()); // never a stale/wrong kernel.
  EXPECT_EQ(res.error,
            "no tuned kernel for gfx1101/fused_matmul_bias_gelu/"
            "M=2048,N=4096,K=11008,bf16; run the autotuner "
            "(polykernel-bench --autotune) to populate the cache");
  EXPECT_FALSE(cache.GetBinary("gfx1101", kOp, kShape).has_value());
  EXPECT_EQ(cache.Size(), 0u);
}

// A missing/unreadable cached binary (the .so was deleted) is also stale: it is
// invalidated and falls back to the graceful miss, never served.
TEST(kernel_cache_persist, MissingBinaryInvalidated) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  const std::string ghost = MakeTempDir() + "/deleted.so"; // never created.
  ASSERT_TRUE(
      cache.PutBinary("gfx1101", kOp, kShape, CachedBinary{ghost, "deadbeef"}));

  EXPECT_EQ(cache.InvalidateStale(), 1u); // unreadable .so => stale.
  EXPECT_FALSE(cache.Select("gfx1101", kOp, kShape).ok());
  EXPECT_EQ(cache.Size(), 0u);
}

// A fresh binary (on-disk hash matches the recorded kernel-hash) is NOT invalidated:
// the entry survives and still selects. Invalidation is precise, not sweeping.
TEST(kernel_cache_persist, FreshBinaryNotInvalidated) {
  const std::string dir = MakeTempDir();
  const std::string so = dir + "/gfx1101.so";
  WriteBytes(so, "binary-v1");
  std::string hash, hash_err;
  ASSERT_TRUE(HashFile(so, hash, hash_err)) << hash_err;

  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(cache.PutBinary("gfx1101", kOp, kShape, CachedBinary{so, hash}));

  EXPECT_EQ(cache.InvalidateStale(), 0u); // hash matches => fresh.
  const SelectionResult res = cache.Select("gfx1101", kOp, kShape);
  ASSERT_TRUE(res.ok()) << res.error;
  EXPECT_EQ(res.entry->best, kGfx1101Best);
  EXPECT_EQ(cache.Size(), 1u);
}

// Parse-don't-validate: a malformed persisted binaries.json (missing so_path) is
// rejected with a field-path error, and the cache is left UNCHANGED (no partial load).
TEST(kernel_cache_persist, LoadPersistedRejectsMalformedBinaries) {
  const std::string dir = MakeTempDir();
  KernelCache writer;
  ASSERT_TRUE(writer.Put(Entry("gfx1101", kGfx1101Best)));
  std::string error;
  ASSERT_TRUE(writer.Persist(dir, error)) << error;

  // Overwrite binaries.json with a record missing the required so_path field.
  WriteBytes(dir + "/binaries.json",
             R"({"version": 1, "binaries": [{"gpu": "gfx1101", "op": "x",
                 "shape": {"M": 1, "N": 1, "K": 1, "dtype": "bf16"},
                 "kernel_hash": "abcd"}]})");

  KernelCache reader;
  ASSERT_TRUE(reader.Put(Entry("gfx942", kGfx942Best))); // pre-existing content.
  EXPECT_FALSE(reader.LoadPersisted(dir, error));
  EXPECT_NE(error.find("binaries[0].so_path"), std::string::npos) << error;
  EXPECT_NE(error.find("missing required string field"), std::string::npos) << error;
  // Unchanged on failure: the pre-existing gfx942 entry survives, no partial load.
  EXPECT_EQ(reader.Size(), 1u);
  EXPECT_TRUE(reader.Select("gfx942", kOp, kShape).ok());
}

// HashFile on a missing file is a clear error (the stale-detection primitive never
// silently hashes a non-existent binary).
TEST(kernel_cache_persist, HashFileMissingFileIsError) {
  std::string hash, error;
  EXPECT_FALSE(HashFile(MakeTempDir() + "/nope.so", hash, error));
  EXPECT_NE(error.find("cannot read binary"), std::string::npos) << error;
}

} // namespace
