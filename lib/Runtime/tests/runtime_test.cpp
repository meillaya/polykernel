//===- runtime_test.cpp - Runtime selection-flow unit tests -----*- C++ -*-===//
//
// GPU-free tests of the production runtime flow (Todo 26): detect GPU arch ->
// select best cached kernel -> load -> serve. The device detector AND the launcher
// are MOCKED (FixedArchDetector + a recording LaunchFn), so the full flow runs with
// NO GPU. Asserts:
//   - selection follows the DETECTED arch: a gfx1101 detection picks the gfx1101
//     cache entry, a gfx942 detection picks the gfx942 entry (namespace isolation
//     end-to-end through the runtime);
//   - serve invokes the registered launcher with the SELECTED entry;
//   - a (gpu, op, shape) with NO cache entry yields the clear "no tuned kernel ...;
//     run the autotuner" error - the launcher is NOT invoked, no wrong kernel, no
//     crash (the pinned negative path);
//   - a present-but-unvalidated entry is treated as a miss (never served);
//   - the contract-H JSON tuning cache loads + selects (reusing ParseTuningCache);
//   - a missing launcher is a clear error (not a crash).
//
// Suite name is `runtime` so `ctest -R runtime` discovers it.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Runtime/Runtime.h"

#include "gtest/gtest.h"

#include <optional>
#include <string>

namespace {

using polykernel::runtime::CacheEntry;
using polykernel::runtime::FixedArchDetector;
using polykernel::runtime::KernelCache;
using polykernel::runtime::LaunchFn;
using polykernel::runtime::ResolveResult;
using polykernel::runtime::Runtime;
using polykernel::runtime::Shape;
using polykernel::autotune::Config;
using polykernel::autotune::Correctness;

const Shape kShape{2048, 4096, 11008, "bf16"};
const char *const kOp = "fused_matmul_bias_gelu";

// gfx1101 (local) and gfx942 (rental) winners DIFFER so a cross-namespace leak or
// a wrong-arch selection is observable.
const Config kGfx1101Best{16, 32, 32, 4, 2, 2, 2};
const Config kGfx942Best{128, 128, 64, 8, 8, 4, 3};

// A validated contract-H cache entry for one gpu namespace.
CacheEntry Entry(const std::string &gpu, const Config &best,
                 bool validated = true) {
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

// A recording launcher: captures the entry it was served with + the call count.
struct RecordingLauncher {
  int calls = 0;
  std::optional<CacheEntry> served_with;

  LaunchFn fn() {
    return [this](const CacheEntry &e, void *, std::string &) {
      ++calls;
      served_with = e;
      return true;
    };
  }
};

// The pinned acceptance: selection picks the cache entry for the DETECTED arch.
// A gfx1101 detection selects the gfx1101 winner; a gfx942 detection selects the
// gfx942 winner - never the sibling namespace's kernel.
TEST(runtime, SelectionPicksDetectedArch) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(cache.Put(Entry("gfx942", kGfx942Best)));

  const FixedArchDetector local("gfx1101");
  const Runtime local_rt(local, cache);
  const ResolveResult local_res = local_rt.Resolve(kOp, kShape);
  ASSERT_TRUE(local_res.ok()) << local_res.error;
  ASSERT_TRUE(local_res.device.has_value());
  EXPECT_EQ(local_res.device->arch, "gfx1101");
  ASSERT_TRUE(local_res.entry.has_value());
  EXPECT_EQ(local_res.entry->gpu, "gfx1101");
  EXPECT_EQ(local_res.entry->best, kGfx1101Best);

  const FixedArchDetector rental("gfx942");
  const Runtime rental_rt(rental, cache);
  const ResolveResult rental_res = rental_rt.Resolve(kOp, kShape);
  ASSERT_TRUE(rental_res.ok()) << rental_res.error;
  ASSERT_TRUE(rental_res.entry.has_value());
  EXPECT_EQ(rental_res.entry->gpu, "gfx942");
  EXPECT_EQ(rental_res.entry->best, kGfx942Best);
}

// Serve: Run invokes the registered launcher with the SELECTED (detected-arch)
// entry - the load + serve step delivers the right compiled variant.
TEST(runtime, RunServesSelectedVariant) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  ASSERT_TRUE(cache.Put(Entry("gfx942", kGfx942Best)));

  RecordingLauncher launcher;
  const FixedArchDetector det("gfx1101");
  Runtime rt(det, cache);
  rt.RegisterLauncher(kOp, launcher.fn());

  std::string error;
  int io = 0; // opaque I/O bundle (the recording launcher ignores it).
  ASSERT_TRUE(rt.Run(kOp, kShape, &io, error)) << error;
  EXPECT_EQ(launcher.calls, 1);
  ASSERT_TRUE(launcher.served_with.has_value());
  EXPECT_EQ(launcher.served_with->gpu, "gfx1101");
  EXPECT_EQ(launcher.served_with->best, kGfx1101Best); // the detected-arch winner.
}

// The pinned negative path: a (gpu, op, shape) with NO cache entry returns the
// clear "no tuned kernel ...; run the autotuner" error. The launcher is NOT
// invoked (no wrong kernel), and nothing crashes.
TEST(runtime, MissReturnsClearErrorAndDoesNotServe) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best))); // gfx1101 only.

  RecordingLauncher launcher;
  const FixedArchDetector det("gfx900"); // an arch with NO cache entry.
  Runtime rt(det, cache);
  rt.RegisterLauncher(kOp, launcher.fn());

  // Resolve surfaces the miss as a clear typed error.
  const ResolveResult res = rt.Resolve(kOp, kShape);
  EXPECT_FALSE(res.ok());
  EXPECT_FALSE(res.entry.has_value());
  EXPECT_NE(res.error.find("no tuned kernel"), std::string::npos) << res.error;
  EXPECT_NE(res.error.find("gfx900"), std::string::npos) << res.error;
  EXPECT_NE(res.error.find("run the autotuner"), std::string::npos) << res.error;

  // Run propagates the same error, does NOT invoke the launcher, and does not crash.
  std::string error;
  int io = 0;
  EXPECT_FALSE(rt.Run(kOp, kShape, &io, error));
  EXPECT_NE(error.find("no tuned kernel"), std::string::npos) << error;
  EXPECT_NE(error.find("run the autotuner"), std::string::npos) << error;
  EXPECT_EQ(launcher.calls, 0); // never served a wrong kernel.
}

// A miss on an untuned SHAPE (same detected arch) is also a clear miss.
TEST(runtime, MissOnUntunedShape) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best))); // shape kShape only.

  const FixedArchDetector det("gfx1101");
  const Runtime rt(det, cache);
  const Shape untuned{128, 128, 128, "bf16"};
  const ResolveResult res = rt.Resolve(kOp, untuned);
  EXPECT_FALSE(res.ok());
  EXPECT_NE(res.error.find("no tuned kernel"), std::string::npos) << res.error;
  EXPECT_NE(res.error.find("M=128,N=128,K=128"), std::string::npos) << res.error;
}

// A present-but-unvalidated entry is treated as a miss - a bench-rejected variant
// is never served.
TEST(runtime, UnvalidatedEntryIsNotServed) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best, /*validated=*/false)));

  RecordingLauncher launcher;
  const FixedArchDetector det("gfx1101");
  Runtime rt(det, cache);
  rt.RegisterLauncher(kOp, launcher.fn());

  const ResolveResult res = rt.Resolve(kOp, kShape);
  EXPECT_FALSE(res.ok());
  EXPECT_NE(res.error.find("not validated"), std::string::npos) << res.error;
  EXPECT_NE(res.error.find("run the autotuner"), std::string::npos) << res.error;

  std::string error;
  int io = 0;
  EXPECT_FALSE(rt.Run(kOp, kShape, &io, error));
  EXPECT_EQ(launcher.calls, 0);
}

// The contract-H JSON tuning cache loads (reusing ParseTuningCache via the
// AmdTuningDb) and selects the detected-arch entry end-to-end.
TEST(runtime, LoadTuningCacheJsonThenSelect) {
  const std::string json = R"({
    "version": 1,
    "entries": [{
      "gpu": "gfx1101", "op": "fused_matmul_bias_gelu",
      "shape": {"M": 2048, "N": 4096, "K": 11008, "dtype": "bf16"},
      "best": {"block_m": 16, "block_n": 32, "block_k": 32, "num_warps": 4,
               "vector_width": 2, "unroll": 2, "shared_memory_stages": 2},
      "scored_by": "measure", "time_ms": 71.33, "validated": true,
      "correctness": {"cosine": 1.0, "max_rel_err": 1e-4, "pcc": 1.0}
    }]
  })";
  KernelCache cache;
  std::string load_error;
  ASSERT_TRUE(cache.LoadTuningCache(json, load_error)) << load_error;
  EXPECT_EQ(cache.Size(), 1u);

  const FixedArchDetector det("gfx1101");
  const Runtime rt(det, cache);
  const ResolveResult res = rt.Resolve(kOp, kShape);
  ASSERT_TRUE(res.ok()) << res.error;
  ASSERT_TRUE(res.entry.has_value());
  EXPECT_EQ(res.entry->gpu, "gfx1101");
  EXPECT_EQ(res.entry->best, kGfx1101Best);
  EXPECT_TRUE(res.entry->validated);
}

// A malformed tuning cache is rejected with a schema error and leaves the cache
// unchanged (parse-don't-validate; no partial load).
TEST(runtime, LoadTuningCacheRejectsMalformed) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));
  std::string error;
  // Missing required `scored_by` field -> contract-H schema error.
  EXPECT_FALSE(cache.LoadTuningCache(
      R"({"version": 1, "entries": [{"gpu": "gfx1101", "op": "x",
          "shape": {"M": 1, "N": 1, "K": 1, "dtype": "bf16"},
          "best": {"block_m": 16, "block_n": 16, "block_k": 32, "num_warps": 4,
                   "vector_width": 1, "unroll": 1, "shared_memory_stages": 2},
          "time_ms": null, "validated": false,
          "correctness": {"cosine": 1.0, "max_rel_err": 0.0, "pcc": 1.0}}]})",
      error));
  EXPECT_NE(error.find("scored_by"), std::string::npos) << error;
  EXPECT_EQ(cache.Size(), 1u); // unchanged on a failed load.
}

// Resolve ok but no launcher registered for the op -> clear error, not a crash.
TEST(runtime, MissingLauncherIsClearError) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));

  const FixedArchDetector det("gfx1101");
  Runtime rt(det, cache); // no launcher registered.

  std::string error;
  int io = 0;
  EXPECT_FALSE(rt.Run(kOp, kShape, &io, error));
  EXPECT_NE(error.find("no compiled launcher"), std::string::npos) << error;
}

// A launch failure (the launcher returns false) propagates its error from Run.
TEST(runtime, LaunchFailurePropagates) {
  KernelCache cache;
  ASSERT_TRUE(cache.Put(Entry("gfx1101", kGfx1101Best)));

  const FixedArchDetector det("gfx1101");
  Runtime rt(det, cache);
  rt.RegisterLauncher(kOp, [](const CacheEntry &, void *, std::string &error) {
    error = "hip launch failed: invalid configuration argument";
    return false;
  });

  std::string error;
  int io = 0;
  EXPECT_FALSE(rt.Run(kOp, kShape, &io, error));
  EXPECT_NE(error.find("hip launch failed"), std::string::npos) << error;
}

} // namespace
