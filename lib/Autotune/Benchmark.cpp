//===- Benchmark.cpp - Correctness-gated benchmarking core ------*- C++ -*-===//
//
// PolyKernel correctness-gated benchmarking harness core (Todo 25 / Wave 5).
// See include/PolyKernel/Autotune/Benchmark.h for the design + the core
// invariant (golden BEFORE timing; a variant is timed/recorded ONLY if it first
// passes the contract-C gate; failing variants are validated:false, never best).
//
// allow: SIZE_OK - this translation unit deliberately bundles three cohesive
// pieces of ONE benchmarking core: (1) the GPU-free correctness-gate + selection
// logic (the gtest-tested source of truth), (2) the backend-event timer
// (HipEventTimer / CudaEventTimer twins, Layer 2), and (3) the hipcc- or
// nvcc-built launch+timing driver. The driver is a per-op .npy launch bridge -
// the same inherently single-responsibility shape as the accepted Todo 20
// lib/Runtime/hip_run_main.cpp (337 LOC) - and is compiled ONLY under
// -DPOLYKERNEL_BENCH_DRIVER (so the GPU-free autotuner lib builds just (1)+(2)).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Autotune/Benchmark.h"

#include <cstddef>
#include <optional>
#include <vector>

namespace polykernel::autotune {

//===----------------------------------------------------------------------===//
// Layer 1: pure correctness-gate + selection logic (GPU-free, always compiled).
//===----------------------------------------------------------------------===//

bool PassesCorrectnessGate(const Correctness &c, double max_rel_ceiling) {
  // Contract C: ALL THREE metrics must hold. cosine + pcc use the pinned global
  // thresholds; max_rel_err uses the (strict or calibrated large-K) ceiling.
  return c.cosine >= kGateCosineThreshold && c.pcc >= kGatePccThreshold &&
         c.max_rel_err <= max_rel_ceiling;
}

std::optional<std::size_t>
SelectBestValidated(const std::vector<VariantResult> &results) {
  std::optional<std::size_t> best;
  for (std::size_t i = 0; i < results.size(); ++i) {
    // Only a validated variant WITH a recorded time is a candidate: an
    // unvalidated variant (gate failed => time_ms == nullopt) is never selected,
    // regardless of speed. The gate, not the timing, decides the winner.
    if (!results[i].validated || !results[i].time_ms.has_value())
      continue;
    if (!best.has_value() ||
        *results[i].time_ms < *results[*best].time_ms)
      best = i; // strict < gives the deterministic lowest-index tie-break.
  }
  return best;
}

} // namespace polykernel::autotune

//===----------------------------------------------------------------------===//
// Layer 2: backend-event timing + launch driver (only under -DPOLYKERNEL_HIP or
// -DPOLYKERNEL_CUDA). The HIP + CUDA twins are mutually exclusive via the same
// backend macro the generated kernels use.
//===----------------------------------------------------------------------===//

#ifdef POLYKERNEL_HIP

#include "PolyKernel/Runtime/HipRuntime.h"

#include <hip/hip_runtime_api.h>

namespace polykernel::autotune {

HipEventTimer::HipEventTimer() {
  PK_HIP_CHECK(hipEventCreate(&start_));
  PK_HIP_CHECK(hipEventCreate(&stop_));
}

HipEventTimer::~HipEventTimer() {
  if (start_)
    (void)hipEventDestroy(start_);
  if (stop_)
    (void)hipEventDestroy(stop_);
}

HipEventTimer::HipEventTimer(HipEventTimer &&o) noexcept
    : start_(o.start_), stop_(o.stop_) {
  o.start_ = nullptr;
  o.stop_ = nullptr;
}

void HipEventTimer::Start(hipStream_t s) { PK_HIP_CHECK(hipEventRecord(start_, s)); }
void HipEventTimer::Stop(hipStream_t s) { PK_HIP_CHECK(hipEventRecord(stop_, s)); }

float HipEventTimer::ElapsedMs() {
  PK_HIP_CHECK(hipEventSynchronize(stop_));
  float ms = 0.0F;
  PK_HIP_CHECK(hipEventElapsedTime(&ms, start_, stop_));
  return ms;
}

} // namespace polykernel::autotune

#endif // POLYKERNEL_HIP

#ifdef POLYKERNEL_CUDA

#include <cstdio>
#include <cstdlib>

// Abort-on-failure CUDA runtime check (the CUDA twin of PK_HIP_CHECK). The
// portable kernel_common.h PK_CHECK only prints, so the host timer + driver
// need their own abort-style guard - exactly like todo 7's cuda_run_main.cpp.
#define PK_CUDA_CHECK(call)                                                     \
  do {                                                                          \
    cudaError_t pk_cuda_err__ = (call);                                         \
    if (pk_cuda_err__ != cudaSuccess) {                                         \
      std::fprintf(stderr, "[polykernel-cuda] %s failed at %s:%d: %s\n", #call, \
                   __FILE__, __LINE__, cudaGetErrorString(pk_cuda_err__));      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

namespace polykernel::autotune {

CudaEventTimer::CudaEventTimer() {
  PK_CUDA_CHECK(cudaEventCreate(&start_));
  PK_CUDA_CHECK(cudaEventCreate(&stop_));
}

CudaEventTimer::~CudaEventTimer() {
  if (start_)
    (void)cudaEventDestroy(start_);
  if (stop_)
    (void)cudaEventDestroy(stop_);
}

CudaEventTimer::CudaEventTimer(CudaEventTimer &&o) noexcept
    : start_(o.start_), stop_(o.stop_) {
  o.start_ = nullptr;
  o.stop_ = nullptr;
}

void CudaEventTimer::Start(cudaStream_t s) {
  PK_CUDA_CHECK(cudaEventRecord(start_, s));
}
void CudaEventTimer::Stop(cudaStream_t s) {
  PK_CUDA_CHECK(cudaEventRecord(stop_, s));
}

float CudaEventTimer::ElapsedMs() {
  PK_CUDA_CHECK(cudaEventSynchronize(stop_));
  float ms = 0.0F;
  PK_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
  return ms;
}

} // namespace polykernel::autotune

#endif // POLYKERNEL_CUDA

//===----------------------------------------------------------------------===//
// Layer 3: the polykernel-bench host driver (only under -DPOLYKERNEL_BENCH_DRIVER,
// which the Python CLI compiles with hipcc or nvcc together with the generated
// kernels + the .npy bridge + (HIP only) HipRuntime). Modes:
//   run        <op> <variant> <inputs...> --out OUT.npy
//              launch the variant ONCE, sync, write the bf16 .npy output (the
//              Python side compares it to the NumPy golden for correctness).
//   time       <op> <variant> <inputs...> --cosine C --rel R --pcc P
//              --ceiling X --warmup W --iters N
//              FIRST evaluate the C++ gate on the caller-supplied metrics; ONLY
//              if it passes, warm up + time N launches with backend events and
//              print "VALIDATED min_ms=.. median_ms=..". A failed gate prints
//              "REJECTED .." and exits non-zero WITHOUT timing (golden before
//              timing, enforced here in C++).
//   write-cache --gpu G --op O --M M --N N --K K --dtype D --config "bm bn bk nw
//              vw un st" --scored-by S --time-ms T --cosine C --rel R --pcc P
//              --validated {true|false} --out FILE
//              build a contract-H CacheEntry and SerializeTuningCache it to FILE
//              (reusing the Todo 24 serializer; the schema is NOT redefined).
//   dev        report the backend's device count; exit 1 with a clear message on
//              no device (the bench probes this BEFORE any sweep work, so a
//              no-device machine SKIPs cleanly instead of failing mid-sweep).
//===----------------------------------------------------------------------===//

#ifdef POLYKERNEL_BENCH_DRIVER

#include "kernel_common.h" // pk_bf16, pk_stream_t (one of -DPOLYKERNEL_HIP/-DPOLYKERNEL_CUDA)
#include "npy_io.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace cpu = polykernel::cpu;
using polykernel::autotune::CacheEntry;
using polykernel::autotune::Config;
using polykernel::autotune::Correctness;
using polykernel::autotune::PassesCorrectnessGate;
using polykernel::autotune::SerializeTuningCache;
using polykernel::autotune::Shape;
using polykernel::autotune::TuningCache;

// The driver's timing primitive: CudaEventTimer under POLYKERNEL_CUDA,
// HipEventTimer otherwise (Layer 2's HIP block is excluded under CUDA, so the
// plain `using ...HipEventTimer` would not compile there).
#ifdef POLYKERNEL_CUDA
using Timer = polykernel::autotune::CudaEventTimer;
#else
using Timer = polykernel::autotune::HipEventTimer;
#endif

// Device-runtime shim: the driver's alloc/copy/sync/free surface, selected by
// the same backend macro as the kernels. HIP -> the checked HipRuntime layer
// (lib/Runtime/HipRuntime.cpp); CUDA -> the raw CUDA runtime API through the
// abort-on-failure PK_CUDA_CHECK. Every Hip* call in the driver routes through
// here, so -DPOLYKERNEL_CUDA -DPOLYKERNEL_BENCH_DRIVER compiles the same TU.
#ifdef POLYKERNEL_CUDA
inline void DeviceMalloc(void **p, std::size_t bytes) {
  PK_CUDA_CHECK(cudaMalloc(p, bytes));
}
inline void DeviceFree(void *p) { PK_CUDA_CHECK(cudaFree(p)); }
inline void DeviceCopyH2D(void *dst, const void *src, std::size_t bytes) {
  PK_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}
inline void DeviceCopyD2H(void *dst, const void *src, std::size_t bytes) {
  PK_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}
inline void DeviceSync() { PK_CUDA_CHECK(cudaDeviceSynchronize()); }
#else
inline void DeviceMalloc(void **p, std::size_t bytes) {
  polykernel::runtime::HipMalloc(p, bytes);
}
inline void DeviceFree(void *p) { polykernel::runtime::HipFree(p); }
inline void DeviceCopyH2D(void *dst, const void *src, std::size_t bytes) {
  polykernel::runtime::HipCopyH2D(dst, src, bytes);
}
inline void DeviceCopyD2H(void *dst, const void *src, std::size_t bytes) {
  polykernel::runtime::HipCopyD2H(dst, src, bytes);
}
inline void DeviceSync() { polykernel::runtime::HipSync(); }
#endif

static_assert(sizeof(pk_bf16) == 2, "pk_bf16 must be 2 bytes (bf16)");

// launch_* ABI from kernels/generated/<op>.cu (default variants) + the generated
// broken variants (<op>_broken) the bench compiles in for the negative path. The
// broken symbols are always linked (the bench builds the driver with the broken
// .cu) so the dispatch below is unconditional.
extern void launch_matmul(const pk_bf16 *, const pk_bf16 *, pk_bf16 *, int, int,
                          int, void *);
extern void launch_matmul_broken(const pk_bf16 *, const pk_bf16 *, pk_bf16 *, int,
                                 int, int, void *);
extern void launch_fused_rmsnorm_matmul(const pk_bf16 *, const pk_bf16 *,
                                        pk_bf16 *, int, int, int, float, void *);
extern void launch_fused_rmsnorm_matmul_broken(const pk_bf16 *, const pk_bf16 *,
                                               pk_bf16 *, int, int, int, float,
                                               void *);
extern void launch_fused_matmul_bias_gelu(const pk_bf16 *, const pk_bf16 *,
                                          const pk_bf16 *, pk_bf16 *, int, int,
                                          int, void *);
extern void launch_fused_matmul_bias_gelu_broken(const pk_bf16 *, const pk_bf16 *,
                                                 const pk_bf16 *, pk_bf16 *, int,
                                                 int, int, void *);

namespace {

int usage(const char *msg) {
  std::fprintf(stderr, "usage: polykernel-bench-driver <run|time|enumerate|write-cache|dev> ...\n"
                       "  %s\n", msg);
  return 2;
}

/// RAII device buffer (mirrors the Todo 20 hip_run driver): checked alloc on
/// construction (bytes==0 -> null), checked free on destruction. Movable only.
struct DevBuf {
  void *p = nullptr;
  DevBuf() = default; ///< null buffer (no allocation).
  explicit DevBuf(std::size_t bytes) {
    if (bytes)
      DeviceMalloc(&p, bytes);
  }
  ~DevBuf() {
    if (p)
      DeviceFree(p);
  }
  DevBuf(DevBuf &&o) noexcept : p(o.p) { o.p = nullptr; }
  DevBuf &operator=(DevBuf &&o) noexcept {
    if (this != &o) {
      if (p)
        DeviceFree(p);
      p = o.p;
      o.p = nullptr;
    }
    return *this;
  }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
  pk_bf16 *ptr() { return static_cast<pk_bf16 *>(p); }
};

/// Upload a loaded .npy tensor's bf16 payload host->device.
DevBuf upload(const cpu::NpyArray &arr) {
  DevBuf d(arr.data.size() * sizeof(uint16_t));
  DeviceCopyH2D(d.p, arr.data.data(), arr.data.size() * sizeof(uint16_t));
  return d;
}

/// A benchmarked op instance: its device buffers + output shape + a launch
/// closure (the chosen variant on the uploaded pointers) + M/N/K identity.
struct OpInstance {
  std::vector<DevBuf> in;            ///< device input buffers (kept alive).
  DevBuf out;                        ///< device output buffer.
  std::vector<int64_t> out_shape;    ///< for the .npy download.
  int m = 0, n = 0, k = 0;
  std::function<void(void *)> launch; ///< launch the variant on stream.

  OpInstance() = default;
  OpInstance(OpInstance &&) = default;
  OpInstance &operator=(OpInstance &&) = default;
  OpInstance(const OpInstance &) = delete;
  OpInstance &operator=(const OpInstance &) = delete;
};

/// matmul-family output shape: A's leading dims (all but K) + N.
std::vector<int64_t> matmul_out_shape(const std::vector<int64_t> &a, int64_t n) {
  std::vector<int64_t> s(a.begin(), a.end() - 1);
  s.push_back(n);
  return s;
}

bool is_broken(const std::string &variant) {
  if (variant == "default" || variant == "broken")
    return variant == "broken";
  std::fprintf(stderr, "unknown variant '%s' (expected default|broken)\n",
               variant.c_str());
  std::exit(2);
}

/// Build an OpInstance for `op` from the input .npy paths, wiring the launch
/// closure to the default or broken implementation per `variant`.
OpInstance setup_op(const std::string &op, const std::string &variant,
                    const std::vector<std::string> &inputs) {
  const bool broken = is_broken(variant);
  OpInstance io;

  if (op == "matmul") {
    if (inputs.size() != 2) { std::fprintf(stderr, "matmul needs A B\n"); std::exit(2); }
    cpu::NpyArray a = cpu::read_npy_bf16(inputs[0]);
    cpu::NpyArray b = cpu::read_npy_bf16(inputs[1]);
    io.k = static_cast<int>(a.shape.back());
    io.m = static_cast<int>(a.size() / io.k);
    io.n = static_cast<int>(b.shape[1]);
    io.in.push_back(upload(a));
    io.in.push_back(upload(b));
    io.out = DevBuf(static_cast<std::size_t>(io.m) * io.n * sizeof(uint16_t));
    io.out_shape = matmul_out_shape(a.shape, io.n);
    const pk_bf16 *A = io.in[0].ptr(), *B = io.in[1].ptr();
    pk_bf16 *C = io.out.ptr();
    const int M = io.m, N = io.n, K = io.k;
    io.launch = [A, B, C, M, N, K, broken](void *s) {
      if (broken) launch_matmul_broken(A, B, C, M, N, K, s);
      else        launch_matmul(A, B, C, M, N, K, s);
    };
    return io;
  }

  if (op == "fused_rmsnorm_matmul") {
    if (inputs.size() != 2) { std::fprintf(stderr, "fused_rmsnorm_matmul needs X W\n"); std::exit(2); }
    cpu::NpyArray x = cpu::read_npy_bf16(inputs[0]);
    cpu::NpyArray w = cpu::read_npy_bf16(inputs[1]);
    io.k = static_cast<int>(x.shape.back());
    io.m = static_cast<int>(x.size() / io.k);
    io.n = static_cast<int>(w.shape[1]);
    io.in.push_back(upload(x));
    io.in.push_back(upload(w));
    io.out = DevBuf(static_cast<std::size_t>(io.m) * io.n * sizeof(uint16_t));
    io.out_shape = matmul_out_shape(x.shape, io.n);
    const pk_bf16 *X = io.in[0].ptr(), *W = io.in[1].ptr();
    pk_bf16 *O = io.out.ptr();
    const int M = io.m, N = io.n, K = io.k;
    const float eps = 1e-5F;
    io.launch = [X, W, O, M, N, K, eps, broken](void *s) {
      if (broken) launch_fused_rmsnorm_matmul_broken(X, W, O, M, N, K, eps, s);
      else        launch_fused_rmsnorm_matmul(X, W, O, M, N, K, eps, s);
    };
    return io;
  }

  if (op == "fused_matmul_bias_gelu") {
    if (inputs.size() != 3) { std::fprintf(stderr, "fused_matmul_bias_gelu needs A B BIAS\n"); std::exit(2); }
    cpu::NpyArray a = cpu::read_npy_bf16(inputs[0]);
    cpu::NpyArray b = cpu::read_npy_bf16(inputs[1]);
    cpu::NpyArray bias = cpu::read_npy_bf16(inputs[2]);
    io.k = static_cast<int>(a.shape.back());
    io.m = static_cast<int>(a.size() / io.k);
    io.n = static_cast<int>(b.shape[1]);
    io.in.push_back(upload(a));
    io.in.push_back(upload(b));
    io.in.push_back(upload(bias));
    io.out = DevBuf(static_cast<std::size_t>(io.m) * io.n * sizeof(uint16_t));
    io.out_shape = matmul_out_shape(a.shape, io.n);
    const pk_bf16 *A = io.in[0].ptr(), *B = io.in[1].ptr(), *BI = io.in[2].ptr();
    pk_bf16 *O = io.out.ptr();
    const int M = io.m, N = io.n, K = io.k;
    io.launch = [A, B, BI, O, M, N, K, broken](void *s) {
      if (broken) launch_fused_matmul_bias_gelu_broken(A, B, BI, O, M, N, K, s);
      else        launch_fused_matmul_bias_gelu(A, B, BI, O, M, N, K, s);
    };
    return io;
  }

  std::fprintf(stderr, "unsupported op '%s' (matmul-family only)\n", op.c_str());
  std::exit(2);
}

/// Download a device output buffer + write it as a '<V2' bf16 .npy.
void download_write(const DevBuf &d, const std::vector<int64_t> &shape,
                    const std::string &path) {
  int64_t n = 1;
  for (int64_t x : shape)
    n *= x;
  std::vector<uint16_t> host(static_cast<std::size_t>(n));
  DeviceCopyD2H(host.data(), d.p, host.size() * sizeof(uint16_t));
  cpu::write_npy_bf16(path, shape, host);
}

/// Collect the positional input paths + a named --opt value from argv[i0..argc).
std::vector<std::string> collect_inputs(int argc, char **argv, int i0) {
  std::vector<std::string> inputs;
  for (int i = i0; i < argc; ++i) {
    const std::string a = argv[i];
    if (a.rfind("--", 0) == 0) {
      ++i; // skip the option's value (caller validates specifics).
      continue;
    }
    inputs.push_back(a);
  }
  return inputs;
}

/// Read a "--name value" string option; `def` if absent.
std::string opt_str(int argc, char **argv, const char *name,
                    const std::string &def) {
  for (int i = 0; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0)
      return argv[i + 1];
  return def;
}
double opt_dbl(int argc, char **argv, const char *name, double def) {
  const std::string s = opt_str(argc, argv, name, "");
  return s.empty() ? def : std::stod(s);
}
int opt_int(int argc, char **argv, const char *name, int def) {
  const std::string s = opt_str(argc, argv, name, "");
  return s.empty() ? def : std::stoi(s);
}

int run_run(int argc, char **argv) {
  // run <op> <variant> <inputs...> --out OUT.npy
  if (argc < 4)
    return usage("run <op> <variant> <inputs...> --out OUT.npy");
  const std::string op = argv[2];
  const std::string variant = argv[3];
  const std::string out = opt_str(argc, argv, "--out", "");
  if (out.empty())
    return usage("run: --out OUT.npy is required");
  OpInstance io = setup_op(op, variant, collect_inputs(argc, argv, 4));
  io.launch(nullptr);
  DeviceSync();
  download_write(io.out, io.out_shape, out);
  return 0;
}

int run_time(int argc, char **argv) {
  // time <op> <variant> <inputs...> --cosine C --rel R --pcc P --ceiling X
  //      --warmup W --iters N
  if (argc < 4)
    return usage("time <op> <variant> <inputs...> --cosine --rel --pcc --ceiling "
                 "--warmup --iters");
  const std::string op = argv[2];
  const std::string variant = argv[3];

  // THE CORE INVARIANT, enforced HERE in C++: gate BEFORE any timing. A variant
  // that fails the contract-C gate is rejected + logged and NEVER timed.
  const Correctness c{opt_dbl(argc, argv, "--cosine", 0.0),
                      opt_dbl(argc, argv, "--rel", 1.0),
                      opt_dbl(argc, argv, "--pcc", 0.0)};
  const double ceiling = opt_dbl(argc, argv, "--ceiling",
                                 polykernel::autotune::kGateMaxRelErrStrict);
  if (!PassesCorrectnessGate(c, ceiling)) {
    std::fprintf(stderr,
                 "REJECTED %s/%s: correctness gate FAILED (cosine=%.6f "
                 "max_rel_err=%.6e pcc=%.6f ceiling=%.1e) - excluded, NOT timed, "
                 "never best\n",
                 op.c_str(), variant.c_str(), c.cosine, c.max_rel_err, c.pcc,
                 ceiling);
    return 3; // distinct exit code: gate-rejected (the bench maps this to
              // validated:false + a logged rejection).
  }

  const int warmup = opt_int(argc, argv, "--warmup", 3);
  const int iters = opt_int(argc, argv, "--iters", 10);
  OpInstance io = setup_op(op, variant, collect_inputs(argc, argv, 4));

  for (int i = 0; i < warmup; ++i) {
    io.launch(nullptr);
    DeviceSync();
  }

  std::vector<float> times;
  times.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    Timer t;
    t.Start(nullptr);
    io.launch(nullptr);
    t.Stop(nullptr);
    times.push_back(t.ElapsedMs());
  }
  std::sort(times.begin(), times.end());
  const float min_ms = times.front();
  const float median_ms = times[times.size() / 2];
  std::printf("VALIDATED %s/%s min_ms=%.5f median_ms=%.5f iters=%d\n",
              op.c_str(), variant.c_str(), min_ms, median_ms, iters);
  return 0;
}

int run_enumerate(int argc, char **argv) {
  // enumerate --limit N : print the first N pruned ConfigSpace configs (the
  // Todo 24 enumerator; default sm_90 ArchLimits) as "bm bn bk nw vw un st"
  // lines, preceded by a "pruned_total=<n>" header. The bench samples a bounded
  // prefix as its variant set (it does NOT compile all ~141 configs).
  const int limit = opt_int(argc, argv, "--limit", 4);
  const auto pruned = polykernel::autotune::ConfigSpace::Enumerate();
  std::printf("pruned_total=%zu\n", pruned.size());
  const int n = static_cast<int>(std::min<std::size_t>(limit, pruned.size()));
  for (int i = 0; i < n; ++i) {
    const Config &c = pruned[i];
    std::printf("%d %d %d %d %d %d %d\n", c.block_m, c.block_n, c.block_k,
                c.num_warps, c.vector_width, c.unroll, c.shared_memory_stages);
  }
  return 0;
}

int run_write_cache(int argc, char **argv) {
  // write-cache --gpu --op --M --N --K --dtype --config "bm bn bk nw vw un st"
  //             --scored-by --time-ms --cosine --rel --pcc --validated --out
  const std::string out = opt_str(argc, argv, "--out", "");
  if (out.empty())
    return usage("write-cache: --out FILE is required");

  CacheEntry e;
  e.gpu = opt_str(argc, argv, "--gpu", "gfx1101");
  e.op = opt_str(argc, argv, "--op", "matmul");
  e.shape = Shape{opt_int(argc, argv, "--M", 0), opt_int(argc, argv, "--N", 0),
                  opt_int(argc, argv, "--K", 0),
                  opt_str(argc, argv, "--dtype", "bf16")};
  // --config "bm bn bk nw vw un st" (the winning variant's representative tile).
  const std::string cfg = opt_str(argc, argv, "--config", "16 16 16 8 1 1 2");
  Config best{};
  if (std::sscanf(cfg.c_str(), "%d %d %d %d %d %d %d", &best.block_m,
                  &best.block_n, &best.block_k, &best.num_warps,
                  &best.vector_width, &best.unroll,
                  &best.shared_memory_stages) != 7)
    return usage("write-cache: --config needs 7 ints \"bm bn bk nw vw un st\"");
  e.best = best;
  e.scored_by = opt_str(argc, argv, "--scored-by", "measure");
  e.time_ms = opt_dbl(argc, argv, "--time-ms", 0.0);
  e.validated = opt_str(argc, argv, "--validated", "true") == "true";
  e.correctness = Correctness{opt_dbl(argc, argv, "--cosine", 0.0),
                              opt_dbl(argc, argv, "--rel", 0.0),
                              opt_dbl(argc, argv, "--pcc", 0.0)};

  TuningCache cache;
  cache.entries.push_back(e);
  const std::string json = SerializeTuningCache(cache); // contract H (Todo 24).

  FILE *f = std::fopen(out.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "write-cache: cannot open %s for writing\n", out.c_str());
    return 1;
  }
  std::fwrite(json.data(), 1, json.size(), f);
  std::fclose(f);
  std::printf("%s\n", json.c_str());
  return 0;
}

int run_dev() {
  // dev : report the backend's device count; exit 1 with a clear message on no
  // device. The bench probes this BEFORE any sweep work, so a no-device machine
  // SKIPs cleanly (never a wrong result / never a mid-sweep cudaMalloc crash).
#ifdef POLYKERNEL_CUDA
  int ndev = 0;
  cudaError_t err = cudaGetDeviceCount(&ndev);
  if (err != cudaSuccess || ndev == 0) {
    std::fprintf(stderr,
                 "[polykernel-bench] dev: no CUDA device (count=%d, %s)\n",
                 ndev, cudaGetErrorString(err));
    return 1;
  }
  std::printf("DEVICE_COUNT %d\n", ndev);
  return 0;
#else
  int ndev = 0;
  hipError_t err = hipGetDeviceCount(&ndev);
  if (err != hipSuccess || ndev == 0) {
    std::fprintf(stderr,
                 "[polykernel-bench] dev: no HIP device (count=%d, %s)\n",
                 ndev, polykernel::runtime::HipErrorString(err));
    return 1;
  }
  std::printf("DEVICE_COUNT %d\n", ndev);
  return 0;
#endif
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage("<run|time|enumerate|write-cache|dev> ...");
  const std::string mode = argv[1];
  try {
    if (mode == "run")
      return run_run(argc, argv);
    if (mode == "time")
      return run_time(argc, argv);
    if (mode == "enumerate")
      return run_enumerate(argc, argv);
    if (mode == "dev")
      return run_dev();
    if (mode == "write-cache")
      return run_write_cache(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[polykernel-bench] error: %s\n", e.what());
    return 1;
  }
  return usage("unknown mode");
}

#endif // POLYKERNEL_BENCH_DRIVER
