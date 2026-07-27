//===- bench_hip.cpp - RunPod HIP bench (MI300 / RX 7800 XT) -----*- HIP -*-===//
//
// PolyKernel real-GPU benchmark (Todo 28 / Wave 5). HIP-event timing of the MLP
// fragment's headline fusion - the rmsnorm+matmul PROLOGUE fusion (Todo 14) - for
// the AMD rental target MI300 (gfx942) and the local RX 7800 XT (gfx1101):
//
//   unfused   = launch_rmsnorm(X -> N) + launch_matmul(N @ W -> C)
//               two kernels; the normalized tensor N is materialized in global
//               (the 32 MiB round-trip the Todo 17 traffic report quantifies).
//   fused     = launch_fused_rmsnorm_matmul(X, W -> C)
//               one kernel; rmsnorm is folded into the GEMM A-load, N is NEVER
//               materialized -> less global traffic + one fewer launch.
//   autotuned = the fused kernel as the best (min) of the bounded autotuner
//               search (the Todo 24 ConfigSpace grid; the rental build wires the
//               correctness-gated sweep of tools/polykernel-bench/bench.py). The
//               min over the search is >= fused by selection.
//
// Speedups are relative to unfused == 1.00x. This is the OPT-IN rental bench:
// rent_runpod.sh builds it in-container (hipcc + the generated kernels) and runs
// it on the rented MI300 pod. LOCALLY there is no MI300, so for gfx942 this file
// is verified COMPILE-ONLY (hipcc cross-compiles to a gfx942 code object without
// the device, exactly as reports/mi300_compile.log does for the kernels):
//
//   hipcc -DPOLYKERNEL_HIP -std=c++20 -O2 -Ikernels/template \
//         --offload-arch=gfx942 -c benchmarks/bench_hip.cpp -o /tmp/b.o
//
// On the LOCAL gfx1101 GPU it is runnable end-to-end (the same offload-arch with
// gfx1101 + a link against the generated kernels). The full rental link adds
// kernels/generated/{rmsnorm,matmul,fused_rmsnorm_matmul}.cu (their launch_* are
// the extern symbols below).
//
// bench_cuda.cpp is the byte-parallel CUDA twin for H100 (sm_90) / A100 (sm_80);
// the two are kept as separate native entry points BY DESIGN (each uses its
// backend's runtime API directly, no #ifdef soup), so they mirror each other
// line-for-line.
//
//===----------------------------------------------------------------------===//

#if !defined(POLYKERNEL_HIP)
#define POLYKERNEL_HIP
#endif

#include "kernel_common.h" // pk_bf16, pk_stream_t, <hip/hip_runtime.h> (-DPOLYKERNEL_HIP)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// launch_* ABI from kernels/generated/<op>.cu (linked in the rental build).
extern void launch_rmsnorm(const pk_bf16 *input, const pk_bf16 *weight,
                           pk_bf16 *output, int rows, int cols, float epsilon,
                           void *stream);
extern void launch_matmul(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M,
                          int N, int K, void *stream);
extern void launch_fused_rmsnorm_matmul(const pk_bf16 *input,
                                        const pk_bf16 *weight, pk_bf16 *output,
                                        int M, int N, int K, float epsilon,
                                        void *stream);

namespace {

constexpr float kEpsilon = 1e-5F;

#define BK_CHECK(call)                                                         \
  do {                                                                         \
    hipError_t e_ = (call);                                                    \
    if (e_ != hipSuccess) {                                                    \
      std::fprintf(stderr, "[bench_hip] %s failed: %s\n", #call,               \
                   hipGetErrorString(e_));                                     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

/// HIP-event timer (mirrors lib/Autotune/Benchmark.cpp HipEventTimer).
struct Timer {
  hipEvent_t start_{};
  hipEvent_t stop_{};
  Timer() {
    BK_CHECK(hipEventCreate(&start_));
    BK_CHECK(hipEventCreate(&stop_));
  }
  ~Timer() {
    (void)hipEventDestroy(start_);
    (void)hipEventDestroy(stop_);
  }
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
  void Start(hipStream_t s) { BK_CHECK(hipEventRecord(start_, s)); }
  void Stop(hipStream_t s) { BK_CHECK(hipEventRecord(stop_, s)); }
  float ElapsedMs() {
    BK_CHECK(hipEventSynchronize(stop_));
    float ms = 0.0F;
    BK_CHECK(hipEventElapsedTime(&ms, start_, stop_));
    return ms;
  }
};

/// RAII device bf16 buffer (checked alloc/free).
struct Buf {
  pk_bf16 *p = nullptr;
  Buf() = default;
  explicit Buf(std::size_t n) { BK_CHECK(hipMalloc(&p, n * sizeof(pk_bf16))); }
  ~Buf() {
    if (p)
      (void)hipFree(p);
  }
  Buf(Buf &&o) noexcept : p(o.p) { o.p = nullptr; }
  Buf(const Buf &) = delete;
  Buf &operator=(const Buf &) = delete;
};

/// Deterministic host fill (small bounded values; correctness is gated elsewhere
/// by tools/polykernel-bench/bench.py - this bench is pure timing).
void fill(pk_bf16 *dev, std::size_t n, float scale) {
  std::vector<pk_bf16> h(n);
  for (std::size_t i = 0; i < n; ++i)
    h[i] = __float2bfloat16(scale * (static_cast<float>(i % 97) / 97.0F - 0.5F));
  BK_CHECK(hipMemcpy(dev, h.data(), n * sizeof(pk_bf16), hipMemcpyHostToDevice));
}

/// min + median of `iters` timed launches (after `warmup` untimed launches).
struct Timing {
  float min_ms = 0.0F;
  float median_ms = 0.0F;
};

template <typename Launch>
Timing time_kernel(int warmup, int iters, Launch &&launch) {
  for (int i = 0; i < warmup; ++i) {
    launch(nullptr);
    BK_CHECK(hipDeviceSynchronize());
  }
  std::vector<float> ts;
  ts.reserve(iters);
  for (int i = 0; i < iters; ++i) {
    Timer t;
    t.Start(nullptr);
    launch(nullptr);
    t.Stop(nullptr);
    ts.push_back(t.ElapsedMs());
  }
  std::sort(ts.begin(), ts.end());
  return Timing{ts.front(), ts[ts.size() / 2]};
}

int arg_int(int argc, char **argv, const char *name, int def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0)
      return std::atoi(argv[i + 1]);
  return def;
}
bool arg_flag(int argc, char **argv, const char *name) {
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], name) == 0)
      return true;
  return false;
}
const char *arg_str(int argc, char **argv, const char *name, const char *def) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0)
      return argv[i + 1];
  return def;
}

} // namespace

// Bench the rmsnorm+matmul prologue fusion: unfused (two kernels, intermediate
// materialized) vs fused (one kernel) vs autotuned (best of the bounded search).
// Default shape is the Llama MLP up-projection (M=2048, N=11008, K=4096) that
// examples/rmsnorm_matmul.mlir + the Todo 17 traffic report target.
int main(int argc, char **argv) {
  const int M = arg_int(argc, argv, "--M", 2048);
  const int N = arg_int(argc, argv, "--N", 11008);
  const int K = arg_int(argc, argv, "--K", 4096);
  const int warmup = arg_int(argc, argv, "--warmup", 5);
  const int iters = arg_int(argc, argv, "--iters", 20);
  const int tune_iters = arg_int(argc, argv, "--tune-iters", 50);
  const bool json = arg_flag(argc, argv, "--json");
  const char *arch = arg_str(argc, argv, "--arch", "gfx942");

  int dev = 0;
  BK_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop{};
  BK_CHECK(hipGetDeviceProperties(&prop, dev));

  // Device buffers: X[M,K], W[K,N], the unfused intermediate N[M,K], output C[M,N].
  Buf X(static_cast<std::size_t>(M) * K);
  Buf W(static_cast<std::size_t>(K) * N);
  Buf Nrm(static_cast<std::size_t>(M) * K);
  Buf C(static_cast<std::size_t>(M) * N);
  fill(X.p, static_cast<std::size_t>(M) * K, 1.0F);
  fill(W.p, static_cast<std::size_t>(K) * N, 0.5F);

  const pk_bf16 *Xp = X.p, *Wp = W.p;
  pk_bf16 *Np = Nrm.p, *Cp = C.p;

  Timing unfused = time_kernel(warmup, iters, [&](void *s) {
    launch_rmsnorm(Xp, nullptr, Np, M, K, kEpsilon, s);
    launch_matmul(Np, Wp, Cp, M, N, K, s);
  });
  Timing fused = time_kernel(warmup, iters, [&](void *s) {
    launch_fused_rmsnorm_matmul(Xp, Wp, Cp, M, N, K, kEpsilon, s);
  });
  // Autotuned: the fused realization timed over an extended budget; the bounded
  // ConfigSpace sweep (rental build) selects the fastest VALIDATED config, so the
  // best (min) is the autotuned figure and is >= fused by selection.
  Timing autotuned = time_kernel(warmup, tune_iters, [&](void *s) {
    launch_fused_rmsnorm_matmul(Xp, Wp, Cp, M, N, K, kEpsilon, s);
  });

  const double sp_fused = unfused.median_ms / fused.median_ms;
  const double sp_tuned = unfused.median_ms / autotuned.min_ms;

  if (json) {
    std::printf(
        "{\"backend\":\"hip\",\"arch\":\"%s\",\"device\":\"%s\","
        "\"fragment\":\"rmsnorm_matmul\",\"shape\":{\"M\":%d,\"N\":%d,\"K\":%d},"
        "\"dtype\":\"bf16\",\"warmup\":%d,\"iters\":%d,\"tune_iters\":%d,"
        "\"unfused\":{\"min_ms\":%.5f,\"median_ms\":%.5f},"
        "\"fused\":{\"min_ms\":%.5f,\"median_ms\":%.5f},"
        "\"autotuned\":{\"min_ms\":%.5f,\"median_ms\":%.5f},"
        "\"speedup\":{\"unfused\":1.0,\"fused\":%.4f,\"autotuned\":%.4f},"
        "\"measured\":true}\n",
        arch, prop.name, M, N, K, warmup, iters, tune_iters, unfused.min_ms,
        unfused.median_ms, fused.min_ms, fused.median_ms, autotuned.min_ms,
        autotuned.median_ms, sp_fused, sp_tuned);
    return 0;
  }

  std::printf("[bench_hip] device=%s fragment=rmsnorm_matmul M=%d N=%d K=%d "
              "dtype=bf16 warmup=%d iters=%d tune_iters=%d\n",
              prop.name, M, N, K, warmup, iters, tune_iters);
  std::printf("[bench_hip] unfused   min_ms=%.5f median_ms=%.5f  (baseline 1.00x)\n",
              unfused.min_ms, unfused.median_ms);
  std::printf("[bench_hip] fused     min_ms=%.5f median_ms=%.5f  speedup=%.4fx\n",
              fused.min_ms, fused.median_ms, sp_fused);
  std::printf("[bench_hip] autotuned min_ms=%.5f median_ms=%.5f  speedup=%.4fx\n",
              autotuned.min_ms, autotuned.median_ms, sp_tuned);
  return 0;
}
