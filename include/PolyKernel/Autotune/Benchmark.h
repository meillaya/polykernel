//===- Benchmark.h - Correctness-gated benchmarking core --------*- C++ -*-===//
//
// PolyKernel correctness-gated benchmarking harness core (Todo 25 / Wave 5).
//
//===----------------------------------------------------------------------===//
//
// THE CORE INVARIANT (correctness-gated benchmarking): a variant's time is
// recorded ONLY if it FIRST passes the golden correctness gate (contract C:
// cosine >= 0.999, max_rel_err <= ceiling, pcc >= 0.99). Failing variants are
// discarded + logged with validated:false and are NEVER "best". The gate, not
// the timing, decides the winner.
//
// Two layers live here:
//
//  1. PURE correctness-gate + selection logic (always compiled; GPU-free; the
//     source of truth unit-tested by lib/Autotune/tests/benchmark_test.cpp):
//       - PassesCorrectnessGate(c, ceiling): the contract-C gate decision.
//       - VariantResult / SelectBestValidated(results): pick the FASTEST variant
//         among those that passed the gate; an unvalidated variant is never
//         selected even if it would have been the fastest.
//     This layer links into the GPU-free PolyKernelAutotune static lib (it reuses
//     the contract-H `Correctness` metrics type from TuningCache.h) and builds
//     under the project's plain clang++ with NO HIP on the include path.
//
//  2. Backend-event timing + a small host driver (compiled ONLY under
//     -DPOLYKERNEL_HIP or -DPOLYKERNEL_CUDA; the driver main ONLY under
//     -DPOLYKERNEL_BENCH_DRIVER):
//       - HipEventTimer / CudaEventTimer: RAII over the backend's event-create /
//         record / synchronize / elapsed-time / destroy API (hipEvent* /
//         cudaEvent*), the latter the CUDA twin selected by the driver's `Timer`
//         alias under POLYKERNEL_CUDA.
//       - the driver main (Benchmark.cpp) reads bf16 .npy inputs, runs a named
//         kernel variant, and in `time` mode FIRST evaluates the C++ gate on the
//         caller-supplied correctness metrics and ONLY times the variant if it
//         passes - so "golden BEFORE timing" is enforced inside C++, not merely
//         by convention in the Python orchestrator.
//     The polykernel-bench Python CLI (tools/polykernel-bench/bench.py) compiles
//     this driver with hipcc (exactly as tests/kernels/test_hip_run.py builds the
//     Todo 20 hip_run launcher) or nvcc (--backend cuda, dropping the HIP layer's
//     HipRuntime.cpp and replacing its calls with the CUDA runtime API) and
//     drives it through the .npy bridge; the Python side owns the NumPy golden +
//     the correctness-metric computation, the C++ side owns the gate + the
//     backend-event timing + the contract-H cache write.
//
// The HIP/CUDA includes are fully preprocessed out unless POLYKERNEL_HIP or
// POLYKERNEL_CUDA is defined, so the GPU-free autotuner lib + its gtest never
// need the ROCm/CUDA headers.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_AUTOTUNE_BENCHMARK_H
#define POLYKERNEL_AUTOTUNE_BENCHMARK_H

#include "PolyKernel/Autotune/TuningCache.h"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

// The HIP host API header is needed only for the HipEventTimer declaration below.
// It MUST be included at file scope (never inside the polykernel::autotune
// namespace, which would pull the HIP/libstdc++ headers into the namespace). It is
// fully preprocessed out unless POLYKERNEL_HIP is defined, so the GPU-free
// autotuner lib + its gtest never need the ROCm headers on the include path.
#ifdef POLYKERNEL_HIP
#include <hip/hip_runtime_api.h>
#endif

// The CUDA host API header is the twin include for CudaEventTimer; same rule
// (file scope, preprocessed out unless POLYKERNEL_CUDA is defined).
#ifdef POLYKERNEL_CUDA
#include <cuda_runtime_api.h>
#endif

namespace polykernel::autotune {

//===----------------------------------------------------------------------===//
// Layer 1: pure correctness-gate + selection logic (GPU-free, always compiled).
//===----------------------------------------------------------------------===//

/// Pinned contract-C correctness thresholds (mirror tests/golden/metrics.py). A
/// variant passes the gate iff cosine >= 0.999 AND pcc >= 0.99 AND max_rel_err
/// <= the rel ceiling. The rel ceiling defaults to the strict single-op value
/// 1e-2; the documented large-K tiled-GEMM reduction-order class (see
/// tests/kernels/test_hip_run.py) uses a calibrated 5e-2 ceiling where an
/// isolated element lands 1 bf16 ULP off while cosine == pcc == 1.0.
inline constexpr double kGateCosineThreshold = 0.999;
inline constexpr double kGatePccThreshold = 0.99;
inline constexpr double kGateMaxRelErrStrict = 1e-2;
inline constexpr double kGateMaxRelErrLargeK = 5e-2;

/// The correctness gate: does a variant's golden comparison pass contract C?
/// `max_rel_ceiling` selects the strict (1e-2) vs calibrated large-K (5e-2)
/// max_rel_err bound; cosine + pcc always use the pinned global thresholds.
[[nodiscard]] bool PassesCorrectnessGate(const Correctness &c,
                                         double max_rel_ceiling =
                                             kGateMaxRelErrStrict);

/// One benchmarked variant: its name, the golden correctness metrics, whether it
/// PASSED the gate (validated), and - ONLY if validated - its measured time.
/// `time_ms` is std::nullopt precisely when the variant failed the gate (it was
/// never timed): a rejected variant carries validated:false + no time, encoding
/// "golden before timing" in the type.
struct VariantResult {
  std::string name;
  Correctness correctness;
  bool validated = false;
  std::optional<double> time_ms; ///< nullopt <=> failed the gate, never timed.

  [[nodiscard]] bool operator==(const VariantResult &) const = default;
};

/// Index of the FASTEST VALIDATED variant (smallest time_ms among validated), or
/// std::nullopt if no variant passed the gate. An unvalidated variant is never
/// selected, even if its (absent) time would have been the smallest - the gate,
/// not the timing, decides the winner. Deterministic tie-break: lowest index.
[[nodiscard]] std::optional<std::size_t>
SelectBestValidated(const std::vector<VariantResult> &results);

//===----------------------------------------------------------------------===//
// Layer 2: HIP-event timing (compiled only under -DPOLYKERNEL_HIP).
//===----------------------------------------------------------------------===//

#ifdef POLYKERNEL_HIP

/// RAII HIP-event timer: hipEventCreate on construction, hipEventDestroy on
/// destruction. Start/Stop record events on a stream; ElapsedMs synchronizes the
/// stop event and returns hipEventElapsedTime in milliseconds. This is the
/// timing primitive the bench wraps around a kernel launch (warmup + N iters,
/// reporting min + median for stability). Movable, not copyable (single owner of
/// the two HIP events).
class HipEventTimer {
public:
  HipEventTimer();
  ~HipEventTimer();
  HipEventTimer(HipEventTimer &&o) noexcept;
  HipEventTimer(const HipEventTimer &) = delete;
  HipEventTimer &operator=(const HipEventTimer &) = delete;

  /// Record the start event on `s` (call immediately before the launch).
  void Start(hipStream_t s);
  /// Record the stop event on `s` (call immediately after the launch).
  void Stop(hipStream_t s);
  /// hipEventSynchronize(stop) then hipEventElapsedTime(start, stop) in ms.
  [[nodiscard]] float ElapsedMs();

private:
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

#endif // POLYKERNEL_HIP

#ifdef POLYKERNEL_CUDA

/// RAII CUDA-event timer: cudaEventCreate on construction, cudaEventDestroy on
/// destruction. Start/Stop record events on a stream; ElapsedMs synchronizes the
/// stop event and returns cudaEventElapsedTime in milliseconds. The CUDA twin of
/// HipEventTimer (same shape, same semantics), selected by the bench driver's
/// `Timer` alias under POLYKERNEL_CUDA. Movable, not copyable.
class CudaEventTimer {
public:
  CudaEventTimer();
  ~CudaEventTimer();
  CudaEventTimer(CudaEventTimer &&o) noexcept;
  CudaEventTimer(const CudaEventTimer &) = delete;
  CudaEventTimer &operator=(const CudaEventTimer &) = delete;

  /// Record the start event on `s` (call immediately before the launch).
  void Start(cudaStream_t s);
  /// Record the stop event on `s` (call immediately after the launch).
  void Stop(cudaStream_t s);
  /// cudaEventSynchronize(stop) then cudaEventElapsedTime(start, stop) in ms.
  [[nodiscard]] float ElapsedMs();

private:
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

#endif // POLYKERNEL_CUDA

} // namespace polykernel::autotune

#endif // POLYKERNEL_AUTOTUNE_BENCHMARK_H
