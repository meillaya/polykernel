//===- Roofline.h - GEMM roofline model -------------------------*- C++ -*-===//
//
// PolyKernel compile-time analyzer (Todo 11 / Wave 2). GPU-FREE roofline model
// for a GEMM C[M,N] = A[M,K] @ B[K,N]:
//   bytes = (M*K + K*N + M*N) * dtype_bytes   (A read + B read + C write)
//   flops = 2 * M * N * K
//   arithmetic_intensity = flops / bytes       (flop per byte)
//   ridge point = peak_flop_s / peak_byte_s    (arch-specific)
//   roofline = compute-bound iff AI >= ridge else memory-bound.
// Arch peak/bandwidth defaults: H100 ~989 TFLOPS fp16 / 3.35 TB/s (ridge ~295),
// A100 ~312 TFLOPS / 1.555 TB/s (ridge ~200).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_ANALYSIS_ROOFLINE_H
#define POLYKERNEL_ANALYSIS_ROOFLINE_H

#include "Occupancy.h" // for Arch

#include <cstdint>
#include <string_view>

namespace polykernel::analysis {

/// Roofline classification. Enum string values pinned by contract H.
enum class RooflineBound { compute_bound, memory_bound };

[[nodiscard]] std::string_view RooflineName(RooflineBound bound);

/// Peak compute + memory bandwidth used for the ridge point. peak_tflops is
/// TFLOP/s (1e12 flop/s); peak_bw_gbps is GB/s (1e9 byte/s).
struct ArchPerf {
  double peak_tflops = 0.0;
  double peak_bw_gbps = 0.0;
};

[[nodiscard]] ArchPerf PerfFor(Arch arch);

/// GEMM problem shape + element size. dtype_bytes is the on-device element
/// width (bf16/fp16 = 2, fp32 = 4).
struct GemmShape {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  int dtype_bytes = 2;
};

/// Roofline outcome. bytes/flops are exact integer counts; the intensity +
/// classification derive from them and the arch ridge point.
struct Roofline {
  std::int64_t bytes = 0;
  std::int64_t flops = 0;
  double arithmetic_intensity_flop_per_byte = 0.0;
  RooflineBound bound = RooflineBound::memory_bound;
};

/// Compute the GEMM roofline for `shape` on `arch`.
[[nodiscard]] Roofline ComputeRoofline(const GemmShape &shape, Arch arch);

} // namespace polykernel::analysis

#endif // POLYKERNEL_ANALYSIS_ROOFLINE_H
