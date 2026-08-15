//===- Roofline.cpp - GEMM roofline model -----------------------*- C++ -*-===//
//
// GEMM roofline: bytes = (M*K + K*N + M*N)*dtype_bytes, flops = 2*M*N*K,
// AI = flops/bytes, ridge = peak_flop_s / peak_byte_s. compute-bound iff the
// arithmetic intensity reaches the ridge point, else memory-bound.
//
//===----------------------------------------------------------------------===//

#include "Roofline.h"

namespace polykernel::analysis {

std::string_view RooflineName(RooflineBound bound) {
  switch (bound) {
  case RooflineBound::compute_bound:
    return "compute-bound";
  case RooflineBound::memory_bound:
    return "memory-bound";
  }
  return "unknown"; // unreachable: RooflineBound is exhaustively matched above.
}

ArchPerf PerfFor(Arch arch) {
  switch (arch) {
  case Arch::sm_80: // A100: ~312 TFLOPS fp16, ~1.555 TB/s HBM2e.
    return ArchPerf{312.0, 1555.0};
  case Arch::sm_89: // RTX 6000 Ada: 960 GB/s (datasheet). Peak FLOPs CONVENTION
    // (load-bearing for the roofline ridge + report projections): per the CUDA
    // C++ BPG "native arithmetic throughput" table, sm_89 bf16 = 128 results/
    // clk/SM = 1x FP32 = 91.1 TFLOPS - the ceiling for THIS project's SCALAR
    // kernels (their FMAs run on the FP32 pipes). fp16 alone = 2x FP32 ~= 182
    // (packed half2, NOT the bf16 rate); tensor-core dense FP16 = 364.25 (mma
    // kernels only). The H100/A100 entries above (989/312) are TENSOR-core
    // dense FP16, NOT scalar rates - do not compare them to 91.1 directly. The
    // datasheet's 1457 is FP8-with-sparsity = 4x FP16-dense; never use it raw.
    return ArchPerf{91.1, 960.0};
  case Arch::sm_90: // H100: ~989 TFLOPS fp16, ~3.35 TB/s HBM3.
    return ArchPerf{989.0, 3350.0};
  }
  return ArchPerf{}; // unreachable: Arch is exhaustively matched above.
}

Roofline ComputeRoofline(const GemmShape &shape, Arch arch) {
  const std::int64_t m = shape.m;
  const std::int64_t n = shape.n;
  const std::int64_t k = shape.k;
  const std::int64_t d = shape.dtype_bytes;

  Roofline r;
  r.bytes = (m * k + k * n + m * n) * d; // A read + B read + C write.
  r.flops = 2 * m * n * k;               // one mul + one add per output per K.

  const ArchPerf perf = PerfFor(arch);
  if (r.bytes > 0)
    r.arithmetic_intensity_flop_per_byte =
        static_cast<double>(r.flops) / static_cast<double>(r.bytes);

  // ridge = peak_flop_s / peak_byte_s = (tflops*1e12) / (gbps*1e9).
  const double ridge =
      (perf.peak_tflops * 1e12) / (perf.peak_bw_gbps * 1e9);
  r.bound = (r.arithmetic_intensity_flop_per_byte >= ridge)
                ? RooflineBound::compute_bound
                : RooflineBound::memory_bound;
  return r;
}

} // namespace polykernel::analysis
