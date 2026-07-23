//===- cpu_reference.h - Host CPU reference kernels -------------*- C++ -*-===//
//
// PolyKernel host-compilable CPU reference (Todo 8 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// The CPU reference is the LOCAL EXECUTABLE SPEC for kernel correctness: pure
// host C++ (no CUDA/HIP) implementing the SAME math + rounding contract as
// tests/golden/golden.py. Because no NVIDIA GPU is available locally, the CPU
// reference validated against the golden (tests/kernels/test_cpu_ref.py) is the
// proof that the algorithm + rounding are correct; the CUDA/HIP kernels share
// that algorithm via the portable template (kernels/template/kernel_common.h).
//
// ROUNDING CONTRACT (must match golden.py exactly):
//   1. inputs/weights are bf16 (round-to-nearest-even);
//   2. all accumulation / reduction is performed in fp32;
//   3. each op's OUTPUT is rounded back to bf16 (RNE);
//   4. comparison is on the bf16 output upcast to fp32 (see metrics.py).
//
// bf16 is carried across the ABI as raw uint16_t bits (the IEEE-754 brain-float
// encoding: 1 sign + 8 exponent + 7 mantissa, the high 16 bits of an fp32).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_KERNELS_CPU_CPU_REFERENCE_H
#define POLYKERNEL_KERNELS_CPU_CPU_REFERENCE_H

#include <cstdint>

namespace polykernel::cpu {

//===----------------------------------------------------------------------===//
// bf16 <-> float helpers (round-to-nearest-even, matching ml_dtypes.bfloat16).
//===----------------------------------------------------------------------===//

/// Upcast a bf16 (raw bits) to fp32. Exact (bf16 is a subset of fp32).
inline float bf16_to_float(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof f);
  return f;
}

/// Round an fp32 to bf16 (raw bits), round-to-nearest-even.
///
/// The low 16 bits of the fp32 are dropped. Adding `0x7FFF + lsb` (where lsb is
/// bit 16 of the value, i.e. the LSB of the kept part) before truncating yields
/// IEEE round-to-nearest-even: values above the midpoint round up, below round
/// down, and exact ties round to an even kept-LSB. NaN is preserved as a quiet
/// NaN (ml_dtypes maps NaN -> NaN); the RNE add would otherwise turn a NaN into
/// an Inf.
inline uint16_t float_to_bf16(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof u);
  // NaN (all-ones exponent, non-zero mantissa) -> quiet NaN, preserve payload MSB.
  if ((u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u)
    return static_cast<uint16_t>((u >> 16) | 0x0040u);
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb; // round-to-nearest-even (see comment above)
  return static_cast<uint16_t>(u >> 16);
}

//===----------------------------------------------------------------------===//
// RMSNorm CPU reference.
//===----------------------------------------------------------------------===//
//
// RMSNorm over the LAST axis: for each of `rows` length-`cols` rows,
//     rms = sqrt(mean(x^2) + epsilon);  out = x / rms [* weight].
// `input`/`weight`/`output` are row-major bf16 (raw uint16 bits). `weight` is a
// length-`cols` per-feature scale; pass nullptr to omit it (matches the golden's
// `weight=None`). Accumulation is fp32; the output is rounded back to bf16.
//
// This is the host `launch_rmsnorm`-style entry point (the same ABI shape the
// CUDA/HIP kernels expose via launch_rmsnorm in the generated .cu).
void launch_rmsnorm_cpu(const uint16_t *input, const uint16_t *weight,
                        uint16_t *output, int64_t rows, int64_t cols,
                        float epsilon);

} // namespace polykernel::cpu

#endif // POLYKERNEL_KERNELS_CPU_CPU_REFERENCE_H
