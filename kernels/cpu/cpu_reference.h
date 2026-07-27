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

//===----------------------------------------------------------------------===//
// GELU CPU reference.
//===----------------------------------------------------------------------===//
//
// Exact erf-based GELU over `n` elements:
//     y = 0.5 * x * (1 + erf(x / sqrt(2))).
// `input`/`output` are bf16 (raw uint16 bits). Compute is fp32 (erf evaluated in
// fp64 via std::erf then rounded, matching golden's math.erf); the output is
// rounded back to bf16. Shape-agnostic: the caller flattens all dims into `n`.
void launch_gelu_cpu(const uint16_t *input, uint16_t *output, int64_t n);

//===----------------------------------------------------------------------===//
// SiLU CPU reference.
//===----------------------------------------------------------------------===//
//
// SiLU / swish over `n` elements:
//     y = x / (1 + exp(-x)).
// `input`/`output` are bf16 (raw uint16 bits). Compute is fp32 (std::exp); the
// output is rounded back to bf16. Shape-agnostic: the caller flattens dims to `n`.
void launch_silu_cpu(const uint16_t *input, uint16_t *output, int64_t n);

//===----------------------------------------------------------------------===//
// MatMul CPU reference.
//===----------------------------------------------------------------------===//
//
// Tiled-baseline matrix multiply: C[M,N] = A[M,K] @ B[K,N]. `a`/`b`/`c` are
// row-major bf16 (raw uint16 bits). Accumulation is fp32 (matching golden's
// np.matmul on fp32(bf16(a)), fp32(bf16(b))); each output element is rounded
// back to bf16 (RNE). Batched leading dims are flattened into M by the caller
// (matching golden's np.matmul on [...,M,K] @ [K,N]).
void launch_matmul_cpu(const uint16_t *a, const uint16_t *b, uint16_t *c,
                       int64_t M, int64_t N, int64_t K);

//===----------------------------------------------------------------------===//
// Fused RMSNorm + MatMul CPU reference.
//===----------------------------------------------------------------------===//
//
// Fused PROLOGUE: out[M,N] = matmul(rmsnorm(input, eps), weight). Composes the
// primitive references EXACTLY as the golden (fused_rmsnorm_matmul = matmul(
// rmsnorm(x, None, eps), w)): rmsnorm over the last axis of `input` (the matmul
// contraction K), then matmul with `weight` [K,N]. `input` is [M,K] row-major bf16
// (leading batch dims flattened into M by the caller), `weight` is [K,N], `output`
// is [M,N]. The rmsnorm has no per-feature scale (golden weight=None). Each stage
// rounds to bf16 as the golden does, so this is bit-identical to running the two
// primitive refs in sequence.
void launch_fused_rmsnorm_matmul_cpu(const uint16_t *input, const uint16_t *weight,
                                     uint16_t *output, int64_t M, int64_t N,
                                     int64_t K, float epsilon);

//===----------------------------------------------------------------------===//
// Fused MatMul + Bias + GELU CPU reference.
//===----------------------------------------------------------------------===//
//
// Fused EPILOGUE: out[M,N] = gelu(bias(matmul(a, b), bias)). Composes the primitive
// references EXACTLY as the golden (fused_matmul_bias_gelu = gelu(bias(matmul(x,w),
// b))): matmul a[M,K] @ b[K,N], add the per-column bias[N], then exact erf GELU.
// `a`/`b`/`bias`/`output` are bf16 (raw uint16 bits); `a` is [M,K] (leading batch
// dims flattened into M), `b` is [K,N], `bias` is [N], `output` is [M,N]. Each stage
// rounds to bf16 as the golden does, so this is bit-identical to running matmul ->
// bias -> gelu in sequence.
void launch_fused_matmul_bias_gelu_cpu(const uint16_t *a, const uint16_t *b,
                                       const uint16_t *bias, uint16_t *output,
                                       int64_t M, int64_t N, int64_t K);

//===----------------------------------------------------------------------===//
// Softmax CPU reference.
//===----------------------------------------------------------------------===//
//
// Numerically stable softmax over the LAST axis: for each of `rows` length-`cols`
// rows, subtract the row max before exp, then normalize:
//     m = max(x);  y = exp(x - m) / sum(exp(x - m)).
// `input`/`output` are row-major bf16 (raw uint16 bits). Compute is fp32
// (std::exp); the output is rounded back to bf16 (RNE). The caller flattens all
// leading dims into `rows` (matching golden's softmax(axis=-1)).
void launch_softmax_cpu(const uint16_t *input, uint16_t *output, int64_t rows,
                        int64_t cols);

} // namespace polykernel::cpu

#endif // POLYKERNEL_KERNELS_CPU_CPU_REFERENCE_H
