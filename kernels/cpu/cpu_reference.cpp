//===- cpu_reference.cpp - Host CPU reference kernels -----------*- C++ -*-===//
//
// PolyKernel host-compilable CPU reference (Todo 8 / Wave 2). See
// cpu_reference.h for the rounding contract.
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace polykernel::cpu {

void launch_rmsnorm_cpu(const uint16_t *input, const uint16_t *weight,
                        uint16_t *output, int64_t rows, int64_t cols,
                        float epsilon) {
  const float inv_cols = 1.0f / static_cast<float>(cols);

  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t *xrow = input + r * cols;
    uint16_t *yrow = output + r * cols;

    // Pass 1: sum of squares in fp32 (input bf16 upcast exactly to fp32).
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < cols; ++i) {
      float x = bf16_to_float(xrow[i]);
      sum_sq += x * x;
    }

    // rms = sqrt(mean(x^2) + epsilon), fp32 (matches golden: np.sqrt(np.mean(...)+eps)).
    float rms = std::sqrt(sum_sq * inv_cols + epsilon);

    // Pass 2: normalize, optional per-feature weight, round output back to bf16.
    for (int64_t i = 0; i < cols; ++i) {
      float x = bf16_to_float(xrow[i]);
      float y = x / rms; // fp32 divide (IEEE-754 correctly rounded), as golden.
      if (weight != nullptr)
        y *= bf16_to_float(weight[i]);
      yrow[i] = float_to_bf16(y);
    }
  }
}

void launch_gelu_cpu(const uint16_t *input, uint16_t *output, int64_t n) {
  constexpr float kInvSqrt2 = 0.70710678118654752440f; // 1 / sqrt(2)
  for (int64_t i = 0; i < n; ++i) {
    float x = bf16_to_float(input[i]);
    // erf in fp64 (std::erf, correctly rounded) then rounded to fp32, matching
    // golden's math.erf; the surrounding multiply/add stay fp32 as in golden.
    float e = static_cast<float>(std::erf(static_cast<double>(x * kInvSqrt2)));
    output[i] = float_to_bf16(0.5f * x * (1.0f + e));
  }
}

void launch_silu_cpu(const uint16_t *input, uint16_t *output, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    float x = bf16_to_float(input[i]);
    output[i] = float_to_bf16(x / (1.0f + std::exp(-x)));
  }
}

void launch_matmul_cpu(const uint16_t *a, const uint16_t *b, uint16_t *c,
                       int64_t M, int64_t N, int64_t K) {
  // C[M,N] = A[M,K] @ B[K,N]; fp32 accumulation, bf16 (RNE) output. The ikj loop
  // order keeps the B row contiguous and matches golden's np.matmul math exactly.
  for (int64_t i = 0; i < M; ++i) {
    const uint16_t *arow = a + i * K;
    uint16_t *crow = c + i * N;
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        acc += bf16_to_float(arow[k]) * bf16_to_float(b[k * N + j]);
      crow[j] = float_to_bf16(acc);
    }
  }
}

void launch_fused_rmsnorm_matmul_cpu(const uint16_t *input, const uint16_t *weight,
                                     uint16_t *output, int64_t M, int64_t N,
                                     int64_t K, float epsilon) {
  // Compose the primitives exactly as golden fused_rmsnorm_matmul: rmsnorm over the
  // last axis (K) -> bf16, then matmul with `weight`. The intermediate normalized
  // tensor is bf16 (the golden rounds rmsnorm output before the matmul reads it), so
  // this is bit-identical to running the two primitive refs in sequence.
  std::vector<uint16_t> normed(static_cast<size_t>(M) * static_cast<size_t>(K));
  launch_rmsnorm_cpu(input, nullptr, normed.data(), M, K, epsilon);
  launch_matmul_cpu(normed.data(), weight, output, M, N, K);
}

void launch_fused_matmul_bias_gelu_cpu(const uint16_t *a, const uint16_t *b,
                                       const uint16_t *bias, uint16_t *output,
                                       int64_t M, int64_t N, int64_t K) {
  // Compose the primitives exactly as golden fused_matmul_bias_gelu: matmul -> bf16,
  // bias (per-column add) -> bf16, gelu -> bf16. matmul + gelu reuse the primitive
  // refs; the bias step matches golden.bias (_bf(_f32(_bf(x)) + _f32(_bf(b)))).
  std::vector<uint16_t> mm(static_cast<size_t>(M) * static_cast<size_t>(N));
  launch_matmul_cpu(a, b, mm.data(), M, N, K);
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j)
      mm[i * N + j] =
          float_to_bf16(bf16_to_float(mm[i * N + j]) + bf16_to_float(bias[j]));
  launch_gelu_cpu(mm.data(), output, M * N);
}

void launch_softmax_cpu(const uint16_t *input, uint16_t *output, int64_t rows,
                        int64_t cols) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t *xrow = input + r * cols;
    uint16_t *yrow = output + r * cols;

    // Pass 1: row max in fp32 (stable softmax subtracts it before exp).
    float row_max = bf16_to_float(xrow[0]);
    for (int64_t i = 1; i < cols; ++i)
      row_max = std::max(row_max, bf16_to_float(xrow[i]));

    // Pass 2: sum of exp(x - max) in fp32.
    float sum = 0.0f;
    for (int64_t i = 0; i < cols; ++i)
      sum += std::exp(bf16_to_float(xrow[i]) - row_max);

    // Pass 3: normalize, round output back to bf16 (RNE).
    float inv_sum = 1.0f / sum;
    for (int64_t i = 0; i < cols; ++i)
      yrow[i] = float_to_bf16(std::exp(bf16_to_float(xrow[i]) - row_max) * inv_sum);
  }
}

} // namespace polykernel::cpu
