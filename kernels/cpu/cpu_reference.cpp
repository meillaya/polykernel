//===- cpu_reference.cpp - Host CPU reference kernels -----------*- C++ -*-===//
//
// PolyKernel host-compilable CPU reference (Todo 8 / Wave 2). See
// cpu_reference.h for the rounding contract.
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h"

#include <cmath>

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

} // namespace polykernel::cpu
