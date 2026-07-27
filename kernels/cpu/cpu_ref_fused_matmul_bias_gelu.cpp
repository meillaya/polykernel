//===- cpu_ref_fused_matmul_bias_gelu.cpp - CPU ref fused driver -*- C++ -*-===//
//
// PolyKernel CPU reference fused MatMul+Bias+GELU driver (Todo 14 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads bf16 A.npy + B.npy +
// BIAS.npy, computes out = gelu(matmul(a, b) + bias) via
// launch_fused_matmul_bias_gelu_cpu (same math + rounding as
// tests/golden/golden.py), and writes a bf16 OUTPUT.npy.
// tests/kernels/test_cpu_ref_fused.py drives this executable and compares the
// output to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_fused_matmul_bias_gelu A.npy B.npy BIAS.npy OUTPUT.npy
//
// Shapes follow golden.fused_matmul_bias_gelu: a is [..., M, K], b is [K, N],
// bias is [N] -> out is [..., M, N]. Leading batch dims of a flatten into the row
// count M; the contraction K must agree across a/b and bias must match the output
// column count N.
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h"
#include "npy_io.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using polykernel::cpu::NpyArray;
using polykernel::cpu::read_npy_bf16;
using polykernel::cpu::write_npy_bf16;

namespace {

void usage(const char *prog) {
  std::fprintf(stderr, "usage: %s A.npy B.npy BIAS.npy OUTPUT.npy\n", prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    usage(argv[0]);
    return 2;
  }

  std::string a_path = argv[1];
  std::string b_path = argv[2];
  std::string bias_path = argv[3];
  std::string out_path = argv[4];

  try {
    NpyArray a = read_npy_bf16(a_path);
    NpyArray b = read_npy_bf16(b_path);
    NpyArray bias = read_npy_bf16(bias_path);

    if (a.shape.size() < 2) {
      std::fprintf(stderr,
                   "error: fused_matmul_bias_gelu needs a of rank >= 2 (got %zu)\n",
                   a.shape.size());
      return 1;
    }
    if (b.shape.size() != 2) {
      std::fprintf(stderr,
                   "error: fused_matmul_bias_gelu needs b of rank 2 (got %zu)\n",
                   b.shape.size());
      return 1;
    }

    int64_t K = a.shape.back();
    int64_t M = a.size() / K; // product of all leading (batch + M) dims
    int64_t Kb = b.shape[0];
    int64_t N = b.shape[1];
    if (K != Kb) {
      std::fprintf(stderr, "error: contraction mismatch: a K=%lld != b K=%lld\n",
                   static_cast<long long>(K), static_cast<long long>(Kb));
      return 1;
    }
    if (bias.size() != N) {
      std::fprintf(stderr, "error: bias length %lld != output columns %lld\n",
                   static_cast<long long>(bias.size()),
                   static_cast<long long>(N));
      return 1;
    }

    std::vector<uint16_t> out(static_cast<size_t>(M) * static_cast<size_t>(N));
    polykernel::cpu::launch_fused_matmul_bias_gelu_cpu(
        a.data.data(), b.data.data(), bias.data.data(), out.data(), M, N, K);

    // Output shape: a's leading dims (everything but K) followed by N.
    std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
    out_shape.push_back(N);
    write_npy_bf16(out_path, out_shape, out);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
