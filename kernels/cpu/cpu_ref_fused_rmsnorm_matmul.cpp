//===- cpu_ref_fused_rmsnorm_matmul.cpp - CPU ref fused driver ---*- C++ -*-===//
//
// PolyKernel CPU reference fused RMSNorm+MatMul driver (Todo 14 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads bf16 INPUT.npy (x) +
// WEIGHT.npy (the matmul weight w), computes out = matmul(rmsnorm(x, eps), w) via
// launch_fused_rmsnorm_matmul_cpu (same math + rounding as tests/golden/golden.py),
// and writes a bf16 OUTPUT.npy. tests/kernels/test_cpu_ref_fused.py drives this
// executable and compares the output to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_fused_rmsnorm_matmul INPUT.npy WEIGHT.npy OUTPUT.npy [--epsilon E]
//
// Shapes follow golden.fused_rmsnorm_matmul: x is [..., M, K] (rmsnorm over the
// last axis K), w is [K, N] -> out is [..., M, N]. Leading batch dims of x flatten
// into the row count M; the contraction K must agree across x/w.
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

constexpr float kDefaultEpsilon = 1e-5f;

void usage(const char *prog) {
  std::fprintf(stderr, "usage: %s INPUT.npy WEIGHT.npy OUTPUT.npy [--epsilon E]\n",
               prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    usage(argv[0]);
    return 2;
  }

  std::string input_path = argv[1];
  std::string weight_path = argv[2];
  std::string output_path = argv[3];
  float epsilon = kDefaultEpsilon;

  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--epsilon" && i + 1 < argc) {
      epsilon = std::stof(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  try {
    NpyArray x = read_npy_bf16(input_path);
    NpyArray w = read_npy_bf16(weight_path);

    if (x.shape.size() < 2) {
      std::fprintf(stderr,
                   "error: fused_rmsnorm_matmul needs x of rank >= 2 (got %zu)\n",
                   x.shape.size());
      return 1;
    }
    if (w.shape.size() != 2) {
      std::fprintf(stderr,
                   "error: fused_rmsnorm_matmul needs w of rank 2 (got %zu)\n",
                   w.shape.size());
      return 1;
    }

    int64_t K = x.shape.back(); // rmsnorm axis == matmul contraction
    int64_t M = x.size() / K;   // product of all leading (batch + M) dims
    int64_t Kw = w.shape[0];
    int64_t N = w.shape[1];
    if (K != Kw) {
      std::fprintf(stderr, "error: contraction mismatch: x K=%lld != w K=%lld\n",
                   static_cast<long long>(K), static_cast<long long>(Kw));
      return 1;
    }

    std::vector<uint16_t> out(static_cast<size_t>(M) * static_cast<size_t>(N));
    polykernel::cpu::launch_fused_rmsnorm_matmul_cpu(x.data.data(), w.data.data(),
                                                     out.data(), M, N, K, epsilon);

    // Output shape: x's leading dims (everything but K) followed by N.
    std::vector<int64_t> out_shape(x.shape.begin(), x.shape.end() - 1);
    out_shape.push_back(N);
    write_npy_bf16(output_path, out_shape, out);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
