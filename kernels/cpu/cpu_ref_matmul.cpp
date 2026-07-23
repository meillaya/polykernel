//===- cpu_ref_matmul.cpp - CPU reference MatMul driver ----------*- C++ -*-===//
//
// PolyKernel CPU reference MatMul driver (Todo 10 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads bf16 A.npy + B.npy,
// computes C = A @ B via launch_matmul_cpu (the same math + rounding contract as
// tests/golden/golden.py: fp32 accumulation, bf16 RNE output), and writes C.npy.
// tests/kernels/test_cpu_ref_matmul_softmax.py drives this executable and compares
// the output to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_matmul A.npy B.npy C.npy
//
// Shapes follow golden.matmul: A is [..., M, K], B is [K, N] -> C is [..., M, N].
// All leading (batch) dims of A are flattened into the row count M (matching
// np.matmul on [...,M,K] @ [K,N]); the contraction dim K must agree across A/B.
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
  std::fprintf(stderr, "usage: %s A.npy B.npy C.npy\n", prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    usage(argv[0]);
    return 2;
  }

  std::string a_path = argv[1];
  std::string b_path = argv[2];
  std::string c_path = argv[3];

  try {
    NpyArray a = read_npy_bf16(a_path);
    NpyArray b = read_npy_bf16(b_path);

    if (a.shape.size() < 2) {
      std::fprintf(stderr, "error: matmul needs A of rank >= 2 (got rank %zu)\n",
                   a.shape.size());
      return 1;
    }
    if (b.shape.size() != 2) {
      std::fprintf(stderr, "error: matmul needs B of rank 2 (got rank %zu)\n",
                   b.shape.size());
      return 1;
    }

    int64_t K = a.shape.back();
    int64_t M = a.size() / K; // product of all leading (batch + M) dims
    int64_t Kb = b.shape[0];
    int64_t N = b.shape[1];
    if (K != Kb) {
      std::fprintf(stderr,
                   "error: matmul contraction mismatch: A K=%lld != B K=%lld\n",
                   static_cast<long long>(K), static_cast<long long>(Kb));
      return 1;
    }

    std::vector<uint16_t> c(static_cast<size_t>(M) * static_cast<size_t>(N));
    polykernel::cpu::launch_matmul_cpu(a.data.data(), b.data.data(), c.data(), M,
                                       N, K);

    // Output shape: A's leading dims (everything but K) followed by N.
    std::vector<int64_t> c_shape(a.shape.begin(), a.shape.end() - 1);
    c_shape.push_back(N);
    write_npy_bf16(c_path, c_shape, c);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
