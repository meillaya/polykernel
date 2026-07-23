//===- cpu_ref_softmax.cpp - CPU reference Softmax driver --------*- C++ -*-===//
//
// PolyKernel CPU reference Softmax driver (Todo 10 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads a bf16 input .npy,
// computes a numerically stable softmax via launch_softmax_cpu (the same math +
// rounding contract as tests/golden/golden.py: subtract the row max before exp,
// fp32 compute, bf16 RNE output), and writes a bf16 output .npy.
// tests/kernels/test_cpu_ref_matmul_softmax.py drives this executable and compares
// the output to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_softmax INPUT.npy OUTPUT.npy [--axis N]
//
// The reduction axis defaults to the LAST dimension (golden's axis=-1); all
// leading dims are flattened into the row count (rows = product(shape[:-1]),
// cols = shape[-1]). The Wave-2 baseline reduces the last axis only — a non-last
// --axis is a clear error (a transpose-to-last lands with the layout work later).
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
  std::fprintf(stderr, "usage: %s INPUT.npy OUTPUT.npy [--axis N]\n", prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  int64_t axis = -1;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--axis" && i + 1 < argc) {
      axis = std::stoll(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  try {
    NpyArray input = read_npy_bf16(input_path);
    if (input.shape.empty()) {
      std::fprintf(stderr, "error: softmax needs a non-scalar input tensor\n");
      return 1;
    }

    int64_t rank = static_cast<int64_t>(input.shape.size());
    int64_t resolved_axis = axis < 0 ? axis + rank : axis;
    if (resolved_axis != rank - 1) {
      std::fprintf(stderr,
                   "error: the Wave-2 softmax baseline reduces the last axis "
                   "only (got axis %lld, last axis %lld)\n",
                   static_cast<long long>(axis),
                   static_cast<long long>(rank - 1));
      return 1;
    }

    int64_t cols = input.shape.back();
    int64_t rows = input.size() / cols;

    std::vector<uint16_t> output(input.data.size());
    polykernel::cpu::launch_softmax_cpu(input.data.data(), output.data(), rows,
                                        cols);

    write_npy_bf16(output_path, input.shape, output);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
