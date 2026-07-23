//===- cpu_ref_rmsnorm.cpp - CPU reference RMSNorm driver --------*- C++ -*-===//
//
// PolyKernel CPU reference RMSNorm driver (Todo 8 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads a bf16 input .npy,
// computes RMSNorm over the last axis via launch_rmsnorm_cpu (the same math +
// rounding contract as tests/golden/golden.py), and writes a bf16 output .npy.
// tests/kernels/test_cpu_ref.py drives this executable and compares the output
// to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_rmsnorm INPUT.npy OUTPUT.npy [--weight WEIGHT.npy] [--epsilon E]
//
// The reduction axis is the LAST dimension; all leading dimensions are flattened
// into the row count (rows = product(shape[:-1]), cols = shape[-1]).
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h"
#include "npy_io.h"

#include <cstdio>
#include <string>
#include <vector>

using polykernel::cpu::NpyArray;
using polykernel::cpu::read_npy_bf16;
using polykernel::cpu::write_npy_bf16;

namespace {

constexpr float kDefaultEpsilon = 1e-5f;

void usage(const char *prog) {
  std::fprintf(stderr,
               "usage: %s INPUT.npy OUTPUT.npy [--weight WEIGHT.npy] "
               "[--epsilon E]\n",
               prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 2;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];
  std::string weight_path;
  float epsilon = kDefaultEpsilon;

  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--weight" && i + 1 < argc) {
      weight_path = argv[++i];
    } else if (arg == "--epsilon" && i + 1 < argc) {
      epsilon = std::stof(argv[++i]);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  try {
    NpyArray input = read_npy_bf16(input_path);
    if (input.shape.empty()) {
      std::fprintf(stderr, "error: rmsnorm needs a non-scalar input tensor\n");
      return 1;
    }

    int64_t cols = input.shape.back();
    int64_t rows = input.size() / cols;

    std::vector<uint16_t> weight;
    const uint16_t *weight_ptr = nullptr;
    if (!weight_path.empty()) {
      NpyArray w = read_npy_bf16(weight_path);
      if (w.size() != cols) {
        std::fprintf(stderr,
                     "error: weight length %lld != last input dim %lld\n",
                     static_cast<long long>(w.size()),
                     static_cast<long long>(cols));
        return 1;
      }
      weight = std::move(w.data);
      weight_ptr = weight.data();
    }

    std::vector<uint16_t> output(input.data.size());
    polykernel::cpu::launch_rmsnorm_cpu(input.data.data(), weight_ptr,
                                        output.data(), rows, cols, epsilon);

    write_npy_bf16(output_path, input.shape, output);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
