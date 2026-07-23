//===- cpu_ref_gelu.cpp - CPU reference GELU driver -------------*- C++ -*-===//
//
// PolyKernel CPU reference GELU driver (Todo 9 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B): reads a bf16 input .npy,
// computes exact erf-based GELU via launch_gelu_cpu (the same math + rounding
// contract as tests/golden/golden.py), and writes a bf16 output .npy.
// tests/kernels/test_cpu_ref_activations.py drives this executable and compares
// the output to the golden via metrics.assert_correct.
//
// Usage:
//   cpu_ref_gelu INPUT.npy OUTPUT.npy
//
// GELU is elementwise + shape-preserving, so the whole tensor is flattened to a
// single element count (n = product(shape)); the output keeps the input shape.
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

void usage(const char *prog) {
  std::fprintf(stderr, "usage: %s INPUT.npy OUTPUT.npy\n", prog);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    usage(argv[0]);
    return 2;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];

  try {
    NpyArray input = read_npy_bf16(input_path);

    std::vector<uint16_t> output(input.data.size());
    polykernel::cpu::launch_gelu_cpu(input.data.data(), output.data(),
                                     input.size());

    write_npy_bf16(output_path, input.shape, output);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }

  return 0;
}
