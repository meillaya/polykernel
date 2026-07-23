//===- npy_io.h - Minimal NumPy .npy I/O for bf16 (<V2>) ---------*- C++ -*-===//
//
// PolyKernel CPU reference .npy bridge (Todo 8 / Wave 2).
//
//===----------------------------------------------------------------------===//
//
// The single I/O contract between the CPU/GPU kernels and the Python golden
// harness is NumPy's .npy file (Pinned contract B). ml_dtypes.bfloat16 has no
// native NumPy dtype, so np.save serializes a bf16 array with the void descr
// '<V2' (2 little-endian bytes per element = the raw bf16 bits). This module
// reads/writes exactly that encoding:
//
//   - read:  parse the v1.0/v2.0 header dict for `shape`, then read the raw
//            uint16 (bf16-bit) payload;
//   - write: emit a v1.0 header with descr '<V2' + the raw uint16 payload, so
//            the Python side does `np.load(path).view(ml_dtypes.bfloat16)`.
//
// Only the bf16 '<V2' encoding is supported (that is all the harness uses); a
// mismatched descr is a hard error so a wrong-dtype file fails loudly.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_KERNELS_CPU_NPY_IO_H
#define POLYKERNEL_KERNELS_CPU_NPY_IO_H

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace polykernel::cpu {

/// A loaded .npy bf16 tensor: its shape + row-major bf16 (raw uint16) payload.
struct NpyArray {
  std::vector<int64_t> shape;
  std::vector<uint16_t> data;

  /// Total element count (product of the shape; 1 for a scalar/empty shape).
  int64_t size() const;
};

/// Thrown when a .npy file is malformed or not a '<V2' bf16 array.
class NpyError : public std::runtime_error {
public:
  explicit NpyError(const std::string &what) : std::runtime_error(what) {}
};

/// Read a bf16 ('<V2') .npy file. Throws NpyError on any format problem.
NpyArray read_npy_bf16(const std::string &path);

/// Write a bf16 ('<V2') .npy v1.0 file (row-major). `data.size()` must equal
/// the product of `shape`.
void write_npy_bf16(const std::string &path, const std::vector<int64_t> &shape,
                    const std::vector<uint16_t> &data);

} // namespace polykernel::cpu

#endif // POLYKERNEL_KERNELS_CPU_NPY_IO_H
