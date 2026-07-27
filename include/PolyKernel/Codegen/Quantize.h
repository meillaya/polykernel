//===- Quantize.h - PolyKernel quantization codegen lowering ----*- C++ -*-===//
//
// PolyKernel quantization codegen lowering (Todo 41 / Wave 8 elite).
//
//===----------------------------------------------------------------------===//
//
// A CUSTOM SOURCE EMITTER (Pinned contract F, the same approach as
// lib/Passes/LowerToCuda.cpp) for the QUANTIZED matmul variants of the closed
// `polykernel.matmul` op. It STRING-EMITS portable CUDA/HIP kernel source written
// against kernels/template/kernel_common.h:
//
//   - Int8WeightOnly: the int8 weight-only matmul kernel — weights stored as int8
//     with a per-output-channel scale[N], activations bf16, dequantize-and-multiply
//     inside the kernel (emitted source is byte-identical to the kernel + launch
//     portion of kernels/generated/matmul_int8.cu).
//   - Fp8SimE4M3 / Fp8SimE5M2: the fp8 (e4m3fn / e5m2) SIMULATION path. This is NOT
//     a hardware fp8 kernel — fp8 is simulated in the golden (ml_dtypes float8) and
//     the CPU ref (portable RNE round); the emitter documents that path. No new op,
//     no new declared attribute (the op set is CLOSED); `path=int8_weight_only` /
//     `path=fp8_sim` are report annotations (contract H).
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_CODEGEN_QUANTIZE_H
#define POLYKERNEL_CODEGEN_QUANTIZE_H

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace polykernel::codegen {

/// The quantization codegen paths this lowering can emit.
enum class QuantPath {
  Int8WeightOnly, // int8 weights (per-channel scale) + bf16 activations
  Fp8SimE4M3,     // fp8 e4m3fn weight SIMULATION (no fp8 hardware)
  Fp8SimE5M2,     // fp8 e5m2 weight SIMULATION (no fp8 hardware)
};

/// Human-readable path name (the contract-H `path` annotation value).
const char *quantPathName(QuantPath path);

/// Emit the int8 weight-only matmul kernel + host launch entry to `os`. The output
/// is byte-identical to the kernel portion of kernels/generated/matmul_int8.cu.
void emitMatMulInt8Cuda(llvm::raw_ostream &os);

/// Emit the source for a quantization path: Int8WeightOnly dispatches to
/// emitMatMulInt8Cuda; the fp8-sim paths emit a clearly-labeled simulation note
/// (fp8 is simulated in the golden + CPU ref, NOT a hardware kernel).
void emitQuantizeCuda(llvm::raw_ostream &os, QuantPath path);

} // namespace polykernel::codegen

#endif // POLYKERNEL_CODEGEN_QUANTIZE_H
