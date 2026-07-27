//===- Quantize.cpp - PolyKernel quantization codegen lowering --*- C++ -*-===//
//
// PolyKernel quantization codegen lowering (Todo 41 / Wave 8 elite). See Quantize.h.
//
//===----------------------------------------------------------------------===//
//
// allow: SIZE_OK — emitMatMulInt8Cuda carries the full int8 kernel + launch source
// VERBATIM as a raw string (a string emitter, Pinned contract F, exactly like the
// per-op emitters in lib/Passes/LowerToCuda.cpp). Most of the file's lines are emitted
// .cu text (stripped by pure-LOC counters, which see ~91 logic lines), not new logic;
// the emitter is kept whole so its output stays byte-identical to matmul_int8.cu.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Codegen/Quantize.h"

#include "llvm/Support/raw_ostream.h"

namespace polykernel::codegen {

const char *quantPathName(QuantPath path) {
  switch (path) {
  case QuantPath::Int8WeightOnly:
    return "int8_weight_only";
  case QuantPath::Fp8SimE4M3:
    return "fp8_sim_e4m3";
  case QuantPath::Fp8SimE5M2:
    return "fp8_sim_e5m2";
  }
  return "unknown"; // unreachable: switch above is exhaustive over QuantPath
}

void emitMatMulInt8Cuda(llvm::raw_ostream &os) {
  os << R"POLYKERNEL(//===- matmul_int8.cu - PolyKernel int8 weight-only MatMul -------*- CUDA -*-===//
//
// AUTO-GENERATED codegen VARIANT of the `polykernel.matmul` op (Todo 41 / Wave 8).
// Emitted by lib/Codegen/Quantize.cpp (emitMatMulInt8Cuda); do not edit by hand.
//
// INT8 WEIGHT-ONLY quantized matmul (ADDITIVE to the bf16 scalar baseline). The
// WEIGHTS are stored as int8 with a per-output-channel scale[N]; the ACTIVATIONS
// stay bf16. Inside the inner loop each int8 weight is DEQUANTIZED on the fly and
// multiplied by the bf16 activation, accumulating in fp32:
//     C[M,N] = A[M,K] @ dequant(Wq[K,N], scale[N])
//     dequant: w_deq[k,n] = Wq[k,n] * scale[n]      (fp32, per output channel n)
//     C[m,n]  = bf16( sum_k fp32(A[m,k]) * (fp32(Wq[k,n]) * scale[n]) )
// This is a codegen variant of the CLOSED `polykernel.matmul` op (NO new op, NO new
// declared attribute); `path=int8_weight_only` is a report annotation (contract H),
// exactly as `path=wmma` is for the WMMA variant. The tiled bf16 SCALAR baseline
// (kernels/generated/matmul.cu) is UNCHANGED and remains the correctness reference.
//
// ROUNDING CONTRACT (matches tests/golden/quant_golden.py): the dequant wq*scale is
// fp32 (wq is an exact int8 -> fp32); the fp32 K-accumulation is rounded back to bf16
// (RNE) on store. Quantization is lossy, so this is gated by the RELAXED quant
// thresholds (cosine >= 0.99, max_rel_err <= 5e-2), NOT the bf16 contract C.
//
// PORTABILITY: written against the portable template (kernels/template/kernel_common.h)
// so the SAME file compiles UNCHANGED under `nvcc -DPOLYKERNEL_CUDA` (sm_80/sm_90) and
// `hipcc -DPOLYKERNEL_HIP` (gfx1101). int8_t is standard <cstdint> (no vendor
// intrinsic); everything backend-specific is behind pk_* / PK_* macros. The standalone
// .npy test driver at the bottom is guarded by POLYKERNEL_QUANT_MAIN (HIP run only).
//
//===----------------------------------------------------------------------===//

#include "../template/kernel_common.h"

#include <cstdint>

namespace {

// Output tile per block + K-tile width (mirrors matmul.cu). One thread (tx, ty)
// computes one output element C[row, col]; blockDim == (kBlockN, kBlockM) == 16x16.
constexpr int kBlockM = 16;
constexpr int kBlockN = 16;
constexpr int kBlockK = 16;

// One block per BLOCK_M x BLOCK_N output tile. gridDim.x tiles N (columns),
// gridDim.y tiles M (rows). Shared memory stages one BLOCK_K-wide slice of the bf16
// activation A (BLOCK_M x BLOCK_K, 512 B) and of the int8 weight Wq (BLOCK_K x
// BLOCK_N, 256 B) per K-tile iteration (768 B total).
PK_GLOBAL void matmul_int8_kernel(const pk_bf16 *__restrict__ A,
                                  const int8_t *__restrict__ Wq,
                                  const float *__restrict__ scale,
                                  pk_bf16 *__restrict__ C, int M, int N, int K) {
  PK_SHARED pk_bf16 As[kBlockM][kBlockK];
  PK_SHARED int8_t Bs[kBlockK][kBlockN];

  const int tx = threadIdx.x; // column within the output tile [0, kBlockN)
  const int ty = threadIdx.y; // row within the output tile    [0, kBlockM)
  const int row = blockIdx.y * kBlockM + ty;
  const int col = blockIdx.x * kBlockN + tx;

  // Per-output-channel dequant scale for this thread's column (constant over K).
  const float s = (col < N) ? scale[col] : 0.0f;

  // fp32 accumulator (matches quant_golden.matmul_int8: fp32(bf16(A)) @ (wq*scale)).
  float acc = 0.0f;

  const int numKTiles = (K + kBlockK - 1) / kBlockK;
  for (int t = 0; t < numKTiles; ++t) {
    // Stage A[row, t*kBlockK + tx] (bf16) and Wq[t*kBlockK + ty, col] (int8) into
    // shared mem. Out-of-bounds loads (non-tile-multiple M/N/K) read as 0 so they
    // never contribute to the accumulator.
    int aCol = t * kBlockK + tx;
    As[ty][tx] = (row < M && aCol < K) ? A[static_cast<size_t>(row) * K + aCol]
                                       : pk_float2bf16(0.0f);
    int bRow = t * kBlockK + ty;
    Bs[ty][tx] = (bRow < K && col < N) ? Wq[static_cast<size_t>(bRow) * N + col] : 0;
    pk_syncthreads();

    // Accumulate this K-tile in fp32: dequantize the int8 weight on the fly
    // (wq * per-channel scale) and multiply by the bf16 activation (upcast exactly).
#pragma unroll
    for (int kk = 0; kk < kBlockK; ++kk)
      acc += pk_bf162float(As[ty][kk]) * (static_cast<float>(Bs[kk][tx]) * s);
    pk_syncthreads();
  }

  // Round the fp32 accumulator back to bf16 (RNE) and store, within bounds.
  if (row < M && col < N)
    C[static_cast<size_t>(row) * N + col] = pk_float2bf16(acc);
}

} // namespace

// Host-callable launch entry (the launch_*-style ABI). C[M,N] =
// A[M,K] @ dequant(Wq[K,N], scale[N]); A is bf16 (batched leading dims flattened
// into M by the caller), Wq is int8, scale is fp32 [N] (per output channel). The
// stream is passed as void* to keep the signature backend-agnostic and cast to the
// backend stream type (pk_stream_t) inside. EVERY launch is error-checked (PK_CHECK).
void launch_matmul_int8(const pk_bf16 *A, const int8_t *Wq, const float *scale,
                        pk_bf16 *C, int M, int N, int K, void *stream) {
  dim3 block(kBlockN, kBlockM);
  dim3 grid((N + kBlockN - 1) / kBlockN, (M + kBlockM - 1) / kBlockM);
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(matmul_int8_kernel, grid, block, 0, s, A, Wq, scale, C, M, N, K);
  PK_CHECK(pk_get_last_error());
}
)POLYKERNEL";
}

namespace {

// Emit the fp8 (e4m3fn / e5m2) weight-SIMULATION path note. This is deliberately NOT
// a hardware fp8 kernel: fp8 is simulated in the golden (ml_dtypes float8) and the
// CPU ref (portable RNE round). The emitter documents that contract so the codegen
// lowering carries the fp8-sim path explicitly (and never claims real fp8 hardware).
void emitFp8SimNote(llvm::raw_ostream &os, QuantPath path, const char *fmt,
                    const char *dtype) {
  os << "//===- matmul_" << fmt
     << ".cu - PolyKernel fp8 weight SIMULATION path -----*- CUDA -*-===//\n"
     << "//\n"
     << "// AUTO-GENERATED by lib/Codegen/Quantize.cpp (Todo 41 / Wave 8 elite).\n"
     << "//\n"
     << "// fp8 (" << fmt
     << ") weight SIMULATION — NOT a hardware fp8 kernel. There is no fp8\n"
     << "// tensor-core / storage path here: the fp8 rounding is SIMULATED in the golden\n"
     << "// (tests/golden/quant_golden.py via ml_dtypes.float8_" << dtype
     << ") and in the CPU\n"
     << "// reference (kernels/cpu/matmul_quant.cpp fp8sim mode, a portable RNE round to\n"
     << "// the fp8 grid). path=" << quantPathName(path)
     << " is a report annotation (contract H), not a\n"
     << "// new op; the op set is CLOSED. The int8 weight-only HARDWARE path is\n"
     << "// emitMatMulInt8Cuda (kernels/generated/matmul_int8.cu).\n"
     << "//===---------------------------------------------------------------===//\n";
}

} // namespace

void emitQuantizeCuda(llvm::raw_ostream &os, QuantPath path) {
  switch (path) {
  case QuantPath::Int8WeightOnly:
    emitMatMulInt8Cuda(os);
    return;
  case QuantPath::Fp8SimE4M3:
    emitFp8SimNote(os, path, "e4m3", "e4m3fn");
    return;
  case QuantPath::Fp8SimE5M2:
    emitFp8SimNote(os, path, "e5m2", "e5m2");
    return;
  }
}

} // namespace polykernel::codegen
