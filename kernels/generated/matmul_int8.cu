//===- matmul_int8.cu - PolyKernel int8 weight-only MatMul -------*- CUDA -*-===//
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

//===----------------------------------------------------------------------===//
// STANDALONE TEST DRIVER (test scaffolding; compiled ONLY when the test builds this
// file with -DPOLYKERNEL_QUANT_MAIN). The file-based .npy validation bridge (Pinned
// contract B) mirroring matmul_wmma.cu's driver: reads bf16 A, int8 Wq, fp32 scale,
// launches the int8 kernel on the GPU, writes bf16 C. Not part of the kernel; guarded
// so the plain kernel build (nvcc --ptx / hipcc -c) never sees it.
//===----------------------------------------------------------------------===//
#if defined(POLYKERNEL_QUANT_MAIN)

#include "PolyKernel/Runtime/HipRuntime.h"
#include "npy_io.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace cpu = polykernel::cpu;
namespace runtime = polykernel::runtime;

static_assert(sizeof(pk_bf16) == 2, "pk_bf16 must be 2 bytes (bf16)");

namespace {

/// Minimal raw .npy reader for the non-bf16 operands (int8 '|i1', fp32 '<f4').
/// bf16 A/C reuse cpu::read_npy_bf16 / write_npy_bf16 (the '<V2' bridge). Parses
/// the v1/v2 header dict for `shape`, validates the descr, returns the raw payload.
struct NpyRaw {
  std::vector<int64_t> shape;
  std::vector<char> bytes;
};

std::string quoted_after(const std::string &h, const std::string &key) {
  auto k = h.find(key);
  if (k == std::string::npos)
    throw std::runtime_error("npy header missing " + key);
  auto o = h.find('\'', k + key.size());
  auto c = h.find('\'', o + 1);
  return h.substr(o + 1, c - o - 1);
}

std::vector<int64_t> parse_shape(const std::string &h) {
  auto o = h.find('(', h.find("'shape'"));
  auto c = h.find(')', o);
  std::vector<int64_t> shape;
  std::string::size_type i = o + 1;
  while (i < c) {
    while (i < c && (h[i] == ' ' || h[i] == ','))
      ++i;
    if (i >= c)
      break;
    auto s = i;
    while (i < c && h[i] >= '0' && h[i] <= '9')
      ++i;
    shape.push_back(std::stoll(h.substr(s, i - s)));
  }
  return shape;
}

NpyRaw read_npy_raw(const std::string &path, const std::string &descr, int esz) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open " + path);
  char magic[6];
  in.read(magic, 6);
  if (std::string(magic, 6) != "\x93"
      "NUMPY")
    throw std::runtime_error("bad npy magic: " + path);
  unsigned char ver[2];
  in.read(reinterpret_cast<char *>(ver), 2);
  uint64_t hlen = 0;
  if (ver[0] == 1) {
    unsigned char b[2];
    in.read(reinterpret_cast<char *>(b), 2);
    hlen = uint64_t(b[0]) | (uint64_t(b[1]) << 8);
  } else {
    unsigned char b[4];
    in.read(reinterpret_cast<char *>(b), 4);
    hlen = uint64_t(b[0]) | (uint64_t(b[1]) << 8) | (uint64_t(b[2]) << 16) |
           (uint64_t(b[3]) << 24);
  }
  std::string header(hlen, '\0');
  in.read(header.data(), static_cast<std::streamsize>(hlen));
  std::string d = quoted_after(header, "'descr'");
  if (d != descr && d != descr.substr(0, 1) + descr.substr(2))
    throw std::runtime_error("npy descr '" + d + "' != expected '" + descr + "'");
  NpyRaw arr;
  arr.shape = parse_shape(header);
  int64_t n = 1;
  for (int64_t x : arr.shape)
    n *= x;
  arr.bytes.resize(static_cast<std::size_t>(n) * esz);
  in.read(arr.bytes.data(), static_cast<std::streamsize>(arr.bytes.size()));
  if (!in)
    throw std::runtime_error("truncated npy payload: " + path);
  return arr;
}

/// RAII device byte buffer (mirrors matmul_wmma.cu): checked alloc / checked free.
struct DevBuf {
  void *p = nullptr;
  explicit DevBuf(std::size_t bytes) {
    if (bytes)
      runtime::HipMalloc(&p, bytes);
  }
  ~DevBuf() {
    if (p)
      runtime::HipFree(p);
  }
  DevBuf(DevBuf &&o) noexcept : p(o.p) { o.p = nullptr; }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
};

int run(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: matmul_int8_run int8 A.npy Wq.npy scale.npy C.npy\n");
    return 2;
  }
  const std::string mode = argv[1];
  if (mode != "int8") {
    std::fprintf(stderr, "matmul_int8_run: unknown mode %s (expected 'int8')\n",
                 mode.c_str());
    return 2;
  }

  cpu::NpyArray a = cpu::read_npy_bf16(argv[2]);           // bf16 [M, K]
  NpyRaw wq = read_npy_raw(argv[3], "|i1", 1);             // int8 [K, N]
  NpyRaw sc = read_npy_raw(argv[4], "<f4", 4);             // fp32 [N]

  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = wq.shape[1];
  if (wq.shape[0] != K) {
    std::fprintf(stderr, "matmul_int8: contraction mismatch A.K=%lld != Wq.K=%lld\n",
                 (long long)K, (long long)wq.shape[0]);
    return 1;
  }
  if (sc.shape[0] != N) {
    std::fprintf(stderr, "matmul_int8: scale length %lld != N=%lld\n",
                 (long long)sc.shape[0], (long long)N);
    return 1;
  }

  // Report annotation (contract H): the quant path + dtype this kernel uses.
  std::fprintf(stderr,
               "[polykernel-quant] path=int8_weight_only dtype=bf16xint8 "
               "scale=per-channel M=%lld N=%lld K=%lld\n",
               (long long)M, (long long)N, (long long)K);

  DevBuf da(a.data.size() * sizeof(uint16_t));
  runtime::HipCopyH2D(da.p, a.data.data(), a.data.size() * sizeof(uint16_t));
  DevBuf dw(wq.bytes.size());
  runtime::HipCopyH2D(dw.p, wq.bytes.data(), wq.bytes.size());
  DevBuf ds(sc.bytes.size());
  runtime::HipCopyH2D(ds.p, sc.bytes.data(), sc.bytes.size());
  DevBuf dc(static_cast<std::size_t>(M * N) * sizeof(uint16_t));

  launch_matmul_int8(static_cast<pk_bf16 *>(da.p), static_cast<int8_t *>(dw.p),
                     static_cast<float *>(ds.p), static_cast<pk_bf16 *>(dc.p),
                     (int)M, (int)N, (int)K, nullptr);
  runtime::HipSync();

  std::vector<uint16_t> host(static_cast<std::size_t>(M * N));
  runtime::HipCopyD2H(host.data(), dc.p, host.size() * sizeof(uint16_t));
  std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
  out_shape.push_back(N);
  cpu::write_npy_bf16(argv[5], out_shape, host);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[polykernel-quant] error: %s\n", e.what());
    return 1;
  }
}

#endif // POLYKERNEL_QUANT_MAIN
