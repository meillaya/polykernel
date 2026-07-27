//===- matmul_wmma.cu - PolyKernel WMMA bf16 MatMul (RDNA3) -----*- CUDA -*-===//
//
// AUTO-GENERATED codegen VARIANT of the `polykernel.matmul` op (Todo 22 / Wave 4).
// Do not edit by hand; re-run the emitter to regenerate.
//
// RDNA3 TENSOR-CORE PATH (ADDITIVE to the scalar baseline). This is the WMMA bf16
// variant the autotuner (Todo 25) can SELECT; the tiled SCALAR baseline
// kernels/generated/matmul.cu (Todo 10) REMAINS the correctness reference and is
// UNCHANGED. WMMA is correctness-gated: if this variant fails golden locally it is
// excluded (validated=false) and the scalar baseline stands — it NEVER blocks the
// wave. The op set is CLOSED: this adds NO new op and NO new declared attribute
// (`path=wmma` is a report annotation, contract H, not a new op).
//
// INSTRUCTION: gfx1101 (RX 7800 XT, Navi 32, wave32) `v_wmma_f32_16x16x16_bf16`,
// driven by the clang builtin
//     v8f __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(v16bf a, v16bf b, v8f c);
// One wave (32 lanes) computes a 16x16 f32 accumulator tile from 16x16 bf16 A and
// 16x16 bf16 B: 16 bf16 elems/lane input (v16bf), 8 f32 elems/lane accumulator (v8f).
//
// FRAGMENT LANE-MAPPING (RDNA3 wave32; validated bit-exact against production RDNA3
// WMMA — AMD GPUOpen "WMMA on RDNA3" layout, the same idiom vLLM/llama.cpp ship):
//     typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
//     typedef float  v8f   __attribute__((ext_vector_type(8)));
//   A load (MxK row-major): lane l -> a[i] = A_tile[l & 15][i],  i = 0..15 (full K-row)
//   B load (KxN):           lane l -> b[k] = B_tile[k][l & 15],  k = 0..15 (full K-col)
//   Store / accumulator:    lane l, elem e (0..7) -> C[2*e + (l >> 4)][l & 15]
// Lanes l and l+16 hold the SAME A-row / B-col (the inputs are duplicated across the
// two lane halves); the accumulator interleaves even rows in lanes 0..15 and odd rows
// in lanes 16..31 (row = 2*e + (l>>4)), columns = l & 15.
//
// ROUNDING CONTRACT (matches tests/golden/golden.py, contract C): bf16 inputs upcast
// exactly to bf16 fragments (pk_bf16 -> fp32 -> __bf16 is lossless); the WMMA unit
// multiplies bf16xbf16 and ACCUMULATES in fp32; the fp32 accumulator is rounded back
// to bf16 (RNE) on store. K is accumulated in 16-wide WMMA tiles (a different fp32
// summation order than NumPy), so at large K an isolated element can land 1 bf16 ULP
// off at small magnitude — the SAME documented reduction-order class as the scalar
// tiled baseline (cosine == pcc == 1.0, >= 99.99% bit-exact; NOT a bug).
//
// PORTABILITY: this variant is RDNA3/HIP-specific by construction (the WMMA builtin
// + __bf16 fragments are AMD). It is written against the portable template for every
// backend-agnostic concern (pk_bf16 / PK_GLOBAL / PK_SHARED / pk_syncthreads /
// pk_float2bf16 / pk_bf162float / PK_LAUNCH / PK_CHECK) and guards the AMD-only
// fragment section behind POLYKERNEL_HIP. Under -DPOLYKERNEL_CUDA it is a hard
// #error: the portable scalar matmul.cu is the CUDA-path baseline (a CUDA WMMA path
// would use nvcuda::wmma and is out of scope for this RDNA3 task).
//
//===----------------------------------------------------------------------===//

#include "../template/kernel_common.h"

#if !defined(POLYKERNEL_HIP)
#error "matmul_wmma.cu is the RDNA3/HIP WMMA variant; compile with hipcc -DPOLYKERNEL_HIP. The portable scalar baseline is kernels/generated/matmul.cu."
#endif

namespace {

//===----------------------------------------------------------------------===//
// AMD RDNA3 WMMA fragment types (wave32). __bf16 is the compiler-native bf16 type
// the v_wmma builtin consumes; it is DISTINCT from pk_bf16 (__hip_bfloat16, a struct
// wrapper), so fragments are populated by casting through fp32 (lossless for bf16).
//===----------------------------------------------------------------------===//
using pk_wmma_a = __bf16 __attribute__((ext_vector_type(16))); // 16 bf16 / lane
using pk_wmma_c = float __attribute__((ext_vector_type(8)));   // 8 f32 / lane

// One wave (32 lanes) computes one 16x16 output tile; gridDim tiles N (x) and M (y).
constexpr int kTile = 16;

// WMMA bf16 matmul: C[M,N] = A[M,K] @ B[K,N]. One block == one wave == one 16x16
// output tile; K is reduced in 16-wide WMMA tiles staged through shared memory.
// BadLane=false is the CORRECT store lane-mapping; BadLane=true swaps it for a
// deliberately WRONG (but believable) contiguous-rows mapping — the negative path
// that proves a bad WMMA variant fails golden and is discarded (see the store below).
template <bool BadLane>
PK_GLOBAL void matmul_wmma_kernel(const pk_bf16 *__restrict__ A,
                                  const pk_bf16 *__restrict__ B,
                                  pk_bf16 *__restrict__ C, int M, int N, int K) {
  PK_SHARED pk_bf16 As[kTile][kTile]; // A tile [16 M-rows][16 K-cols]
  PK_SHARED pk_bf16 Bs[kTile][kTile]; // B tile [16 K-rows][16 N-cols]

  const int lane = threadIdx.x & 31; // wave32 lane id (blockDim.x == 32)
  const int base_m = blockIdx.y * kTile;
  const int base_n = blockIdx.x * kTile;

  pk_wmma_c acc = pk_wmma_c{0, 0, 0, 0, 0, 0, 0, 0};

  const int numKTiles = (K + kTile - 1) / kTile;
  for (int t = 0; t < numKTiles; ++t) {
    const int k_base = t * kTile;
    // Cooperatively stage the 16x16 A and B tiles (256 elems each / 32 lanes = 8
    // per lane). Out-of-bounds loads (non-tile-multiple M/N/K) read 0 so they never
    // contribute to the accumulator — the kernel is correct for ANY shape, though the
    // autotuner selects WMMA only for 16-multiples (the clean, no-tail case).
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
      const int r = idx / kTile;
      const int c = idx % kTile;
      const int gm = base_m + r;
      const int gk = k_base + c;
      As[r][c] = (gm < M && gk < K) ? A[static_cast<size_t>(gm) * K + gk]
                                    : pk_float2bf16(0.0f);
      const int gk2 = k_base + r;
      const int gn = base_n + c;
      Bs[r][c] = (gk2 < K && gn < N) ? B[static_cast<size_t>(gk2) * N + gn]
                                     : pk_float2bf16(0.0f);
    }
    pk_syncthreads();

    // Build the per-lane WMMA fragments from shared memory (validated lane-mapping).
    pk_wmma_a a;
    pk_wmma_a b;
#pragma unroll
    for (int i = 0; i < kTile; ++i)
      a[i] = static_cast<__bf16>(pk_bf162float(As[lane & 15][i])); // A-row l&15
#pragma unroll
    for (int k = 0; k < kTile; ++k)
      b[k] = static_cast<__bf16>(pk_bf162float(Bs[k][lane & 15])); // B-col l&15

    acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, acc);
    pk_syncthreads();
  }

  // Store the 8 f32 accumulator elems/lane, rounded to bf16 (RNE), within bounds.
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    // CORRECT: rows interleave across the two lane halves (row = 2*e + (lane>>4)).
    // BAD (negative path): the believable bug of assuming contiguous rows per half
    // (row = 8*(lane>>4) + e) — scrambles the output rows so golden fails.
    const int r = BadLane ? (base_m + 8 * (lane >> 4) + e)
                          : (base_m + 2 * e + (lane >> 4));
    const int c = base_n + (lane & 15);
    if (r < M && c < N)
      C[static_cast<size_t>(r) * N + c] = pk_float2bf16(acc[e]);
  }
}

} // namespace

// Host-callable launch entries (mirror launch_matmul's ABI exactly). C[M,N] =
// A[M,K] @ B[K,N]; batched leading dims are flattened into M by the caller. The
// stream is void* (backend-agnostic), cast to pk_stream_t inside.
void launch_matmul_wmma(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M,
                        int N, int K, void *stream) {
  dim3 block(32); // one wave per block
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(matmul_wmma_kernel<false>, grid, block, 0, s, A, B, C, M, N, K);
  PK_CHECK(pk_get_last_error());
}

// Negative-path launcher: the SAME kernel with a deliberately WRONG store lane-mapping
// (BadLane=true). Exists only so the test can prove a bad WMMA variant fails golden
// and is discarded while the scalar baseline still passes (WMMA is safely additive).
void launch_matmul_wmma_bad(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M,
                            int N, int K, void *stream) {
  dim3 block(32);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(matmul_wmma_kernel<true>, grid, block, 0, s, A, B, C, M, N, K);
  PK_CHECK(pk_get_last_error());
}

//===----------------------------------------------------------------------===//
// STANDALONE TEST DRIVER (test scaffolding; compiled ONLY when the test builds this
// file with -DPOLYKERNEL_WMMA_MAIN). The file-based .npy validation bridge (Pinned
// contract B) mirroring lib/Runtime/hip_run_main.cpp's run_matmul, plus a `scalar`
// mode that launches the UNCHANGED scalar baseline (kernels/generated/matmul.cu) so
// the negative test can show the baseline still passes while a bad WMMA variant fails.
// Not part of the kernel; guarded so the plain kernel build never sees it.
//===----------------------------------------------------------------------===//
#if defined(POLYKERNEL_WMMA_MAIN)

#include "PolyKernel/Runtime/HipRuntime.h"
#include "npy_io.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Scalar baseline ABI (kernels/generated/matmul.cu, Todo 10) — linked alongside.
void launch_matmul(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M, int N,
                   int K, void *stream);

namespace cpu = polykernel::cpu;
namespace runtime = polykernel::runtime;

static_assert(sizeof(pk_bf16) == 2, "pk_bf16 must be 2 bytes (bf16)");

namespace {

/// RAII device buffer (mirrors hip_run_main.cpp): checked alloc / checked free.
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
  pk_bf16 *ptr() { return static_cast<pk_bf16 *>(p); }
};

DevBuf upload(const cpu::NpyArray &arr) {
  DevBuf d(arr.data.size() * sizeof(uint16_t));
  runtime::HipCopyH2D(d.p, arr.data.data(), arr.data.size() * sizeof(uint16_t));
  return d;
}

void download_write(const DevBuf &d, const std::vector<int64_t> &shape,
                    const std::string &path) {
  int64_t n = 1;
  for (int64_t x : shape)
    n *= x;
  std::vector<uint16_t> host(static_cast<std::size_t>(n));
  runtime::HipCopyD2H(host.data(), d.p, host.size() * sizeof(uint16_t));
  cpu::write_npy_bf16(path, shape, host);
}

int run(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: wmma_run <wmma|wmma_bad|scalar> A.npy B.npy C.npy\n");
    return 2;
  }
  const std::string mode = argv[1];
  cpu::NpyArray a = cpu::read_npy_bf16(argv[2]);
  cpu::NpyArray b = cpu::read_npy_bf16(argv[3]);
  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = b.shape[1];
  if (b.shape[0] != K) {
    std::fprintf(stderr, "matmul_wmma: contraction mismatch A.K=%lld != B.K=%lld\n",
                 (long long)K, (long long)b.shape[0]);
    return 1;
  }
  DevBuf da = upload(a);
  DevBuf db = upload(b);
  DevBuf dc(static_cast<std::size_t>(M * N) * sizeof(uint16_t));

  // Report annotation (contract H): which path produced C.
  std::fprintf(stderr, "[polykernel-wmma] path=%s M=%lld N=%lld K=%lld\n",
               mode.c_str(), (long long)M, (long long)N, (long long)K);
  if (mode == "wmma")
    launch_matmul_wmma(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K, nullptr);
  else if (mode == "wmma_bad")
    launch_matmul_wmma_bad(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K,
                           nullptr);
  else if (mode == "scalar")
    launch_matmul(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K, nullptr);
  else
    return 2;

  runtime::HipSync();
  // Output shape = A's leading dims (all but K) + N (mirrors matmul_out_shape).
  std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
  out_shape.push_back(N);
  download_write(dc, out_shape, argv[4]);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[polykernel-wmma] error: %s\n", e.what());
    return 1;
  }
}

#endif // POLYKERNEL_WMMA_MAIN
