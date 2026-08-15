//===- matmul_mma.cu - PolyKernel CUDA MMA bf16 MatMul (sm_80+) --*- CUDA -*-===//
//
// AUTO-GENERATED codegen VARIANT of the `polykernel.matmul` op (Todo 10 / Wave 4
// pass 2). Do not edit by hand; re-run the emitter to regenerate.
//
// CUDA TENSOR-CORE PATH (ADDITIVE to the scalar baseline). This is the nvcuda::wmma
// bf16 variant the autotuner (Todo 15) can SELECT on NVIDIA GPUs; the tiled SCALAR
// baseline kernels/generated/matmul.cu (Todo 10, Wave 2) REMAINS the correctness
// reference and is UNCHANGED. MMA is correctness-gated: if this variant fails golden
// locally it is excluded (validated=false) and the scalar baseline stands — it NEVER
// blocks the wave. The op set is CLOSED: this adds NO new op and NO new declared
// attribute (`path=mma` is a report annotation, contract H, not a new op). This
// closes the docs' false implication that CUDA had no tensor-core path (the WMMA
// sibling matmul_wmma.cu is RDNA3/HIP-only by design).
//
// INSTRUCTION: `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` on sm_80+
// (the RTX 6000 Ada target is sm_89), driven by the nvcuda::wmma API from
// <mma.h>. One warp (32 lanes) computes a 16x16 f32 accumulator tile from 16x16
// bf16 A and 16x16 bf16 B per K-tile. The fragment API ABSTRACTs the hardware
// lane mapping: `load_matrix_sync` / `store_matrix_sync` interpret memory per the
// fragment's declared layout, so the kernel only has to stage tiles in the right
// shape.
//
// FRAGMENT LAYOUTS (m16n16k16 bf16; contract K, the CUDA twin of contract K's WMMA
// side). A is an MxK operand, B a KxN operand:
//     fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major>
//         As[16][16] row-major tile (As[m][k] = A[base_m+m][k_base+k]);
//         load_matrix_sync(a, &As[0][0], 16) — ldm = K = 16.
//     fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major>
//         B MUST be col_major in the m16n16k16 layout (NOT row_major — the
//         row_major variant reads an NxK/B^T memory view and would transpose
//         the operand, failing golden). col_major means each N-COLUMN's 16
//         K-values are CONTIGUOUS, so the shared B tile is staged TRANSPOSED:
//         Bs[n][k] = B[k_base+k][base_n+n], then
//         load_matrix_sync(b, &Bs[0][0], 16) — ldm = K = 16.
//     fragment<accumulator, 16, 16, 16, float> + mma_sync(acc, a, b, acc);
//         store_matrix_sync(&Cs[0][0], acc, 16, mem_row_major) — the fp32 C
//         tile lands row-major in shared Cs[16][16], then threads round it to
//         bf16 (RNE) and store to global C within bounds.
//
// ROUNDING CONTRACT (matches tests/golden/golden.py, contract C): bf16 inputs
// upcast losslessly to __nv_bfloat16 fragments; the WMMA unit multiplies
// bf16xbf16 and ACCUMULATES in fp32; the fp32 accumulator is rounded back to
// bf16 (RNE) on store. K is accumulated in 16-wide WMMA tiles (a different fp32
// summation order than NumPy), so at large K an isolated element can land 1 bf16
// ULP off at small magnitude — the SAME documented reduction-order class as the
// scalar tiled baseline (cosine == pcc == 1.0, >= 99.99% bit-exact; the large-K
// rel ceiling is 5e-2, NOT a bug).
//
// PORTABILITY: this variant is CUDA/nvcuda::wmma-specific by construction (the
// fragment API + __nv_bfloat16 are NVIDIA). It is written against the portable
// template for every backend-agnostic concern (pk_bf16 / PK_GLOBAL / PK_SHARED /
// pk_syncthreads / pk_float2bf16 / pk_bf162float / PK_LAUNCH / PK_CHECK) and
// guards the NVIDIA-only fragment section behind -DPOLYKERNEL_CUDA. Under
// -DPOLYKERNEL_HIP it is a hard #error: the portable scalar matmul.cu is the
// HIP-path baseline (the RDNA3 tensor-core path is matmul_wmma.cu).
//
//===----------------------------------------------------------------------===//

#include "../template/kernel_common.h"

#if !defined(POLYKERNEL_CUDA)
#error "matmul_mma.cu is the CUDA nvcuda::wmma MMA variant; compile with nvcc -DPOLYKERNEL_CUDA. The portable scalar baseline is kernels/generated/matmul.cu (and the RDNA3 tensor-core sibling is matmul_wmma.cu, HIP-only)."
#endif

#include <mma.h>

namespace {

//===----------------------------------------------------------------------===//
// CUDA WMMA m16n16k16 bf16 fragment types (sm_80+). __nv_bfloat16 == pk_bf16
// under -DPOLYKERNEL_CUDA (kernel_common.h), so shared tiles alias the fragment
// element type exactly and load_matrix_sync reads them directly.
//===----------------------------------------------------------------------===//
using pk_mma_a = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                        __nv_bfloat16, nvcuda::wmma::row_major>;
using pk_mma_b = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                        __nv_bfloat16, nvcuda::wmma::col_major>;
using pk_mma_c = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16,
                                        float>;

// One warp (32 lanes) computes one 16x16 output tile; gridDim tiles N (x) and M (y).
constexpr int kTile = 16;

// WMMA bf16 matmul: C[M,N] = A[M,K] @ B[K,N]. One block == one warp == one 16x16
// output tile; K is reduced in 16-wide WMMA tiles staged through shared memory.
// BadStore=false is the CORRECT store layout; BadStore=true swaps it for a
// deliberately WRONG (but believable) col-major store that TRANSPOSES the C tile
// — the negative path that proves a bad MMA variant fails golden and is discarded
// (see the store below).
template <bool BadStore>
PK_GLOBAL void matmul_mma_kernel(const pk_bf16 *__restrict__ A,
                                 const pk_bf16 *__restrict__ B,
                                 pk_bf16 *__restrict__ C, int M, int N, int K) {
  PK_SHARED pk_bf16 As[kTile][kTile]; // A tile [16 M-rows][16 K-cols] (row-major)
  PK_SHARED pk_bf16 Bs[kTile][kTile]; // B tile [16 N-cols][16 K-rows] (col-major,
                                      // TRANSPOSED staging for the B fragment)
  PK_SHARED float Cs[kTile][kTile];   // C tile [16 M-rows][16 N-cols] (row-major)

  const int lane = threadIdx.x; // one warp per block (blockDim.x == 32)
  const int base_m = blockIdx.y * kTile;
  const int base_n = blockIdx.x * kTile;

  pk_mma_c acc;
  nvcuda::wmma::fill_fragment(acc, 0.0f);

  const int numKTiles = (K + kTile - 1) / kTile;
  for (int t = 0; t < numKTiles; ++t) {
    const int k_base = t * kTile;
    // Cooperatively stage the 16x16 A and B tiles (256 elems each / 32 lanes =
    // 8 per lane). Out-of-bounds loads (non-tile-multiple M/N/K) read 0 so they
    // never contribute to the accumulator — the kernel is correct for ANY shape,
    // though the autotuner selects MMA only for 16-multiples (the clean, no-tail
    // case). K is padded to a multiple of 16 in shared memory by the zero-fill.
    for (int idx = lane; idx < kTile * kTile; idx += 32) {
      const int r = idx / kTile;
      const int c = idx % kTile;
      // A tile row-major: As[r][c] = A[base_m + r][k_base + c].
      const int gm = base_m + r;
      const int gk = k_base + c;
      As[r][c] = (gm < M && gk < K) ? A[static_cast<size_t>(gm) * K + gk]
                                    : pk_float2bf16(0.0f);
      // B tile TRANSPOSED for the col_major B fragment: Bs[row=n][col=k] =
      // B[k_base + k][base_n + n] — each N-column's 16 K-values are contiguous
      // (the KxN B tile in column-major order), which is what the col_major
      // fragment load expects.
      const int gk2 = k_base + c;
      const int gn = base_n + r;
      Bs[r][c] = (gk2 < K && gn < N) ? B[static_cast<size_t>(gk2) * N + gn]
                                     : pk_float2bf16(0.0f);
    }
    pk_syncthreads();

    // Build the fragments from shared memory (the fragment API abstracts the
    // per-lane mapping): A row-major 16x16 from As, B col-major 16x16 from Bs.
    pk_mma_a a_frag;
    pk_mma_b b_frag;
    nvcuda::wmma::load_matrix_sync(a_frag, &As[0][0], kTile);
    nvcuda::wmma::load_matrix_sync(b_frag, &Bs[0][0], kTile);
    nvcuda::wmma::mma_sync(acc, a_frag, b_frag, acc);
    pk_syncthreads();
  }

  // Store the fp32 accumulator tile to shared, then round to bf16 (RNE) and
  // store to global C within bounds.
  // CORRECT: mem_row_major (Cs[16][16] row-major, element (m,n) at Cs[m][n]).
  // BAD (negative path): the believable bug of storing with mem_col_major —
  // element (m,n) lands at Cs[n][m], so the whole C tile comes out TRANSPOSED
  // and golden fails decisively.
  nvcuda::wmma::store_matrix_sync(&Cs[0][0], acc, kTile,
                                  BadStore ? nvcuda::wmma::mem_col_major
                                           : nvcuda::wmma::mem_row_major);
  pk_syncthreads();
  for (int idx = lane; idx < kTile * kTile; idx += 32) {
    const int r = idx / kTile;
    const int c = idx % kTile;
    const int gm = base_m + r;
    const int gn = base_n + c;
    if (gm < M && gn < N)
      C[static_cast<size_t>(gm) * N + gn] = pk_float2bf16(Cs[r][c]);
  }
}

} // namespace

// Host-callable launch entries (mirror launch_matmul's ABI exactly). C[M,N] =
// A[M,K] @ B[K,N]; batched leading dims are flattened into M by the caller. The
// stream is void* (backend-agnostic), cast to pk_stream_t inside.
void launch_matmul_mma(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M,
                       int N, int K, void *stream) {
  dim3 block(32); // one warp per block
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(matmul_mma_kernel<false>, grid, block, 0, s, A, B, C, M, N, K);
  PK_CHECK(pk_get_last_error());
}

// Negative-path launcher: the SAME kernel with a deliberately WRONG store layout
// (BadStore=true, mem_col_major -> transposed C). Exists only so the test can
// prove a bad MMA variant fails golden and is discarded while the scalar baseline
// still passes (MMA is safely additive).
void launch_matmul_mma_bad(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C,
                           int M, int N, int K, void *stream) {
  dim3 block(32);
  dim3 grid((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(matmul_mma_kernel<true>, grid, block, 0, s, A, B, C, M, N, K);
  PK_CHECK(pk_get_last_error());
}

//===----------------------------------------------------------------------===//
// STANDALONE TEST DRIVER (test scaffolding; compiled ONLY when the test builds this
// file with -DPOLYKERNEL_MMA_MAIN). The file-based .npy validation bridge (Pinned
// contract B) mirroring lib/Runtime/cuda_run_main.cpp's shape, plus a `scalar`
// mode that launches the UNCHANGED scalar baseline (kernels/generated/matmul.cu) so
// the negative test can show the baseline still passes while a bad MMA variant fails.
// The CUDA runtime is used DIRECTLY (the thin checked-call layer below is the local
// twin of the HIP layer's HipMalloc/HipCopyH2D/HipCopyD2H/HipSync/HipFree); this file
// stays disjoint from lib/Runtime/. Not part of the kernel; guarded so the plain
// kernel build never sees it.
//===----------------------------------------------------------------------===//
#if defined(POLYKERNEL_MMA_MAIN)

#include "npy_io.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

// Scalar baseline ABI (kernels/generated/matmul.cu, Todo 10) — linked alongside.
void launch_matmul(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M, int N,
                   int K, void *stream);

namespace cpu = polykernel::cpu;

static_assert(sizeof(pk_bf16) == 2, "pk_bf16 must be 2 bytes (bf16)");

// Checked CUDA runtime call: on failure print the call + CUDA error string and
// abort cleanly (non-zero exit) — never a silent continue that corrupts results.
#define PK_CUDA_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t pk_cuda_err__ = (call);                                        \
    if (pk_cuda_err__ != cudaSuccess) {                                        \
      std::fprintf(stderr, "[polykernel-mma] %s failed: %s\n", #call,          \
                   cudaGetErrorString(pk_cuda_err__));                         \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

namespace {

/// RAII device buffer (mirrors cuda_run_main.cpp): checked alloc / checked free.
struct DevBuf {
  void *p = nullptr;
  explicit DevBuf(std::size_t bytes) {
    if (bytes)
      PK_CUDA_CHECK(cudaMalloc(&p, bytes));
  }
  ~DevBuf() {
    if (p)
      PK_CUDA_CHECK(cudaFree(p));
  }
  DevBuf(DevBuf &&o) noexcept : p(o.p) { o.p = nullptr; }
  DevBuf(const DevBuf &) = delete;
  DevBuf &operator=(const DevBuf &) = delete;
  pk_bf16 *ptr() { return static_cast<pk_bf16 *>(p); }
};

DevBuf upload(const cpu::NpyArray &arr) {
  DevBuf d(arr.data.size() * sizeof(uint16_t));
  PK_CUDA_CHECK(cudaMemcpy(d.p, arr.data.data(), arr.data.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice));
  return d;
}

void download_write(const DevBuf &d, const std::vector<int64_t> &shape,
                    const std::string &path) {
  int64_t n = 1;
  for (int64_t x : shape)
    n *= x;
  std::vector<uint16_t> host(static_cast<std::size_t>(n));
  PK_CUDA_CHECK(cudaMemcpy(host.data(), d.p, host.size() * sizeof(uint16_t),
                           cudaMemcpyDeviceToHost));
  cpu::write_npy_bf16(path, shape, host);
}

int run(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: mma_run <mma|mma_bad|scalar> A.npy B.npy C.npy\n");
    return 2;
  }
  const std::string mode = argv[1];
  cpu::NpyArray a = cpu::read_npy_bf16(argv[2]);
  cpu::NpyArray b = cpu::read_npy_bf16(argv[3]);
  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = b.shape[1];
  if (b.shape[0] != K) {
    std::fprintf(stderr, "matmul_mma: contraction mismatch A.K=%lld != B.K=%lld\n",
                 (long long)K, (long long)b.shape[0]);
    return 1;
  }
  DevBuf da = upload(a);
  DevBuf db = upload(b);
  DevBuf dc(static_cast<std::size_t>(M * N) * sizeof(uint16_t));

  // Report annotation (contract H): which path produced C.
  std::fprintf(stderr, "[polykernel-mma] path=%s M=%lld N=%lld K=%lld\n",
               mode.c_str(), (long long)M, (long long)N, (long long)K);
  if (mode == "mma")
    launch_matmul_mma(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K,
                      nullptr);
  else if (mode == "mma_bad")
    launch_matmul_mma_bad(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K,
                          nullptr);
  else if (mode == "scalar")
    launch_matmul(da.ptr(), db.ptr(), dc.ptr(), (int)M, (int)N, (int)K, nullptr);
  else
    return 2;

  PK_CUDA_CHECK(cudaDeviceSynchronize());
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
    std::fprintf(stderr, "[polykernel-mma] error: %s\n", e.what());
    return 1;
  }
}

#endif // POLYKERNEL_MMA_MAIN
