//===- attention_prefill.cu - PolyKernel FlashAttention-lite ----*- CUDA -*-===//
//
// PolyKernel generated FlashAttention-lite prefill kernel (Todo 40 / Wave 8).
//
//===----------------------------------------------------------------------===//
//
// Flash-style TILED scaled dot-product attention (prefill): for each query row i,
//     scores[i, j] = (Q[i] . K[j]) / sqrt(D)      (causal-masked: j > limit -> -inf)
//     attn[i, :]   = softmax(scores[i, :])         (over the key axis)
//     out[i, :]    = attn[i, :] @ V                (= sum_j attn[i,j] * V[j])
//
// ONLINE SOFTMAX (the FlashAttention core): the key axis is processed in tiles of
// kKeyTile keys. Each tile updates a RUNNING MAX (m) and RUNNING SUM (l) and a
// running output accumulator (o), rescaling the running state by exp(m_old - m_new)
// when the tile raises the max. The full S_q x S_kv attention matrix is therefore
// NEVER materialized - only a kKeyTile-wide score tile lives in shared memory at a
// time. This is "FlashAttention-lite": the online-softmax tiling of FlashAttention
// without the warp-level GEMM micro-kernel (one block per query row, one thread per
// head_dim lane; the per-key dot product is a block reduction). Correctness over
// peak throughput - no SOTA claim.
//
// PARALLELIZATION: gridDim = (S_q, B*H) - one block per (batch, head, query-row);
// blockDim.x = D rounded up to a warp/wave multiple, one thread per head_dim lane d
// (padding lanes idle, guarded by `t < D`). The score dot product is a reduction
// over the D lanes (warp-shuffle + shared); the output accumulation is
// embarrassingly parallel over D (each thread owns o[d]). Causal masking: query row
// i has absolute position (S_kv - S_q + i) and attends to keys j <= that limit (for
// pure prefill S_q == S_kv this is the usual j <= i). Masked positions are set to
// -FLT_MAX so exp(score - max) == 0.
//
// ROUNDING CONTRACT (matches golden fused_kv_append_attention): Q/K/V are bf16,
// upcast exactly to fp32; ALL intermediate compute (scores, softmax weights, the PV
// accumulation) is fp32 and is NOT rounded to bf16 between stages; only the FINAL
// output o[d] / l is rounded to bf16 (RNE). The online-softmax state (m, l, o) is
// fp32 throughout, exactly mirroring the golden's fp32 softmax + fp32 PV matmul.
// Written against the portable template (kernel_common.h) so it compiles UNCHANGED
// under `nvcc -DPOLYKERNEL_CUDA` and `hipcc -DPOLYKERNEL_HIP` (expf + the pk_shfl_*
// warp shuffles are device math on both; PK_FULL_MASK is backend-guarded).
//
//===----------------------------------------------------------------------===//

#include "../template/kernel_common.h"

#include <cfloat>

namespace {

// Keys processed per online-softmax tile. The score tile (kKeyTile floats) is the
// only attention-state held in shared memory, so the full attention matrix is never
// materialized regardless of S_kv.
constexpr int kKeyTile = 32;
// Max warps per block for the block-reduction scratch (blockDim.x = D <= 1024 -> at
// most 32 warps). head_dim is always far below this in practice.
constexpr int kMaxWarps = 32;

// Butterfly warp sum; result lands in every lane of the warp.
PK_DEVICE_INLINE float warp_reduce_sum(float v) {
#pragma unroll
  for (int offset = PK_WARP_SIZE / 2; offset > 0; offset >>= 1)
    v += pk_shfl_xor_sync(v, offset);
  return v;
}

// Block sum reduction for blockDim.x == D (any D <= 1024, need not be a multiple of
// the warp size): per-warp partials -> shared -> warp 0 finalizes, then the result
// is BROADCAST to every thread via warp_buf[0] (so all lanes see the same score).
PK_DEVICE float block_reduce_sum_broadcast(float v, float *warp_buf) {
  const int lane = threadIdx.x & (PK_WARP_SIZE - 1);
  const int warp = threadIdx.x / PK_WARP_SIZE;
  const int nwarps = (blockDim.x + PK_WARP_SIZE - 1) / PK_WARP_SIZE;
  v = warp_reduce_sum(v);
  if (lane == 0)
    warp_buf[warp] = v;
  pk_syncthreads();
  v = (threadIdx.x < nwarps) ? warp_buf[threadIdx.x] : 0.0f;
  if (warp == 0)
    v = warp_reduce_sum(v);
  if (threadIdx.x == 0)
    warp_buf[0] = v;
  pk_syncthreads();
  return warp_buf[0];
}

PK_GLOBAL void attention_prefill_kernel(const pk_bf16 *__restrict__ Q,
                                        const pk_bf16 *__restrict__ K,
                                        const pk_bf16 *__restrict__ V,
                                        pk_bf16 *__restrict__ O, int S_q,
                                        int S_kv, int D, float scale, int causal) {
  PK_SHARED float smem_score[kKeyTile];
  PK_SHARED float smem_warp[kMaxWarps];

  const int i = blockIdx.x;  // query row
  const int bh = blockIdx.y; // b*H + h
  const int t = threadIdx.x; // head_dim lane (active when t < D)
  const bool active = (t < D);

  const long rowBase = (static_cast<long>(bh) * S_q + i) * D;
  const long kvBase = static_cast<long>(bh) * S_kv * D;
  const float q_reg = active ? pk_bf162float(Q[rowBase + t]) : 0.0f;

  // Causal limit: query row i (absolute position S_kv - S_q + i) attends to keys
  // j <= limit. Non-causal attends to all keys (limit = S_kv - 1).
  const int limit = causal ? (S_kv - S_q + i) : (S_kv - 1);

  float m = -FLT_MAX; // running max
  float l = 0.0f;     // running sum
  float o_reg = 0.0f; // running output for lane t

  for (int t0 = 0; t0 < S_kv; t0 += kKeyTile) {
    const int tileLen = min(kKeyTile, S_kv - t0);

    // Phase 1: scores for this key tile (each a D-wide block reduction), masked.
    for (int jj = 0; jj < tileLen; ++jj) {
      const int j = t0 + jj;
      float contrib = 0.0f;
      if (active && j <= limit)
        contrib = q_reg * pk_bf162float(K[kvBase + static_cast<long>(j) * D + t]);
      const float dot = block_reduce_sum_broadcast(contrib, smem_warp);
      if (t == 0)
        smem_score[jj] = (j <= limit) ? (dot * scale) : -FLT_MAX;
    }
    pk_syncthreads(); // publish smem_score[0..tileLen) to all lanes

    // Phase 2: online-softmax update. tile_max / tile_sum are derived from shared
    // (identical in every lane), so m/l stay consistent across the block.
    float tile_max = -FLT_MAX;
    for (int jj = 0; jj < tileLen; ++jj)
      tile_max = fmaxf(tile_max, smem_score[jj]);
    const float m_new = fmaxf(m, tile_max);
    const float correction = (m == -FLT_MAX) ? 0.0f : expf(m - m_new);

    float tile_sum = 0.0f;
    for (int jj = 0; jj < tileLen; ++jj) {
      const float w = expf(smem_score[jj] - m_new);
      smem_score[jj] = w; // reuse the tile buffer for the softmax weights
      tile_sum += w;
    }
    l = l * correction + tile_sum;

    // Phase 3: PV accumulation for lane t (parallel over D, no reduction).
    float o_acc = o_reg * correction;
    for (int jj = 0; jj < tileLen; ++jj) {
      const int j = t0 + jj;
      const float v = active ? pk_bf162float(V[kvBase + static_cast<long>(j) * D + t])
                             : 0.0f;
      o_acc += smem_score[jj] * v;
    }
    o_reg = o_acc;
    m = m_new;
    pk_syncthreads(); // tile buffer reused next iteration
  }

  if (active)
    O[rowBase + t] = pk_float2bf16(l > 0.0f ? o_reg / l : 0.0f);
}

} // namespace

// Host-callable launch entry (the launch_*-style ABI). Q [B,H,S_q,D], K/V
// [B,H,S_kv,D] (the FULL key/value set - the appended KV cache), O [B,H,S_q,D].
// scale = 1/sqrt(D); causal != 0 enables the causal mask. blockDim.x is D rounded
// UP to a multiple of the warp/wave size (32): one thread per head_dim lane, with
// the padding lanes idle (the kernel guards every access with `t < D`). Rounding up
// guarantees full waves so the warp shuffles are well-defined (a partial wave - e.g.
// D = 40 launched as 40 threads - reads inactive shuffle lanes and faults on RDNA3).
// D <= 1024 (always true for transformer head dims). The stream is passed as void*
// to keep the signature backend-agnostic and cast to the backend stream type
// (pk_stream_t) inside.
void launch_attention_prefill(const pk_bf16 *Q, const pk_bf16 *K,
                              const pk_bf16 *V, pk_bf16 *O, int B, int H, int S_q,
                              int S_kv, int D, float scale, int causal,
                              void *stream) {
  const int blockThreads = ((D + PK_WARP_SIZE - 1) / PK_WARP_SIZE) * PK_WARP_SIZE;
  dim3 grid(static_cast<unsigned>(S_q), static_cast<unsigned>(B * H));
  dim3 block(static_cast<unsigned>(blockThreads));
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(attention_prefill_kernel, grid, block, 0, s, Q, K, V, O, S_q, S_kv, D,
            scale, causal);
  PK_CHECK(pk_get_last_error());
}
