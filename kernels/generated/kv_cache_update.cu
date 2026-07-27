//===- kv_cache_update.cu - PolyKernel KV-cache append kernel ---*- CUDA -*-===//
//
// PolyKernel generated KV-cache append kernel (Todo 40 / Wave 8).
//
//===----------------------------------------------------------------------===//
//
// Appends the NEW key (or value) rows onto a KV cache along the sequence axis:
//     out[b, h, s, d] = (s < S_cache) ? cache[b, h, s, d]
//                                     : new  [b, h, s - S_cache, d]
// Layout is [B, H, S, D] (batch, heads, sequence, head_dim), row-major; the cache
// holds S_cache past tokens and `new` holds the S_new freshly-projected tokens, so
// the updated cache has S_cache + S_new rows. This is the standalone KV-cache half
// of `polykernel.kv_cache_update`; the attention_prefill kernel consumes the
// appended cache (and `fused_kv_append_attention` fuses the two — see
// lib/Passes/FuseKvAppendAttention.cpp).
//
// ROUNDING CONTRACT (matches golden fused_kv_append_attention): the updated cache
// is just the bf16 concatenation of the bf16 inputs (new_k/new_v = concat(cache,
// new)), so each output element is a verbatim bf16 copy — no arithmetic, no
// re-rounding. One thread copies one element; bounds are exact (the grid covers
// B*H*(S_cache+S_new)*D threads). Written against the portable template
// (kernel_common.h) so it compiles UNCHANGED under `nvcc -DPOLYKERNEL_CUDA` and
// `hipcc -DPOLYKERNEL_HIP` (pure index math + bf16 copy; nothing backend-specific).
//
//===----------------------------------------------------------------------===//

#include "../template/kernel_common.h"

namespace {

// One thread per output element. The grid is a flat 1-D launch over the whole
// updated-cache tensor (B*H*(S_cache+S_new)*D elements); a 1-D grid keeps the
// launch simple and bounds-exact for any shape (no tile multiples required).
PK_GLOBAL void kv_cache_update_kernel(const pk_bf16 *__restrict__ cache,
                                      const pk_bf16 *__restrict__ new_rows,
                                      pk_bf16 *__restrict__ out, int S_cache,
                                      int S_new, int D, long total) {
  const long gid = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (gid >= total)
    return;

  // Decompose the flat index into (bh, s, d): the sequence axis is the second-fastest
  // stride (D is fastest). bh = b*H+h is kept combined (the copy is per-(bh,s,d)).
  const int d = static_cast<int>(gid % D);
  const long bh_s = gid / D;
  const int s = static_cast<int>(bh_s % (S_cache + S_new));
  const long bh = bh_s / (S_cache + S_new);

  // First S_cache rows come from the cache; the remaining S_new rows from `new`.
  const long stride = static_cast<long>(S_cache + S_new) * D;
  if (s < S_cache)
    out[gid] = cache[bh * static_cast<long>(S_cache) * D +
                     static_cast<long>(s) * D + d];
  else
    out[gid] = new_rows[bh * static_cast<long>(S_new) * D +
                        static_cast<long>(s - S_cache) * D + d];
  (void)stride;
}

} // namespace

// Host-callable launch entry (the launch_*-style ABI). Appends `new_rows`
// [B,H,S_new,D] onto `cache` [B,H,S_cache,D] along the sequence axis, writing the
// updated cache [B,H,S_cache+S_new,D] into `out` (which the caller sizes to the
// updated shape). The stream is passed as void* to keep the signature
// backend-agnostic and cast to the backend stream type (pk_stream_t) inside.
void launch_kv_cache_update(const pk_bf16 *cache, const pk_bf16 *new_rows,
                            pk_bf16 *out, int B, int H, int S_cache, int S_new,
                            int D, void *stream) {
  const long total =
      static_cast<long>(B) * H * (S_cache + S_new) * D;
  constexpr int kThreads = 256;
  const long blocks = (total + kThreads - 1) / kThreads;
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(kv_cache_update_kernel, dim3(static_cast<unsigned>(blocks)),
            dim3(kThreads), 0, s, cache, new_rows, out, S_cache, S_new, D, total);
  PK_CHECK(pk_get_last_error());
}
