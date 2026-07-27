//===- attention.cpp - CPU reference attention + KV-cache -------*- C++ -*-===//
//
// PolyKernel CPU reference attention driver (Todo 40 / Wave 8).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B) for the two Wave-8 attention
// ops: the LOCAL EXECUTABLE SPEC the CUDA/HIP kernels share their algorithm with.
// It implements the SAME math + rounding contract as tests/golden/golden.py
// (fused_kv_append_attention): bf16 inputs upcast exactly to fp32, ALL intermediate
// compute (scores, softmax weights, PV accumulation) in fp32 with NO inter-stage
// bf16 rounding, and only the FINAL attention output rounded back to bf16 (RNE).
// The updated KV cache is a verbatim bf16 concatenation (no arithmetic).
//
// Two subcommands (argv[1]):
//   attention prefill Q.npy K.npy V.npy OUT.npy [--causal N]
//       Scaled dot-product attention. Q [B,H,S_q,D], K/V [B,H,S_kv,D] (the full
//       key/value set, i.e. the appended cache). scores = Q@K^T / sqrt(D); with
//       --causal 1 (default) query row i (absolute position S_kv - S_q + i) attends
//       only to keys j <= that limit (masked positions get -inf so their softmax
//       weight is 0); out = softmax(scores) @ V, rounded to bf16. --causal 0
//       disables masking (the NEGATIVE path: leaks future tokens).
//   attention append CACHE.npy NEW.npy OUT.npy
//       KV-cache append: OUT [B,H,S_cache+S_new,D] = concat(CACHE, NEW) along the
//       sequence axis (axis 2) - a verbatim bf16 copy.
//
// tests/kernels/test_attention.py drives this executable and compares the output to
// the golden via metrics.assert_correct.
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h" // bf16_to_float / float_to_bf16 (RNE, matches ml_dtypes)
#include "npy_io.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using polykernel::cpu::NpyArray;
using polykernel::cpu::read_npy_bf16;
using polykernel::cpu::write_npy_bf16;
using polykernel::cpu::bf16_to_float;
using polykernel::cpu::float_to_bf16;

namespace {

void usage(const char *prog) {
  std::fprintf(stderr,
               "usage:\n"
               "  %s prefill Q.npy K.npy V.npy OUT.npy [--causal N]\n"
               "  %s append CACHE.npy NEW.npy OUT.npy\n",
               prog, prog);
}

/// Return the dim at `axis` (rank-checked), or -1 if the rank is too small.
int64_t dim(const NpyArray &a, int axis) {
  return axis < static_cast<int>(a.shape.size()) ? a.shape[axis] : -1;
}

/// KV-cache append: out [B,H,Sc+Sn,D] = concat(cache [B,H,Sc,D], new [B,H,Sn,D]).
int run_append(int argc, char **argv) {
  if (argc != 5) {
    usage(argv[0]);
    return 2;
  }
  NpyArray cache = read_npy_bf16(argv[2]);
  NpyArray neu = read_npy_bf16(argv[3]);
  const int64_t B = dim(cache, 0), H = dim(cache, 1), Sc = dim(cache, 2),
                D = dim(cache, 3);
  const int64_t Sn = dim(neu, 2);
  if (B < 0 || dim(neu, 0) != B || dim(neu, 1) != H || dim(neu, 3) != D) {
    std::fprintf(stderr, "append: cache/new shape mismatch\n");
    return 1;
  }
  const int64_t BH = B * H;
  std::vector<uint16_t> out(static_cast<size_t>(BH * (Sc + Sn) * D));
  for (int64_t bh = 0; bh < BH; ++bh) {
    const uint16_t *c = cache.data.data() + bh * Sc * D;
    const uint16_t *n = neu.data.data() + bh * Sn * D;
    uint16_t *o = out.data() + bh * (Sc + Sn) * D;
    for (int64_t s = 0; s < Sc; ++s)
      for (int64_t d = 0; d < D; ++d)
        o[s * D + d] = c[s * D + d];
    for (int64_t s = 0; s < Sn; ++s)
      for (int64_t d = 0; d < D; ++d)
        o[(Sc + s) * D + d] = n[s * D + d];
  }
  write_npy_bf16(argv[4], {B, H, Sc + Sn, D}, out);
  return 0;
}

/// Scaled dot-product attention (fp32 throughout, bf16 output - matches golden).
int run_prefill(int argc, char **argv) {
  if (argc < 6) {
    usage(argv[0]);
    return 2;
  }
  int causal = 1;
  for (int i = 6; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--causal" && i + 1 < argc)
      causal = std::stoi(argv[++i]);
    else {
      usage(argv[0]);
      return 2;
    }
  }
  NpyArray Q = read_npy_bf16(argv[2]);
  NpyArray K = read_npy_bf16(argv[3]);
  NpyArray V = read_npy_bf16(argv[4]);
  const int64_t B = dim(Q, 0), H = dim(Q, 1), Sq = dim(Q, 2), D = dim(Q, 3);
  const int64_t Skv = dim(K, 2);
  if (B < 0 || dim(K, 0) != B || dim(K, 1) != H || dim(K, 3) != D ||
      dim(V, 0) != B || dim(V, 1) != H || dim(V, 2) != Skv || dim(V, 3) != D) {
    std::fprintf(stderr, "prefill: Q/K/V shape mismatch\n");
    return 1;
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  const int64_t BH = B * H;
  std::vector<uint16_t> out(static_cast<size_t>(BH * Sq * D));
  std::vector<float> score(static_cast<size_t>(Skv));

  for (int64_t bh = 0; bh < BH; ++bh) {
    const uint16_t *q = Q.data.data() + bh * Sq * D;
    const uint16_t *k = K.data.data() + bh * Skv * D;
    const uint16_t *v = V.data.data() + bh * Skv * D;
    uint16_t *o = out.data() + bh * Sq * D;
    for (int64_t i = 0; i < Sq; ++i) {
      const int64_t limit = causal ? (Skv - Sq + i) : (Skv - 1);
      // scores = Q[i] . K[j] / sqrt(D), causal-masked (fp32, no bf16 rounding).
      float m = -HUGE_VALF;
      for (int64_t j = 0; j < Skv; ++j) {
        if (j > limit) {
          score[j] = -HUGE_VALF;
          continue;
        }
        float dot = 0.0f;
        for (int64_t d = 0; d < D; ++d)
          dot += bf16_to_float(q[i * D + d]) * bf16_to_float(k[j * D + d]);
        score[j] = dot * scale;
        m = std::max(m, score[j]);
      }
      // softmax weights (fp32) + their sum.
      float sum = 0.0f;
      for (int64_t j = 0; j < Skv; ++j) {
        score[j] = std::exp(score[j] - m);
        sum += score[j];
      }
      // out[i, d] = sum_j weight[j] * V[j, d], rounded to bf16 only here.
      const float inv = 1.0f / sum;
      for (int64_t d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int64_t j = 0; j < Skv; ++j)
          acc += score[j] * bf16_to_float(v[j * D + d]);
        o[i * D + d] = float_to_bf16(acc * inv);
      }
    }
  }
  write_npy_bf16(argv[5], {B, H, Sq, D}, out);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }
  try {
    // The subcommand parsers index the original argv: argv[1] is the subcommand,
    // argv[2..] are the .npy paths (matching their usage() strings).
    const std::string sub = argv[1];
    if (sub == "prefill")
      return run_prefill(argc, argv);
    if (sub == "append")
      return run_append(argc, argv);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  usage(argv[0]);
  return 2;
}
