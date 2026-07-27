//===- matmul_quant.cpp - CPU reference quantized MatMul driver --*- C++ -*-===//
//
// PolyKernel CPU reference quantized-matmul driver (Todo 41 / Wave 8 elite).
//
//===----------------------------------------------------------------------===//
//
// File-based validation bridge (Pinned contract B) for the QUANTIZED matmul, the
// host sibling of kernels/generated/matmul_int8.cu. Two modes:
//
//   matmul_quant int8   A.npy Wq.npy scale.npy C.npy
//       int8 WEIGHT-ONLY: A is bf16 [M,K], Wq is int8 [K,N], scale is fp32 [N]
//       (per output channel). Dequantize-and-multiply, fp32 accumulate, bf16 out:
//           C[m,n] = bf16( sum_k fp32(A[m,k]) * (fp32(Wq[k,n]) * scale[n]) )
//       This is the SAME math + rounding as tests/golden/quant_golden.matmul_int8
//       and the GPU kernel (only the fp32 K-reduction order differs).
//
//   matmul_quant fp8sim <e4m3|e5m2> A.npy W.npy C.npy
//       fp8 weight SIMULATION (NO fp8 hardware): A is bf16 [M,K], W is bf16 [K,N].
//       Round each weight onto the fp8 e4m3fn / e5m2 grid (portable RNE round), then
//           C[m,n] = bf16( sum_k fp32(A[m,k]) * fp8(W[k,n]) )
//       matching tests/golden/quant_golden.matmul_fp8_sim (ml_dtypes float8).
//
// tests/kernels/test_quant.py drives this executable and compares the output to the
// quantization-aware golden via quant_golden.assert_quant_correct (relaxed: cosine
// >= 0.99, max_rel_err <= 5e-2, pcc >= 0.99).
//
//===----------------------------------------------------------------------===//

#include "cpu_reference.h"
#include "npy_io.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using polykernel::cpu::NpyArray;
using polykernel::cpu::read_npy_bf16;
using polykernel::cpu::write_npy_bf16;

namespace {

//===----------------------------------------------------------------------===//
// Minimal raw .npy reader for the non-bf16 operands (int8 '|i1', fp32 '<f4').
// bf16 A/W/C reuse read_npy_bf16 / write_npy_bf16 (the '<V2' bridge). Mirrors the
// reader in matmul_int8.cu's test driver.
//===----------------------------------------------------------------------===//

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
  if (d != descr)
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

//===----------------------------------------------------------------------===//
// Portable fp8 (e4m3fn / e5m2) round-to-nearest-even SIMULATION. Returns the fp8
// grid point as fp32. This is what lets the CPU ref match the golden's ml_dtypes
// float8 cast WITHOUT any fp8 hardware: round the fp32 magnitude to `mbits`
// mantissa bits at its own exponent (RNE via nearbyint, FE_TONEAREST default),
// saturating to the format's max-finite. Subnormals use the fixed smallest ulp.
//===----------------------------------------------------------------------===//

float fp8_round(float x, int mbits, int bias, float fmax, bool has_inf) {
  if (std::isnan(x))
    return x;
  float mag = std::fabs(x);
  if (std::isinf(mag))
    return has_inf ? x : std::copysign(fmax, x); // e5m2 keeps inf; e4m3fn saturates
  if (mag == 0.0f)
    return x; // preserve signed zero
  if (mag > fmax)
    mag = fmax; // saturate to max-finite
  const int emin = 1 - bias; // smallest normal exponent
  int e = 0;
  std::frexp(mag, &e); // mag = [0.5, 1) * 2^e  ->  unbiased exponent = e - 1
  const int exp = e - 1;
  const float ulp = (exp < emin) ? std::ldexp(1.0f, emin - mbits)
                                 : std::ldexp(1.0f, exp - mbits);
  float r = std::nearbyint(mag / ulp) * ulp; // RNE to the fp8 grid
  if (r > fmax)
    r = fmax; // rounding up may exceed fmax; clamp
  return std::copysign(r, x);
}

// e4m3fn: 3 mantissa bits, bias 7, max-finite 448, NO inf (saturates).
// e5m2:   2 mantissa bits, bias 15, max-finite 57344, IEEE inf/nan.
float fp8_sim(float x, const std::string &fmt) {
  if (fmt == "e4m3")
    return fp8_round(x, 3, 7, 448.0f, false);
  if (fmt == "e5m2")
    return fp8_round(x, 2, 15, 57344.0f, true);
  throw std::runtime_error("unknown fp8 format '" + fmt + "' (e4m3|e5m2)");
}

void usage(const char *prog) {
  std::fprintf(stderr,
               "usage: %s int8 A.npy Wq.npy scale.npy C.npy\n"
               "       %s fp8sim <e4m3|e5m2> A.npy W.npy C.npy\n",
               prog, prog);
}

int run_int8(char **argv) {
  NpyArray a = read_npy_bf16(argv[0]);          // bf16 [M, K]
  NpyRaw wq = read_npy_raw(argv[1], "|i1", 1);  // int8 [K, N]
  NpyRaw sc = read_npy_raw(argv[2], "<f4", 4);  // fp32 [N]
  const int8_t *wq_p = reinterpret_cast<const int8_t *>(wq.bytes.data());
  const float *sc_p = reinterpret_cast<const float *>(sc.bytes.data());

  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = wq.shape[1];
  if (wq.shape[0] != K || sc.shape[0] != N) {
    std::fprintf(stderr, "matmul_quant int8: shape mismatch (K=%lld N=%lld)\n",
                 (long long)K, (long long)N);
    return 1;
  }
  std::fprintf(stderr,
               "[polykernel-quant-cpu] path=int8_weight_only dtype=bf16xint8 "
               "scale=per-channel M=%lld N=%lld K=%lld\n",
               (long long)M, (long long)N, (long long)K);

  std::vector<uint16_t> c(static_cast<std::size_t>(M * N));
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      const float s = sc_p[n];
      for (int64_t k = 0; k < K; ++k)
        acc += polykernel::cpu::bf16_to_float(a.data[m * K + k]) *
               (static_cast<float>(wq_p[k * N + n]) * s);
      c[m * N + n] = polykernel::cpu::float_to_bf16(acc);
    }
  }
  std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
  out_shape.push_back(N);
  write_npy_bf16(argv[3], out_shape, c);
  return 0;
}

int run_fp8sim(const std::string &fmt, char **argv) {
  NpyArray a = read_npy_bf16(argv[0]); // bf16 [M, K]
  NpyArray w = read_npy_bf16(argv[1]); // bf16 [K, N]
  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = w.shape[1];
  if (w.shape[0] != K) {
    std::fprintf(stderr, "matmul_quant fp8sim: contraction mismatch\n");
    return 1;
  }
  std::fprintf(stderr,
               "[polykernel-quant-cpu] path=fp8_sim fmt=%s dtype=bf16xfp8(sim) "
               "M=%lld N=%lld K=%lld\n",
               fmt.c_str(), (long long)M, (long long)N, (long long)K);

  std::vector<uint16_t> c(static_cast<std::size_t>(M * N));
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        float w8 = fp8_sim(polykernel::cpu::bf16_to_float(w.data[k * N + n]), fmt);
        acc += polykernel::cpu::bf16_to_float(a.data[m * K + k]) * w8;
      }
      c[m * N + n] = polykernel::cpu::float_to_bf16(acc);
    }
  }
  std::vector<int64_t> out_shape(a.shape.begin(), a.shape.end() - 1);
  out_shape.push_back(N);
  write_npy_bf16(argv[2], out_shape, c);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }
  try {
    const std::string mode = argv[1];
    if (mode == "int8" && argc == 6)
      return run_int8(argv + 2);
    if (mode == "fp8sim" && argc == 6)
      return run_fp8sim(argv[2], argv + 3);
    usage(argv[0]);
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
