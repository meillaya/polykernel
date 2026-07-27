//===- hip_run_main.cpp - HIP kernel launcher driver -------------*- C++ -*-===//
//
// PolyKernel HIP launcher driver (Todo 20 / Wave 4).
//
//===----------------------------------------------------------------------===//
//
// The GPU sibling of the CPU-reference drivers (kernels/cpu/cpu_ref_*.cpp): it is
// the file-based .npy validation bridge (Pinned contract B) executed on the real
// gfx1101 GPU. For one op it reads bf16 .npy inputs, allocates device buffers +
// copies H2D (via the error-checked HipRuntime layer), launches the GENERATED
// portable kernel (kernels/generated/<op>.cu, compiled by hipcc -DPOLYKERNEL_HIP
// - the SAME source the CUDA backend uses), synchronizes, copies D2H and writes
// the bf16 .npy output. tests/kernels/test_hip_run.py drives this executable and
// compares the output to the golden via metrics.assert_correct.
//
// Built by the test with hipcc (NOT CMake - it links device code):
//   hipcc --offload-arch=gfx1101 -DPOLYKERNEL_HIP -Ikernels/template \
//         -Iinclude -Ikernels/cpu \
//         lib/Runtime/HipRuntime.cpp lib/Runtime/hip_run_main.cpp \
//         kernels/cpu/npy_io.cpp kernels/generated/<ops>.cu -o hip_run
//
// Usage (argv[1] = op):
//   hip_run rmsnorm IN OUT [--weight W.npy] [--epsilon E]
//   hip_run gelu IN OUT
//   hip_run matmul A B C
//   hip_run softmax IN OUT [--axis A]        (last-axis only; A == -1 or rank-1)
//   hip_run fused_rmsnorm_matmul X W OUT [--epsilon E]
//   hip_run fused_matmul_bias_gelu A B BIAS OUT
//   hip_run neg                              (negative: invalid launch config)
//
//===----------------------------------------------------------------------===//

#include "kernel_common.h" // pk_bf16, PK_GLOBAL, pk_float2bf16 (-DPOLYKERNEL_HIP)
#include "npy_io.h"
#include "PolyKernel/Runtime/HipRuntime.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace cpu = polykernel::cpu;
namespace runtime = polykernel::runtime;

// bf16 is 2 bytes on both backends; the .npy bridge stores raw uint16 bf16 bits,
// so a uint16 payload reinterprets exactly as pk_bf16 (__hip_bfloat16) on device.
static_assert(sizeof(pk_bf16) == 2, "pk_bf16 must be 2 bytes (bf16)");

// launch_* ABI from kernels/generated/<op>.cu (host-callable; the stream is a
// backend-agnostic void*). Types match the .cu exactly (pk_bf16 == __hip_bfloat16
// under -DPOLYKERNEL_HIP), so the C++ mangled names link.
void launch_rmsnorm(const pk_bf16 *input, const pk_bf16 *weight, pk_bf16 *output,
                    int rows, int cols, float epsilon, void *stream);
void launch_gelu(const pk_bf16 *input, pk_bf16 *output, int n, void *stream);
void launch_matmul(const pk_bf16 *A, const pk_bf16 *B, pk_bf16 *C, int M, int N,
                   int K, void *stream);
void launch_softmax(const pk_bf16 *input, pk_bf16 *output, int rows, int cols,
                    void *stream);
void launch_fused_rmsnorm_matmul(const pk_bf16 *input, const pk_bf16 *weight,
                                 pk_bf16 *output, int M, int N, int K,
                                 float epsilon, void *stream);
void launch_fused_matmul_bias_gelu(const pk_bf16 *a, const pk_bf16 *b,
                                   const pk_bf16 *bias, pk_bf16 *output, int M,
                                   int N, int K, void *stream);

namespace {

int usage(const char *msg) {
  std::fprintf(stderr, "usage: hip_run <op> ...\n  %s\n", msg);
  return 2;
}

/// RAII device buffer: checked alloc on construction (bytes==0 -> null, no alloc),
/// checked free on destruction. Movable, not copyable (single owner).
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

/// Allocate a device buffer for `arr` + copy its bf16 payload host->device.
DevBuf upload(const cpu::NpyArray &arr) {
  DevBuf d(arr.data.size() * sizeof(uint16_t));
  runtime::HipCopyH2D(d.p, arr.data.data(), arr.data.size() * sizeof(uint16_t));
  return d;
}

/// Copy a device buffer's bf16 payload back to host + write it as a '<V2' .npy.
void download_write(const DevBuf &d, const std::vector<int64_t> &shape,
                    const std::string &path) {
  int64_t n = 1;
  for (int64_t x : shape)
    n *= x;
  std::vector<uint16_t> host(static_cast<std::size_t>(n));
  runtime::HipCopyD2H(host.data(), d.p, host.size() * sizeof(uint16_t));
  cpu::write_npy_bf16(path, shape, host);
}

/// Output shape for a matmul-family op: A's leading dims (all but K) + N.
std::vector<int64_t> matmul_out_shape(const std::vector<int64_t> &a, int64_t N) {
  std::vector<int64_t> s(a.begin(), a.end() - 1);
  s.push_back(N);
  return s;
}

int run_rmsnorm(int argc, char **argv) {
  if (argc < 4)
    return usage("rmsnorm IN OUT [--weight W.npy] [--epsilon E]");
  std::string weight_path;
  float epsilon = 1e-5f;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--weight" && i + 1 < argc)
      weight_path = argv[++i];
    else if (a == "--epsilon" && i + 1 < argc)
      epsilon = std::stof(argv[++i]);
    else
      return usage("rmsnorm IN OUT [--weight W.npy] [--epsilon E]");
  }
  cpu::NpyArray in = cpu::read_npy_bf16(argv[2]);
  const int64_t cols = in.shape.back();
  const int64_t rows = in.size() / cols;
  const bool has_w = !weight_path.empty();
  cpu::NpyArray w = has_w ? cpu::read_npy_bf16(weight_path) : cpu::NpyArray{};
  DevBuf din = upload(in);
  DevBuf dw(has_w ? w.data.size() * sizeof(uint16_t) : 0);
  const pk_bf16 *wptr = nullptr;
  if (has_w) {
    runtime::HipCopyH2D(dw.p, w.data.data(), w.data.size() * sizeof(uint16_t));
    wptr = static_cast<const pk_bf16 *>(dw.p);
  }
  DevBuf dout(in.data.size() * sizeof(uint16_t));
  launch_rmsnorm(din.ptr(), wptr, dout.ptr(), static_cast<int>(rows),
                 static_cast<int>(cols), epsilon, nullptr);
  runtime::HipSync();
  download_write(dout, in.shape, argv[3]);
  return 0;
}

int run_gelu(int argc, char **argv) {
  if (argc != 4)
    return usage("gelu IN OUT");
  cpu::NpyArray in = cpu::read_npy_bf16(argv[2]);
  DevBuf din = upload(in);
  DevBuf dout(in.data.size() * sizeof(uint16_t));
  launch_gelu(din.ptr(), dout.ptr(), static_cast<int>(in.size()), nullptr);
  runtime::HipSync();
  download_write(dout, in.shape, argv[3]);
  return 0;
}

int run_matmul(int argc, char **argv) {
  if (argc != 5)
    return usage("matmul A B C");
  cpu::NpyArray a = cpu::read_npy_bf16(argv[2]);
  cpu::NpyArray b = cpu::read_npy_bf16(argv[3]);
  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = b.shape[1];
  if (b.shape[0] != K) {
    std::fprintf(stderr, "matmul: contraction mismatch A.K=%lld != B.K=%lld\n",
                 (long long)K, (long long)b.shape[0]);
    return 1;
  }
  DevBuf da = upload(a);
  DevBuf db = upload(b);
  DevBuf dc(static_cast<std::size_t>(M * N) * sizeof(uint16_t));
  launch_matmul(da.ptr(), db.ptr(), dc.ptr(), static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), nullptr);
  runtime::HipSync();
  download_write(dc, matmul_out_shape(a.shape, N), argv[4]);
  return 0;
}

int run_softmax(int argc, char **argv) {
  if (argc < 4)
    return usage("softmax IN OUT [--axis A]");
  int axis = -1;
  for (int i = 4; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--axis" && i + 1 < argc)
      axis = std::stoi(argv[++i]);
    else
      return usage("softmax IN OUT [--axis A]");
  }
  cpu::NpyArray in = cpu::read_npy_bf16(argv[2]);
  if (axis != -1 && axis != static_cast<int>(in.shape.size()) - 1) {
    std::fprintf(stderr, "softmax: only last-axis reduction supported (axis=-1)\n");
    return 1;
  }
  const int64_t cols = in.shape.back();
  const int64_t rows = in.size() / cols;
  DevBuf din = upload(in);
  DevBuf dout(in.data.size() * sizeof(uint16_t));
  launch_softmax(din.ptr(), dout.ptr(), static_cast<int>(rows),
                 static_cast<int>(cols), nullptr);
  runtime::HipSync();
  download_write(dout, in.shape, argv[3]);
  return 0;
}

int run_fused_rmsnorm_matmul(int argc, char **argv) {
  if (argc < 5)
    return usage("fused_rmsnorm_matmul X W OUT [--epsilon E]");
  float epsilon = 1e-5f;
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--epsilon" && i + 1 < argc)
      epsilon = std::stof(argv[++i]);
    else
      return usage("fused_rmsnorm_matmul X W OUT [--epsilon E]");
  }
  cpu::NpyArray x = cpu::read_npy_bf16(argv[2]);
  cpu::NpyArray w = cpu::read_npy_bf16(argv[3]);
  const int64_t K = x.shape.back();
  const int64_t M = x.size() / K;
  const int64_t N = w.shape[1];
  if (w.shape[0] != K) {
    std::fprintf(stderr, "fused_rmsnorm_matmul: K mismatch X.K=%lld != W.K=%lld\n",
                 (long long)K, (long long)w.shape[0]);
    return 1;
  }
  DevBuf dx = upload(x);
  DevBuf dw = upload(w);
  DevBuf dout(static_cast<std::size_t>(M * N) * sizeof(uint16_t));
  launch_fused_rmsnorm_matmul(dx.ptr(), dw.ptr(), dout.ptr(), static_cast<int>(M),
                              static_cast<int>(N), static_cast<int>(K), epsilon,
                              nullptr);
  runtime::HipSync();
  download_write(dout, matmul_out_shape(x.shape, N), argv[4]);
  return 0;
}

int run_fused_matmul_bias_gelu(int argc, char **argv) {
  if (argc != 6)
    return usage("fused_matmul_bias_gelu A B BIAS OUT");
  cpu::NpyArray a = cpu::read_npy_bf16(argv[2]);
  cpu::NpyArray b = cpu::read_npy_bf16(argv[3]);
  cpu::NpyArray bias = cpu::read_npy_bf16(argv[4]);
  const int64_t K = a.shape.back();
  const int64_t M = a.size() / K;
  const int64_t N = b.shape[1];
  if (b.shape[0] != K || bias.size() != N) {
    std::fprintf(stderr, "fused_matmul_bias_gelu: shape mismatch "
                         "(A.K=%lld B.K=%lld N=%lld bias=%lld)\n",
                 (long long)K, (long long)b.shape[0], (long long)N,
                 (long long)bias.size());
    return 1;
  }
  DevBuf da = upload(a);
  DevBuf db = upload(b);
  DevBuf dbias = upload(bias);
  DevBuf dout(static_cast<std::size_t>(M * N) * sizeof(uint16_t));
  launch_fused_matmul_bias_gelu(da.ptr(), db.ptr(), dbias.ptr(), dout.ptr(),
                                static_cast<int>(M), static_cast<int>(N),
                                static_cast<int>(K), nullptr);
  runtime::HipSync();
  download_write(dout, matmul_out_shape(a.shape, N), argv[5]);
  return 0;
}

// A trivial kernel for the negative path (compiled by hipcc, so device code).
PK_GLOBAL void neg_probe_kernel(pk_bf16 *out) {
  if (threadIdx.x == 0)
    out[0] = pk_float2bf16(1.0f);
}

int run_neg() {
  // Negative QA: an intentionally INVALID launch config (2048 threads/block > the
  // gfx1101 device max of 1024). The launch is asynchronous, so the error surfaces
  // at hipGetLastError / hipDeviceSynchronize and MUST be caught + reported - never
  // a silent wrong result or a crash. Raw (non-aborting) HIP calls are used here so
  // we can OBSERVE the error code rather than exit on it.
  pk_bf16 *d = nullptr;
  const hipError_t merr =
      hipMalloc(reinterpret_cast<void **>(&d), 16 * sizeof(pk_bf16));
  if (merr != hipSuccess) {
    std::fprintf(stderr, "[polykernel-hip] neg: hipMalloc failed: %s\n",
                 runtime::HipErrorString(merr));
    return 1;
  }
  neg_probe_kernel<<<dim3(1), dim3(2048), 0, nullptr>>>(d); // INVALID block dim
  const hipError_t lerr =
      runtime::ProbeLaunchError("neg out-of-bounds block (2048 > 1024)");
  const hipError_t serr = hipDeviceSynchronize();
  std::fprintf(stderr, "[polykernel-hip] neg: launch-config error = %s\n",
               runtime::HipErrorString(lerr));
  std::fprintf(stderr, "[polykernel-hip] neg: sync error          = %s\n",
               runtime::HipErrorString(serr));
  (void)hipFree(d); // raw: a sticky error must not abort cleanup here
  if (lerr == hipSuccess && serr == hipSuccess) {
    std::fprintf(stderr, "[polykernel-hip] neg: FAIL - invalid launch produced "
                         "NO error (silent corruption risk)\n");
    return 2;
  }
  std::fprintf(stderr, "[polykernel-hip] neg: PASS - invalid launch CAUGHT + "
                       "reported; no output written, no crash\n");
  return 1; // non-zero: the test asserts the error path is reached
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage("<op> ...");
  const std::string op = argv[1];
  try {
    if (op == "rmsnorm")
      return run_rmsnorm(argc, argv);
    if (op == "gelu")
      return run_gelu(argc, argv);
    if (op == "matmul")
      return run_matmul(argc, argv);
    if (op == "softmax")
      return run_softmax(argc, argv);
    if (op == "fused_rmsnorm_matmul")
      return run_fused_rmsnorm_matmul(argc, argv);
    if (op == "fused_matmul_bias_gelu")
      return run_fused_matmul_bias_gelu(argc, argv);
    if (op == "neg")
      return run_neg();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "[polykernel-hip] error: %s\n", e.what());
    return 1;
  }
  return usage("unknown op");
}
