//===- KernelReport.cpp - Assemble the per-kernel report --------*- C++ -*-===//
//
// BuildReport wires the three models together and applies the rule tables:
//   bottleneck: spills>0 -> latency; else memory-bound roofline -> memory;
//               else occupancy<50% -> latency; else compute.
//   suggested_fixes (in rule order):
//     spills>0                          -> reduce register pressure
//     register-limited AND occ<100%     -> reduce unroll/block size
//     occupancy<50%                     -> increase occupancy
//     memory-bound                      -> wider vectorized loads / fuse
// ToJson emits the contract H schema with exact field names + enum values.
//
//===----------------------------------------------------------------------===//

#include "KernelReport.h"

#include <cstdio>
#include <string>

namespace polykernel::analysis {

std::string_view BackendName(Backend backend) {
  switch (backend) {
  case Backend::cuda:
    return "cuda";
  case Backend::hip:
    return "hip";
  }
  return "unknown"; // unreachable: Backend is exhaustively matched above.
}

std::string_view BottleneckName(Bottleneck bottleneck) {
  switch (bottleneck) {
  case Bottleneck::compute:
    return "compute";
  case Bottleneck::memory:
    return "memory";
  case Bottleneck::latency:
    return "latency";
  }
  return "unknown"; // unreachable: Bottleneck is exhaustively matched above.
}

std::string_view PathName(Path path) {
  switch (path) {
  case Path::scalar:
    return "scalar";
  case Path::wmma:
    return "wmma";
  case Path::mma:
    return "mma";
  }
  return "unknown"; // unreachable: Path is exhaustively matched above.
}

namespace {

/// Compact double rendering (e.g. 100, 25, 42.6667) - no fixed trailing zeros.
std::string FmtDouble(double v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.6g", v);
  return std::string(buf);
}

Bottleneck DeriveBottleneck(const KernelReport &r) {
  if (r.spill_stores_bytes > 0 || r.spill_loads_bytes > 0)
    return Bottleneck::latency; // spills stall on memory latency.
  if (r.roofline == RooflineBound::memory_bound)
    return Bottleneck::memory;
  if (r.occupancy.occupancy_pct < 50.0)
    return Bottleneck::latency; // too few warps to hide latency.
  return Bottleneck::compute;
}

std::vector<std::string> DeriveFixes(const KernelReport &r) {
  std::vector<std::string> fixes;
  if (r.spill_stores_bytes > 0 || r.spill_loads_bytes > 0)
    fixes.emplace_back(
        "reduce register pressure (smaller tile/fewer live values)");
  if (r.occupancy.limiter == Limiter::registers &&
      r.occupancy.occupancy_pct < 100.0)
    fixes.emplace_back("reduce unroll/block size to lower register pressure");
  if (r.occupancy.occupancy_pct < 50.0)
    fixes.emplace_back(
        "increase occupancy (smaller tile / fewer registers / less smem)");
  if (r.roofline == RooflineBound::memory_bound)
    fixes.emplace_back("wider vectorized loads / fuse to reduce traffic");
  return fixes;
}

} // namespace

KernelReport BuildReport(const ReportInputs &in) {
  KernelReport r;
  r.kernel = in.kernel;
  r.backend = in.backend;
  r.arch = in.arch;
  r.registers_per_thread = in.ptxas.registers;
  r.smem_per_block_bytes = in.ptxas.smem_bytes;
  r.spill_stores_bytes = in.ptxas.spill_stores_bytes;
  r.spill_loads_bytes = in.ptxas.spill_loads_bytes;
  r.path = in.path;

  r.occupancy = ComputeOccupancy(in.ptxas.registers, in.ptxas.smem_bytes,
                                 in.threads_per_block, in.arch);

  // GEMM traffic: read A[M,K] + B[K,N], write C[M,N].
  const std::int64_t d = in.shape.dtype_bytes;
  r.global_read_bytes = (in.shape.m * in.shape.k + in.shape.k * in.shape.n) * d;
  r.global_write_bytes = in.shape.m * in.shape.n * d;

  const Roofline roof = ComputeRoofline(in.shape, in.arch);
  r.arithmetic_intensity_flop_per_byte = roof.arithmetic_intensity_flop_per_byte;
  r.roofline = roof.bound;

  r.bottleneck = DeriveBottleneck(r);
  r.suggested_fixes = DeriveFixes(r);
  return r;
}

std::string ToJson(const KernelReport &r) {
  std::string j;
  j += "{\n";
  j += "  \"kernel\": \"" + r.kernel + "\",\n";
  j += "  \"backend\": \"" + std::string(BackendName(r.backend)) + "\",\n";
  j += "  \"arch\": \"" + std::string(ArchName(r.arch)) + "\",\n";
  j += "  \"registers_per_thread\": " + std::to_string(r.registers_per_thread) +
       ",\n";
  j += "  \"smem_per_block_bytes\": " + std::to_string(r.smem_per_block_bytes) +
       ",\n";
  j += "  \"spill_stores_bytes\": " + std::to_string(r.spill_stores_bytes) +
       ",\n";
  j += "  \"spill_loads_bytes\": " + std::to_string(r.spill_loads_bytes) +
       ",\n";
  j += "  \"occupancy\": {\n";
  j += "    \"active_warps_per_sm\": " +
       std::to_string(r.occupancy.active_warps_per_sm) + ",\n";
  j += "    \"max_warps_per_sm\": " +
       std::to_string(r.occupancy.max_warps_per_sm) + ",\n";
  j += "    \"occupancy_pct\": " + FmtDouble(r.occupancy.occupancy_pct) +
       ",\n";
  j += "    \"limiter\": \"" + std::string(LimiterName(r.occupancy.limiter)) +
       "\"\n";
  j += "  },\n";
  j += "  \"traffic\": {\n";
  j += "    \"global_read_bytes\": " + std::to_string(r.global_read_bytes) +
       ",\n";
  j += "    \"global_write_bytes\": " + std::to_string(r.global_write_bytes) +
       ",\n";
  j += "    \"arithmetic_intensity_flop_per_byte\": " +
       FmtDouble(r.arithmetic_intensity_flop_per_byte) + ",\n";
  j += "    \"roofline\": \"" + std::string(RooflineName(r.roofline)) + "\"\n";
  j += "  },\n";
  j += "  \"bottleneck\": \"" + std::string(BottleneckName(r.bottleneck)) +
       "\",\n";
  j += "  \"suggested_fixes\": [";
  for (std::size_t i = 0; i < r.suggested_fixes.size(); ++i) {
    j += (i == 0 ? std::string() : ", ") + "\"" + r.suggested_fixes[i] + "\"";
  }
  j += "],\n";
  j += "  \"path\": \"" + std::string(PathName(r.path)) + "\"\n";
  j += "}\n";
  return j;
}

std::string ToText(const KernelReport &r) {
  std::string t;
  t += "kernel:                 " + r.kernel + "\n";
  t += "backend:                " + std::string(BackendName(r.backend)) + "\n";
  t += "arch:                   " + std::string(ArchName(r.arch)) + "\n";
  t += "registers_per_thread:   " + std::to_string(r.registers_per_thread) +
       "\n";
  t += "smem_per_block_bytes:   " + std::to_string(r.smem_per_block_bytes) +
       "\n";
  t += "spill_stores_bytes:     " + std::to_string(r.spill_stores_bytes) +
       "\n";
  t += "spill_loads_bytes:      " + std::to_string(r.spill_loads_bytes) + "\n";
  t += "occupancy:\n";
  t += "  active_warps_per_sm:  " +
       std::to_string(r.occupancy.active_warps_per_sm) + "\n";
  t += "  max_warps_per_sm:     " +
       std::to_string(r.occupancy.max_warps_per_sm) + "\n";
  t += "  occupancy_pct:        " + FmtDouble(r.occupancy.occupancy_pct) +
       "\n";
  t += "  limiter:              " +
       std::string(LimiterName(r.occupancy.limiter)) + "\n";
  t += "traffic:\n";
  t += "  global_read_bytes:    " + std::to_string(r.global_read_bytes) + "\n";
  t += "  global_write_bytes:   " + std::to_string(r.global_write_bytes) +
       "\n";
  t += "  arithmetic_intensity: " +
       FmtDouble(r.arithmetic_intensity_flop_per_byte) + " flop/byte\n";
  t += "  roofline:             " + std::string(RooflineName(r.roofline)) +
       "\n";
  t += "bottleneck:             " + std::string(BottleneckName(r.bottleneck)) +
       "\n";
  t += "path:                   " + std::string(PathName(r.path)) + "\n";
  t += "suggested_fixes:\n";
  if (r.suggested_fixes.empty())
    t += "  (none)\n";
  for (const std::string &f : r.suggested_fixes)
    t += "  - " + f + "\n";
  return t;
}

} // namespace polykernel::analysis
