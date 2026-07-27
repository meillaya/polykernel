//===- EmitKernelReport.cpp - PolyKernel per-kernel report pass -*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--emit-kernel-report` pass
// (Todo 27 / Wave 5).
//
//===----------------------------------------------------------------------===//
//
// The IR half of the full per-kernel report pipeline. Walks the matmul-structured
// ops (matmul, fused_rmsnorm_matmul, fused_matmul_bias_gelu — the same scope as
// --infer-tile-layout / --plan-memory) and, for each generated kernel:
//
//   1. ATTACHES the IR-derivable contract-H report fields as DISCARDABLE attrs
//      (polykernel.report_kernel/backend/arch/path) — the op set stays CLOSED,
//      consistent with the tile attrs (Todo 15), the memory-plan attrs (Todo 16),
//      and the fusion-tracking attrs (Todos 12/13).
//   2. EMITS a per-kernel report manifest (kernel_report.json) into --output-dir,
//      using contract-H field names for everything the IR alone can derive: the
//      kernel name, backend/arch/path, the GEMM shape (M/N/K + dtype bytes), the
//      tile shape, the shared-memory budget (read from the --plan-memory attr),
//      and the fusion provenance.
//
// The COMPILE-TIME contract-H fields — registers_per_thread, spill bytes,
// occupancy{}, traffic{} (global r/w + arithmetic intensity + roofline),
// bottleneck, and suggested_fixes — need the nvcc/ptxas (CUDA) or hipcc
// --save-temps (AMD) analyzers and are filled by tools/polykernel-report/report.py,
// which merges this skeleton with the CUDA compile-time analysis (Todo 11), the
// AMD ISA analysis (Todo 21), and the autotuner result (Todo 25). This pass does
// NOT fabricate register/spill numbers — the manifest names the analyzer as the
// source of those fields rather than emitting zeros that could be mistaken for
// real measurements.
//
// Shape model (matches InferTileLayout / PlanMemory): operand 0 = [..,M,K],
// operand 1 = [K,N]; K = last dim of operand 0, N = last dim of operand 1,
// M = product of operand 0's leading dims. Ops with a dynamic operand/result dim
// are skipped (no computable shape). The IR is otherwise unchanged (emitter).
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/Passes.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <vector>

namespace mlir::polykernel {
#define GEN_PASS_DEF_EMITKERNELREPORT
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable attributes READ (attached by earlier Wave 3 passes).
constexpr llvm::StringLiteral kTileMAttr = "polykernel.tile_m";
constexpr llvm::StringLiteral kTileNAttr = "polykernel.tile_n";
constexpr llvm::StringLiteral kTileKAttr = "polykernel.tile_k";
constexpr llvm::StringLiteral kSmemBytesAttr = "polykernel.smem_bytes";
constexpr llvm::StringLiteral kFusedFromAttr = "polykernel.fused_from";

/// Discardable attributes ATTACHED by this pass (the report identity).
constexpr llvm::StringLiteral kReportKernelAttr = "polykernel.report_kernel";
constexpr llvm::StringLiteral kReportBackendAttr = "polykernel.report_backend";
constexpr llvm::StringLiteral kReportArchAttr = "polykernel.report_arch";
constexpr llvm::StringLiteral kReportPathAttr = "polykernel.report_path";

/// Fallback tile when the --infer-tile-layout attrs are absent (the baseline CUDA
/// GEMM tile; matches PlanMemory's fallback).
constexpr int64_t kFallbackTile = 16;

/// One kernel's IR-derivable contract-H report skeleton.
struct KernelSkeleton {
  std::string name;
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t dtypeBytes = 0;
  int64_t tileM = kFallbackTile;
  int64_t tileN = kFallbackTile;
  int64_t tileK = kFallbackTile;
  int64_t smemBytes = 0;
  bool fused = false;
  std::string fusedFrom;
};

/// The kernel name for an op kind — matches the emitted .cu basenames
/// (kernels/generated/<name>.cu) so polykernel-report can locate the source.
std::string kernelName(Operation *op) {
  if (isa<polykernel::FusedRmsNormMatMulOp>(op))
    return "fused_rmsnorm_matmul";
  if (isa<polykernel::FusedMatMulBiasGeluOp>(op))
    return "fused_matmul_bias_gelu";
  return "matmul";
}

/// True when `type` is a ranked tensor with no dynamic dim (shape computable).
bool isStaticRankedTensor(Type type) {
  auto shaped = dyn_cast<RankedTensorType>(type);
  return shaped && shaped.hasStaticShape();
}

/// Read an I64 attr, falling back to `fallback` when absent.
int64_t readI64Attr(Operation *op, llvm::StringLiteral name, int64_t fallback) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(name))
    return attr.getInt();
  return fallback;
}

/// Render the manifest as a contract-H-shaped JSON document. The compile-time
/// fields are named (under "compile_time_analysis") but left to the analyzer —
/// never emitted as fake zeros.
std::string renderManifest(const std::vector<KernelSkeleton> &kernels,
                           StringRef backend, StringRef arch) {
  std::string j;
  j += "{\n";
  j += "  \"backend\": \"" + backend.str() + "\",\n";
  j += "  \"arch\": \"" + arch.str() + "\",\n";
  j += "  \"kernels\": [";
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    const KernelSkeleton &s = kernels[i];
    j += (i == 0 ? "\n" : ",\n");
    j += "    {\n";
    j += "      \"kernel\": \"" + s.name + "\",\n";
    j += "      \"backend\": \"" + backend.str() + "\",\n";
    j += "      \"arch\": \"" + arch.str() + "\",\n";
    j += "      \"shape\": { \"m\": " + std::to_string(s.m) +
         ", \"n\": " + std::to_string(s.n) + ", \"k\": " + std::to_string(s.k) +
         ", \"dtype_bytes\": " + std::to_string(s.dtypeBytes) + " },\n";
    j += "      \"tile\": { \"m\": " + std::to_string(s.tileM) +
         ", \"n\": " + std::to_string(s.tileN) +
         ", \"k\": " + std::to_string(s.tileK) + " },\n";
    j += "      \"smem_per_block_bytes\": " + std::to_string(s.smemBytes) + ",\n";
    j += "      \"fused\": " + std::string(s.fused ? "true" : "false") + ",\n";
    j += "      \"fused_from\": \"" + s.fusedFrom + "\",\n";
    j += "      \"compile_time_analysis\": \"registers_per_thread, spill bytes, "
         "occupancy, traffic, roofline, bottleneck, suggested_fixes filled by "
         "polykernel-report (nvcc/ptxas + hipcc --save-temps)\"\n";
    j += "    }";
  }
  j += kernels.empty() ? std::string("]\n") : std::string("\n  ]\n");
  j += "}\n";
  return j;
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class EmitKernelReportPass
    : public impl::EmitKernelReportBase<EmitKernelReportPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    std::vector<KernelSkeleton> kernels;

    getOperation().walk([&](Operation *op) {
      if (!isa<polykernel::MatMulOp, polykernel::FusedRmsNormMatMulOp,
               polykernel::FusedMatMulBiasGeluOp>(op))
        return;

      Type aType = op->getOperand(0).getType();
      Type bType = op->getOperand(1).getType();
      if (!isStaticRankedTensor(aType) || !isStaticRankedTensor(bType))
        return;

      auto aShape = cast<RankedTensorType>(aType).getShape();
      auto bShape = cast<RankedTensorType>(bType).getShape();

      KernelSkeleton s;
      s.name = kernelName(op);
      s.k = aShape.back();
      s.n = bShape.back();
      s.m = 1;
      for (int64_t dim : aShape.drop_back())
        s.m *= dim;
      s.dtypeBytes =
          cast<RankedTensorType>(aType).getElementType().getIntOrFloatBitWidth() /
          8;
      s.tileM = readI64Attr(op, kTileMAttr, kFallbackTile);
      s.tileN = readI64Attr(op, kTileNAttr, kFallbackTile);
      s.tileK = readI64Attr(op, kTileKAttr, kFallbackTile);
      s.smemBytes = readI64Attr(op, kSmemBytesAttr, 0);
      if (auto fusedFrom = op->getAttrOfType<StringAttr>(kFusedFromAttr)) {
        s.fused = true;
        s.fusedFrom = fusedFrom.getValue().str();
      }
      kernels.push_back(s);

      // Attach the report identity as discardable attrs (op set stays closed).
      op->setAttr(kReportKernelAttr, StringAttr::get(ctx, s.name));
      op->setAttr(kReportBackendAttr, StringAttr::get(ctx, backend));
      op->setAttr(kReportArchAttr, StringAttr::get(ctx, arch));
      op->setAttr(kReportPathAttr, StringAttr::get(ctx, path));
    });

    if (failed(writeManifest(renderManifest(kernels, backend, arch))))
      return signalPassFailure();
  }

private:
  /// Create --output-dir if needed and write kernel_report.json into it.
  LogicalResult writeManifest(const std::string &json) {
    if (std::error_code ec = llvm::sys::fs::create_directories(outputDir)) {
      getOperation().emitError("cannot create output directory '")
          << outputDir << "': " << ec.message();
      return failure();
    }
    llvm::SmallString<128> manifestPath(outputDir);
    llvm::sys::path::append(manifestPath, "kernel_report.json");

    std::error_code ec;
    llvm::raw_fd_ostream out(manifestPath, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      getOperation().emitError("cannot open '")
          << manifestPath << "': " << ec.message();
      return failure();
    }
    out << json;
    out.flush();
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createEmitKernelReportPass() {
  return std::make_unique<EmitKernelReportPass>();
}

} // namespace mlir::polykernel
