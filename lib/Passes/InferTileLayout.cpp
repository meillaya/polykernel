//===- InferTileLayout.cpp - PolyKernel tile-layout pass --------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--infer-tile-layout` pass
// (Todo 15 / Wave 3).
//
//===----------------------------------------------------------------------===//
//
// Assigns each matmul-structured op an INITIAL tile shape + operand layout as
// DISCARDABLE attributes (consistent with the fusion tracking attrs the Wave 3
// fusion passes attach — the op set stays CLOSED; no new declared op attributes
// are added to PolyKernelOps.td):
//
//   - `polykernel.tile_m` / `polykernel.tile_n` / `polykernel.tile_k` (I64Attr):
//     the BLOCK_M / BLOCK_N / BLOCK_K tile the lowering targets. The baseline
//     CUDA GEMM (kernels/generated/matmul.cu) uses 16x16x16; these inferred tiles
//     are sensible DEFAULTS the autotuner (Todo 24) later refines — they need not
//     be optimal, only deterministic + clamped to the problem dims.
//   - `polykernel.layout` (StringAttr): the operand memory layout. Always
//     "row_major" today (the baseline; both operands are row-major [..,M,K] /
//     [K,N]). Transposed / packed layouts are a FUTURE refinement (Todo 24+).
//
// Scope: the three PRIMARY matmul ops — `matmul` (a[M,K] @ b[K,N] -> [M,N]),
// `fused_rmsnorm_matmul` (input[M,K], weight[K,N] -> [M,N]), and
// `fused_matmul_bias_gelu` (a[M,K], b[K,N], bias -> [M,N]). All three share the
// same operand convention: operand 0 is the [..,M,K] lhs and operand 1 is the
// [K,N] rhs, so one extractor covers them. (qkv_projection / fused_kv_append_
// attention are matmul-LIKE but have a head-split / cache structure, not a plain
// [M,K]x[K,N]; they are intentionally out of scope for this initial pass.)
//
// Shape extraction (matches the kernel's "flatten leading dims into M" model):
//   K = last dim of operand 0;  N = last dim of operand 1;
//   M = product of operand 0's leading dims (all but the last).
// Ops with any dynamic dim are skipped (the pass runs on typed IR after
// `--infer-shapes`, where shapes are concrete).
//
// Tile heuristic (simple, deterministic, documented): for a problem dim D, pick
// the LARGEST power-of-two candidate in {128, 64, 32, 16} that is <= D; if
// D < 16, clamp the tile to D itself (SMALL-DIM CLAMP: tile_m <= M, tile_n <= N,
// tile_k <= K, always). This keeps the baseline 16x16x16 for moderate dims,
// scales up to 128 for large dims, and never emits a tile larger than the
// problem it tiles (so small problems get small tiles). Power-of-two tiles align
// with the GPU shared-memory tiling in the baseline kernel.
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/InferTileLayout.h"

#include "PolyKernel/IR/PolyKernelOps.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include <optional>

namespace mlir::polykernel {
#define GEN_PASS_DEF_INFERTILELAYOUT
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace {

/// Discardable-attribute names attached to each matmul-structured op (the initial
/// tile shape + operand layout the autotuner refines later).
constexpr llvm::StringLiteral kTileMAttr = "polykernel.tile_m";
constexpr llvm::StringLiteral kTileNAttr = "polykernel.tile_n";
constexpr llvm::StringLiteral kTileKAttr = "polykernel.tile_k";
constexpr llvm::StringLiteral kLayoutAttr = "polykernel.layout";
/// Value of `polykernel.layout`: the baseline operand layout (both operands
/// row-major). Transposed / packed layouts are a future refinement.
constexpr llvm::StringLiteral kLayoutRowMajor = "row_major";

/// The matmul problem dimensions extracted from an op's operand shapes.
struct MatmulDims {
  int64_t m;
  int64_t n;
  int64_t k;
};

/// Extract (M, N, K) from a matmul-structured op's first two operands:
/// operand 0 is [..,M,K] (M = product of the leading dims), operand 1 is [K,N].
/// Returns std::nullopt if either operand is unranked or has a dynamic dim.
std::optional<MatmulDims> extractMatmulDims(Operation *op) {
  auto aType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto bType = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 1)
    return std::nullopt;

  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  for (int64_t d : aShape)
    if (ShapedType::isDynamic(d))
      return std::nullopt;
  for (int64_t d : bShape)
    if (ShapedType::isDynamic(d))
      return std::nullopt;

  int64_t m = 1;
  for (int64_t d : aShape.drop_back())
    m *= d;
  return MatmulDims{m, /*n=*/bShape.back(), /*k=*/aShape.back()};
}

/// Choose the tile for a problem dim D: the largest power-of-two candidate in
/// {128, 64, 32, 16} that is <= D; if D < 16, clamp to D itself (small-dim
/// clamp). The result is always <= D.
int64_t chooseTile(int64_t dim) {
  constexpr int64_t kCandidates[] = {128, 64, 32, 16};
  for (int64_t candidate : kCandidates)
    if (candidate <= dim)
      return candidate;
  return dim; // dim < 16 -> tile == problem dim.
}

} // namespace

//===----------------------------------------------------------------------===//
// The pass
//===----------------------------------------------------------------------===//

namespace mlir::polykernel {
namespace {

class InferTileLayoutPass
    : public impl::InferTileLayoutBase<InferTileLayoutPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto i64Type = IntegerType::get(ctx, 64);

    getOperation().walk([&](Operation *op) {
      // Scope: the three primary matmul ops (see the file header for why the
      // matmul-like qkv_projection / fused_kv_append_attention are excluded).
      if (!isa<polykernel::MatMulOp, polykernel::FusedRmsNormMatMulOp,
               polykernel::FusedMatMulBiasGeluOp>(op))
        return;

      std::optional<MatmulDims> dims = extractMatmulDims(op);
      if (!dims)
        return; // unranked / dynamic operand — nothing concrete to tile.

      op->setAttr(kTileMAttr,
                  IntegerAttr::get(i64Type, chooseTile(dims->m)));
      op->setAttr(kTileNAttr,
                  IntegerAttr::get(i64Type, chooseTile(dims->n)));
      op->setAttr(kTileKAttr,
                  IntegerAttr::get(i64Type, chooseTile(dims->k)));
      op->setAttr(kLayoutAttr, StringAttr::get(ctx, kLayoutRowMajor));
    });
  }
};

} // namespace

std::unique_ptr<Pass> createInferTileLayoutPass() {
  return std::make_unique<InferTileLayoutPass>();
}

} // namespace mlir::polykernel
