// RUN: polykernel-opt %s --infer-shapes --infer-tile-layout | FileCheck %s

// Todo 15 (Wave 3): --infer-tile-layout walks the matmul-structured ops (matmul,
// fused_rmsnorm_matmul, fused_matmul_bias_gelu) and assigns each an INITIAL tile
// shape + operand layout as DISCARDABLE attributes:
//   polykernel.tile_m / polykernel.tile_n / polykernel.tile_k (I64Attr) and
//   polykernel.layout (StringAttr, "row_major").
//
// It runs AFTER --infer-shapes so the operand shapes are concrete.
//
// Tile heuristic (deterministic): for a problem dim D, the tile is the LARGEST
// power-of-two in {128, 64, 32, 16} that is <= D; if D < 16 the tile is clamped to
// D itself (SMALL-DIM CLAMP: tile_m <= M, tile_n <= N, tile_k <= K, always).
// M = product of operand 0's leading dims, K = last dim of operand 0, N = last dim
// of operand 1. Attributes print sorted, so the CHECK-SAME order below is
// layout, tile_k, tile_m, tile_n.
//
// Cases:
//   POSITIVE (large):  [2048,4096] @ [4096,11008] -> 128/128/128 (all dims >= 128).
//   POSITIVE (scaled): fused_rmsnorm_matmul [48,64] x [64,24] -> M=48->32, K=64->64,
//                      N=24->16 (exercises the mid tiers of the heuristic).
//   POSITIVE (clamp):  [8,8] @ [8,8] -> 8/8/8 (every dim < 16, so tile == dim).

// ----- POSITIVE (large): all dims >= 128 -> tile 128x128x128, row_major -----

// CHECK-LABEL: polykernel.func @matmul_large(
// CHECK: %{{.*}} = polykernel.matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.tile_k = 128 : i64
// CHECK-SAME: polykernel.tile_m = 128 : i64
// CHECK-SAME: polykernel.tile_n = 128 : i64
// CHECK-SAME: : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xbf16>
polykernel.func @matmul_large(%a: tensor<2048x4096xbf16>,
                              %b: tensor<4096x11008xbf16>)
    -> tensor<2048x11008xbf16> {
  %0 = polykernel.matmul %a, %b
      : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
  polykernel.return %0 : tensor<2048x11008xbf16>
}

// ----- POSITIVE (scaled): fused_rmsnorm_matmul [48,64] x [64,24] -----
// M = 48 -> 32, K = 64 -> 64, N = 24 -> 16 (mid heuristic tiers).

// CHECK-LABEL: polykernel.func @fused_rmsnorm_matmul_scaled(
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.tile_k = 64 : i64
// CHECK-SAME: polykernel.tile_m = 32 : i64
// CHECK-SAME: polykernel.tile_n = 16 : i64
// CHECK-SAME: : tensor<48x64xbf16>, tensor<64x24xbf16> -> tensor<48x24xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<48x24xbf16>
polykernel.func @fused_rmsnorm_matmul_scaled(%x: tensor<48x64xbf16>,
                                             %w: tensor<64x24xbf16>)
    -> tensor<48x24xbf16> {
  %0 = polykernel.fused_rmsnorm_matmul %x, %w
      : tensor<48x64xbf16>, tensor<64x24xbf16> -> tensor<48x24xbf16>
  polykernel.return %0 : tensor<48x24xbf16>
}

// ----- POSITIVE (clamp): [8,8] @ [8,8] -> every dim < 16, tile clamped to dim --
// The tiles are 8 (== the problem dim), NOT the 16 baseline / 128 max: this is the
// small-dim clamp (tile_m <= M, tile_n <= N, tile_k <= K).

// CHECK-LABEL: polykernel.func @matmul_small_clamped(
// CHECK: %{{.*}} = polykernel.matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.tile_k = 8 : i64
// CHECK-SAME: polykernel.tile_m = 8 : i64
// CHECK-SAME: polykernel.tile_n = 8 : i64
// CHECK-SAME: : tensor<8x8xf32>, tensor<8x8xf32> -> tensor<8x8xf32>
// CHECK: polykernel.return %{{.*}} : tensor<8x8xf32>
polykernel.func @matmul_small_clamped(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
    -> tensor<8x8xf32> {
  %0 = polykernel.matmul %a, %b
      : tensor<8x8xf32>, tensor<8x8xf32> -> tensor<8x8xf32>
  polykernel.return %0 : tensor<8x8xf32>
}
