// RUN: polykernel-opt %s --infer-shapes --fuse-rmsnorm-matmul --infer-tile-layout --plan-memory | FileCheck %s
// RUN: polykernel-opt %s --infer-shapes --plan-memory | FileCheck %s --check-prefix=FALLBACK

// Todo 16 (Wave 3): --plan-memory walks the matmul-structured ops (matmul,
// fused_rmsnorm_matmul, fused_matmul_bias_gelu) and computes a per-kernel memory
// plan, attaching DISCARDABLE attributes:
//   polykernel.smem_bytes (I64Attr), polykernel.workspace_bytes (I64Attr), and
//   polykernel.smem_over_budget (BoolAttr=true) when the budget exceeds the limit.
//
// MEMORY MODEL (documented in lib/Passes/PlanMemory.cpp):
//   smem_bytes = (tile_m*tile_k + tile_k*tile_n) * dtype_bytes * stages
//     - tile_m/n/k from the --infer-tile-layout attrs; ABSENT -> fallback 16x16x16.
//     - dtype_bytes = operand-0 element width / 8 (bf16 = 2, fp32 = 4).
//     - stages = 2 (default double-buffer).
//   workspace_bytes = 0 for a fused op (carries polykernel.fused_from; its
//     eliminated intermediates are fused away), else numElements(result)*dtype_bytes.
//   over-budget limit = 163 * 1024 = 166912 bytes (sm_80 per-block budget).
//
// The main RUN line runs the full pipeline (fusion + tile-layout + plan-memory); the
// FALLBACK RUN line omits --infer-tile-layout to prove the 16x16x16 tile fallback.
// Attributes print sorted, so CHECK-SAME order below is alphabetical.
//
// Cases (hand-computed):
//   @standalone  [32,32]@[32,32] bf16, tiles 32/32/32:
//                smem = (32*32 + 32*32) * 2 * 2 = 8192; workspace = 32*32*2 = 2048.
//   @fused       rmsnorm([2,4,8]f32)+matmul([2,4,8]f32@[8,16]f32) -> fused, tiles
//                8/8/16 (M=2*4=8,K=8,N=16): smem = (8*8 + 8*16) * 4 * 2 = 1536;
//                workspace = 0 (fused away).
//   @over_budget [256,256]@[256,256] f32, tiles 128/128/128:
//                smem = (128*128 + 128*128) * 4 * 2 = 262144 > 166912 -> over budget;
//                workspace = 256*256*4 = 262144.
//   FALLBACK     @standalone with NO tile attrs -> 16x16x16:
//                smem = (16*16 + 16*16) * 2 * 2 = 2048 (NOT the 8192 of the tiled
//                case, proving the fallback); workspace = 2048.

// ----- @standalone: plain matmul, tiled 32/32/32, smem + workspace, not over -----

// CHECK-LABEL: polykernel.func @standalone(
// CHECK: %{{.*}} = polykernel.matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.smem_bytes = 8192 : i64
// CHECK-SAME: polykernel.tile_k = 32 : i64
// CHECK-SAME: polykernel.tile_m = 32 : i64
// CHECK-SAME: polykernel.tile_n = 32 : i64
// CHECK-SAME: polykernel.workspace_bytes = 2048 : i64
// CHECK-SAME: : tensor<32x32xbf16>, tensor<32x32xbf16> -> tensor<32x32xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<32x32xbf16>
polykernel.func @standalone(%a: tensor<32x32xbf16>, %b: tensor<32x32xbf16>)
    -> tensor<32x32xbf16> {
  %0 = polykernel.matmul %a, %b
      : tensor<32x32xbf16>, tensor<32x32xbf16> -> tensor<32x32xbf16>
  polykernel.return %0 : tensor<32x32xbf16>
}

// ----- @fused: rmsnorm + matmul fuses; fused op gets workspace_bytes = 0 --------

// CHECK-LABEL: polykernel.func @fused(
// CHECK-NOT: polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.eliminated_type = tensor<2x4x8xf32>
// CHECK-SAME: polykernel.fused_from = "rmsnorm_matmul"
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.smem_bytes = 1536 : i64
// CHECK-SAME: polykernel.tile_k = 8 : i64
// CHECK-SAME: polykernel.tile_m = 8 : i64
// CHECK-SAME: polykernel.tile_n = 16 : i64
// CHECK-SAME: polykernel.workspace_bytes = 0 : i64
// CHECK-SAME: : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fused(%x: tensor<2x4x8xf32>, %w: tensor<8x16xf32>)
    -> tensor<2x4x16xf32> {
  %rms = polykernel.rmsnorm %x {epsilon = 1.0e-5 : f32} : tensor<2x4x8xf32>
  %out = polykernel.matmul %rms, %w
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- @over_budget: large fp32 matmul, tiles 128/128/128 -> smem over limit ----

// CHECK-LABEL: polykernel.func @over_budget(
// CHECK: %{{.*}} = polykernel.matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: polykernel.layout = "row_major"
// CHECK-SAME: polykernel.smem_bytes = 262144 : i64
// CHECK-SAME: polykernel.smem_over_budget = true
// CHECK-SAME: polykernel.tile_k = 128 : i64
// CHECK-SAME: polykernel.tile_m = 128 : i64
// CHECK-SAME: polykernel.tile_n = 128 : i64
// CHECK-SAME: polykernel.workspace_bytes = 262144 : i64
// CHECK-SAME: : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
// CHECK: polykernel.return %{{.*}} : tensor<256x256xf32>
polykernel.func @over_budget(%a: tensor<256x256xf32>, %b: tensor<256x256xf32>)
    -> tensor<256x256xf32> {
  %0 = polykernel.matmul %a, %b
      : tensor<256x256xf32>, tensor<256x256xf32> -> tensor<256x256xf32>
  polykernel.return %0 : tensor<256x256xf32>
}

// ----- FALLBACK case: no --infer-tile-layout -> tile falls back to 16x16x16 -----
// @standalone smem = (16*16 + 16*16) * 2 * 2 = 2048 (vs 8192 tiled above). The
// matmul carries ONLY the two plan attrs (no tile/layout attrs in this pipeline).

// FALLBACK-LABEL: polykernel.func @standalone(
// FALLBACK: %{{.*}} = polykernel.matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// FALLBACK-SAME: polykernel.smem_bytes = 2048 : i64
// FALLBACK-SAME: polykernel.workspace_bytes = 2048 : i64
// FALLBACK-SAME: : tensor<32x32xbf16>, tensor<32x32xbf16> -> tensor<32x32xbf16>
