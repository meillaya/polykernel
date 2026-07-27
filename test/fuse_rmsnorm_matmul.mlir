// RUN: polykernel-opt %s --fuse-rmsnorm-matmul | FileCheck %s

// Todo 12 (Wave 3): --fuse-rmsnorm-matmul fuses a polykernel.matmul whose input
// (A) operand is produced by a SINGLE-USE polykernel.rmsnorm into one
// polykernel.fused_rmsnorm_matmul, carrying the rmsnorm's optional `epsilon`
// and producing the matmul's (MxN) result type. The dead rmsnorm is erased.
//
// Each fused op is tagged for the Todo 17 traffic report with the discardable
// attributes `polykernel.fused_from = "rmsnorm_matmul"` and
// `polykernel.eliminated_type = <the eliminated rmsnorm result's tensor type>`.
//
// Guardrails exercised below:
//   POSITIVE: single-use rmsnorm -> matmul fuses (with and without epsilon).
//   NEGATIVE: a multi-use rmsnorm (matmul AND residual add) is NOT fused.
//   NEGATIVE: a rmsnorm feeding the matmul's B operand (not A) is NOT fused.

// ----- POSITIVE: single-use rmsnorm + matmul fuses; epsilon is carried -----

// CHECK-LABEL: polykernel.func @fuse_single_use_eps(
// CHECK-NOT: polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{[^[:space:]]*}}, %{{[^[:space:]]*}}
// CHECK-SAME: epsilon = {{.*}} : f32
// CHECK-SAME: polykernel.eliminated_type = tensor<2x4x8xf32>
// CHECK-SAME: polykernel.fused_from = "rmsnorm_matmul"
// CHECK-SAME: : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fuse_single_use_eps(%x: tensor<2x4x8xf32>,
                                     %w: tensor<8x16xf32>)
    -> tensor<2x4x16xf32> {
  %rms = polykernel.rmsnorm %x {epsilon = 1.0e-5 : f32} : tensor<2x4x8xf32>
  %out = polykernel.matmul %rms, %w
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- POSITIVE: single-use rmsnorm WITHOUT epsilon fuses (no epsilon attr) ---
// The attr dict is pinned literally to contain ONLY the two tracking attributes:
// it opens `{polykernel.eliminated_type` (attributes print sorted, so an `epsilon`
// attr — if wrongly present — would sort first and break the literal match).

// CHECK-LABEL: polykernel.func @fuse_single_use_no_eps(
// CHECK-NOT: polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{.*}}, %{{.*}} {polykernel.eliminated_type = tensor<2x4x8xf32>, polykernel.fused_from = "rmsnorm_matmul"} : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fuse_single_use_no_eps(%x: tensor<2x4x8xf32>,
                                        %w: tensor<8x16xf32>)
    -> tensor<2x4x16xf32> {
  %rms = polykernel.rmsnorm %x : tensor<2x4x8xf32>
  %out = polykernel.matmul %rms, %w
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- NEGATIVE: multi-use rmsnorm (matmul AND residual add) is NOT fused -----
// The rmsnorm result feeds two ops, so fusing would duplicate the normalization;
// the standalone rmsnorm + matmul + add must all SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use(
// CHECK-NOT: polykernel.fused_rmsnorm_matmul
// CHECK: %{{.*}} = polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.matmul
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use(%x: tensor<2x4x8xf32>, %w: tensor<8x16xf32>,
                                   %r: tensor<2x4x8xf32>)
    -> (tensor<2x4x16xf32>, tensor<2x4x8xf32>) {
  %rms = polykernel.rmsnorm %x {epsilon = 1.0e-5 : f32} : tensor<2x4x8xf32>
  %out = polykernel.matmul %rms, %w
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  %res = polykernel.add %rms, %r : tensor<2x4x8xf32>
  polykernel.return %out, %res : tensor<2x4x16xf32>, tensor<2x4x8xf32>
}

// ----- NEGATIVE: rmsnorm feeding the matmul's B operand (not A) is NOT fused --

// CHECK-LABEL: polykernel.func @no_fuse_rmsnorm_as_b(
// CHECK-NOT: polykernel.fused_rmsnorm_matmul
// CHECK: %{{.*}} = polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.matmul
// CHECK: polykernel.return
polykernel.func @no_fuse_rmsnorm_as_b(%x: tensor<2x4x8xf32>,
                                      %w: tensor<8x16xf32>)
    -> tensor<2x4x16xf32> {
  %rw = polykernel.rmsnorm %w : tensor<8x16xf32>
  %out = polykernel.matmul %x, %rw
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}
