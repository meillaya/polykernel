// RUN: polykernel-opt %s --fuse-residual-rmsnorm | FileCheck %s

// Todo 13 (Wave 3): --fuse-residual-rmsnorm fuses
//   add(residual, input); rmsnorm(sum, eps)
// into one polykernel.fused_residual_rmsnorm(residual, input, eps) when the add
// result is single-use (feeds exactly one rmsnorm). Carries the rmsnorm's optional
// `epsilon` and produces the rmsnorm's result type. The dead add is erased.
//
// The fused op uses the EXACT operand order from PolyKernelOps.td:
// (residual, input, epsilon?). OPERAND MAPPING (positional): the add's `lhs` ->
// fused `residual`, `rhs` -> fused `input` (add is commutative, so numerically
// equivalent either way; pinned below to prove the convention).
//
// This fusion eliminates ONE intermediate (the add result), so it uses the
// UNINDEXED tracking attrs `polykernel.eliminated_type` +
// `polykernel.fused_from = "residual_rmsnorm"`.
//
// Guardrails exercised below:
//   POSITIVE: single-use add -> rmsnorm fuses WITH epsilon (epsilon carried).
//   POSITIVE: single-use add -> rmsnorm fuses WITHOUT epsilon (attr dict pinned
//             literally to ONLY the two tracking attrs — a wrongly-present
//             epsilon would sort first and break the literal match).
//   NEGATIVE: a multi-use add (rmsnorm AND another add) is NOT fused.

// ----- POSITIVE: single-use add + rmsnorm fuses; epsilon is carried -----
// Operands pinned to prove the positional mapping: polykernel.func renumbers args
// to %argN, so %arg0 = x (add.lhs -> fused residual) and %arg1 = r (add.rhs ->
// fused input). epsilon reprints as its rounded f32 value, so match it with a
// wildcard (not a pinned literal).

// CHECK-LABEL: polykernel.func @fuse_residual_eps(
// CHECK-NOT: polykernel.add
// CHECK-NOT: polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.fused_residual_rmsnorm %arg0, %arg1
// CHECK-SAME: epsilon = {{.*}} : f32
// CHECK-SAME: polykernel.eliminated_type = tensor<2x4x8xf32>
// CHECK-SAME: polykernel.fused_from = "residual_rmsnorm"
// CHECK-SAME: : tensor<2x4x8xf32>, tensor<2x4x8xf32> -> tensor<2x4x8xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x8xf32>
polykernel.func @fuse_residual_eps(%x: tensor<2x4x8xf32>, %r: tensor<2x4x8xf32>)
    -> tensor<2x4x8xf32> {
  %sum = polykernel.add %x, %r : tensor<2x4x8xf32>
  %out = polykernel.rmsnorm %sum {epsilon = 1.0e-5 : f32} : tensor<2x4x8xf32>
  polykernel.return %out : tensor<2x4x8xf32>
}

// ----- POSITIVE: single-use add + rmsnorm WITHOUT epsilon fuses -----
// The attr dict is pinned literally to contain ONLY the two tracking attributes:
// it opens `{polykernel.eliminated_type` (attributes print sorted, so an `epsilon`
// attr — if wrongly present — would sort first and break the literal match).

// CHECK-LABEL: polykernel.func @fuse_residual_no_eps(
// CHECK-NOT: polykernel.add
// CHECK-NOT: polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.fused_residual_rmsnorm %arg0, %arg1 {polykernel.eliminated_type = tensor<2x4x8xf32>, polykernel.fused_from = "residual_rmsnorm"} : tensor<2x4x8xf32>, tensor<2x4x8xf32> -> tensor<2x4x8xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x8xf32>
polykernel.func @fuse_residual_no_eps(%x: tensor<2x4x8xf32>,
                                      %r: tensor<2x4x8xf32>)
    -> tensor<2x4x8xf32> {
  %sum = polykernel.add %x, %r : tensor<2x4x8xf32>
  %out = polykernel.rmsnorm %sum : tensor<2x4x8xf32>
  polykernel.return %out : tensor<2x4x8xf32>
}

// ----- NEGATIVE: multi-use add (rmsnorm AND another add) is NOT fused -----
// The add result feeds two ops, so fusing would duplicate the residual sum; the
// standalone add + rmsnorm + second add must all SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use_add(
// CHECK-NOT: polykernel.fused_residual_rmsnorm
// CHECK: %{{.*}} = polykernel.add
// CHECK: %{{.*}} = polykernel.rmsnorm
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use_add(%x: tensor<2x4x8xf32>,
                                       %r: tensor<2x4x8xf32>,
                                       %r2: tensor<2x4x8xf32>)
    -> (tensor<2x4x8xf32>, tensor<2x4x8xf32>) {
  %sum = polykernel.add %x, %r : tensor<2x4x8xf32>
  %out = polykernel.rmsnorm %sum {epsilon = 1.0e-5 : f32} : tensor<2x4x8xf32>
  %res = polykernel.add %sum, %r2 : tensor<2x4x8xf32>
  polykernel.return %out, %res : tensor<2x4x8xf32>, tensor<2x4x8xf32>
}
