// RUN: polykernel-opt %s --fuse-softmax-mask | FileCheck %s

// Todo 13 (Wave 3): --fuse-softmax-mask fuses
//   add(input, mask); softmax(sum)
// into one polykernel.fused_softmax_mask(input, mask) when the add result is
// single-use (feeds exactly one softmax). Produces the softmax's result type. The
// dead add is erased.
//
// The fused op uses the EXACT operand order from PolyKernelOps.td: (input, mask).
// OPERAND MAPPING (positional): the add's `lhs` -> fused `input`, `rhs` -> fused
// `mask` (add is commutative, so numerically equivalent either way; pinned below
// to prove the convention). fused_softmax_mask declares NO `axis` attribute
// (PolyKernelOps.td), so the softmax's optional `axis` is intentionally dropped —
// proven below by pinning the attr dict literally.
//
// This fusion eliminates ONE intermediate (the add result), so it uses the
// UNINDEXED tracking attrs `polykernel.eliminated_type` +
// `polykernel.fused_from = "softmax_mask"`.
//
// Guardrails exercised below:
//   POSITIVE: single-use add -> softmax fuses (operand order + tracking pinned).
//   POSITIVE: single-use add -> softmax WITH an axis attr fuses and DROPS the axis
//             (attr dict pinned literally to ONLY the two tracking attrs — a
//             wrongly-carried `axis` would sort first and break the literal match).
//   NEGATIVE: a multi-use add (softmax AND another add) is NOT fused.

// ----- POSITIVE: single-use add + softmax fuses; producers gone -----
// Operands pinned to prove the positional mapping: polykernel.func renumbers args
// to %argN, so %arg0 = x (add.lhs -> fused input) and %arg1 = m (add.rhs -> fused
// mask); the attr dict is pinned literally (no axis here).

// CHECK-LABEL: polykernel.func @fuse_mask(
// CHECK-NOT: polykernel.add
// CHECK-NOT: polykernel.softmax
// CHECK: %{{.*}} = polykernel.fused_softmax_mask %arg0, %arg1 {polykernel.eliminated_type = tensor<2x4x16xf32>, polykernel.fused_from = "softmax_mask"} : tensor<2x4x16xf32>, tensor<2x4x16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fuse_mask(%x: tensor<2x4x16xf32>, %m: tensor<2x4x16xf32>)
    -> tensor<2x4x16xf32> {
  %sum = polykernel.add %x, %m : tensor<2x4x16xf32>
  %out = polykernel.softmax %sum : tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- POSITIVE: softmax WITH axis fuses and DROPS the axis -----
// fused_softmax_mask declares no `axis` attr; the literal attr-dict match below
// (opening `{polykernel.eliminated_type`) proves the axis is NOT carried — `axis`
// sorts before `polykernel.*`, so a carried axis would break the literal match.

// CHECK-LABEL: polykernel.func @fuse_mask_drops_axis(
// CHECK-NOT: polykernel.add
// CHECK-NOT: polykernel.softmax
// CHECK: %{{.*}} = polykernel.fused_softmax_mask %arg0, %arg1 {polykernel.eliminated_type = tensor<2x4x16xf32>, polykernel.fused_from = "softmax_mask"} : tensor<2x4x16xf32>, tensor<2x4x16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fuse_mask_drops_axis(%x: tensor<2x4x16xf32>,
                                      %m: tensor<2x4x16xf32>)
    -> tensor<2x4x16xf32> {
  %sum = polykernel.add %x, %m : tensor<2x4x16xf32>
  %out = polykernel.softmax %sum {axis = 2 : i64} : tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- NEGATIVE: multi-use add (softmax AND another add) is NOT fused -----
// The add result feeds two ops, so fusing would duplicate the masked sum; the
// standalone add + softmax + second add must all SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use_add(
// CHECK-NOT: polykernel.fused_softmax_mask
// CHECK: %{{.*}} = polykernel.add
// CHECK: %{{.*}} = polykernel.softmax
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use_add(%x: tensor<2x4x16xf32>,
                                       %m: tensor<2x4x16xf32>,
                                       %r: tensor<2x4x16xf32>)
    -> (tensor<2x4x16xf32>, tensor<2x4x16xf32>) {
  %sum = polykernel.add %x, %m : tensor<2x4x16xf32>
  %out = polykernel.softmax %sum : tensor<2x4x16xf32>
  %res = polykernel.add %sum, %r : tensor<2x4x16xf32>
  polykernel.return %out, %res : tensor<2x4x16xf32>, tensor<2x4x16xf32>
}
