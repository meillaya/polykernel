// RUN: polykernel-opt %s --canonicalize | FileCheck %s

// Todo 5 (Wave 1): --canonicalize applies the PolyKernel canonicalization
// patterns with a greedy driver that also folds + DCEs:
//   1. identity residual-add elimination: add(x, 0) -> x (and add(0, x) -> x),
//      where the zero is a dense arith.constant (DCE removes the dead constant);
//   2. constant folding of pure elementwise ops: gelu/silu of a dense constant
//      fold to a computed arith.constant (GELU: exact erf-based definition;
//      SiLU: x / (1 + exp(-x)));
//   3. no-op fold: softmax(splat(c)) -> splat(1/N), N = the axis dim size.
// Non-foldable ops (add of two non-constant operands; softmax of a NON-splat
// constant; gelu of a non-constant) must SURVIVE unchanged (no over-folding).

// ----- 1a. add(x, 0) -> x (rhs zero); the dead zero constant is DCE'd -----

// CHECK-LABEL: polykernel.func @add_zero_rhs(
// CHECK-NOT: polykernel.add
// CHECK-NOT: arith.constant
// CHECK: polykernel.return %{{.*}} : tensor<2x4xf32>
polykernel.func @add_zero_rhs(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant dense<0.0> : tensor<2x4xf32>
  %0 = polykernel.add %x, %zero : tensor<2x4xf32>
  polykernel.return %0 : tensor<2x4xf32>
}

// ----- 1b. add(0, x) -> x (lhs zero) -----

// CHECK-LABEL: polykernel.func @add_zero_lhs(
// CHECK-NOT: polykernel.add
// CHECK-NOT: arith.constant
// CHECK: polykernel.return %{{.*}} : tensor<2x4xf32>
polykernel.func @add_zero_lhs(%x: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %zero = arith.constant dense<0.0> : tensor<2x4xf32>
  %0 = polykernel.add %zero, %x : tensor<2x4xf32>
  polykernel.return %0 : tensor<2x4xf32>
}

// ----- 2a. gelu of a dense constant folds to a computed constant -----
// gelu(0.0) = 0.0 exactly; gelu(1.0) = 0.5 * (1 + erf(1/sqrt(2))). Both folds
// are returned so DCE keeps them (unused folds are correctly DCE'd).

// CHECK-LABEL: polykernel.func @gelu_const_fold(
// CHECK-NOT: polykernel.gelu
// CHECK: %[[G0:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK: %[[G1:.*]] = arith.constant dense<0.841344773> : tensor<4xf32>
// CHECK: polykernel.return %[[G0]], %[[G1]]
polykernel.func @gelu_const_fold() -> (tensor<4xf32>, tensor<4xf32>) {
  %c0 = arith.constant dense<0.0> : tensor<4xf32>
  %c1 = arith.constant dense<1.0> : tensor<4xf32>
  %0 = polykernel.gelu %c0 : tensor<4xf32>
  %1 = polykernel.gelu %c1 : tensor<4xf32>
  polykernel.return %0, %1 : tensor<4xf32>, tensor<4xf32>
}

// ----- 2b. silu of a dense constant folds to a computed constant -----
// silu(1.0) = 1 / (1 + exp(-1)).

// CHECK-LABEL: polykernel.func @silu_const_fold(
// CHECK-NOT: polykernel.silu
// CHECK: %[[S1:.*]] = arith.constant dense<0.731058597> : tensor<4xf32>
// CHECK: polykernel.return %[[S1]]
polykernel.func @silu_const_fold() -> tensor<4xf32> {
  %c1 = arith.constant dense<1.0> : tensor<4xf32>
  %0 = polykernel.silu %c1 : tensor<4xf32>
  polykernel.return %0 : tensor<4xf32>
}

// ----- 3. no-op fold: softmax(splat(c)) -> splat(1/N) -----
// Default axis (last dim, N = 4): uniform 1/4 = 0.25.
// Explicit axis = 0 (N = 2): uniform 1/2 = 0.5.

// CHECK-LABEL: polykernel.func @softmax_splat_fold(
// CHECK-NOT: polykernel.softmax
// CHECK: %[[U4:.*]] = arith.constant dense<2.500000e-01> : tensor<2x4xf32>
// CHECK: %[[U2:.*]] = arith.constant dense<5.000000e-01> : tensor<2x4xf32>
// CHECK: polykernel.return %[[U4]], %[[U2]]
polykernel.func @softmax_splat_fold() -> (tensor<2x4xf32>, tensor<2x4xf32>) {
  %c = arith.constant dense<2.0> : tensor<2x4xf32>
  %0 = polykernel.softmax %c : tensor<2x4xf32>
  %1 = polykernel.softmax %c {axis = 0 : i64} : tensor<2x4xf32>
  polykernel.return %0, %1 : tensor<2x4xf32>, tensor<2x4xf32>
}

// ----- NEGATIVE: non-foldable add (two non-constant operands) SURVIVES -----

// CHECK-LABEL: polykernel.func @add_nonfoldable(
// CHECK: %{{.*}} = polykernel.add %{{.*}}, %{{.*}} : tensor<2x4xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4xf32>
polykernel.func @add_nonfoldable(%a: tensor<2x4xf32>, %b: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %0 = polykernel.add %a, %b : tensor<2x4xf32>
  polykernel.return %0 : tensor<2x4xf32>
}

// ----- NEGATIVE: softmax of a NON-splat constant is NOT folded -----

// CHECK-LABEL: polykernel.func @softmax_nonsplat_const_survives(
// CHECK: %{{.*}} = polykernel.softmax %{{.*}} : tensor<2xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2xf32>
polykernel.func @softmax_nonsplat_const_survives() -> tensor<2xf32> {
  %c = arith.constant dense<[1.0, 2.0]> : tensor<2xf32>
  %0 = polykernel.softmax %c : tensor<2xf32>
  polykernel.return %0 : tensor<2xf32>
}

// ----- NEGATIVE: gelu of a non-constant operand SURVIVES -----

// CHECK-LABEL: polykernel.func @gelu_nonconst_survives(
// CHECK: %{{.*}} = polykernel.gelu %{{.*}} : tensor<2x4xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4xf32>
polykernel.func @gelu_nonconst_survives(%x: tensor<2x4xf32>)
    -> tensor<2x4xf32> {
  %0 = polykernel.gelu %x : tensor<2x4xf32>
  polykernel.return %0 : tensor<2x4xf32>
}
