// RUN: polykernel-opt %s --fuse-matmul-bias-gelu | FileCheck %s

// Todo 13 (Wave 3): --fuse-matmul-bias-gelu fuses the chain
//   matmul(a, b); bias(mm, bias); gelu(bs)
// into one polykernel.fused_matmul_bias_gelu(a, b, bias) when BOTH absorbed
// intermediates are single-use (the matmul result feeds exactly one bias AND the
// bias result feeds exactly one gelu). Produces the gelu's result type. The dead
// matmul + bias are erased.
//
// The fused op uses the EXACT operand order from PolyKernelOps.td: (a, b, bias).
// fused_matmul_bias_gelu declares NO attributes, so none are carried.
//
// This fusion eliminates TWO intermediates, so it uses the INDEXED tracking attrs
// `polykernel.eliminated_type_0` (matmul result) + `polykernel.eliminated_type_1`
// (bias result), plus `polykernel.fused_from = "matmul_bias_gelu"`.
//
// Guardrails exercised below:
//   POSITIVE: single-use matmul -> bias -> gelu fuses; producers gone; both
//             indexed eliminated-type attrs + fused_from present.
//   NEGATIVE: a multi-use MATMUL (feeds the bias AND a residual add) is NOT fused.
//   NEGATIVE: a multi-use BIAS (feeds the gelu AND a residual add) is NOT fused
//             (proves the SECOND single-use check fires independently).

// ----- POSITIVE: single-use matmul + bias + gelu fuses; producers gone -----
// Operands are pinned to the func args to prove the .td operand order
// (a, b, bias): polykernel.func renumbers args to %argN, so %arg0 = a (matmul A),
// %arg1 = b (matmul B), %arg2 = bias. The distinct operand types on the type line
// confirm the order too.

// CHECK-LABEL: polykernel.func @fuse_chain(
// CHECK-NOT: polykernel.matmul
// CHECK-NOT: polykernel.bias
// CHECK-NOT: polykernel.gelu
// CHECK: %{{.*}} = polykernel.fused_matmul_bias_gelu %arg0, %arg1, %arg2
// CHECK-SAME: polykernel.eliminated_type_0 = tensor<2x4x16xf32>
// CHECK-SAME: polykernel.eliminated_type_1 = tensor<2x4x16xf32>
// CHECK-SAME: polykernel.fused_from = "matmul_bias_gelu"
// CHECK-SAME: : tensor<2x4x8xf32>, tensor<8x16xf32>, tensor<16xf32> -> tensor<2x4x16xf32>
// CHECK: polykernel.return %{{.*}} : tensor<2x4x16xf32>
polykernel.func @fuse_chain(%a: tensor<2x4x8xf32>, %b: tensor<8x16xf32>,
                            %bias: tensor<16xf32>) -> tensor<2x4x16xf32> {
  %mm = polykernel.matmul %a, %b
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  %bs = polykernel.bias %mm, %bias
      : tensor<2x4x16xf32>, tensor<16xf32> -> tensor<2x4x16xf32>
  %out = polykernel.gelu %bs : tensor<2x4x16xf32>
  polykernel.return %out : tensor<2x4x16xf32>
}

// ----- NEGATIVE: multi-use MATMUL (bias AND residual add) is NOT fused -----
// The matmul result feeds two ops, so fusing would duplicate the matmul; the
// standalone matmul + bias + gelu + add must all SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use_matmul(
// CHECK-NOT: polykernel.fused_matmul_bias_gelu
// CHECK: %{{.*}} = polykernel.matmul
// CHECK: %{{.*}} = polykernel.bias
// CHECK: %{{.*}} = polykernel.gelu
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use_matmul(%a: tensor<2x4x8xf32>,
                                          %b: tensor<8x16xf32>,
                                          %bias: tensor<16xf32>,
                                          %r: tensor<2x4x16xf32>)
    -> (tensor<2x4x16xf32>, tensor<2x4x16xf32>) {
  %mm = polykernel.matmul %a, %b
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  %bs = polykernel.bias %mm, %bias
      : tensor<2x4x16xf32>, tensor<16xf32> -> tensor<2x4x16xf32>
  %out = polykernel.gelu %bs : tensor<2x4x16xf32>
  %res = polykernel.add %mm, %r : tensor<2x4x16xf32>
  polykernel.return %out, %res : tensor<2x4x16xf32>, tensor<2x4x16xf32>
}

// ----- NEGATIVE: multi-use BIAS (gelu AND residual add) is NOT fused -----
// The matmul is single-use, but the bias result feeds two ops, so the SECOND
// single-use check fails and nothing fuses; all four producers SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use_bias(
// CHECK-NOT: polykernel.fused_matmul_bias_gelu
// CHECK: %{{.*}} = polykernel.matmul
// CHECK: %{{.*}} = polykernel.bias
// CHECK: %{{.*}} = polykernel.gelu
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use_bias(%a: tensor<2x4x8xf32>,
                                        %b: tensor<8x16xf32>,
                                        %bias: tensor<16xf32>,
                                        %r: tensor<2x4x16xf32>)
    -> (tensor<2x4x16xf32>, tensor<2x4x16xf32>) {
  %mm = polykernel.matmul %a, %b
      : tensor<2x4x8xf32>, tensor<8x16xf32> -> tensor<2x4x16xf32>
  %bs = polykernel.bias %mm, %bias
      : tensor<2x4x16xf32>, tensor<16xf32> -> tensor<2x4x16xf32>
  %out = polykernel.gelu %bs : tensor<2x4x16xf32>
  %res = polykernel.add %bs, %r : tensor<2x4x16xf32>
  polykernel.return %out, %res : tensor<2x4x16xf32>, tensor<2x4x16xf32>
}
