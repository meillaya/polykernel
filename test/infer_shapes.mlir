// RUN: polykernel-opt %s --infer-shapes | FileCheck %s

// Todo 4 (Wave 1): --infer-shapes computes every PolyKernel compute op's result
// type from its operands (via InferTypeOpInterface) and refines the IR.
//
// The shape-preserving ops (rmsnorm/gelu/silu/softmax/add) have ELIDED result
// types in their assembly format, so their result type is inferred from the input
// (checked here via the function result / return type). matmul / bias / the fused
// ops carry explicit result types that the pass recomputes from the operands
// (MxN for the matmul family). The mismatch case (mismatched matmul contraction
// dimension) is covered by infer_shapes_mismatch.mlir.

// ===== bf16 =====

// CHECK-LABEL: polykernel.func @matmul_bf16(
// CHECK-SAME: -> tensor<2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.matmul %{{.*}}, %{{.*}} : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xbf16>
polykernel.func @matmul_bf16(%a: tensor<2048x4096xbf16>,
                             %b: tensor<4096x11008xbf16>)
    -> tensor<2048x11008xbf16> {
  %0 = polykernel.matmul %a, %b
      : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
  polykernel.return %0 : tensor<2048x11008xbf16>
}

// CHECK-LABEL: polykernel.func @shape_preserving_bf16(
// CHECK-SAME: -> tensor<2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.rmsnorm %{{.*}} {epsilon = {{.*}} : f32} : tensor<2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.gelu %{{.*}} : tensor<2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.silu %{{.*}} : tensor<2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.softmax %{{.*}} : tensor<2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.add %{{.*}}, %{{.*}} : tensor<2048x4096xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x4096xbf16>
polykernel.func @shape_preserving_bf16(%x: tensor<2048x4096xbf16>,
                                       %res: tensor<2048x4096xbf16>)
    -> tensor<2048x4096xbf16> {
  %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<2048x4096xbf16>
  %1 = polykernel.gelu %0 : tensor<2048x4096xbf16>
  %2 = polykernel.silu %1 : tensor<2048x4096xbf16>
  %3 = polykernel.softmax %2 : tensor<2048x4096xbf16>
  %4 = polykernel.add %3, %res : tensor<2048x4096xbf16>
  polykernel.return %4 : tensor<2048x4096xbf16>
}

// CHECK-LABEL: polykernel.func @fused_rmsnorm_matmul_bf16(
// CHECK-SAME: -> tensor<2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{.*}}, %{{.*}} {epsilon = {{.*}} : f32} : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xbf16>
polykernel.func @fused_rmsnorm_matmul_bf16(%x: tensor<2048x4096xbf16>,
                                           %w: tensor<4096x11008xbf16>)
    -> tensor<2048x11008xbf16> {
  %0 = polykernel.fused_rmsnorm_matmul %x, %w {epsilon = 1.0e-05 : f32}
      : tensor<2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<2048x11008xbf16>
  polykernel.return %0 : tensor<2048x11008xbf16>
}

// CHECK-LABEL: polykernel.func @fused_matmul_bias_gelu_bf16(
// CHECK-SAME: -> tensor<2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.fused_matmul_bias_gelu %{{.*}}, %{{.*}}, %{{.*}} : tensor<2048x4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16> -> tensor<2048x11008xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xbf16>
polykernel.func @fused_matmul_bias_gelu_bf16(%a: tensor<2048x4096xbf16>,
                                             %b: tensor<4096x11008xbf16>,
                                             %bias: tensor<11008xbf16>)
    -> tensor<2048x11008xbf16> {
  %0 = polykernel.fused_matmul_bias_gelu %a, %b, %bias
      : tensor<2048x4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16>
      -> tensor<2048x11008xbf16>
  polykernel.return %0 : tensor<2048x11008xbf16>
}

// CHECK-LABEL: polykernel.func @qkv_projection_bf16(
// CHECK-SAME: -> tensor<1x32x2048x128xbf16>
// CHECK: %{{.*}}, %{{.*}}, %{{.*}} = polykernel.qkv_projection %{{.*}}, %{{.*}} {num_heads = 32 : i64} : tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
polykernel.func @qkv_projection_bf16(%x: tensor<1x2048x4096xbf16>,
                                     %w: tensor<4096x12288xbf16>)
    -> tensor<1x32x2048x128xbf16> {
  %q:3 = polykernel.qkv_projection %x, %w {num_heads = 32 : i64}
      : tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16>
      -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
  polykernel.return %q#0 : tensor<1x32x2048x128xbf16>
}

// ===== fp16 =====

// CHECK-LABEL: polykernel.func @matmul_fp16(
// CHECK-SAME: -> tensor<2048x11008xf16>
// CHECK: %{{.*}} = polykernel.matmul %{{.*}}, %{{.*}} : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xf16>
polykernel.func @matmul_fp16(%a: tensor<2048x4096xf16>,
                             %b: tensor<4096x11008xf16>)
    -> tensor<2048x11008xf16> {
  %0 = polykernel.matmul %a, %b
      : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
  polykernel.return %0 : tensor<2048x11008xf16>
}

// CHECK-LABEL: polykernel.func @shape_preserving_fp16(
// CHECK-SAME: -> tensor<2048x4096xf16>
// CHECK: %{{.*}} = polykernel.rmsnorm %{{.*}} {epsilon = {{.*}} : f32} : tensor<2048x4096xf16>
// CHECK: %{{.*}} = polykernel.gelu %{{.*}} : tensor<2048x4096xf16>
// CHECK: %{{.*}} = polykernel.add %{{.*}}, %{{.*}} : tensor<2048x4096xf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x4096xf16>
polykernel.func @shape_preserving_fp16(%x: tensor<2048x4096xf16>,
                                       %res: tensor<2048x4096xf16>)
    -> tensor<2048x4096xf16> {
  %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<2048x4096xf16>
  %1 = polykernel.gelu %0 : tensor<2048x4096xf16>
  %2 = polykernel.add %1, %res : tensor<2048x4096xf16>
  polykernel.return %2 : tensor<2048x4096xf16>
}

// CHECK-LABEL: polykernel.func @fused_rmsnorm_matmul_fp16(
// CHECK-SAME: -> tensor<2048x11008xf16>
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{.*}}, %{{.*}} {epsilon = {{.*}} : f32} : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
// CHECK: polykernel.return %{{.*}} : tensor<2048x11008xf16>
polykernel.func @fused_rmsnorm_matmul_fp16(%x: tensor<2048x4096xf16>,
                                           %w: tensor<4096x11008xf16>)
    -> tensor<2048x11008xf16> {
  %0 = polykernel.fused_rmsnorm_matmul %x, %w {epsilon = 1.0e-05 : f32}
      : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
  polykernel.return %0 : tensor<2048x11008xf16>
}
