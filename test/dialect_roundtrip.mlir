// RUN: polykernel-opt %s | FileCheck %s

// Todo 3 (Wave 1): round-trip EVERY named PolyKernel op. The input is parsed
// and re-printed by polykernel-opt; the CHECK lines match the canonical printed
// form (SSA values are matched with {{.*}} since the printer canonicalizes block
// arg / result names; f32 epsilon is matched with {{.*}} for the same reason).
// Together with op_set_closed.mlir this proves the op set is exactly the named
// transformer-inference set and that every op parses + round-trips with explicit
// tensor types.

builtin.module {
  // ----- Base ops -----
  polykernel.func @base_ops(%x: tensor<1x2048x4096xbf16>,
                            %w: tensor<4096x11008xbf16>,
                            %b: tensor<11008xbf16>,
                            %res: tensor<1x2048x4096xbf16>,
                            %cos: tensor<1x2048x32x128xbf16>,
                            %sin: tensor<1x2048x32x128xbf16>,
                            %q: tensor<1x32x2048x128xbf16>,
                            %k: tensor<1x32x2048x128xbf16>,
                            %v: tensor<1x32x2048x128xbf16>,
                            %cache: tensor<1x32x2048x128xbf16>,
                            %nk: tensor<1x32x1x128xbf16>,
                            %nv: tensor<1x32x1x128xbf16>)
      -> tensor<1x32x1x128xbf16> {
    %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>
    %1 = polykernel.matmul %0, %w : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
    %2 = polykernel.bias %1, %b : tensor<1x2048x11008xbf16>, tensor<11008xbf16> -> tensor<1x2048x11008xbf16>
    %3 = polykernel.gelu %2 : tensor<1x2048x11008xbf16>
    %4 = polykernel.silu %3 : tensor<1x2048x11008xbf16>
    %5 = polykernel.add %x, %res : tensor<1x2048x4096xbf16>
    %6 = polykernel.softmax %5 {axis = 2 : i64} : tensor<1x2048x4096xbf16>
    %7 = polykernel.rope %q, %cos, %sin : tensor<1x32x2048x128xbf16>, tensor<1x2048x32x128xbf16>, tensor<1x2048x32x128xbf16> -> tensor<1x32x2048x128xbf16>
    %8 = polykernel.attention %q, %k, %v : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16> -> tensor<1x32x2048x128xbf16>
    %9 = polykernel.kv_cache_update %cache, %nk, %nv : tensor<1x32x2048x128xbf16>, tensor<1x32x1x128xbf16>, tensor<1x32x1x128xbf16> -> tensor<1x32x1x128xbf16>
    polykernel.return %9 : tensor<1x32x1x128xbf16>
  }

  // ----- Fused ops -----
  polykernel.func @fused_ops(%x: tensor<1x2048x4096xbf16>,
                             %w: tensor<4096x11008xbf16>,
                             %b: tensor<11008xbf16>,
                             %res: tensor<1x2048x4096xbf16>,
                             %mask: tensor<1x1x2048x2048xbf16>,
                             %qkvw: tensor<4096x12288xbf16>,
                             %q: tensor<1x32x2048x128xbf16>,
                             %cache: tensor<1x32x2048x128xbf16>,
                             %nk: tensor<1x32x1x128xbf16>,
                             %nv: tensor<1x32x1x128xbf16>)
      -> tensor<1x32x2048x128xbf16> {
    %0 = polykernel.fused_rmsnorm_matmul %x, %w {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
    %1 = polykernel.fused_matmul_bias_gelu %x, %w, %b : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16> -> tensor<1x2048x11008xbf16>
    %2 = polykernel.fused_residual_rmsnorm %res, %x {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
    %3 = polykernel.fused_softmax_mask %x, %mask : tensor<1x2048x4096xbf16>, tensor<1x1x2048x2048xbf16> -> tensor<1x2048x4096xbf16>
    %4:3 = polykernel.qkv_projection %x, %qkvw : tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
    %5:2 = polykernel.fused_kv_append_attention %q, %cache, %nk, %nv : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x1x128xbf16>, tensor<1x32x1x128xbf16> -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
    polykernel.return %5#0 : tensor<1x32x2048x128xbf16>
  }
}

// CHECK-LABEL: polykernel.func @base_ops(
// CHECK-SAME: -> tensor<1x32x1x128xbf16>
// CHECK: %{{.*}} = polykernel.rmsnorm %{{.*}} {epsilon = {{.*}} : f32} : tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.matmul %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.bias %{{.*}}, %{{.*}} : tensor<1x2048x11008xbf16>, tensor<11008xbf16> -> tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.gelu %{{.*}} : tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.silu %{{.*}} : tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.add %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.softmax %{{.*}} {axis = 2 : i64} : tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.rope %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x32x2048x128xbf16>, tensor<1x2048x32x128xbf16>, tensor<1x2048x32x128xbf16> -> tensor<1x32x2048x128xbf16>
// CHECK: %{{.*}} = polykernel.attention %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16> -> tensor<1x32x2048x128xbf16>
// CHECK: %{{.*}} = polykernel.kv_cache_update %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x32x2048x128xbf16>, tensor<1x32x1x128xbf16>, tensor<1x32x1x128xbf16> -> tensor<1x32x1x128xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<1x32x1x128xbf16>

// CHECK-LABEL: polykernel.func @fused_ops(
// CHECK-SAME: -> tensor<1x32x2048x128xbf16>
// CHECK: %{{.*}} = polykernel.fused_rmsnorm_matmul %{{.*}}, %{{.*}} {epsilon = {{.*}} : f32} : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.fused_matmul_bias_gelu %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16>, tensor<11008xbf16> -> tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.fused_residual_rmsnorm %{{.*}}, %{{.*}} {epsilon = {{.*}} : f32} : tensor<1x2048x4096xbf16>, tensor<1x2048x4096xbf16> -> tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.fused_softmax_mask %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>, tensor<1x1x2048x2048xbf16> -> tensor<1x2048x4096xbf16>
// CHECK: %{{.*}}, %{{.*}}, %{{.*}} = polykernel.qkv_projection %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16> -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
// CHECK: %{{.*}}, %{{.*}} = polykernel.fused_kv_append_attention %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x1x128xbf16>, tensor<1x32x1x128xbf16> -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<1x32x2048x128xbf16>
