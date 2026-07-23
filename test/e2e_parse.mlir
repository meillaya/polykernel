// RUN: polykernel-opt %S/../examples/mlp_block.mlir --infer-shapes --canonicalize | FileCheck %s

// Todo 6 (Wave 1): end-to-end parse + pipeline of a shipped example. This lit
// test parses examples/mlp_block.mlir (relative to this test's source dir via
// %S) and runs the full --infer-shapes --canonicalize pipeline, proving the
// example parses, infers shapes consistently and canonicalizes cleanly.
//
// mlp_block pipeline: rmsnorm -> matmul -> gelu -> matmul -> add (residual),
// bf16 tensors, result tensor<1x2048x4096xbf16>. SSA names + the printed f32
// epsilon are matched with {{.*}} (the printer canonicalizes both).

// CHECK-LABEL: polykernel.func @mlp_block(
// CHECK-SAME: -> tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.rmsnorm %{{.*}} {epsilon = {{.*}} : f32} : tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.matmul %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.gelu %{{.*}} : tensor<1x2048x11008xbf16>
// CHECK: %{{.*}} = polykernel.matmul %{{.*}}, %{{.*}} : tensor<1x2048x11008xbf16>, tensor<11008x4096xbf16> -> tensor<1x2048x4096xbf16>
// CHECK: %{{.*}} = polykernel.add %{{.*}}, %{{.*}} : tensor<1x2048x4096xbf16>
// CHECK: polykernel.return %{{.*}} : tensor<1x2048x4096xbf16>
