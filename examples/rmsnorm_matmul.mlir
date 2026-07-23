// examples/rmsnorm_matmul.mlir
//
// Todo 6 (Wave 1): example demonstrating the fused target
// polykernel.fused_rmsnorm_matmul alongside the equivalent unfused
// rmsnorm + matmul pair, for comparison.
//
// Shapes (bf16):
//   %x : tensor<1x2048x4096xbf16>   batch=1, seq=2048, d_model=4096
//   %w : tensor<4096x11008xbf16>    projection (d_model -> ffn=11008)
//   result : tensor<1x2048x11008xbf16>
//
// Both functions compute the SAME result type. The fused op's shape inference
// runs rmsnorm (shape-preserving) then matmul(input, weight) -> MxN, so:
//   fused_rmsnorm_matmul : <1x2048x4096> x <4096x11008> (K=4096==4096)
//                          -> <1x2048x11008>
//   unfused rmsnorm      : shape-preserving -> <1x2048x4096>
//   unfused matmul       : <1x2048x4096> x <4096x11008> -> <1x2048x11008>
//
// Only polykernel ops (fused_rmsnorm_matmul, rmsnorm, matmul, func, return) +
// builtin tensors are used. No arith/func dialect ops.

// ----- Fused: single op -----
polykernel.func @fused(%x: tensor<1x2048x4096xbf16>,
                       %w: tensor<4096x11008xbf16>)
    -> tensor<1x2048x11008xbf16> {
  %0 = polykernel.fused_rmsnorm_matmul %x, %w {epsilon = 1.0e-05 : f32}
      : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
  polykernel.return %0 : tensor<1x2048x11008xbf16>
}

// ----- Unfused: rmsnorm then matmul (same result type) -----
polykernel.func @unfused(%x: tensor<1x2048x4096xbf16>,
                         %w: tensor<4096x11008xbf16>)
    -> tensor<1x2048x11008xbf16> {
  %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>
  %1 = polykernel.matmul %0, %w
      : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
  polykernel.return %1 : tensor<1x2048x11008xbf16>
}
