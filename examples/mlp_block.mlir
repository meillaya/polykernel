// examples/mlp_block.mlir
//
// Todo 6 (Wave 1): example MLP fragment as a polykernel.func.
//
// Pipeline: rmsnorm -> matmul -> gelu -> matmul -> add (residual).
//
// Shapes (bf16), Llama-2-7B MLP dims:
//   %x  : tensor<1x2048x4096xbf16>   batch=1, seq=2048, d_model=4096
//   %w1 : tensor<4096x11008xbf16>    up-projection  (d_model -> ffn=11008)
//   %w2 : tensor<11008x4096xbf16>    down-projection (ffn -> d_model)
//   result : tensor<1x2048x4096xbf16>
//
// Shape-inference check (--infer-shapes):
//   rmsnorm  : shape-preserving            -> <1x2048x4096>
//   matmul   : <1x2048x4096> x <4096x11008>  (K=4096==4096) -> <1x2048x11008>
//   gelu     : shape-preserving            -> <1x2048x11008>
//   matmul   : <1x2048x11008> x <11008x4096> (K=11008==11008) -> <1x2048x4096>
//   add      : <1x2048x4096> + <1x2048x4096> (equal shapes)  -> <1x2048x4096>
//
// Only the 18 defined polykernel ops are used (rmsnorm, matmul, gelu, add,
// func, return); tensors are builtin. No arith/func dialect ops.

polykernel.func @mlp_block(%x: tensor<1x2048x4096xbf16>,
                           %w1: tensor<4096x11008xbf16>,
                           %w2: tensor<11008x4096xbf16>)
    -> tensor<1x2048x4096xbf16> {
  %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>
  %1 = polykernel.matmul %0, %w1
      : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16> -> tensor<1x2048x11008xbf16>
  %2 = polykernel.gelu %1 : tensor<1x2048x11008xbf16>
  %3 = polykernel.matmul %2, %w2
      : tensor<1x2048x11008xbf16>, tensor<11008x4096xbf16> -> tensor<1x2048x4096xbf16>
  %4 = polykernel.add %3, %x : tensor<1x2048x4096xbf16>
  polykernel.return %4 : tensor<1x2048x4096xbf16>
}
