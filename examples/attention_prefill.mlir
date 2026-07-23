// examples/attention_prefill.mlir
//
// Todo 6 (Wave 1): example transformer-block (prefill) attention fragment.
//
// Pipeline: rmsnorm -> qkv_projection (num_heads=32) -> rope (cos + sin)
//           -> attention -> matmul (output projection) -> add (residual).
//
// Shapes (bf16): batch=1, seq=2048, d_model=4096, heads=32, head_dim=128.
//   %x    : tensor<1x2048x4096xbf16>      d_model stream
//   %qkvw : tensor<4096x12288xbf16>       qkv weight [hidden, 3*hidden]
//   %cos  : tensor<1x32x2048x128xbf16>    RoPE cos table
//   %sin  : tensor<1x32x2048x128xbf16>    RoPE sin table
//   %wo   : tensor<128x128xbf16>          per-head output projection
//   %res  : tensor<1x32x2048x128xbf16>    residual (multi-head layout; see note)
//   result: tensor<1x32x2048x128xbf16>
//
// Shape-inference check (--infer-shapes):
//   rmsnorm        : shape-preserving -> <1x2048x4096>
//   qkv_projection : hidden=4096, num_heads=32 (4096 % 32 == 0), head_dim=128;
//                    weight [4096, 12288] == [hidden, 3*hidden];
//                    Q = K = V = <1x32x2048x128>
//   rope (x2)      : shape-preserving (result == input) -> <1x32x2048x128>
//   attention      : output preserves the query shape -> <1x32x2048x128>
//   matmul         : <1x32x2048x128> x <128x128> (K=128==128) -> <1x32x2048x128>
//   add            : <1x32x2048x128> + <1x32x2048x128> (equal) -> <1x32x2048x128>
//
// NOTE (op-signature surprise / Wave-2 risk): polykernel.attention emits the
// multi-head layout <1x32x2048x128> and the 18-op set has NO reshape /
// transpose / concat op to collapse the heads back to the d_model stream
// <1x2048x4096>. The output projection and the residual add therefore stay in
// the per-head layout (the output projection is per-head head_dim->head_dim).
// A faithful d_model output projection (Wo: <4096x4096>) + d_model residual
// needs a reshape/concat op; adding one is a Wave-2 consideration.
//
// Only polykernel ops (rmsnorm, qkv_projection, rope, attention, matmul, add,
// func, return) + builtin tensors are used. No arith/func dialect ops.

polykernel.func @attention_prefill(%x: tensor<1x2048x4096xbf16>,
                                   %qkvw: tensor<4096x12288xbf16>,
                                   %cos: tensor<1x32x2048x128xbf16>,
                                   %sin: tensor<1x32x2048x128xbf16>,
                                   %wo: tensor<128x128xbf16>,
                                   %res: tensor<1x32x2048x128xbf16>)
    -> tensor<1x32x2048x128xbf16> {
  %0 = polykernel.rmsnorm %x {epsilon = 1.0e-05 : f32} : tensor<1x2048x4096xbf16>
  %q, %k, %v = polykernel.qkv_projection %0, %qkvw {num_heads = 32 : i64}
      : tensor<1x2048x4096xbf16>, tensor<4096x12288xbf16>
      -> tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
  %qr = polykernel.rope %q, %cos, %sin
      : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
      -> tensor<1x32x2048x128xbf16>
  %kr = polykernel.rope %k, %cos, %sin
      : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
      -> tensor<1x32x2048x128xbf16>
  %attn = polykernel.attention %qr, %kr, %v
      : tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>, tensor<1x32x2048x128xbf16>
      -> tensor<1x32x2048x128xbf16>
  %proj = polykernel.matmul %attn, %wo
      : tensor<1x32x2048x128xbf16>, tensor<128x128xbf16> -> tensor<1x32x2048x128xbf16>
  %out = polykernel.add %proj, %res : tensor<1x32x2048x128xbf16>
  polykernel.return %out : tensor<1x32x2048x128xbf16>
}
