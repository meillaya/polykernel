// RUN: polykernel-opt %s --fuse-kv-append-attention | FileCheck %s

// Todo 40 (Wave 8): --fuse-kv-append-attention fuses
//   kv_cache_update(cache, nk, nv); attention(q, nk, nv)
// into one polykernel.fused_kv_append_attention(q, cache, nk, nv) when the new K/V
// are single-use (each feeds EXACTLY the attention + the update). Produces the
// attention output type AND the updated-cache type. The dead attention + update are
// erased.
//
// The fused op uses the EXACT operand order from PolyKernelOps.td:
// (query, cache, new_keys, new_values); results are (output, updated_cache).
//
// This fusion eliminates TWO intermediates (the standalone updated-cache + the
// standalone attention output), so it uses the INDEXED tracking attrs
// `polykernel.eliminated_type_0` (update result) + `polykernel.eliminated_type_1`
// (attention result) + `polykernel.fused_from = "kv_append_attention"`.
//
// Guardrails exercised below:
//   POSITIVE: single-use new K/V shared by update + attention fuses (operand order
//             + indexed tracking pinned literally).
//   NEGATIVE: multi-use new_keys (attention AND another add) is NOT fused.
//   NEGATIVE: attention with NO matching kv_cache_update is NOT fused.

// ----- POSITIVE: single-use update + attention fuses; producers gone -----
// polykernel.func renumbers args to %argN: %arg0 = q (fused query), %arg1 = cache
// (fused cache), %arg2 = nk (fused new_keys), %arg3 = nv (fused new_values). The
// attr dict is pinned literally (attributes print sorted: eliminated_type_0,
// eliminated_type_1, fused_from).

// CHECK-LABEL: polykernel.func @fuse_kv_append(
// CHECK-NOT: polykernel.kv_cache_update
// CHECK-NOT: polykernel.attention
// CHECK: %{{.*}}, %{{.*}} = polykernel.fused_kv_append_attention %arg0, %arg1, %arg2, %arg3 {polykernel.eliminated_type_0 = tensor<1x2x4x8xf32>, polykernel.eliminated_type_1 = tensor<1x2x4x8xf32>, polykernel.fused_from = "kv_append_attention"} : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32> -> tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
// CHECK: polykernel.return
polykernel.func @fuse_kv_append(%q: tensor<1x2x4x8xf32>,
                                %cache: tensor<1x2x4x8xf32>,
                                %nk: tensor<1x2x4x8xf32>,
                                %nv: tensor<1x2x4x8xf32>)
    -> (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) {
  %updated = polykernel.kv_cache_update %cache, %nk, %nv
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
      -> tensor<1x2x4x8xf32>
  %out = polykernel.attention %q, %nk, %nv
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
      -> tensor<1x2x4x8xf32>
  polykernel.return %out, %updated : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
}

// ----- NEGATIVE: multi-use new_keys (attention AND another add) is NOT fused -----
// nk feeds three ops (update + attention + add), so fusing would duplicate the
// appended tensor; the standalone update + attention + add must all SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_multi_use_nk(
// CHECK-NOT: polykernel.fused_kv_append_attention
// CHECK: %{{.*}} = polykernel.kv_cache_update
// CHECK: %{{.*}} = polykernel.attention
// CHECK: %{{.*}} = polykernel.add
// CHECK: polykernel.return
polykernel.func @no_fuse_multi_use_nk(%q: tensor<1x2x4x8xf32>,
                                      %cache: tensor<1x2x4x8xf32>,
                                      %nk: tensor<1x2x4x8xf32>,
                                      %nv: tensor<1x2x4x8xf32>,
                                      %r: tensor<1x2x4x8xf32>)
    -> (tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>) {
  %updated = polykernel.kv_cache_update %cache, %nk, %nv
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
      -> tensor<1x2x4x8xf32>
  %out = polykernel.attention %q, %nk, %nv
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
      -> tensor<1x2x4x8xf32>
  %extra = polykernel.add %nk, %r : tensor<1x2x4x8xf32>
  polykernel.return %out, %updated, %extra
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
}

// ----- NEGATIVE: attention with NO matching kv_cache_update is NOT fused -----
// The attention's K/V are bare func args (no update appends them), so there is
// nothing to fuse; the standalone attention must SURVIVE.

// CHECK-LABEL: polykernel.func @no_fuse_no_update(
// CHECK-NOT: polykernel.fused_kv_append_attention
// CHECK: %{{.*}} = polykernel.attention
// CHECK: polykernel.return
polykernel.func @no_fuse_no_update(%q: tensor<1x2x4x8xf32>,
                                   %k: tensor<1x2x4x8xf32>,
                                   %v: tensor<1x2x4x8xf32>)
    -> tensor<1x2x4x8xf32> {
  %out = polykernel.attention %q, %k, %v
      : tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>, tensor<1x2x4x8xf32>
      -> tensor<1x2x4x8xf32>
  polykernel.return %out : tensor<1x2x4x8xf32>
}
