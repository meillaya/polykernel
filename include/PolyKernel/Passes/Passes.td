//===- Passes.td - PolyKernel pass definitions -------------*- tablegen -*-===//
//
// PolyKernel out-of-tree MLIR dialect: pass declarations (Todo 4 / Wave 1).
//
//===----------------------------------------------------------------------===//
//
// This file declares the PolyKernel transformation passes via ODS/TableGen
// (`-gen-pass-decls`). It is the single home for ALL PolyKernel pass decls so
// later todos extend it in place:
//
//   - Todo 4 (this file): `InferShapes` (`--infer-shapes`), shape + dtype
//     inference for every polykernel compute op.
//   - Todo 5 (future):    add a `Canonicalize`-style pass decl here too; the
//     `GEN_PASS_*` machinery below picks it up automatically (just add another
//     `def ... : Pass<...>` and a matching create/register helper).
//
// The generated registration is EXPLICIT (no registerAllPasses / MLIRRegisterAll*
// — nixpkgs MLIR does not ship the aggregate object archives; see
// docs/compiler_pipeline.md). `--gen-pass-decls -name PolyKernel` emits:
//   - GEN_PASS_DECL_<PASSNAME>  per-pass decl macros (e.g. INFER_SHAPES)
//   - GEN_PASS_REGISTRATION     registerInferShapesPass() + registerPolyKernelPasses()
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_PASSES
#define POLYKERNEL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// InferShapes
//===----------------------------------------------------------------------===//

def InferShapes : Pass<"infer-shapes", "ModuleOp"> {
  let summary = "Shape + dtype inference for all PolyKernel ops";
  let description = [{
    Walks every op in the module and computes each PolyKernel compute op's
    result type(s) from its operands via `InferTypeOpInterface::inferReturnTypes`,
    then sets/refines the op's result types to the inferred types.

    Inference rules (per op):
      - matmul:                A<...xMxK> x B<KxN> -> <...xMxN> (contraction dim
                               K must match; mismatch -> diagnostic + failure).
      - rmsnorm/gelu/silu/softmax: result type == input type.
      - add:                   two operands must have EQUAL shapes; result == that
                               shape (unequal -> diagnostic + failure).
      - bias:                  operand<...xN> + bias<N> -> operand shape (bias
                               broadcasts over the last dim).
      - rope/attention:        result preserves the input / query shape.
      - kv_cache_update:       result == updated cache shape.
      - fused_rmsnorm_matmul:  matmul(input, weight) -> MxN.
      - fused_matmul_bias_gelu: matmul(a, b) -> MxN.
      - fused_residual_rmsnorm: result == residual shape (residual == input).
      - fused_softmax_mask:    result == input shape.
      - qkv_projection:        input + qkv weight -> Q,K,V head split (uses the
                               `num_heads` attribute).
      - fused_kv_append_attention: output == query shape, updated_cache == cache
                               shape.

    On any inconsistency (e.g. mismatched matmul contraction dimension, unequal
    `add` operand shapes) the offending op's `inferReturnTypes` emits a diagnostic
    and this pass calls `signalPassFailure()`.
  }];

  // The pass is registered explicitly via the TableGen-generated
  // registerInferShapesPass() / registerPolyKernelPasses() (see Passes.h) — never
  // registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createInferShapesPass()";
}

#endif // POLYKERNEL_PASSES
