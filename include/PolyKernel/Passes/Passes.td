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
//   - Todo 4: `InferShapes` (`--infer-shapes`), shape + dtype inference for
//     every polykernel compute op.
//   - Todo 5: `Canonicalize` (`--canonicalize`), canonicalization patterns
//     (identity residual-add elimination, elementwise constant folding,
//     softmax-of-splat no-op fold) + DCE via the greedy pattern driver.
//   - Todo 8: `LowerToCuda` (`--lower-to-cuda`), portable CUDA/HIP source emit.
//   - Todo 12: `FuseRmsnormMatmul` (`--fuse-rmsnorm-matmul`), first fusion pass
//     (Wave 3): single-use rmsnorm + matmul -> fused_rmsnorm_matmul.
//
// The generated registration is EXPLICIT (no registerAllPasses / MLIRRegisterAll*
// — nixpkgs MLIR does not ship the aggregate object archives; see
// docs/compiler_pipeline.md). `--gen-pass-decls -name PolyKernel` emits:
//   - GEN_PASS_DECL_<PASSNAME>  per-pass decl macros (e.g. INFER_SHAPES,
//                               CANONICALIZE)
//   - GEN_PASS_REGISTRATION     registerInferShapesPass() /
//                               registerCanonicalizePass() /
//                               registerFuseRmsnormMatmulPass() /
//                               registerPolyKernelPasses()
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

//===----------------------------------------------------------------------===//
// Canonicalize
//===----------------------------------------------------------------------===//

def Canonicalize : Pass<"canonicalize", "ModuleOp"> {
  let summary = "Canonicalize PolyKernel ops: identity-add elimination, "
                "constant folding, DCE";
  let description = [{
    Applies the PolyKernel canonicalization patterns with the greedy pattern
    rewrite driver (`applyPatternsGreedily`), which also folds ops, removes
    ops left dead by the rewrites (DCE) and simplifies regions. This is a
    self-contained PolyKernel canonicalizer (design A): the patterns live in
    the pass, not on the upstream `--canonicalize` infrastructure.

    Patterns:
      - Identity residual-add elimination: `polykernel.add(x, zero) -> x` and
        `add(zero, x) -> x`, where `zero` is a dense-zero `arith.constant`
        (splat or general). `add`'s result type equals its operand types
        (SameTypeOperands), so replacing the op with the non-zero operand
        keeps types consistent. The dead zero constant is DCE'd.
      - Constant folding of pure elementwise ops: `polykernel.gelu` /
        `polykernel.silu` of a dense floating-point `arith.constant` fold to a
        computed `arith.constant` (every element is mapped through the
        activation, evaluated in double precision and rounded to the element
        type; works for splat and general dense constants). GELU uses the
        exact erf-based definition `0.5 * x * (1 + erf(x / sqrt(2)))`; SiLU
        uses `x / (1 + exp(-x))`.
      - No-op fold: `polykernel.softmax` of a splat constant folds to the
        uniform distribution `splat(1/N)`, where `N` is the size of the
        reduction axis (the `axis` attribute, defaulting to the last dim):
        softmax of equal values is uniform because the `exp(c)` cancels in
        the normalization — no transcendental math needed.

    Non-foldable cases (e.g. `add` of two non-constant operands, softmax of a
    NON-splat constant) are left unchanged.
  }];

  // Registered explicitly via the TableGen-generated registerCanonicalizePass()
  // / registerPolyKernelPasses() (see Passes.h) — never registerAllPasses (nix
  // out-of-tree constraint).
  let constructor = "::mlir::polykernel::createCanonicalizePass()";
}

//===----------------------------------------------------------------------===//
// LowerToCuda
//===----------------------------------------------------------------------===//

def LowerToCuda : Pass<"lower-to-cuda", "ModuleOp"> {
  let summary = "Emit portable CUDA/HIP kernel source (.cu) for PolyKernel ops";
  let description = [{
    A CUSTOM SOURCE EMITTER (Pinned contract F): walks the PolyKernel IR and, for
    each recognized compute op kind, string-emits a portable CUDA/HIP kernel
    source file (.cu) into `--output-dir`. It does NOT use gpu-to-llvm /
    convert-to-llvm; the emitted .cu is written against
    kernels/template/kernel_common.h and compiled to PTX by the nvcc driver. The
    IR itself is left unchanged.

    Todo 8 emits RMSNorm (`kernels/generated/rmsnorm.cu`); Todo 9/10 extend the
    dispatch to gelu/silu/matmul/softmax.
  }];

  let options = [
    Option<"outputDir", "output-dir", "std::string",
           /*default=*/"\"kernels/generated\"",
           "Directory to write the generated .cu kernel sources">
  ];

  // Registered explicitly via the TableGen-generated registerLowerToCudaPass()
  // / registerPolyKernelPasses() (see Passes.h) — never registerAllPasses (nix
  // out-of-tree constraint).
  let constructor = "::mlir::polykernel::createLowerToCudaPass()";
}

//===----------------------------------------------------------------------===//
// FuseRmsnormMatmul
//===----------------------------------------------------------------------===//

def FuseRmsnormMatmul : Pass<"fuse-rmsnorm-matmul", "ModuleOp"> {
  let summary = "Fuse single-use rmsnorm + matmul into fused_rmsnorm_matmul";
  let description = [{
    First PolyKernel fusion pass (Wave 3). Matches a `polykernel.matmul` whose
    input (A) operand is produced by a `polykernel.rmsnorm` and rewrites
    `rmsnorm(x, eps); matmul(rms, w)` into `fused_rmsnorm_matmul(x, w, eps)`,
    producing the matmul's (MxN) result type and carrying the rmsnorm's optional
    `epsilon` attribute. Applied with the greedy pattern driver
    (`applyPatternsGreedily`), which also folds + DCEs + simplifies regions.

    Guardrails:
      - The matmul's A (input) operand must be DEFINED by a `polykernel.rmsnorm`
        (a rmsnorm feeding the matmul's B operand is NOT fused).
      - SINGLE-USE CHECK (essential): the rmsnorm result must have EXACTLY ONE
        use (the matmul). A rmsnorm feeding two or more ops (e.g. the matmul AND
        a residual add) is NOT fused — fusing would duplicate the normalization.

    Eliminated-intermediate tracking (for the Todo 17 traffic report): each fused
    op is tagged with the discardable attributes
    `polykernel.fused_from = "rmsnorm_matmul"` and
    `polykernel.eliminated_type = <the eliminated rmsnorm result's tensor type>`,
    so the report can identify the fusion and compute the saved global-memory
    round-trip as numElements(type) * sizeof(element).
  }];

  // Registered explicitly via the TableGen-generated
  // registerFuseRmsnormMatmulPass() / registerPolyKernelPasses() (see Passes.h)
  // — never registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createFuseRmsnormMatmulPass()";
}

#endif // POLYKERNEL_PASSES
