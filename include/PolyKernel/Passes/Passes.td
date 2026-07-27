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
//   - Todo 19: `LowerToHip` (`--lower-to-hip`), HIP sibling of --lower-to-cuda:
//     emits the SAME portable kernel source (shared kernel_common.h template);
//     the POLYKERNEL_HIP define is applied by the hipcc driver, not the emitter.
//   - Todo 12: `FuseRmsnormMatmul` (`--fuse-rmsnorm-matmul`), first fusion pass
//     (Wave 3): single-use rmsnorm + matmul -> fused_rmsnorm_matmul.
//   - Todo 13: `FuseMatmulBiasGelu` (`--fuse-matmul-bias-gelu`),
//     `FuseResidualRmsnorm` (`--fuse-residual-rmsnorm`),
//     `FuseSoftmaxMask` (`--fuse-softmax-mask`) — the three remaining Wave 3
//     fusion passes (single-use matmul+bias+gelu, residual-add+rmsnorm,
//     mask-add+softmax).
//   - Todo 15: `InferTileLayout` (`--infer-tile-layout`), assign tile shapes
//     (BLOCK_M/N/K) + operand layout to the matmul ops via shape heuristics.
//   - Todo 16: `PlanMemory` (`--plan-memory`), compute a per-kernel memory plan
//     (shared-memory budget + non-fused workspace + over-budget guard) for the
//     matmul ops as discardable attributes.
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
// LowerToHip
//===----------------------------------------------------------------------===//

def LowerToHip : Pass<"lower-to-hip", "ModuleOp"> {
  let summary = "Emit portable HIP kernel source (.cu) for PolyKernel ops";
  let description = [{
    The HIP sibling of `--lower-to-cuda` (Todo 19 / Wave 4). A CUSTOM SOURCE
    EMITTER (Pinned contract F): walks the PolyKernel IR and, for each recognized
    compute op kind, string-emits a portable CUDA/HIP kernel source file (.cu)
    into `--output-dir`. It SHARES the codegen template with `--lower-to-cuda`:
    the emitted .cu is written against kernels/template/kernel_common.h and is
    byte-identical portable compute — the two passes differ ONLY in that the HIP
    variant is compiled by the hipcc driver with `-DPOLYKERNEL_HIP`
    (tools/polykernel-bench/hipcc_driver.py), while the CUDA variant is compiled
    by the nvcc driver with `-DPOLYKERNEL_CUDA`. The POLYKERNEL_HIP define is a
    DRIVER concern, not an emitter concern; the emitter never sees it. It does NOT
    use gpu-to-llvm / convert-to-llvm; the IR itself is left unchanged.

    Emits the same seven kernels as --lower-to-cuda: rmsnorm, gelu, silu, matmul,
    softmax, fused_rmsnorm_matmul, fused_matmul_bias_gelu. hipcc compiles every
    one for gfx1101 UNCHANGED (the portability proof; see
    reports/w4_hipcc_compile.log).
  }];

  let options = [
    Option<"outputDir", "output-dir", "std::string",
           /*default=*/"\"kernels/generated\"",
           "Directory to write the generated .cu kernel sources">
  ];

  // Registered explicitly via the TableGen-generated registerLowerToHipPass()
  // / registerPolyKernelPasses() (see Passes.h) — never registerAllPasses (nix
  // out-of-tree constraint).
  let constructor = "::mlir::polykernel::createLowerToHipPass()";
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

//===----------------------------------------------------------------------===//
// FuseMatmulBiasGelu
//===----------------------------------------------------------------------===//

def FuseMatmulBiasGelu : Pass<"fuse-matmul-bias-gelu", "ModuleOp"> {
  let summary = "Fuse single-use matmul + bias + gelu into fused_matmul_bias_gelu";
  let description = [{
    PolyKernel fusion pass (Wave 3, Todo 13). Matches the chain
    `matmul(a, b); bias(mm, bias); gelu(bs)` and rewrites it into
    `fused_matmul_bias_gelu(a, b, bias)`, producing the gelu's result type.
    Applied with the greedy pattern driver (`applyPatternsGreedily`), which also
    folds + DCEs + simplifies regions.

    Guardrails:
      - The gelu's input must be DEFINED by a `polykernel.bias`, and that bias's
        input must be DEFINED by a `polykernel.matmul` (the exact 3-op chain).
      - SINGLE-USE CHECK (essential, per absorbed intermediate): the matmul result
        must have EXACTLY ONE use (the bias) AND the bias result must have EXACTLY
        ONE use (the gelu). A multi-use matmul OR bias is NOT fused — fusing would
        duplicate the shared producer.

    Eliminated-intermediate tracking (for the Todo 17 traffic report): this fusion
    eliminates TWO intermediates, so each fused op is tagged with the discardable
    attributes `polykernel.fused_from = "matmul_bias_gelu"` and the INDEXED
    `polykernel.eliminated_type_0 = <matmul result type>`,
    `polykernel.eliminated_type_1 = <bias result type>` (producer order). See the
    single-vs-indexed scheme note in FuseMatmulBiasGelu.cpp; the report sums
    numElements(type) * sizeof(element) over every eliminated intermediate.
  }];

  // Registered explicitly via the TableGen-generated
  // registerFuseMatmulBiasGeluPass() / registerPolyKernelPasses() (see Passes.h)
  // — never registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createFuseMatmulBiasGeluPass()";
}

//===----------------------------------------------------------------------===//
// FuseResidualRmsnorm
//===----------------------------------------------------------------------===//

def FuseResidualRmsnorm : Pass<"fuse-residual-rmsnorm", "ModuleOp"> {
  let summary = "Fuse single-use residual add + rmsnorm into fused_residual_rmsnorm";
  let description = [{
    PolyKernel fusion pass (Wave 3, Todo 13). Matches
    `add(residual, input); rmsnorm(sum, eps)` and rewrites it into
    `fused_residual_rmsnorm(residual, input, eps)`, producing the rmsnorm's result
    type and carrying the rmsnorm's optional `epsilon`. Applied with the greedy
    pattern driver (`applyPatternsGreedily`), which also folds + DCEs + simplifies
    regions.

    Guardrails:
      - The rmsnorm's input must be DEFINED by a `polykernel.add` (residual add).
      - SINGLE-USE CHECK (essential): the add result must have EXACTLY ONE use (the
        rmsnorm). A multi-use add is NOT fused — fusing would duplicate the sum.
      - OPERAND MAPPING (positional): the add's `lhs` -> fused `residual`, `rhs` ->
        fused `input` (add is commutative, so this is numerically equivalent either
        way; the positional convention keeps the rewrite deterministic).

    Eliminated-intermediate tracking (for the Todo 17 traffic report): each fused
    op is tagged with the discardable attributes
    `polykernel.fused_from = "residual_rmsnorm"` and
    `polykernel.eliminated_type = <the eliminated add result's tensor type>`
    (unindexed — this fusion eliminates ONE intermediate).
  }];

  // Registered explicitly via the TableGen-generated
  // registerFuseResidualRmsnormPass() / registerPolyKernelPasses() (see Passes.h)
  // — never registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createFuseResidualRmsnormPass()";
}

//===----------------------------------------------------------------------===//
// FuseSoftmaxMask
//===----------------------------------------------------------------------===//

def FuseSoftmaxMask : Pass<"fuse-softmax-mask", "ModuleOp"> {
  let summary = "Fuse single-use mask add + softmax into fused_softmax_mask";
  let description = [{
    PolyKernel fusion pass (Wave 3, Todo 13). Matches
    `add(input, mask); softmax(sum)` and rewrites it into
    `fused_softmax_mask(input, mask)`, producing the softmax's result type.
    Applied with the greedy pattern driver (`applyPatternsGreedily`), which also
    folds + DCEs + simplifies regions.

    Guardrails:
      - The softmax's input must be DEFINED by a `polykernel.add` (the mask add).
      - SINGLE-USE CHECK (essential): the add result must have EXACTLY ONE use (the
        softmax). A multi-use add is NOT fused — fusing would duplicate the sum.
      - OPERAND MAPPING (positional): the add's `lhs` -> fused `input`, `rhs` ->
        fused `mask` (add is commutative, so this is numerically equivalent either
        way; the positional convention keeps the rewrite deterministic).
      - `fused_softmax_mask` declares NO `axis` attribute (PolyKernelOps.td), so the
        softmax's optional `axis` is intentionally NOT carried.

    Eliminated-intermediate tracking (for the Todo 17 traffic report): each fused
    op is tagged with the discardable attributes
    `polykernel.fused_from = "softmax_mask"` and
    `polykernel.eliminated_type = <the eliminated add result's tensor type>`
    (unindexed — this fusion eliminates ONE intermediate).
  }];

  // Registered explicitly via the TableGen-generated
  // registerFuseSoftmaxMaskPass() / registerPolyKernelPasses() (see Passes.h)
  // — never registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createFuseSoftmaxMaskPass()";
}

//===----------------------------------------------------------------------===//
// InferTileLayout
//===----------------------------------------------------------------------===//

def InferTileLayout : Pass<"infer-tile-layout", "ModuleOp"> {
  let summary = "Assign tile shapes (BLOCK_M/N/K) + operand layout to matmul ops";
  let description = [{
    PolyKernel tile-layout pass (Wave 3, Todo 15). Walks the matmul-structured ops
    (`matmul`, `fused_rmsnorm_matmul`, `fused_matmul_bias_gelu`) and assigns each an
    INITIAL tile shape + operand layout as DISCARDABLE attributes (the op set stays
    closed — no new declared op attributes are added to PolyKernelOps.td):

      - `polykernel.tile_m` / `polykernel.tile_n` / `polykernel.tile_k` (I64Attr):
        the BLOCK_M / BLOCK_N / BLOCK_K the lowering targets. The baseline CUDA GEMM
        (kernels/generated/matmul.cu) uses 16x16x16.
      - `polykernel.layout` (StringAttr): the operand memory layout — always
        "row_major" today (the baseline; transposed / packed layouts are a future
        refinement).

    Shape extraction (operand 0 = [..,M,K], operand 1 = [K,N], matching the kernel's
    "flatten leading dims into M" model): K = last dim of operand 0, N = last dim of
    operand 1, M = product of operand 0's leading dims.

    Tile heuristic (simple, deterministic): for a problem dim D, pick the LARGEST
    power-of-two candidate in {128, 64, 32, 16} that is <= D; if D < 16, clamp the
    tile to D itself (SMALL-DIM CLAMP — tile_m <= M, tile_n <= N, tile_k <= K,
    always). These are DEFAULTS the autotuner (Todo 24) refines later; they need only
    be sensible + deterministic + clamped, not optimal. Runs on typed IR (after
    `--infer-shapes`); ops with a dynamic operand dim are skipped.

    Scope: the three PRIMARY matmul ops. qkv_projection / fused_kv_append_attention
    are matmul-LIKE but have a head-split / cache structure (not a plain [M,K]x[K,N])
    and are intentionally out of scope for this initial pass.
  }];

  // Registered explicitly via the TableGen-generated
  // registerInferTileLayoutPass() / registerPolyKernelPasses() (see Passes.h)
  // — never registerAllPasses (nix out-of-tree constraint).
  let constructor = "::mlir::polykernel::createInferTileLayoutPass()";
}

//===----------------------------------------------------------------------===//
// PlanMemory
//===----------------------------------------------------------------------===//

def PlanMemory : Pass<"plan-memory", "ModuleOp"> {
  let summary = "Compute a per-kernel memory plan (smem budget + workspace + "
                "over-budget guard) for the matmul ops";
  let description = [{
    PolyKernel memory-plan pass (Wave 3, Todo 16). Walks the matmul-structured ops
    (`matmul`, `fused_rmsnorm_matmul`, `fused_matmul_bias_gelu`) and computes a
    per-kernel memory plan, attaching DISCARDABLE attributes (the op set stays
    closed — no new declared op attributes are added to PolyKernelOps.td). These are
    PLANNING / REPORTING estimates, not cycle-accurate.

      - `polykernel.smem_bytes` (I64Attr): the tiled shared-memory budget,
        `(tile_m*tile_k + tile_k*tile_n) * dtype_bytes * stages`. tile_m/n/k are read
        from the `--infer-tile-layout` attrs (ABSENT -> fallback 16x16x16, the
        baseline CUDA GEMM tile); dtype_bytes is the operand-0 element width / 8
        (bf16/fp16 = 2, fp32 = 4); stages = 2 (a documented default double-buffer).
        `polykernel.layout` is READ (fallback "row_major"), not hardcoded; the
        baseline footprint is layout-agnostic.
      - `polykernel.workspace_bytes` (I64Attr): global scratch for intermediates NOT
        fused away — 0 for a fused op (carries `polykernel.fused_from`; its
        `polykernel.eliminated_type*` intermediates are gone), else the standalone
        result-tensor size numElements(result) * dtype_bytes.
      - `polykernel.smem_over_budget` (BoolAttr=true): set only when `smem_bytes`
        exceeds the per-block limit (163 * 1024 bytes — the sm_80 budget, a single
        conservative default; sm_90 allows 227KB). A warning is also emitted.

    Runs on typed IR (after `--infer-shapes`, and after `--infer-tile-layout` when
    tile-accurate budgets are wanted). Ops with a dynamic operand/result dim are
    skipped (consistent with --infer-tile-layout).

    Scope: the three PRIMARY matmul ops. qkv_projection / fused_kv_append_attention
    are matmul-LIKE but have a head-split / cache structure and are out of scope.
  }];

  // Registered explicitly via the TableGen-generated registerPlanMemoryPass() /
  // registerPolyKernelPasses() (see Passes.h) — never registerAllPasses (nix
  // out-of-tree constraint).
  let constructor = "::mlir::polykernel::createPlanMemoryPass()";
}

//===----------------------------------------------------------------------===//
// EmitKernelReport
//===----------------------------------------------------------------------===//

def EmitKernelReport : Pass<"emit-kernel-report", "ModuleOp"> {
  let summary = "Emit/attach the contract-H per-kernel report skeleton for the "
                "matmul-structured kernels";
  let description = [{
    PolyKernel full per-kernel report pass (Wave 5, Todo 27). Walks the
    matmul-structured ops (`matmul`, `fused_rmsnorm_matmul`,
    `fused_matmul_bias_gelu` — the same scope as --infer-tile-layout /
    --plan-memory) and, for each generated kernel, ATTACHES the IR-derivable
    contract-H report fields as DISCARDABLE attributes (the op set stays CLOSED —
    no new declared op attributes are added to PolyKernelOps.td) and EMITS a
    per-kernel report manifest (`kernel_report.json`) into `--output-dir`.

    The fields this pass can derive from the IR alone are attached / emitted:
    the kernel name (from the op kind), backend + arch + path (from the pass
    options), the GEMM shape (M/N/K from the operands) + dtype bytes, the tile
    shape (read from the --infer-tile-layout attrs, fallback 16x16x16), the
    shared-memory budget (read from the --plan-memory `polykernel.smem_bytes`
    attr), and the fusion provenance (`polykernel.fused_from`).

    The COMPILE-TIME fields — registers/thread, spill bytes, occupancy, global
    traffic, roofline, bottleneck, and the suggested fixes — require the
    nvcc/ptxas (CUDA) or hipcc --save-temps (AMD) analyzers and are filled by the
    `polykernel-report` CLI (tools/polykernel-report/report.py), which merges this
    IR skeleton with the CUDA compile-time analysis (Todo 11), the AMD ISA
    analysis (Todo 21), and the autotuner result (Todo 25) into the full
    contract-H report. This pass is the IR half of that pipeline; it does NOT
    fabricate register/spill numbers (those need a compiler).

    Attached discardable attributes (per kernel op):
      - `polykernel.report_kernel`  (StringAttr): the kernel name.
      - `polykernel.report_backend` (StringAttr): "cuda" | "hip".
      - `polykernel.report_arch`    (StringAttr): the target arch string.
      - `polykernel.report_path`    (StringAttr): "scalar" | "wmma" | "mma".

    Runs on typed IR (after `--infer-shapes`); ops with a dynamic operand dim are
    skipped (consistent with --infer-tile-layout / --plan-memory). The IR is
    otherwise left unchanged (report emitter, not a transform).
  }];

  let options = [
    Option<"outputDir", "output-dir", "std::string",
           /*default=*/"\"build/report\"",
           "Directory to write the kernel_report.json manifest into">,
    Option<"backend", "backend", "std::string", /*default=*/"\"cuda\"",
           "Target backend recorded in the report: cuda or hip">,
    Option<"arch", "arch", "std::string", /*default=*/"\"sm_90\"",
           "Target arch recorded in the report, e.g. sm_90 or gfx1101">,
    Option<"path", "path", "std::string", /*default=*/"\"scalar\"",
           "Kernel implementation path recorded in the report: scalar, wmma, "
           "or mma">
  ];

  // Registered explicitly via the TableGen-generated registerEmitKernelReportPass()
  // / registerPolyKernelPasses() (see Passes.h) — never registerAllPasses (nix
  // out-of-tree constraint).
  let constructor = "::mlir::polykernel::createEmitKernelReportPass()";
}

#endif // POLYKERNEL_PASSES
