//===- PolyKernelOps.td - PolyKernel dialect operations ----*- tablegen -*-===//
//
// PolyKernel out-of-tree MLIR dialect: operation definitions (Todo 3 / Wave 1).
//
//===----------------------------------------------------------------------===//
//
// This file defines EXACTLY the PolyKernel transformer-inference op set and
// nothing else (op-zoo guardrail). The ops are:
//
//   Base ops:
//     polykernel.func            function-like op (region container; see below)
//     polykernel.return          terminator for polykernel.func (NOT a compute op)
//     polykernel.rmsnorm         RMS normalization
//     polykernel.matmul          matrix multiply (A:MxK, B:KxN -> MxN)
//     polykernel.bias            bias add
//     polykernel.gelu            GELU activation
//     polykernel.silu            SiLU activation
//     polykernel.add             elementwise (residual) add
//     polykernel.softmax         softmax
//     polykernel.rope            rotary position embedding
//     polykernel.attention       attention (prefill), Q/K/V -> out
//     polykernel.kv_cache_update append new K/V into a KV cache
//
//   Fused ops:
//     polykernel.fused_rmsnorm_matmul
//     polykernel.fused_matmul_bias_gelu
//     polykernel.fused_residual_rmsnorm
//     polykernel.fused_softmax_mask
//     polykernel.qkv_projection
//     polykernel.fused_kv_append_attention
//
// `polykernel.func` / `polykernel.return` are the function container + its
// terminator (modeled on func.func / func.return via FunctionOpInterface); they
// are NOT counted among the "compute" ops, so the compute op-zoo guardrail still
// holds. No other ops are defined.
//
// Semantics note (Todo 3 scope): the ops only need to PARSE and ROUND-TRIP with
// explicit tensor types in the IR. Full shape inference is Todo 4. A few ops
// carry light verifiers (rank sanity) where cheap; nothing here infers shapes.
//
//===----------------------------------------------------------------------===//

#ifndef POLYKERNEL_OPS
#define POLYKERNEL_OPS

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "PolyKernel/IR/PolyKernelDialect.td"

//===----------------------------------------------------------------------===//
// PolyKernel op base class + type constraints.
//===----------------------------------------------------------------------===//

class PolyKernel_Op<string mnemonic, list<Trait> traits = []>
    : Op<PolyKernel_Dialect, mnemonic, traits>;

// All PolyKernel compute ops operate on builtin tensors (AnyTensor, from
// CommonTypeConstraints.td).

//===----------------------------------------------------------------------===//
// PolyKernel compute-op base class with shape inference (Todo 4 / Wave 1).
//===----------------------------------------------------------------------===//
//
// Every compute op implements `InferTypeOpInterface` so its result type(s) can be
// computed from its operands. The inference logic lives in each op's
// adaptor-based `inferReturnTypes` (defined in PolyKernelOps.cpp); the
// `--infer-shapes` pass calls it (via the interface) to refine result types and
// signal failure on inconsistency.
//
// We declare BOTH `inferReturnTypes` and `refineReturnTypes`. `refineReturnTypes`
// is overridden to a no-op so the InferTypeOpInterface verification hook does NOT
// reject IR at parse time: shape consistency is enforced by the `--infer-shapes`
// pass (which calls `inferReturnTypes` + `signalPassFailure`), not by the
// verifier. This keeps fully-explicit round-trip IR valid even when a result type
// is a not-yet-inferred placeholder, and makes the pass the meaningful enforcer.
class PolyKernel_InferTypeOp<string mnemonic, list<Trait> traits = []>
    : PolyKernel_Op<mnemonic, !listconcat(traits, [
        DeclareOpInterfaceMethods<InferTypeOpInterface,
                                  ["inferReturnTypes", "refineReturnTypes"]>
      ])> {
  let extraClassDeclaration = [{
    /// Per-op shape inference (adaptor form), defined in PolyKernelOps.cpp:
    /// compute the result type(s) from the operands, emitting a diagnostic +
    /// returning failure on any inconsistency. The InferTypeOpInterface
    /// full-signature `inferReturnTypes` (declared by DeclareOpInterfaceMethods)
    /// builds the Adaptor and delegates here; `refineReturnTypes` is a no-op
    /// (defined in PolyKernelOps.cpp) so the interface verification hook does not
    /// reject IR at parse time — shape consistency is enforced by --infer-shapes.
    static ::llvm::LogicalResult inferReturnTypes(
        ::mlir::MLIRContext *context, std::optional<::mlir::Location> location,
        Adaptor adaptor,
        ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes);
  }];
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def PolyKernel_FuncOp : PolyKernel_Op<"func", [
    AutomaticAllocationScope, FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "PolyKernel function: a named SSACFG region with a tensor signature";
  let description = [{
    A function-like operation modeled on `func.func`. Carries a symbol name, a
    function type (tensor argument types -> tensor result types) and a body
    region terminated by `polykernel.return`. `FunctionOpInterface` brings in
    `SymbolOpInterface` + `CallableOpInterface`.

    Example:

    ```mlir
    polykernel.func @mlp_block(%x: tensor<1x2048x4096xbf16>,
                               %w1: tensor<4096x11008xbf16>)
        -> tensor<1x2048x4096xbf16> {
      %0 = polykernel.matmul %x, %w1
          : tensor<1x2048x4096xbf16>, tensor<4096x11008xbf16>
          -> tensor<1x2048x11008xbf16>
      polykernel.return %0 : tensor<1x2048x11008xbf16>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface methods
    //===------------------------------------------------------------------===//

    /// Region that is callable; null for an external (body-less) declaration.
    ::mlir::Region *getCallableRegion() {
      return isExternal() ? nullptr : &getBody();
    }

    /// Argument types of the function.
    ::llvm::ArrayRef<::mlir::Type> getArgumentTypes() {
      return getFunctionType().getInputs();
    }

    /// Result types of the function.
    ::llvm::ArrayRef<::mlir::Type> getResultTypes() {
      return getFunctionType().getResults();
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }

    /// True when the function has no body (an external declaration).
    bool isExternal();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReturnOp (terminator for polykernel.func; NOT a compute op)
//===----------------------------------------------------------------------===//

def PolyKernel_ReturnOp : PolyKernel_Op<"return",
    [Pure, HasParent<"FuncOp">, ReturnLike, Terminator]> {
  let summary = "PolyKernel function return (terminator)";
  let description = [{
    Terminator for `polykernel.func`. Takes a variable number of operands and
    produces no results; the operand types must match the enclosing function's
    result types. This is the function terminator, not a compute op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise / shape-preserving base ops (result type == input type, inferred
// via InferTypeOpInterface).
//===----------------------------------------------------------------------===//

def PolyKernel_RmsNormOp : PolyKernel_InferTypeOp<"rmsnorm", [Pure]> {
  let summary = "Root-mean-square normalization (shape-preserving)";
  let description = [{
    RMS normalization: `y = x / sqrt(mean(x^2) + epsilon)`. Result has the same
    shape/type as the input. `epsilon` is an optional stabilization constant.
  }];
  let arguments = (ins AnyTensor:$input,
                       OptionalAttr<F32Attr>:$epsilon);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PolyKernel_GeluOp : PolyKernel_InferTypeOp<"gelu", [Pure]> {
  let summary = "GELU activation (elementwise, shape-preserving)";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PolyKernel_SiluOp : PolyKernel_InferTypeOp<"silu", [Pure]> {
  let summary = "SiLU / swish activation (elementwise, shape-preserving)";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def PolyKernel_AddOp : PolyKernel_InferTypeOp<"add", [Pure, SameTypeOperands]> {
  let summary = "Elementwise (residual) add of two same-shaped tensors";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

def PolyKernel_SoftmaxOp : PolyKernel_InferTypeOp<"softmax", [Pure]> {
  let summary = "Softmax (shape-preserving)";
  let description = [{
    Softmax over `axis` (defaults to the last dimension when unset). Result has
    the same shape/type as the input.
  }];
  let arguments = (ins AnyTensor:$input,
                       OptionalAttr<I64Attr>:$axis);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// Base ops with distinct operand/result shapes (explicit result types).
//===----------------------------------------------------------------------===//

def PolyKernel_MatMulOp : PolyKernel_InferTypeOp<"matmul", [Pure]> {
  let summary = "Matrix multiply: A (MxK) x B (KxN) -> C (MxN)";
  let arguments = (ins AnyTensor:$a, AnyTensor:$b);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$a `,` $b attr-dict `:` type($a) `,` type($b) `->` type($result)";
  let hasVerifier = 1;
}

def PolyKernel_BiasOp : PolyKernel_InferTypeOp<"bias", [Pure]> {
  let summary = "Bias add: input + bias -> result";
  let description = [{
    Adds a bias tensor (typically 1-D over the last dim, broadcast) to the input.
    Result shape follows the input.
  }];
  let arguments = (ins AnyTensor:$input, AnyTensor:$bias);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$input `,` $bias attr-dict `:` type($input) `,` type($bias) `->` "
      "type($result)";
}

def PolyKernel_RopeOp : PolyKernel_InferTypeOp<"rope", [Pure]> {
  let summary = "Rotary position embedding";
  let description = [{
    Applies rotary position embedding to `input` using the precomputed `cos`
    and `sin` tables. Result has the same shape/type as the input.

    Note (Todo 3): `cos`/`sin` are modeled as explicit operands so the op has a
    clean declarative assembly format (optional operands would require
    AttrSizedOperandSegments + a custom parser). Whether they become truly
    optional is a Todo 4 concern.
  }];
  let arguments = (ins AnyTensor:$input,
                       AnyTensor:$cos,
                       AnyTensor:$sin);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$input `,` $cos `,` $sin attr-dict `:` type($input) `,` type($cos) "
      "`,` type($sin) `->` type($result)";
}

def PolyKernel_AttentionOp : PolyKernel_InferTypeOp<"attention", [Pure]> {
  let summary = "Attention (prefill): Q, K, V -> output";
  let arguments = (ins AnyTensor:$query,
                       AnyTensor:$key,
                       AnyTensor:$value);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$query `,` $key `,` $value attr-dict `:` type($query) `,` type($key) "
      "`,` type($value) `->` type($result)";
  let hasVerifier = 1;
}

def PolyKernel_KVCacheUpdateOp : PolyKernel_InferTypeOp<"kv_cache_update", [Pure]> {
  let summary = "Append new K/V into a KV cache -> updated cache";
  let arguments = (ins AnyTensor:$cache,
                       AnyTensor:$new_keys,
                       AnyTensor:$new_values);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$cache `,` $new_keys `,` $new_values attr-dict `:` type($cache) `,` "
      "type($new_keys) `,` type($new_values) `->` type($result)";
}

//===----------------------------------------------------------------------===//
// Fused ops (combine the corresponding base ops' operands/results).
//===----------------------------------------------------------------------===//

def PolyKernel_FusedRmsNormMatMulOp : PolyKernel_InferTypeOp<"fused_rmsnorm_matmul",
    [Pure]> {
  let summary = "Fused RMSNorm + MatMul: input + weight -> output";
  let arguments = (ins AnyTensor:$input,
                       AnyTensor:$weight,
                       OptionalAttr<F32Attr>:$epsilon);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$input `,` $weight attr-dict `:` type($input) `,` type($weight) `->` "
      "type($result)";
}

def PolyKernel_FusedMatMulBiasGeluOp : PolyKernel_InferTypeOp<"fused_matmul_bias_gelu",
    [Pure]> {
  let summary = "Fused MatMul + Bias + GELU: A + B + bias -> output";
  let arguments = (ins AnyTensor:$a,
                       AnyTensor:$b,
                       AnyTensor:$bias);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$a `,` $b `,` $bias attr-dict `:` type($a) `,` type($b) `,` type($bias) "
      "`->` type($result)";
}

def PolyKernel_FusedResidualRmsNormOp : PolyKernel_InferTypeOp<"fused_residual_rmsnorm",
    [Pure]> {
  let summary = "Fused residual add + RMSNorm: residual + input -> output";
  let arguments = (ins AnyTensor:$residual,
                       AnyTensor:$input,
                       OptionalAttr<F32Attr>:$epsilon);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$residual `,` $input attr-dict `:` type($residual) `,` type($input) "
      "`->` type($result)";
}

def PolyKernel_FusedSoftmaxMaskOp : PolyKernel_InferTypeOp<"fused_softmax_mask",
    [Pure]> {
  let summary = "Fused mask + softmax: input + mask -> output";
  let arguments = (ins AnyTensor:$input, AnyTensor:$mask);
  let results = (outs AnyTensor:$result);
  let assemblyFormat =
      "$input `,` $mask attr-dict `:` type($input) `,` type($mask) `->` "
      "type($result)";
}

def PolyKernel_QKVProjectionOp : PolyKernel_InferTypeOp<"qkv_projection", [Pure]> {
  let summary = "QKV projection: input + qkv weight -> Q, K, V";
  let description = [{
    Projects `input` (shape `[batch, seq, hidden]`) by `weight` (shape
    `[hidden, 3*hidden]`) and splits the result into the Q, K, V heads. Each of
    Q/K/V has shape `[batch, num_heads, seq, hidden/num_heads]`, so `num_heads`
    must divide `hidden`. `num_heads` is required for shape inference (the head
    split is otherwise under-determined from the operand shapes alone).
  }];
  let arguments = (ins AnyTensor:$input, AnyTensor:$weight,
                       OptionalAttr<I64Attr>:$num_heads);
  let results = (outs AnyTensor:$q,
                      AnyTensor:$k,
                      AnyTensor:$v);
  let assemblyFormat =
      "$input `,` $weight attr-dict `:` type($input) `,` type($weight) `->` "
      "type($q) `,` type($k) `,` type($v)";
}

def PolyKernel_FusedKVAppendAttentionOp
    : PolyKernel_InferTypeOp<"fused_kv_append_attention", [Pure]> {
  let summary = "Fused KV-append + attention: Q + cache + new K/V -> "
                "output + updated cache";
  let arguments = (ins AnyTensor:$query,
                       AnyTensor:$cache,
                       AnyTensor:$new_keys,
                       AnyTensor:$new_values);
  let results = (outs AnyTensor:$output,
                      AnyTensor:$updated_cache);
  let assemblyFormat =
      "$query `,` $cache `,` $new_keys `,` $new_values attr-dict `:` "
      "type($query) `,` type($cache) `,` type($new_keys) `,` type($new_values) "
      "`->` type($output) `,` type($updated_cache)";
}

#endif // POLYKERNEL_OPS
