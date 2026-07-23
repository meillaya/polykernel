// RUN: not polykernel-opt %s --infer-shapes 2>&1 | FileCheck %s

// Todo 4 (Wave 1) negative test: a matmul whose contraction dimension K does not
// match (A is 4x8 -> K=8, B is 16x4 -> K=16). --infer-shapes computes the result
// type via InferTypeOpInterface::inferReturnTypes, detects the mismatch, emits a
// shape-mismatch diagnostic and signals pass failure (non-zero exit). On failure
// polykernel-opt prints only the diagnostic (no IR), so we check the diagnostic.

// CHECK: matmul shape mismatch: contraction dimension K differs (lhs K = 8 vs rhs K = 16)
polykernel.func @matmul_mismatched_k(%a: tensor<4x8xbf16>,
                                     %b: tensor<16x4xbf16>)
    -> tensor<4x4xbf16> {
  %0 = polykernel.matmul %a, %b
      : tensor<4x8xbf16>, tensor<16x4xbf16> -> tensor<4x4xbf16>
  polykernel.return %0 : tensor<4x4xbf16>
}
