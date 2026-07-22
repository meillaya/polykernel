// RUN: not polykernel-opt %s 2>&1 | FileCheck %s

// Todo 3 (Wave 1): the PolyKernel op set is CLOSED. An undefined op
// (polykernel.conv2d) must FAIL to parse with an "unknown op" error - the
// negative half of the op-zoo guardrail. The positive half (every named op
// round-trips) is dialect_roundtrip.mlir.

builtin.module {
  polykernel.func @bad(%x: tensor<1x3x224x224xbf16>) -> tensor<1x3x224x224xbf16> {
    %0 = polykernel.conv2d %x : tensor<1x3x224x224xbf16> -> tensor<1x3x224x224xbf16>
    polykernel.return %0 : tensor<1x3x224x224xbf16>
  }
}

// CHECK: custom op 'polykernel.conv2d' is unknown
