// RUN: polykernel-opt %s | FileCheck %s

// W1 smoke test: parse an empty builtin module through polykernel-opt.
// Proves the out-of-tree tool builds, the PolyKernel dialect is registered
// (the tool loads without error), and round-trips a trivial module.

module {}

// CHECK: module
