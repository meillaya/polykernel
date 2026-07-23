//===- InferShapes.cpp - PolyKernel shape inference pass --------*- C++ -*-===//
//
// PolyKernel out-of-tree MLIR dialect: the `--infer-shapes` pass (Todo 4 / Wave 1).
//
//===----------------------------------------------------------------------===//
//
// Walks every op in the module and, for each op implementing
// InferTypeOpInterface (all PolyKernel compute ops), recomputes the result
// type(s) from the operands via inferReturnTypes and refines the IR to them. On
// any inconsistency inferReturnTypes emits a diagnostic and this pass calls
// signalPassFailure().
//
//===----------------------------------------------------------------------===//

#include "PolyKernel/Passes/InferShapes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

namespace mlir::polykernel {
#define GEN_PASS_DEF_INFERSHAPES
#include "PolyKernel/Passes/Passes.h.inc"
} // namespace mlir::polykernel

using namespace mlir;

namespace mlir::polykernel {
namespace {

class InferShapesPass : public impl::InferShapesBase<InferShapesPass> {
public:
  void runOnOperation() override {
    bool hadFailure = false;

    // A single forward walk suffices: ops are in topological order within a
    // block, so refining an op's result type is visible to its (later) users.
    getOperation().walk([&](Operation *op) -> WalkResult {
      auto inferOp = dyn_cast<InferTypeOpInterface>(op);
      if (!inferOp)
        return WalkResult::advance();

      SmallVector<Type, 4> inferredTypes;
      if (failed(inferOp.inferReturnTypes(
              op->getContext(), op->getLoc(), op->getOperands(),
              op->getRawDictionaryAttrs(), op->getPropertiesStorage(),
              op->getRegions(), inferredTypes))) {
        // inferReturnTypes already emitted a shape-mismatch diagnostic.
        hadFailure = true;
        return WalkResult::interrupt();
      }

      if (inferredTypes.size() != op->getNumResults()) {
        op->emitOpError("inferred ")
            << inferredTypes.size() << " result type(s) but the op has "
            << op->getNumResults();
        hadFailure = true;
        return WalkResult::interrupt();
      }

      for (unsigned i = 0, e = inferredTypes.size(); i != e; ++i)
        op->getResult(i).setType(inferredTypes[i]);

      return WalkResult::advance();
    });

    if (hadFailure)
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createInferShapesPass() {
  return std::make_unique<InferShapesPass>();
}

} // namespace mlir::polykernel
