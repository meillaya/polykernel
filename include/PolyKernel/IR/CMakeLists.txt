# ===- PolyKernel dialect TableGen (dialect decl + ops via ODS) -===
#
# Direct mlir_tablegen (sanctioned alongside `add_mlir_dialect`). Two separate
# TableGen targets so the dialect decls and the op decls/defs are generated
# independently and NEVER double-generated:
#   - MLIRPolyKernelDialectIncGen: -gen-dialect-decls/-defs from PolyKernelDialect.td
#     (named PolyKernelDialect{.h,.cpp}.inc, not the doubled
#     "PolyKernelDialectDialect.*.inc" add_mlir_dialect would produce).
#   - MLIRPolyKernelOpsIncGen: -gen-op-decls/-defs from PolyKernelOps.td
#     (named PolyKernelOps{.h,.cpp}.inc). PolyKernelOps.td includes
#     PolyKernelDialect.td but defines NO dialect record of its own, so
#     -gen-op-* emits only the op decls/defs.

set(LLVM_TARGET_DEFINITIONS PolyKernelDialect.td)
mlir_tablegen(PolyKernelDialect.h.inc -gen-dialect-decls -dialect=polykernel)
mlir_tablegen(PolyKernelDialect.cpp.inc -gen-dialect-defs -dialect=polykernel)
add_public_tablegen_target(MLIRPolyKernelDialectIncGen)

set(LLVM_TARGET_DEFINITIONS PolyKernelOps.td)
mlir_tablegen(PolyKernelOps.h.inc -gen-op-decls)
mlir_tablegen(PolyKernelOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPolyKernelOpsIncGen)

add_dependencies(mlir-headers MLIRPolyKernelDialectIncGen MLIRPolyKernelOpsIncGen)
