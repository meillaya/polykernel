# ===- PolyKernel dialect TableGen (W1 spike: dialect decl only, NO ops) -===
#
# Direct mlir_tablegen for the dialect decls/defs (the task sanctions
# `add_mlir_dialect` OR `mlir_tablegen`; we use mlir_tablegen directly so the
# generated files are named PolyKernelDialect{.h,.cpp}.inc rather than the
# doubled "PolyKernelDialectDialect.*.inc" that add_mlir_dialect(PolyKernelDialect ...)
# would produce). Ops/Types/Attrs are intentionally absent - they land in Todo 3.

set(LLVM_TARGET_DEFINITIONS PolyKernelDialect.td)
mlir_tablegen(PolyKernelDialect.h.inc -gen-dialect-decls -dialect=polykernel)
mlir_tablegen(PolyKernelDialect.cpp.inc -gen-dialect-defs -dialect=polykernel)
add_public_tablegen_target(MLIRPolyKernelDialectIncGen)
add_dependencies(mlir-headers MLIRPolyKernelDialectIncGen)
