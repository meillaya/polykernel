# ===- PolyKernel pass TableGen (-gen-pass-decls) -===
#
# Generates Passes.h.inc from Passes.td with the per-pass decl macros
# (GEN_PASS_DECL_<PASSNAME>) and the registration block (GEN_PASS_REGISTRATION:
# registerInferShapesPass() + registerPolyKernelPasses()). `-name PolyKernel`
# prefixes the aggregate registration helper.
#
# Todo 5 adds the Canonicalize pass decl to Passes.td; this target regenerates
# Passes.h.inc to include it (no CMake change needed here).

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PolyKernel)
add_public_tablegen_target(MLIRPolyKernelPassesIncGen)
