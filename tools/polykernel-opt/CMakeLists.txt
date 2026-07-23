# ===- polykernel-opt: MLIR optimizer driver with the PolyKernel dialect -===

# nixpkgs MLIR (MLIR_INSTALL_AGGREGATE_OBJECTS=1) does not install
# libMLIRRegisterAll{Dialects,Passes}.a, so those are intentionally omitted -
# see polykernel-opt.cpp for the rationale.
set(LIBS
  MLIRPolyKernel
  MLIRPolyKernelPasses
  MLIROptLib
  MLIRIR
  MLIRSupport
)

add_llvm_executable(polykernel-opt polykernel-opt.cpp)

llvm_update_compile_flags(polykernel-opt)
target_link_libraries(polykernel-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(polykernel-opt)
