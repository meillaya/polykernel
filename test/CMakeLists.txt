# ===- PolyKernel lit test suite (W1 spike) -===

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Out-of-tree against an installed (nix) LLVM: FileCheck/count/not are prebuilt
# binaries resolved at lit runtime via config.llvm_tools_dir - they are NOT
# CMake targets here, so only polykernel-opt is a build dependency.
set(POLYKERNEL_TEST_DEPENDS
  polykernel-opt
)

# check-polykernel: run lit over test/ using LLVM_EXTERNAL_LIT (the nix lit
# wrapper, passed via -DLLVM_EXTERNAL_LIT=...).
add_lit_testsuite(check-polykernel
  "Running the PolyKernel regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${POLYKERNEL_TEST_DEPENDS}
)
set_target_properties(check-polykernel PROPERTIES FOLDER "Tests")
