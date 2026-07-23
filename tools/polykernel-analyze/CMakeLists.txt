# ===- polykernel-analyze: compile-time analyzer CLI (Todo 11 / Wave 2) -===
#
# GPU-FREE standalone CLI over the PolyKernelAnalysis library: reads a captured
# `ptxas -v` log + GEMM shape + arch and emits the contract H report (JSON/text).
# Wrapped by tools/polykernel-bench/analyze.py. Lands at
# build/tools/polykernel-analyze/polykernel-analyze (no LLVM_RUNTIME_OUTPUT_INTDIR
# is set project-wide).

add_executable(polykernel-analyze polykernel-analyze.cpp)
target_link_libraries(polykernel-analyze PRIVATE PolyKernelAnalysis)
