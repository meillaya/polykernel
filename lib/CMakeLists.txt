# PolyKernel dialect implementation libraries.
add_subdirectory(IR)
add_subdirectory(Passes)
# GPU-free compile-time analyzer (Todo 11): PtxasParser + Occupancy + Roofline +
# KernelReport + gtest unit tests.
add_subdirectory(Analysis)
