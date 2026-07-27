# PolyKernel dialect implementation libraries.
add_subdirectory(IR)
add_subdirectory(Passes)
# GPU-free compile-time analyzer (Todo 11): PtxasParser + Occupancy + Roofline +
# KernelReport + gtest unit tests.
add_subdirectory(Analysis)
# GPU-free autotuner infrastructure (Todo 24): bounded ConfigSpace + pruner +
# contract-H JSON TuningCache + gtest unit tests.
add_subdirectory(Autotune)
# HIP runtime launch layer (Todo 20): error-checked hipMalloc/hipMemcpy/hipSync
# host wrapper (PolyKernelRuntime); guarded by hipcc availability.
add_subdirectory(Runtime)
