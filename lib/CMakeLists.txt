# PolyKernel dialect implementation libraries.
add_subdirectory(IR)
add_subdirectory(Passes)
# GPU-free compile-time analyzer (Todo 11): PtxasParser + Occupancy + Roofline +
# KernelReport + gtest unit tests.
add_subdirectory(Analysis)
# GPU-free autotuner infrastructure (Todo 24): bounded ConfigSpace + pruner +
# contract-H JSON TuningCache + gtest unit tests.
add_subdirectory(Autotune)
# GPU-free quantization codegen lowering (Todo 41 / Wave 8): int8 weight-only +
# fp8 e4m3/e5m2 simulation source emitter + gtest unit tests.
add_subdirectory(Codegen)
# GPU-free dataflow simulator core (Todo 35 / Wave 7): 64x64 PE mesh, CE +
# 5-port router + 48KB SRAM, 32b wavelet, 24 colors, single-cycle routing.
add_subdirectory(DataflowSim)
# HIP runtime launch layer (Todo 20): error-checked hipMalloc/hipMemcpy/hipSync
# host wrapper (PolyKernelRuntime); guarded by hipcc availability.
add_subdirectory(Runtime)
