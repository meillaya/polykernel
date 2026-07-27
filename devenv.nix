{ pkgs, lib, config, inputs, ... }:

let
  # Out-of-tree MLIR compiler toolchain. nixpkgs-unstable ships llvmPackages_21
  # as the treewide default; MLIR CMake config lives in llvmPackages_21.mlir.dev
  # (lib/cmake/mlir), mlir-tblgen in llvmPackages_21.tblgen, FileCheck in
  # llvmPackages_21.llvm. We use llvmPackages_21.clang for CC/CXX (NOT clang_18)
  # to avoid MLIR-21 header skew.
  llvmPkgs = pkgs.llvmPackages_21;

  # CUDA redistributables (compile/analyze-only path locally; sm_80/sm_90).
  cudaPackages = pkgs.cudaPackages_12_6;

  # ROCm/HIP. rocmPackages.clr provides hipcc + the HIP runtime and auto-sets
  # HIP_PATH/ROCM_PATH/HIP_CLANG_PATH/DEVICE_LIB_PATH/HSA_PATH; gpuTargets
  # include gfx1101 (RX 7800 XT, officially supported since ROCm 7.0).
  rocmPackages = pkgs.rocmPackages;

  # Python golden harness: NumPy + ml_dtypes.bfloat16 (bf16 round / fp32 accum).
  # `modal` (Todo 29 / Wave 6) is the Modal GPU-deployment client; withPackages
  # pulls it in WITH its deps (protobuf, grpcio, ...) as one unified Python env.
  # Used ONLY to validate the modal/image.py + modal/app.py definitions locally
  # (image-definition construction); cloud deploy is opt-in behind a MODAL_TOKEN.
  python = pkgs.python3.withPackages (ps: with ps; [
    numpy
    ml-dtypes
    pytest
    modal
  ]);
in
{
  # NOTE on nixpkgs config (allowUnfree / cudaSupport / rocmSupport):
  #   devenv.lib.mkShell hands this module a pre-instantiated `pkgs` and only
  #   overlays the module's `overlays`; the devenv module system has NO
  #   `nixpkgs.config` option (verified on devenv v1.11.2 and v2.1.2: setting
  #   `nixpkgs.config.*` here errors with "The option `nixpkgs' does not
  #   exist"). The CUDA-EULA / cudaSupport / rocmSupport config is therefore
  #   set where nixpkgs is instantiated - in flake.nix:
  #       pkgs = import nixpkgs { config = { allowUnfree = true;
  #         cudaSupport = true; rocmSupport = true; cudaCapabilities = [...]; }; };
  #   Toggling allowUnfree=false there makes cudaPackages fail to evaluate
  #   (the EULA gate), which is the QA negative test (reports/w1_allowunfree_negative.log).

  packages = [
    # --- LLVM/MLIR 21 (out-of-tree dialect build) ---
    llvmPkgs.mlir      # MLIR libs + headers (.dev: lib/cmake/mlir)
    llvmPkgs.tblgen    # mlir-tblgen
    llvmPkgs.llvm      # FileCheck, llvm-objdump, llvm-readobj
    llvmPkgs.clang     # CC / CXX

    # --- Build tools ---
    pkgs.lit           # LLVM Integrated Tester (lit)
    pkgs.cmake
    pkgs.ninja

    # --- CUDA redistributables (preferred over monolithic cudatoolkit) ---
    cudaPackages.cuda_nvcc        # nvcc + ptxas
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvrtc
    cudaPackages.libcublas
    cudaPackages.cuda_cuobjdump

    # --- ROCm / HIP (first-class local run on gfx1101) ---
    rocmPackages.clr              # hipcc + HIP runtime
    rocmPackages.rocminfo
    rocmPackages.rocm-smi

    # --- Unit testing (REQUIRED: gtest unit tests won't configure without it) ---
    pkgs.gtest

    # --- Python golden harness ---
    python
  ];

  # Use llvmPackages_21.clang for CC/CXX (NOT clang_18 - avoid MLIR-21 header
  # skew). CC = clang, CXX = clang++ (the C++ driver of the same package, so
  # CMake C++ targets link the C++ runtime correctly).
  env.CC = lib.getExe llvmPkgs.clang;
  env.CXX = "${llvmPkgs.clang}/bin/clang++";

  # CUDA_HOME: search path over the CUDA "dev" outputs (headers + cmake).
  env.CUDA_HOME = lib.makeSearchPathOutput "dev" "" (with cudaPackages; [
    cuda_nvcc
    cuda_cudart
    cuda_nvrtc
    libcublas
  ]);

  # Runtime library path: CUDA libs + HIP runtime + the system NVIDIA driver.
  env.LD_LIBRARY_PATH = lib.makeLibraryPath [
    cudaPackages.cuda_cudart
    cudaPackages.libcublas
    cudaPackages.cuda_nvrtc
    rocmPackages.clr
    "/run/opengl-driver"   # system NVIDIA driver
  ];

  enterShell = ''
    # nixpkgs stdenv's setup hook exports CC/CXX=gcc/g++; re-export the LLVM-21
    # clang here (enterShell runs last) so the compiler the CMake build picks up
    # is llvmPackages_21.clang, matching MLIR-21 headers.
    export CC="${lib.getExe llvmPkgs.clang}"
    export CXX="${llvmPkgs.clang}/bin/clang++"
    echo "PolyKernel dev shell (llvmPackages_21 MLIR / cudaPackages_12_6 / rocmPackages.clr)"
    echo "  CC/CXX:      $CC / $CXX"
    echo "  mlir-tblgen: $(mlir-tblgen --version 2>&1 | head -1 || echo 'not in PATH yet')"
    echo "  nvcc:        $(nvcc --version 2>&1 | grep release || echo 'not in PATH yet')"
    echo "  hipcc:       $(hipcc --version 2>&1 | head -1 || echo 'not in PATH yet')"
    echo "  cmake:       $(cmake --version 2>&1 | head -1)"
    echo "  clang:       $(clang --version 2>&1 | head -1)"
  '';
}
