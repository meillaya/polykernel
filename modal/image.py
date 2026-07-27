"""Modal CUDA-toolkit image that compiles the PolyKernel engine at build time.

Todo 29 / Wave 6. This module defines the *image definition* only. Constructing
the `modal.Image` object is pure: it needs NO Modal token and NO Docker. The
image is materialised (layers executed, engine compiled) only when an app that
references it is deployed or served - the owner-gated cloud step (Todo 33) that
sits behind a `MODAL_TOKEN`.

Pipeline (modal 1.5.2 - pinned in requirements.txt; every method below verified
current by introspection of `modal.image`):

    from_registry(nvidia/cuda devel, add_python="3.12")   # CUDA toolkit + Python
      .entrypoint([])                                     # clear base ENTRYPOINT
      .apt_install("cmake", "build-essential")            # C++ build tools
      .add_local_dir("./", "/app", copy=True)             # bake engine source in
      .run_commands("cd /app && cmake -B build && cmake --build build")  # compile

`add_python` is REQUIRED: the NVIDIA base images ship no interpreter, and Modal's
runtime needs one. `copy=True` on `add_local_dir` is REQUIRED for build-time
compilation: it bakes the source tree into an image LAYER so it exists during the
`run_commands` build step (the default `copy=False` only mounts the tree at
container STARTUP, which is too late for a build-time compile).

The final `run_commands` is a REAL compile step, not a no-op: a bad compiler flag
in it fails the image build with the compiler error surfaced (demonstrated in
reports/w6_image_build_neg.log via a bad `nvcc` flag, nvcc being present in the
CUDA devel base).

NOTE (production hardening, out of skeleton scope): the engine's CMake build
resolves MLIR/LLVM-21 via the nix dev shell locally (`find_package(MLIR)` +
`MLIR_TABLEGEN_EXE`, see CMakeLists.txt). For a *green* engine build inside this
standalone CUDA image, the same LLVM/MLIR-21 toolchain must additionally be
provisioned in the image (and `MLIR_TABLEGEN_EXE` exported) before the cmake
step. The skeleton wires the build-time compile step; toolchain provisioning for
a fully self-contained image build is a documented deploy-time follow-up.
"""

from __future__ import annotations

import modal

# CUDA *devel* toolkit image (nvcc + ptxas + CUDA headers) on Ubuntu 24.04 - the
# engine's GPU kernels compile against this. "devel" (not "runtime"/"base") is
# required so nvcc + headers are present for build-time compilation.
CUDA_BASE = "nvidia/cuda:12.8.1-devel-ubuntu24.04"

# The NVIDIA base images ship NO Python; Modal's runtime requires an interpreter,
# so add_python is mandatory (not optional) for from_registry images.
PYTHON_VERSION = "3.12"

# Where the project source is baked into the image; the engine build runs here.
REMOTE_APP_DIR = "/app"


def build_image() -> modal.Image:
    """Return the PolyKernel engine image definition (pure; no token needed).

    The returned object is a declarative spec. Modal materialises it (runs the
    layers, compiles the engine) only on deploy/serve - never at construction.
    """
    return (
        modal.Image.from_registry(CUDA_BASE, add_python=PYTHON_VERSION)
        # Clear the base image ENTRYPOINT so Modal's own runtime entrypoint
        # (its container init) takes over when the app runs.
        .entrypoint([])
        # C++ toolchain + CMake to build the engine (nvcc is already in the
        # CUDA devel base). git is pulled in as a common build-time dep.
        .apt_install("cmake", "build-essential")
        # Bake the project source (engine: CMakeLists.txt + include/ + lib/ +
        # tools/ + kernels/) into /app at BUILD time. copy=True makes this a
        # layer (present during the compile below); the default mount-only mode
        # appears at container startup and would be too late.
        .add_local_dir("./", remote_path=REMOTE_APP_DIR, copy=True)
        # BUILD-TIME compilation of the PolyKernel engine via the project's
        # CMake build (configure + build). A bad flag here fails the image
        # build with the compiler error surfaced - this step is real.
        .run_commands(
            f"cd {REMOTE_APP_DIR} && cmake -B build && cmake --build build"
        )
    )


# Module-level image so consumers can `from <this module> import image`. Building
# this object performs NO image build and needs NO token.
image = build_image()
