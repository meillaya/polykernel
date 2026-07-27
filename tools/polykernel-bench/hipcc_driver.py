#!/usr/bin/env python3
"""hipcc driver: compile a generated PolyKernel .cu for an AMDGPU arch (gfx1101).

Todo 19 / Wave 4. The HIP sibling of nvcc_driver.py. Compiles a generated portable
kernel (kernels/generated/<kernel>.cu, emitted by `polykernel-opt --lower-to-cuda`
OR `--lower-to-hip` — the emitted source is byte-identical) for one or more
`--offload-arch` targets using hipcc (from the nix shell rocmPackages.clr), with
`-DPOLYKERNEL_HIP`, producing a code object — all GPU-FREE (compile only; the
kernels are NOT launched here — runtime launch + golden correctness is Todo 20).

This is the PORTABILITY PROOF: the SAME .cu that nvcc builds with -DPOLYKERNEL_CUDA
compiles UNCHANGED under `hipcc -DPOLYKERNEL_HIP --offload-arch=gfx1101`, because
every kernel is written against kernels/template/kernel_common.h and uses only the
portable pk_* / PK_* names (the backend is selected by the -D define, never by an
edit to the source).

Usage:
    hipcc_driver.py --arch gfx1101 --all          # compile every generated kernel
    hipcc_driver.py --kernel matmul --arch gfx1101 # compile one kernel

Output (per kernel x arch):
    <out-dir>/<kernel>_<arch>.o    the compiled object (host + device code object)
    stdout: the hipcc command + any diagnostics + a per-kernel PASS/FAIL line.

Exits non-zero if hipcc is missing, a kernel source is absent, or any compile
fails (so a portability bug — e.g. an unguarded CUDA-only intrinsic — surfaces as
a clear captured hipcc error; see reports/w4_hipcc_portability_neg.log).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# hipcc device-codegen for a cold arch can take 10-90s; bound it so a wedged
# compile cannot hang QA.
DEFAULT_TIMEOUT_S = 240

# The closed set of generated PolyKernel kernels (matches --lower-to-cuda /
# --lower-to-hip dispatch; the op set is closed — no new ops in Wave 4 codegen).
ALL_KERNELS: tuple[str, ...] = (
    "rmsnorm",
    "gelu",
    "silu",
    "matmul",
    "softmax",
    "fused_rmsnorm_matmul",
    "fused_matmul_bias_gelu",
)


def find_tool(name: str) -> Path:
    """Locate a tool on PATH (inside `nix develop`); raise clearly if absent."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"error: '{name}' not found on PATH. Run inside `nix develop` "
            f"(rocmPackages.clr provides hipcc)."
        )
    return Path(path)


def compile_object(
    hipcc: Path, cu: Path, arch: str, obj: Path, include_dir: Path, timeout_s: int
) -> None:
    """Compile `cu` to an object for `arch` with -DPOLYKERNEL_HIP. Raises on failure.

    `-c` compiles + assembles host and device code to a single object (no link, so
    no main() is required); a non-zero exit means the portable source failed to
    compile for this arch (a real portability bug).
    """
    cmd = [
        str(hipcc),
        f"--offload-arch={arch}",
        "-DPOLYKERNEL_HIP",
        f"-I{include_dir}",
        "-c",
        str(cu),
        "-o",
        str(obj),
    ]
    print(f"[hipcc] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        # hipcc prints warnings to stderr even on success; surface them but only
        # treat a non-zero exit as a failure.
        sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"error: hipcc failed for {cu.name} ({arch}, exit {proc.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", default="rmsnorm", help="kernel name (basename of the .cu)")
    ap.add_argument("--all", action="store_true", dest="all_kernels",
                    help="compile every generated kernel (rmsnorm, gelu, silu, matmul, softmax + fused)")
    ap.add_argument("--arch", default="gfx1101", help="comma-separated --offload-arch list")
    ap.add_argument("--kernel-dir", default="kernels/generated", help="dir holding <kernel>.cu")
    ap.add_argument("--include-dir", default="kernels/template", help="dir holding kernel_common.h")
    ap.add_argument("--out-dir", default="build/hipcc", help="dir for .o output")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="per-compile timeout (s)")
    args = ap.parse_args()

    hipcc = find_tool("hipcc")

    kernels = list(ALL_KERNELS) if args.all_kernels else [args.kernel]
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_dir = Path(args.include_dir)
    kernel_dir = Path(args.kernel_dir)

    # Resolve + validate every source up front so a missing kernel fails fast with
    # a clear message (mirrors the nvcc driver's existence check).
    sources: list[tuple[str, Path]] = []
    for name in kernels:
        cu = kernel_dir / f"{name}.cu"
        if not cu.exists():
            raise SystemExit(
                f"error: kernel source not found: {cu}. Run "
                f"`polykernel-opt <example.mlir> --lower-to-hip` (or --lower-to-cuda) first."
            )
        sources.append((name, cu))

    print(f"kernels: {', '.join(kernels)}")
    print(f"archs:   {', '.join(archs)}")
    print(f"hipcc:   {hipcc}")
    print("=" * 72)

    compiled = 0
    for name, cu in sources:
        for arch in archs:
            obj = out_dir / f"{name}_{arch}.o"
            compile_object(hipcc, cu, arch, obj, include_dir, args.timeout)
            print(f"  [PASS] {name} ({arch}) -> {obj}")
            compiled += 1
        print("=" * 72)

    print(f"OK: compiled {compiled} kernel x arch target(s) for {', '.join(archs)} clean.")


if __name__ == "__main__":
    main()
