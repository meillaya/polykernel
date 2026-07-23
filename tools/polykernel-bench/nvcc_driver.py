#!/usr/bin/env python3
"""nvcc driver: compile a generated PolyKernel .cu to PTX + ptxas stats.

Todo 8 / Wave 2. Compiles a generated CUDA kernel (kernels/generated/<kernel>.cu,
emitted by `polykernel-opt --lower-to-cuda`) for one or more `-arch` targets using
nvcc (from the nix shell), emits PTX, and runs `ptxas -v` to capture
register / shared-memory / spill usage — all GPU-FREE (compile + analyze only).

Usage:
    nvcc_driver.py --kernel rmsnorm --arch sm_80,sm_90 --ptx

Output (per arch):
    <out-dir>/<kernel>_<arch>.ptx    the emitted PTX
    <out-dir>/<kernel>_<arch>.cubin  the ptxas-produced cubin (ptxas -v side effect)
    stdout: the raw `ptxas -v` lines + a parsed register/smem/spill summary.

Exits non-zero if nvcc or ptxas is missing, or any compile/assemble fails (so a
codegen bug — e.g. an undefined macro or broken include — surfaces as a clear
captured nvcc error; see reports/w2_nvcc_fail.log).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# nvcc/ptxas can take 10-60s per arch; bound it so a wedged compile cannot hang QA.
DEFAULT_TIMEOUT_S = 180


@dataclass(frozen=True)
class PtxasStats:
    """Parsed `ptxas -v` resource usage (0 = not reported by ptxas)."""

    registers: int = 0
    smem_bytes: int = 0
    spill_stores_bytes: int = 0
    spill_loads_bytes: int = 0
    stack_frame_bytes: int = 0
    cmem0_bytes: int = 0
    raw_lines: tuple[str, ...] = field(default_factory=tuple)


def find_tool(name: str) -> Path:
    """Locate a tool on PATH (inside `nix develop`); raise clearly if absent."""
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"error: '{name}' not found on PATH. Run inside `nix develop` "
            f"(cudaPackages_12_6 provides nvcc + ptxas)."
        )
    return Path(path)


def compile_ptx(
    nvcc: Path, cu: Path, arch: str, ptx: Path, include_dir: Path, timeout_s: int
) -> None:
    """Compile `cu` to PTX for `arch` with -DPOLYKERNEL_CUDA. Raises on failure."""
    cmd = [
        str(nvcc),
        f"-arch={arch}",
        "-DPOLYKERNEL_CUDA",
        f"-I{include_dir}",
        "--ptx",
        str(cu),
        "-o",
        str(ptx),
    ]
    print(f"[nvcc] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"error: nvcc failed for {arch} (exit {proc.returncode})")


def run_ptxas_verbose(
    ptxas: Path, ptx: Path, arch: str, cubin: Path, timeout_s: int
) -> str:
    """Assemble PTX -> cubin with `ptxas -v`; return the verbose stderr."""
    cmd = [str(ptxas), "-v", f"-arch={arch}", str(ptx), "-o", str(cubin)]
    print(f"[ptxas] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    # capture_output=True guarantees stderr is a str; coerce None for the type checker.
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        sys.stderr.write(stderr)
        raise SystemExit(f"error: ptxas failed for {arch} (exit {proc.returncode})")
    return stderr


def parse_ptxas_stats(stderr: str) -> PtxasStats:
    """Parse register/smem/spill/stack/cmem from `ptxas -v` stderr.

    Handles both the OLD format (`Used N registers, N bytes smem, N bytes
    cmem[0]`) and the NEW format (`Used N registers, used N barriers, N bytes
    smem`), plus spill/stack lines. Missing fields stay 0.
    """
    info = [ln for ln in stderr.splitlines() if "ptxas" in ln]

    def grab(pattern: str) -> int:
        m = re.search(pattern, stderr)
        return int(m.group(1)) if m else 0

    return PtxasStats(
        registers=grab(r"(\d+) registers"),
        smem_bytes=grab(r"(\d+) bytes smem"),
        spill_stores_bytes=grab(r"(\d+) bytes spill stores"),
        spill_loads_bytes=grab(r"(\d+) bytes spill loads"),
        stack_frame_bytes=grab(r"(\d+) bytes stack frame"),
        cmem0_bytes=grab(r"(\d+) bytes cmem\[0\]"),
        raw_lines=tuple(info),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", default="rmsnorm", help="kernel name (basename of the .cu)")
    ap.add_argument("--arch", default="sm_80,sm_90", help="comma-separated arch list")
    ap.add_argument("--ptx", action="store_true", help="emit + keep PTX (always on; accepted for CLI compat)")
    ap.add_argument("--kernel-dir", default="kernels/generated", help="dir holding <kernel>.cu")
    ap.add_argument("--include-dir", default="kernels/template", help="dir holding kernel_common.h")
    ap.add_argument("--out-dir", default="build/nvcc", help="dir for .ptx / .cubin output")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S, help="per-tool timeout (s)")
    args = ap.parse_args()

    nvcc = find_tool("nvcc")
    ptxas = find_tool("ptxas")

    cu = Path(args.kernel_dir) / f"{args.kernel}.cu"
    if not cu.exists():
        raise SystemExit(
            f"error: kernel source not found: {cu}. Run "
            f"`polykernel-opt <example.mlir> --lower-to-cuda` first."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_dir = Path(args.include_dir)
    archs = [a.strip() for a in args.arch.split(",") if a.strip()]

    print(f"kernel: {cu}")
    print(f"archs:  {', '.join(archs)}")
    print(f"nvcc:   {nvcc}")
    print(f"ptxas:  {ptxas}")
    print("=" * 72)

    for arch in archs:
        ptx = out_dir / f"{args.kernel}_{arch}.ptx"
        cubin = out_dir / f"{args.kernel}_{arch}.cubin"
        compile_ptx(nvcc, cu, arch, ptx, include_dir, args.timeout)
        stderr = run_ptxas_verbose(ptxas, ptx, arch, cubin, args.timeout)
        stats = parse_ptxas_stats(stderr)

        print(f"--- ptxas -v ({arch}) ---")
        for ln in stats.raw_lines:
            print(f"  {ln.strip()}")
        print(f"  [parsed] registers={stats.registers} "
              f"smem={stats.smem_bytes}B "
              f"spill_stores={stats.spill_stores_bytes}B "
              f"spill_loads={stats.spill_loads_bytes}B "
              f"stack={stats.stack_frame_bytes}B "
              f"cmem0={stats.cmem0_bytes}B")
        print(f"  PTX: {ptx}")
        print("=" * 72)

    print(f"OK: compiled {args.kernel} for {', '.join(archs)}; PTX emitted.")


if __name__ == "__main__":
    main()
