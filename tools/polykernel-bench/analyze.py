#!/usr/bin/env python3
"""GPU-free compile-time analyzer driver (Todo 11 / Wave 2).

Compiles a generated PolyKernel .cu with nvcc + captures `ptxas -v` (NO GPU -
nvcc/ptxas run without a device), then runs the C++ `polykernel-analyze`
analyzer over the captured log + a GEMM shape + arch to emit the contract H
per-kernel report (registers, smem, occupancy, traffic, roofline, bottleneck,
suggested fixes). Reuses nvcc_driver.py for the compile + ptxas capture so the
parsing/compilation logic is shared, not duplicated.

Usage:
    analyze.py --backend=cuda --mode=analyze --kernel matmul \
        --shape 128,128,128 --arch sm_90
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
from pathlib import Path

# Reuse the nvcc/ptxas compile + capture logic (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvcc_driver  # noqa: E402

DEFAULT_TIMEOUT_S = 180


def find_analyzer(explicit: str | None) -> Path:
    """Locate the polykernel-analyze binary (built by CMake under build/)."""
    if explicit:
        path = Path(explicit)
    else:
        # Repo root = two levels up from tools/polykernel-bench/analyze.py.
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "build" / "tools" / "polykernel-analyze" / "polykernel-analyze"
    if not path.exists():
        raise SystemExit(
            f"error: analyzer not found: {path}. Build it first: "
            f"`cmake --build build --target polykernel-analyze`."
        )
    return path


def capture_ptxas_log(
    kernel: str, arch: str, kernel_dir: Path, include_dir: Path, out_dir: Path,
    timeout_s: int,
) -> str:
    """Compile `kernel` to PTX for `arch` and return the `ptxas -v` stderr."""
    nvcc = nvcc_driver.find_tool("nvcc")
    ptxas = nvcc_driver.find_tool("ptxas")
    cu = kernel_dir / f"{kernel}.cu"
    if not cu.exists():
        raise SystemExit(
            f"error: kernel source not found: {cu}. Run "
            f"`polykernel-opt <example.mlir> --lower-to-cuda` first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    ptx = out_dir / f"{kernel}_{arch}.ptx"
    cubin = out_dir / f"{kernel}_{arch}.cubin"
    # nvcc_driver prints its [nvcc]/[ptxas] progress to stdout; reroute it to
    # stderr so the analyzer's report stays the ONLY thing on stdout (clean
    # machine-readable JSON for downstream consumers).
    with contextlib.redirect_stdout(sys.stderr):
        nvcc_driver.compile_ptx(nvcc, cu, arch, ptx, include_dir, timeout_s)
        return nvcc_driver.run_ptxas_verbose(ptxas, ptx, arch, cubin, timeout_s)


def run_analyzer(
    analyzer: Path, ptxas_log: str, kernel: str, backend: str, arch: str,
    shape: str, dtype_bytes: int, threads_per_block: int, path: str, fmt: str,
) -> str:
    """Feed the ptxas log to polykernel-analyze; return the emitted report."""
    cmd = [
        str(analyzer),
        "--kernel", kernel,
        "--backend", backend,
        "--arch", arch,
        "--shape", shape,
        "--dtype-bytes", str(dtype_bytes),
        "--threads-per-block", str(threads_per_block),
        "--path", path,
        "--ptxas-log", "-",
        "--format", fmt,
    ]
    print(f"[analyze] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, input=ptxas_log, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"error: polykernel-analyze failed (exit {proc.returncode})")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="cuda", choices=["cuda", "hip"])
    ap.add_argument("--mode", default="analyze", help="only 'analyze' is supported")
    ap.add_argument("--kernel", default="matmul")
    ap.add_argument("--shape", default="128,128,128", help="GEMM shape M,N,K")
    ap.add_argument("--arch", default="sm_90")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="element width (bf16/fp16=2)")
    ap.add_argument("--threads-per-block", type=int, default=256)
    ap.add_argument("--path", default="scalar", choices=["scalar", "wmma", "mma"])
    ap.add_argument("--format", default="json", choices=["json", "text"])
    ap.add_argument("--kernel-dir", default="kernels/generated")
    ap.add_argument("--include-dir", default="kernels/template")
    ap.add_argument("--out-dir", default="build/nvcc")
    ap.add_argument("--analyzer", default=None, help="path to polykernel-analyze")
    ap.add_argument("--save", default=None, help="also write the report to this file")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    if args.mode != "analyze":
        raise SystemExit(f"error: unsupported --mode '{args.mode}' (only 'analyze')")

    analyzer = find_analyzer(args.analyzer)
    ptxas_log = capture_ptxas_log(
        args.kernel, args.arch, Path(args.kernel_dir), Path(args.include_dir),
        Path(args.out_dir), args.timeout,
    )
    report = run_analyzer(
        analyzer, ptxas_log, args.kernel, args.backend, args.arch, args.shape,
        args.dtype_bytes, args.threads_per_block, args.path, args.format,
    )
    print(report, end="")
    if args.save:
        Path(args.save).write_text(report)
        print(f"[analyze] report saved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
