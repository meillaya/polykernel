#!/usr/bin/env python3
"""PolyKernel full per-kernel report (Todo 27 / Wave 5).

The CLI half of the per-kernel report pipeline (the IR half is the MLIR pass
`--emit-kernel-report`, lib/Passes/EmitKernelReport.cpp). For one kernel +
backend + arch it MERGES the three analyses into the AUTHORITATIVE contract-H
report (registers/smem/spills, nested occupancy{} + traffic{}, bottleneck,
suggested_fixes, path) and renders JSON + human-readable text:

  * CUDA compile-time analysis (Todo 11): compile kernels/generated/<k>.cu with
    nvcc + capture `ptxas -v` (GPU-FREE), then drive the C++ `polykernel-analyze`
    analyzer — which REUSES lib/Analysis/KernelReport.cpp BuildReport/ToJson for
    the occupancy model, roofline model, traffic, bottleneck, and suggested-fix
    rules. (A pre-captured log can be fed via --ptxas-log.)
  * AMD ISA analysis (Todo 21): compile with `hipcc --save-temps` (GPU-FREE) and
    parse the AMDGPU `.s` for VGPR/SGPR/LDS/scratch + gfx1101 occupancy, reusing
    tools/polykernel-bench/amd_analyze.py. (A pre-captured `.s` via --assembly.)
  * Autotuner result (Todo 25): an optional --tuning-cache lookup by
    (gpu, op, shape); a validated match contributes the kernel `path` (derived
    from the tuned vector_width) and is embedded as `tuned_config`.

Suggested-fix rules (the heart of Todo 27) live in the reused analyzers:
  spills>0 -> reduce register pressure; register-limited & occ<100% -> reduce
  unroll; occ<50% -> increase occupancy; memory-bound -> wider vectorized loads.

Usage:
    report.py --kernel fused_rmsnorm_matmul --backend cuda --arch sm_90
    report.py --kernel matmul --backend hip --arch gfx1101 --format text
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BENCH_DIR = _PROJECT_ROOT / "tools" / "polykernel-bench"

# Reuse the existing CUDA + AMD analyzer drivers (same repo, sibling tool dir).
sys.path.insert(0, str(_BENCH_DIR))
import amd_analyze  # noqa: E402
import nvcc_driver  # noqa: E402

DEFAULT_TIMEOUT_S = 240

# Per-kernel default GEMM shape (M,N,K) — the MLP shapes the kernels target;
# overridable with --shape. fused_rmsnorm_matmul matches examples/rmsnorm_matmul.mlir.
_DEFAULT_SHAPES: dict[str, tuple[int, int, int]] = {
    "matmul": (128, 128, 128),
    "fused_rmsnorm_matmul": (2048, 11008, 4096),
    "fused_matmul_bias_gelu": (2048, 4096, 11008),
}
_DEFAULT_SHAPE = (128, 128, 128)


def find_analyzer(explicit: str | None) -> Path:
    """Locate the polykernel-analyze binary (built by CMake under build/)."""
    path = (
        Path(explicit)
        if explicit
        else _PROJECT_ROOT / "build/tools/polykernel-analyze/polykernel-analyze"
    )
    if not path.exists():
        raise SystemExit(
            f"error: analyzer not found: {path}. Build it first: "
            f"`cmake --build build --target polykernel-analyze`."
        )
    return path


def resolve_shape(kernel: str, explicit: str | None) -> tuple[int, int, int]:
    if explicit:
        vals = [int(x) for x in explicit.split(",")]
        if len(vals) != 3:
            raise SystemExit(f"error: --shape must be M,N,K (got '{explicit}')")
        return (vals[0], vals[1], vals[2])
    return _DEFAULT_SHAPES.get(kernel, _DEFAULT_SHAPE)


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
    # nvcc_driver prints [nvcc]/[ptxas] progress to stdout; reroute to stderr so
    # the report stays the ONLY thing on stdout (clean machine-readable output).
    with contextlib.redirect_stdout(sys.stderr):
        nvcc_driver.compile_ptx(nvcc, cu, arch, ptx, include_dir, timeout_s)
        return nvcc_driver.run_ptxas_verbose(ptxas, ptx, arch, cubin, timeout_s)


def cuda_report(
    analyzer: Path, kernel: str, arch: str, shape: tuple[int, int, int],
    dtype_bytes: int, threads_per_block: int, path: str, ptxas_log: str,
) -> dict:
    """Drive the C++ polykernel-analyze (KernelReport.cpp) over a ptxas log."""
    cmd = [
        str(analyzer),
        "--kernel", kernel,
        "--backend", "cuda",
        "--arch", arch,
        "--shape", f"{shape[0]},{shape[1]},{shape[2]}",
        "--dtype-bytes", str(dtype_bytes),
        "--threads-per-block", str(threads_per_block),
        "--path", path,
        "--ptxas-log", "-",
        "--format", "json",
    ]
    print(f"[report] {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, input=ptxas_log, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or "")
        raise SystemExit(f"error: polykernel-analyze failed (exit {proc.returncode})")
    return json.loads(proc.stdout)


def hip_report(
    kernel: str, arch: str, shape: tuple[int, int, int], dtype_bytes: int,
    threads_per_block: int, path: str, assembly: str,
) -> dict:
    """Parse AMDGPU `.s` via the Todo 21 analyzer + fill exact GEMM traffic."""
    stats = amd_analyze.parse_amd_isa(assembly)
    report = amd_analyze.build_report(kernel, arch, stats, threads_per_block, path)

    # The AMD analyzer leaves traffic at zero; fill the exact GEMM bytes +
    # arithmetic intensity from the shape (read A[M,K] + B[K,N], write C[M,N]).
    m, n, k = shape
    d = dtype_bytes
    total_bytes = (m * k + k * n + m * n) * d
    traffic = report["traffic"]
    traffic["global_read_bytes"] = (m * k + k * n) * d
    traffic["global_write_bytes"] = m * n * d
    traffic["arithmetic_intensity_flop_per_byte"] = (
        round((2 * m * n * k) / total_bytes, 4) if total_bytes else 0.0
    )
    return report


def derive_path_from_config(best: dict) -> str:
    """Documented heuristic: the baseline kernels are scalar; a tuned config with
    a vectorized load width >= 4 is a vectorized/tensor-core path (wmma)."""
    return "wmma" if best.get("vector_width", 1) >= 4 else "scalar"


def lookup_tuning(
    cache_path: str | None, gpu: str, op: str, shape: tuple[int, int, int]
) -> dict | None:
    """Find a VALIDATED tuning-cache entry matching (gpu, op, shape), else None."""
    if not cache_path:
        return None
    cache = json.loads(Path(cache_path).read_text())
    for entry in cache.get("entries", []):
        if entry.get("gpu") != gpu or entry.get("op") != op:
            continue
        if not entry.get("validated", False):
            continue
        es = entry.get("shape", {})
        if (es.get("M"), es.get("N"), es.get("K")) != shape:
            continue
        return entry
    return None


def merge_autotuner(report: dict, entry: dict | None, explicit_path: bool) -> None:
    """Fold a validated tuning-cache entry into the report (path + tuned_config).

    The autotuner contributes the kernel `path` (unless --path was given
    explicitly, which wins) and is recorded transparently as `tuned_config`."""
    if entry is None:
        return
    report["tuned_config"] = entry.get("best")
    report["tuning_validated"] = entry.get("validated", False)
    if not explicit_path:
        report["path"] = derive_path_from_config(entry.get("best", {}))


def to_text(report: dict) -> str:
    """Human-readable summary mirroring KernelReport::ToText (contract-H fields)."""
    occ = report["occupancy"]
    traffic = report["traffic"]
    lines = [
        f"kernel:                 {report['kernel']}",
        f"backend:                {report['backend']}",
        f"arch:                   {report['arch']}",
        f"registers_per_thread:   {report['registers_per_thread']}",
        f"smem_per_block_bytes:   {report['smem_per_block_bytes']}",
        f"spill_stores_bytes:     {report['spill_stores_bytes']}",
        f"spill_loads_bytes:      {report['spill_loads_bytes']}",
        "occupancy:",
        f"  active_warps_per_sm:  {occ['active_warps_per_sm']}",
        f"  max_warps_per_sm:     {occ['max_warps_per_sm']}",
        f"  occupancy_pct:        {occ['occupancy_pct']}",
        f"  limiter:              {occ['limiter']}",
        "traffic:",
        f"  global_read_bytes:    {traffic['global_read_bytes']}",
        f"  global_write_bytes:   {traffic['global_write_bytes']}",
        f"  arithmetic_intensity: {traffic['arithmetic_intensity_flop_per_byte']} flop/byte",
        f"  roofline:             {traffic['roofline']}",
        f"bottleneck:             {report['bottleneck']}",
        f"path:                   {report['path']}",
    ]
    if "tuned_config" in report:
        lines.append(f"tuned_config:           {json.dumps(report['tuned_config'])}")
        lines.append(f"tuning_validated:       {report.get('tuning_validated')}")
    lines.append("suggested_fixes:")
    if not report["suggested_fixes"]:
        lines.append("  (none)")
    for fix in report["suggested_fixes"]:
        lines.append(f"  - {fix}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kernel", default="fused_rmsnorm_matmul")
    ap.add_argument("--backend", default="cuda", choices=["cuda", "hip"])
    ap.add_argument("--arch", default="sm_90")
    ap.add_argument("--shape", default=None, help="GEMM shape M,N,K (per-kernel default if omitted)")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="element width (bf16/fp16=2)")
    ap.add_argument("--threads-per-block", type=int, default=256)
    ap.add_argument("--path", default=None, choices=["scalar", "wmma", "mma"],
                    help="override the kernel path (else autotuner-derived or scalar)")
    ap.add_argument("--tuning-cache", default=None, help="tuning-cache JSON for the autotuner merge")
    ap.add_argument("--format", default="json", choices=["json", "text"])
    ap.add_argument("--kernel-dir", default="kernels/generated")
    ap.add_argument("--include-dir", default="kernels/template")
    ap.add_argument("--out-dir", default="build/nvcc")
    ap.add_argument("--analyzer", default=None, help="path to polykernel-analyze (CUDA)")
    ap.add_argument("--ptxas-log", default=None, help="feed a pre-captured ptxas -v log (CUDA)")
    ap.add_argument("--assembly", default=None, help="feed a pre-captured AMDGPU .s (HIP)")
    ap.add_argument("--save", default=None, help="also write the report to this file")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    shape = resolve_shape(args.kernel, args.shape)
    entry = lookup_tuning(args.tuning_cache, args.arch, args.kernel, shape)
    path = args.path if args.path else (
        derive_path_from_config(entry["best"]) if entry else "scalar"
    )

    if args.backend == "cuda":
        log = (
            Path(args.ptxas_log).read_text()
            if args.ptxas_log
            else capture_ptxas_log(
                args.kernel, args.arch, Path(args.kernel_dir),
                Path(args.include_dir), Path(args.out_dir), args.timeout,
            )
        )
        report = cuda_report(
            find_analyzer(args.analyzer), args.kernel, args.arch, shape,
            args.dtype_bytes, args.threads_per_block, path, log,
        )
    else:
        assembly = (
            Path(args.assembly).read_text()
            if args.assembly
            else amd_analyze.capture_assembly(
                args.kernel, args.arch, Path(args.kernel_dir),
                Path(args.include_dir), args.timeout,
            )
        )
        report = hip_report(
            args.kernel, args.arch, shape, args.dtype_bytes,
            args.threads_per_block, path, assembly,
        )

    merge_autotuner(report, entry, args.path is not None)

    output = to_text(report) if args.format == "text" else json.dumps(report, indent=2) + "\n"
    print(output, end="")
    if args.save:
        Path(args.save).write_text(output)
        print(f"[report] saved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
