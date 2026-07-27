#!/usr/bin/env python3
"""GPU-free AMDGPU ISA analyzer driver (Todo 21 / Wave 4).

Compiles a generated PolyKernel .cu with `hipcc --save-temps` (NO GPU - compile
only), parses the emitted AMDGPU `.s` for resource usage (VGPR/SGPR/LDS/scratch),
computes gfx1101 occupancy from per-CU limits (verified via rocminfo), and emits
the contract-H per-kernel report (JSON). The AMD sibling of analyze.py (CUDA).

Usage:
    amd_analyze.py --backend=hip --mode=analyze --kernel matmul --arch gfx1101
    amd_analyze.py --backend=hip --mode=analyze --kernel matmul --arch gfx1101 --format text

Per-CU limits (gfx1101 / RX 7800 XT, from rocminfo):
    VGPR file = 3072 (1536/SIMD x 2 SIMDs), SGPR file = 3200 (1600/SIMD x 2),
    LDS = 64 KiB = 65536 B (GROUP pool), max waves/CU = 32, wave32.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Reuse hipcc_driver for tool discovery (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hipcc_driver  # noqa: E402

DEFAULT_TIMEOUT_S = 240

# --- gfx1101 per-CU hardware limits (verified via rocminfo) ---
VGPR_FILE_PER_CU = 3072    # 1536 per SIMD x 2 SIMDs/CU.
SGPR_FILE_PER_CU = 3200    # 1600 per SIMD x 2 SIMDs/CU.
LDS_PER_CU_BYTES = 65536   # 64 KiB (rocminfo GROUP pool).
MAX_WAVES_PER_CU = 32      # rocminfo "Max Waves Per CU".
WAVE_SIZE = 32             # wave32 (rocminfo "Wavefront Size").


def capture_assembly(
    kernel: str, arch: str, kernel_dir: Path, include_dir: Path, timeout_s: int
) -> str:
    """Compile `kernel` with hipcc --save-temps and return the AMDGPU .s text."""
    hipcc = hipcc_driver.find_tool("hipcc")
    cu = kernel_dir / f"{kernel}.cu"
    if not cu.exists():
        raise SystemExit(
            f"error: kernel source not found: {cu}. Run "
            f"`polykernel-opt <example.mlir> --lower-to-hip` first."
        )
    with tempfile.TemporaryDirectory(prefix="amd_analyze_") as tmp:
        obj = Path(tmp) / f"{kernel}.o"
        cmd = [
            str(hipcc),
            "--save-temps",
            f"--offload-arch={arch}",
            "-DPOLYKERNEL_HIP",
            f"-I{include_dir.resolve()}",
            "-c",
            str(cu.resolve()),
            "-o",
            str(obj),
        ]
        print(f"[amd_analyze] {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=tmp
        )
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr or "")
            raise SystemExit(f"error: hipcc failed (exit {proc.returncode})")
        # hipcc --save-temps writes <basename>-hip-amdgcn-amd-amdhsa-<arch>.s
        # into the cwd (our temp dir).
        s_file = Path(tmp) / f"{kernel}-hip-amdgcn-amd-amdhsa-{arch}.s"
        if not s_file.exists():
            # Fallback: glob for any .s with amdgcn in the name.
            candidates = list(Path(tmp).glob("*amdgcn*.s"))
            if not candidates:
                raise SystemExit(
                    f"error: no AMDGPU .s produced by --save-temps in {tmp}"
                )
            s_file = candidates[0]
        return s_file.read_text()


def parse_amd_isa(assembly: str) -> dict[str, int]:
    """Extract VGPR/SGPR/LDS/scratch from AMDGPU .s metadata."""

    def grab(pattern: str) -> int:
        m = re.search(pattern, assembly)
        return int(m.group(1)) if m else 0

    if not re.search(r"\.amdhsa_kernel\s", assembly):
        raise SystemExit(
            "error: no '.amdhsa_kernel' directive found; not an AMDGPU .s"
        )
    return {
        "vgpr": grab(r"\.amdhsa_next_free_vgpr\s+(\d+)"),
        "sgpr": grab(r"\.amdhsa_next_free_sgpr\s+(\d+)"),
        "lds_bytes": grab(r"\.amdhsa_group_segment_fixed_size\s+(\d+)"),
        "scratch_bytes": grab(r"\.amdhsa_private_segment_fixed_size\s+(\d+)"),
        "scratch_store_count": len(re.findall(r"scratch_store_\w+", assembly)),
        "scratch_load_count": len(re.findall(r"scratch_load_\w+", assembly)),
    }


def compute_occupancy(
    vgpr: int, sgpr: int, lds_bytes: int, threads_per_block: int
) -> dict:
    """Compute gfx1101 occupancy = min(VGPR, SGPR, LDS, wave) limited."""
    waves_per_block = (threads_per_block + WAVE_SIZE - 1) // WAVE_SIZE
    unlimited = 10**9

    vgpr_per_block = vgpr * waves_per_block
    sgpr_per_block = sgpr * waves_per_block
    wg_by_vgpr = VGPR_FILE_PER_CU // vgpr_per_block if vgpr_per_block > 0 else unlimited
    wg_by_sgpr = SGPR_FILE_PER_CU // sgpr_per_block if sgpr_per_block > 0 else unlimited
    wg_by_lds = LDS_PER_CU_BYTES // lds_bytes if lds_bytes > 0 else unlimited
    wg_by_waves = MAX_WAVES_PER_CU // waves_per_block

    # Deterministic tie-break: vgpr, sgpr, lds, waves.
    candidates = [
        ("registers", wg_by_vgpr),
        ("blocks", wg_by_sgpr),
        ("smem", wg_by_lds),
        ("warps", wg_by_waves),
    ]
    limiter, workgroups = min(candidates, key=lambda c: c[1])

    active_waves = workgroups * waves_per_block
    occupancy_pct = 100.0 * active_waves / MAX_WAVES_PER_CU
    return {
        "active_warps_per_sm": active_waves,
        "max_warps_per_sm": MAX_WAVES_PER_CU,
        "occupancy_pct": round(occupancy_pct, 4),
        "limiter": limiter,
    }


def derive_bottleneck(stats: dict[str, int], occupancy: dict) -> str:
    """Classify the performance bottleneck (contract H rules)."""
    if stats["scratch_store_count"] > 0 or stats["scratch_load_count"] > 0:
        return "latency"  # register pressure / spill.
    if stats["scratch_bytes"] > 0:
        return "latency"
    if occupancy["occupancy_pct"] < 50.0:
        return "latency"
    return "compute"


def build_report(
    kernel: str, arch: str, stats: dict[str, int], threads_per_block: int,
    path: str,
) -> dict:
    """Assemble the contract-H per-kernel report for the AMD backend."""
    occupancy = compute_occupancy(
        stats["vgpr"], stats["sgpr"], stats["lds_bytes"], threads_per_block
    )
    bottleneck = derive_bottleneck(stats, occupancy)

    # Spill bytes: scratch_store/load count x 4 (b32 width) as a lower bound;
    # private_segment_fixed_size is the authoritative per-thread scratch total.
    spill_stores = stats["scratch_store_count"] * 4
    spill_loads = stats["scratch_load_count"] * 4

    report = {
        "kernel": kernel,
        "backend": "hip",
        "arch": arch,
        "registers_per_thread": stats["vgpr"],
        "smem_per_block_bytes": stats["lds_bytes"],
        "spill_stores_bytes": spill_stores,
        "spill_loads_bytes": spill_loads,
        "occupancy": occupancy,
        "traffic": {
            "global_read_bytes": 0,
            "global_write_bytes": 0,
            "arithmetic_intensity_flop_per_byte": 0.0,
            "roofline": "memory-bound",
        },
        "bottleneck": bottleneck,
        "suggested_fixes": _derive_fixes(stats, occupancy),
        "path": path,
        # AMD-specific extras (separate namespace from CUDA tuning DB).
        "amd_isa": {
            "sgpr": stats["sgpr"],
            "scratch_bytes_per_thread": stats["scratch_bytes"],
            "scratch_store_count": stats["scratch_store_count"],
            "scratch_load_count": stats["scratch_load_count"],
            "has_spills": (
                stats["scratch_store_count"] > 0
                or stats["scratch_load_count"] > 0
                or stats["scratch_bytes"] > 0
            ),
        },
    }
    return report


def _derive_fixes(stats: dict[str, int], occupancy: dict) -> list[str]:
    fixes: list[str] = []
    if stats["scratch_store_count"] > 0 or stats["scratch_load_count"] > 0:
        fixes.append("reduce register pressure (smaller tile/fewer live values)")
    if occupancy["limiter"] == "registers" and occupancy["occupancy_pct"] < 100.0:
        fixes.append("reduce unroll/block size to lower VGPR pressure")
    if occupancy["occupancy_pct"] < 50.0:
        fixes.append("increase occupancy (smaller tile / fewer VGPRs / less LDS)")
    return fixes


def format_text(report: dict) -> str:
    """Human-readable multi-line summary (mirrors KernelReport::ToText)."""
    occ = report["occupancy"]
    amd = report["amd_isa"]
    lines = [
        f"kernel:                 {report['kernel']}",
        f"backend:                {report['backend']}",
        f"arch:                   {report['arch']}",
        f"registers_per_thread:   {report['registers_per_thread']}  (VGPR)",
        f"sgpr:                   {amd['sgpr']}",
        f"smem_per_block_bytes:   {report['smem_per_block_bytes']}  (LDS)",
        f"spill_stores_bytes:     {report['spill_stores_bytes']}",
        f"spill_loads_bytes:      {report['spill_loads_bytes']}",
        f"scratch_per_thread:     {amd['scratch_bytes_per_thread']}",
        f"has_spills:             {amd['has_spills']}",
        "occupancy:",
        f"  active_warps_per_sm:  {occ['active_warps_per_sm']}",
        f"  max_warps_per_sm:     {occ['max_warps_per_sm']}",
        f"  occupancy_pct:        {occ['occupancy_pct']}",
        f"  limiter:              {occ['limiter']}",
        f"bottleneck:             {report['bottleneck']}",
        f"path:                   {report['path']}",
        "suggested_fixes:",
    ]
    if not report["suggested_fixes"]:
        lines.append("  (none)")
    for f in report["suggested_fixes"]:
        lines.append(f"  - {f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="hip", choices=["hip"])
    ap.add_argument("--mode", default="analyze", help="only 'analyze' is supported")
    ap.add_argument("--kernel", default="matmul")
    ap.add_argument("--arch", default="gfx1101")
    ap.add_argument("--threads-per-block", type=int, default=256)
    ap.add_argument("--path", default="scalar", choices=["scalar", "wmma", "mma"])
    ap.add_argument("--format", default="json", choices=["json", "text"])
    ap.add_argument("--kernel-dir", default="kernels/generated")
    ap.add_argument("--include-dir", default="kernels/template")
    ap.add_argument("--save", default=None, help="also write the report to this file")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()

    if args.mode != "analyze":
        raise SystemExit(f"error: unsupported --mode '{args.mode}' (only 'analyze')")

    assembly = capture_assembly(
        args.kernel, args.arch, Path(args.kernel_dir), Path(args.include_dir),
        args.timeout,
    )
    stats = parse_amd_isa(assembly)
    report = build_report(
        args.kernel, args.arch, stats, args.threads_per_block, args.path
    )

    if args.format == "text":
        output = format_text(report)
    else:
        output = json.dumps(report, indent=2) + "\n"

    print(output, end="")
    if args.save:
        Path(args.save).write_text(output)
        print(f"[amd_analyze] report saved to {args.save}", file=sys.stderr)


if __name__ == "__main__":
    main()
