#!/usr/bin/env python3
"""polykernel-bench: correctness-gated benchmarking harness (Todo 25 / Wave 5).

THE CORE INVARIANT (correctness-gated benchmarking): a variant's time is recorded
ONLY if it FIRST passes the golden correctness gate (contract C: cosine >= 0.999,
max_rel_err <= ceiling, pcc >= 0.99). Failing variants are discarded + logged with
validated:false and are NEVER "best". The gate, not the timing, decides the winner.

Pipeline for `--autotune --op <op> --shape M,N,K --dtype bf16 --backend hip
--arch gfx1101` (or `--backend cuda --arch sm_89`):

  1. Build the C++ bench driver (lib/Autotune/Benchmark.cpp) with hipcc
     `--offload-arch=<arch>` or nvcc `-arch=<arch>` for the target arch -
     exactly as tests/kernels/test_hip_run.py / test_cuda_run.py build their
     launchers - linking the generated matmul-family kernels, the .npy bridge,
     the HIP runtime (HIP build only; the CUDA build calls the CUDA runtime API
     directly), the Todo 24 ConfigSpace + TuningCache (via the monolithic
     libLLVM), and a generated broken-kernel variant for the negative path. The
     driver lands at a per-backend path (polykernel-bench-driver-{hip,cuda}) so
     both builds coexist.
  2. Probe the backend device (`driver dev`); a no-device machine degrades with
     a clear SKIPPED error instead of a mid-sweep allocation crash.
  3. Enumerate a BOUNDED prefix of pruned variants from the Todo 24 ConfigSpace
     (the driver's `enumerate` mode calls ConfigSpace::Enumerate(); we do NOT
     compile all ~141 configs). Each candidate config is REALIZED by the available
     kernel implementation for the op (the scalar/fused kernel today; the WMMA
     kernel of Todo 22 becomes a second realization when present) - so every
     candidate runs the same correct kernel and the correctness gate is the real
     discriminator while per-config timings are a genuine (if near-tied)
     backend-event measurement.
  4. For EACH variant: run it once (driver `run`) -> compare the GPU output to the
     NumPy golden (tests/golden) -> compute cosine/max_rel_err/pcc -> driver `time`
     evaluates the C++ gate on those metrics and ONLY times the variant if it
     passes (golden BEFORE timing, enforced inside C++). A failed gate logs a
     rejection and records validated:false with NO time.
  5. Select the FASTEST VALIDATED variant (smallest measured median among those
     that passed; an unvalidated variant is never selected) and write it to the
     Todo 24 TuningCache via the driver's `write-cache` mode (SerializeTuningCache;
     contract H is reused, never redefined) with validated:true.

`--inject-broken` adds a fast-but-wrong variant (a kernel that fills the output
with a constant - trivially fast, numerically garbage) to prove the gate excludes
a variant that would otherwise WIN a timing-only race.

Usage:
    polykernel-bench --autotune --op fused_matmul_bias_gelu \
        --shape 2048,4096,11008 --dtype bf16 --backend hip --arch gfx1101
    polykernel-bench --autotune --op fused_matmul_bias_gelu \
        --shape 2048,4096,11008 --dtype bf16 --backend cuda --arch sm_89
"""

from __future__ import annotations

# allow: SIZE_OK - one cohesive CLI orchestrator for the correctness-gated bench:
# driver build + ConfigSpace enumeration + per-op .npy I/O + golden gate +
# backend-event timing + contract-H cache write. The bulk is the embedded
# broken-variant CUDA template (broken_kernels_source) and the closed 3-op I/O
# dispatch; the deliverable constrains this tool to a single file
# (tools/polykernel-bench/bench.py).
import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GOLDEN_DIR = _PROJECT_ROOT / "tests" / "golden"
if str(_GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(_GOLDEN_DIR))

import golden as G  # noqa: E402
from metrics import cosine, max_rel_err, pcc  # noqa: E402

_BF16 = ml_dtypes.bfloat16
SEED = 0xC0FFEE  # fixed seed -> identical tensors every run (defeats flake).
_DRIVER_DIR = _PROJECT_ROOT / "build" / "polykernel-bench"
_BROKEN_CU = _DRIVER_DIR / "broken_kernels.cu"


def driver_path(backend: str) -> Path:
    """Per-backend driver binary, so a hip build and a cuda build coexist without
    clobbering each other's cached binary."""
    return _DRIVER_DIR / f"polykernel-bench-driver-{backend}"

# The matmul-family ops the autotuner tunes (the ConfigSpace is a matmul tile
# space). The acceptance op fused_matmul_bias_gelu is one of these.
MATMUL_FAMILY = ("matmul", "fused_rmsnorm_matmul", "fused_matmul_bias_gelu")

# Contract-C rel-err ceiling (mirror tests/golden/metrics.py). In the small-K,
# non-fused regime the strict single-op value 1e-2 is meaningful and gates the
# scaled-wrong class. Outside it (large K or a fused nonlinearity) max_rel_err is
# dominated by benign near-zero-reference noise - a CORRECT kernel shows
# max_rel_err ~2e3 at K=11008 with cosine == pcc == 1.0 - so the gate relies on the
# robust global metrics cosine >= 0.999 + pcc >= 0.99 there (ceiling = inf), exactly
# as tests/kernels/test_hip_run.py drops the full assert_correct for that class.
_REL_STRICT = 1e-2
_LARGE_K_ONSET = 256


def find_tool(name: str) -> Path:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(
            f"error: '{name}' not found on PATH. Run inside `nix develop` "
            f"(rocmPackages.clr provides hipcc; cuda_nvcc provides nvcc; "
            f"llvmPackages_21 provides llvm-config)."
        )
    return Path(path)


def broken_kernels_source() -> str:
    """The fast-but-wrong variant: trivial constant-fill kernels (no reduction =>
    very fast, numerically garbage => pcc == 0, huge max_rel_err => gate rejects).
    One launch_<op>_broken per matmul-family op, signature-identical to the real
    launch_<op> so the driver dispatches uniformly."""
    return r'''
#include "kernel_common.h"

namespace {
PK_GLOBAL void broken_fill(pk_bf16 *out, long n) {
  long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = pk_float2bf16(7.0f); // wrong constant: fast, but garbage.
}
void fill_broken(pk_bf16 *out, long n, void *stream) {
  const int t = 256;
  const long b = (n + t - 1) / t;
  pk_stream_t s = static_cast<pk_stream_t>(stream);
  PK_LAUNCH(broken_fill, dim3(static_cast<unsigned>(b)), dim3(t), 0, s, out, n);
  PK_CHECK(pk_get_last_error());
}
} // namespace

void launch_matmul_broken(const pk_bf16 *, const pk_bf16 *, pk_bf16 *C, int M,
                          int N, int, void *stream) {
  fill_broken(C, static_cast<long>(M) * N, stream);
}
void launch_fused_rmsnorm_matmul_broken(const pk_bf16 *, const pk_bf16 *,
                                        pk_bf16 *out, int M, int N, int, float,
                                        void *stream) {
  fill_broken(out, static_cast<long>(M) * N, stream);
}
void launch_fused_matmul_bias_gelu_broken(const pk_bf16 *, const pk_bf16 *,
                                          const pk_bf16 *, pk_bf16 *out, int M,
                                          int N, int, void *stream) {
  fill_broken(out, static_cast<long>(M) * N, stream);
}
'''


def build_driver(arch: str, timeout: int, rebuild: bool = False,
                 backend: str = "hip") -> Path:
    """Compile the bench driver for `backend` (hipcc or nvcc) for `arch` (once;
    cached per backend).

    HIP: one hipcc command (Todo 20 pattern) - hipcc's clang handles every TU
    including TuningCache.cpp's LLVM headers.

    CUDA: a three-phase build, because nvcc's bundled host compiler (gcc-13.4)
    cannot compile LLVM 21's headers and nvcc's `-x` flag applies to EVERY input:
      1. the pure-host sources (ConfigSpace/TuningCache/npy_io; backend-agnostic)
         compile with the project's clang++ into .o;
      2. the kernel_common.h-including sources (Benchmark.cpp + the generated /
         broken kernels) compile with nvcc -x cu (device intrinsics such as
         __syncthreads are only declared for CUDA TUs; the -include cstdio
         force-include covers kernel_common.h's PK_CHECK's un-included <cstdio>);
      3. nvcc links all objects against the monolithic libLLVM.
    The driver shim calls the CUDA runtime API directly, so HipRuntime.cpp is NOT
    in the CUDA source set (it is only for the HIP build)."""
    out = driver_path(backend)
    if out.exists() and not rebuild:
        return out
    compiler = find_tool("hipcc" if backend == "hip" else "nvcc")
    host_cc = find_tool("clang++")
    llvm_config = find_tool("llvm-config")
    _DRIVER_DIR.mkdir(parents=True, exist_ok=True)
    _BROKEN_CU.write_text(broken_kernels_source())

    llvm_inc = subprocess.run([str(llvm_config), "--includedir"],
                              capture_output=True, text=True, check=True).stdout.strip()
    llvm_lib = subprocess.run([str(llvm_config), "--libdir"],
                              capture_output=True, text=True, check=True).stdout.strip()

    common_incs = ["-Ikernels/template", "-Iinclude", "-Ikernels/cpu", f"-I{llvm_inc}"]

    if backend == "hip":
        sources = [
            "lib/Autotune/Benchmark.cpp",
            "lib/Autotune/ConfigSpace.cpp",
            "lib/Autotune/TuningCache.cpp",
            "lib/Runtime/HipRuntime.cpp",
            "kernels/cpu/npy_io.cpp",
            *(f"kernels/generated/{op}.cu" for op in MATMUL_FAMILY),
            str(_BROKEN_CU),
        ]
        cmd = [str(compiler), f"--offload-arch={arch}", "-std=c++20",
               "-DPOLYKERNEL_HIP", "-DPOLYKERNEL_BENCH_DRIVER", "-O2", *common_incs,
               *sources, f"-L{llvm_lib}", "-lLLVM-21", "-o", str(out)]
        print(f"[bench] building driver: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=_PROJECT_ROOT)
        if proc.returncode != 0:
            raise SystemExit(
                f"error: hipcc failed to build the bench driver "
                f"(exit {proc.returncode}):\n{proc.stderr}")
        return out

    obj_dir = _DRIVER_DIR / "cuda_obj"
    obj_dir.mkdir(parents=True, exist_ok=True)

    def run_or_raise(cmd: list[str], what: str) -> None:
        print(f"[bench] {what}: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=_PROJECT_ROOT)
        if proc.returncode != 0:
            raise SystemExit(
                f"error: {what} failed (exit {proc.returncode}):\n{proc.stderr}")

    objects: list[str] = []
    for src in ["lib/Autotune/ConfigSpace.cpp", "lib/Autotune/TuningCache.cpp",
                "kernels/cpu/npy_io.cpp"]:
        obj = obj_dir / f"{Path(src).name}.o"
        run_or_raise([str(host_cc), "-std=c++20", "-O2", "-Iinclude", "-Ikernels/cpu",
                      f"-I{llvm_inc}", "-c", src, "-o", str(obj)], "host compile")
        objects.append(str(obj))
    cu_sources = ["lib/Autotune/Benchmark.cpp",
                  *(f"kernels/generated/{op}.cu" for op in MATMUL_FAMILY),
                  str(_BROKEN_CU)]
    for src in cu_sources:
        obj = obj_dir / f"{Path(src).stem}.cu.o"
        run_or_raise([str(compiler), "-x", "cu", f"-arch={arch}", "-DPOLYKERNEL_CUDA",
                      "-include", "cstdio", "-std=c++20", "-DPOLYKERNEL_BENCH_DRIVER",
                      "-O2", *common_incs, "-c", src, "-o", str(obj)],
                     "cuda compile")
        objects.append(str(obj))
    run_or_raise([str(compiler), *objects, f"-L{llvm_lib}", "-lLLVM-21",
                  "-o", str(out)], "link")
    return out


def enumerate_configs(driver: Path, limit: int) -> list[tuple[int, ...]]:
    """The bounded variant set: a prefix of the Todo 24 pruned ConfigSpace."""
    proc = subprocess.run([str(driver), "enumerate", "--limit", str(limit)],
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"enumerate failed:\n{proc.stderr}")
    configs = []
    for line in proc.stdout.splitlines():
        if line.startswith("pruned_total=") or not line.strip():
            continue
        configs.append(tuple(int(x) for x in line.split()))
    return configs


def probe_device(driver: Path, backend: str) -> None:
    """No-device SKIPPED gate: `driver dev` exits 0 iff a usable backend device
    exists. On a no-device machine the sweep degrades with a clear SKIPPED error
    BEFORE any enumerate/run/time work (never a mid-sweep allocation crash, never
    a wrong result - the gate, not the timing, still decides on a device)."""
    proc = subprocess.run([str(driver), "dev"], capture_output=True, text=True,
                          timeout=60)
    if proc.returncode != 0:
        log = (proc.stdout + proc.stderr).strip()
        # Keep the phrase "no device" (tests/autotune/conftest.py's skip detector
        # matches the substring to degrade to SKIPPED instead of FAILED).
        raise SystemExit(
            f"SKIPPED: no device on this machine (backend={backend}) - the "
            f"correctness-gated sweep cannot run.\n{log}"
        )


def make_inputs(op: str, shape: tuple[int, int, int]) -> tuple[list[np.ndarray], np.ndarray]:
    """Fixed-seed bf16 inputs for `op` (in the driver's positional order) + the
    NumPy golden reference output."""
    m, n, k = shape
    rng = np.random.default_rng(SEED)

    def rand(dims: tuple[int, ...]) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=dims).astype(_BF16)

    if op == "matmul":
        a, b = rand((m, k)), rand((k, n))
        return [a, b], G.matmul(a, b)
    if op == "fused_rmsnorm_matmul":
        x, w = rand((m, k)), rand((k, n))
        return [x, w], G.fused_rmsnorm_matmul(x, w)
    if op == "fused_matmul_bias_gelu":
        a, b, bias = rand((m, k)), rand((k, n)), rand((n,))
        return [a, b, bias], G.fused_matmul_bias_gelu(a, b, bias)
    raise SystemExit(f"error: unsupported op '{op}' (matmul-family: {MATMUL_FAMILY})")


def rel_ceiling(op: str, k: int, override: float | None) -> float:
    """The max_rel_err gate ceiling. Strict 1e-2 (contract C) in the small-K,
    non-fused regime where its single-op reduction-order assumption holds; inf
    elsewhere (see the constant's comment) so the gate falls back to the robust
    cosine >= 0.999 + pcc >= 0.99 global metrics."""
    if override is not None:
        return override
    if k < _LARGE_K_ONSET and not op.startswith("fused_"):
        return _REL_STRICT
    return float("inf")


@dataclass(frozen=True)
class Variant:
    """One benchmarked candidate: a pruned ConfigSpace config realized by a kernel
    implementation (`impl`: "default" real kernel, or "broken" injected variant)."""
    name: str
    impl: str
    config: tuple[int, ...]


@dataclass(frozen=True)
class VariantOutcome:
    """A variant's gated result: golden correctness, whether it PASSED the gate
    (validated), and - only if validated - its backend-event min/median time.
    time is None precisely when the gate rejected it (it was never timed)."""
    variant: Variant
    corr: tuple[float, float, float]
    validated: bool
    min_ms: float | None
    median_ms: float | None
    log: str


@dataclass(frozen=True)
class Bench:
    """The bench run context: the built driver + the op + its uploaded .npy inputs
    + timing knobs. Owns the three driver interactions (run / time / write-cache)."""
    driver: Path
    op: str
    in_paths: tuple[Path, ...]
    out_path: Path
    warmup: int
    iters: int
    timeout: int

    def run(self, variant: Variant) -> None:
        """Launch the variant ONCE + write its bf16 .npy output (for the golden diff)."""
        cmd = [str(self.driver), "run", self.op, variant.impl,
               *map(str, self.in_paths), "--out", str(self.out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"run {self.op}/{variant.impl} failed (exit {proc.returncode}):\n{proc.stderr}")

    def time(self, variant: Variant, corr: tuple[float, float, float],
             ceiling: float) -> VariantOutcome:
        """Driver `time`: the C++ gate decides. A rejected variant is NEVER timed
        (driver exit 3) -> validated:False + no time."""
        c, r, p = corr
        cmd = [str(self.driver), "time", self.op, variant.impl, *map(str, self.in_paths),
               "--cosine", repr(c), "--rel", repr(r), "--pcc", repr(p),
               "--ceiling", repr(ceiling), "--warmup", str(self.warmup),
               "--iters", str(self.iters)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        log = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 3:  # gate-rejected (distinct exit code from the driver).
            return VariantOutcome(variant, corr, False, None, None, log)
        if proc.returncode != 0:
            raise RuntimeError(
                f"time {self.op}/{variant.impl} failed (exit {proc.returncode}):\n{log}")
        # "VALIDATED <op>/<impl> min_ms=X median_ms=Y iters=N"
        fields = dict(tok.split("=") for tok in proc.stdout.split() if "=" in tok)
        return VariantOutcome(variant, corr, True,
                              float(fields["min_ms"]), float(fields["median_ms"]), log)

    def write_cache(self, args: argparse.Namespace, best: VariantOutcome,
                    out: Path) -> str:
        """Build the contract-H CacheEntry + SerializeTuningCache it (the driver
        reuses the Todo 24 serializer; the schema is NOT redefined here)."""
        m, n, k = args.shape
        c, r, p = best.corr
        cmd = [str(self.driver), "write-cache",
               "--gpu", args.arch, "--op", self.op,
               "--M", str(m), "--N", str(n), "--K", str(k), "--dtype", args.dtype,
               "--config", " ".join(str(x) for x in best.variant.config),
               "--scored-by", "measure", "--time-ms", repr(best.median_ms),
               "--cosine", repr(c), "--rel", repr(r), "--pcc", repr(p),
               "--validated", "true", "--out", str(out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"write-cache failed (exit {proc.returncode}):\n{proc.stderr}")
        return proc.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--autotune", action="store_true", help="run the correctness-gated autotune sweep")
    ap.add_argument("--op", default="fused_matmul_bias_gelu", choices=MATMUL_FAMILY)
    ap.add_argument("--shape", default="2048,4096,11008", help="GEMM shape M,N,K")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--backend", default="hip", choices=["hip", "cuda"])
    ap.add_argument("--arch", default="gfx1101")
    ap.add_argument("--variants", type=int, default=4, help="bounded prefix of pruned ConfigSpace configs to race")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--inject-broken", action="store_true", help="add a fast-but-wrong variant (negative path)")
    ap.add_argument("--rel-ceiling", type=float, default=None, help="override the max_rel_err gate ceiling")
    ap.add_argument("--cache-out", default=None, help="tuning-cache JSON output path")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--rebuild", action="store_true", help="force a driver rebuild")
    args = ap.parse_args()

    if not args.autotune:
        raise SystemExit("error: only --autotune is supported")

    args.shape = tuple(int(x) for x in args.shape.split(","))
    if len(args.shape) != 3:
        raise SystemExit("error: --shape must be M,N,K")
    m, n, k = args.shape
    ceiling = rel_ceiling(args.op, k, args.rel_ceiling)

    driver = build_driver(args.arch, args.timeout, args.rebuild, args.backend)
    probe_device(driver, args.backend)
    configs = enumerate_configs(driver, args.variants)
    print(f"[bench] op={args.op} shape=M={m},N={n},K={k} dtype={args.dtype} "
          f"arch={args.arch} rel_ceiling={ceiling:.0e}")
    print(f"[bench] variant set: {len(configs)} pruned ConfigSpace configs"
          f"{' + injected broken' if args.inject_broken else ''}")

    inputs, golden_ref = make_inputs(args.op, args.shape)
    work = _DRIVER_DIR / "work"
    work.mkdir(parents=True, exist_ok=True)
    in_paths = []
    for i, arr in enumerate(inputs):
        p = work / f"in{i}.npy"
        np.save(p, arr)
        in_paths.append(p)
    bench = Bench(driver, args.op, tuple(in_paths), work / "out.npy",
                  args.warmup, args.iters, args.timeout)

    # The variant race: each pruned config realized by the default kernel, plus
    # (optionally) the injected fast-but-wrong broken variant.
    variants = [Variant(f"config_{i}", "default", cfg) for i, cfg in enumerate(configs)]
    if args.inject_broken:
        variants.append(Variant("broken_fast", "broken", configs[0]))

    outcomes: list[VariantOutcome] = []
    print("=" * 78)
    for v in variants:
        # GOLDEN BEFORE TIMING: run once, diff vs golden, THEN offer to the C++ gate.
        bench.run(v)
        out = np.load(bench.out_path).view(_BF16)
        corr = (cosine(out, golden_ref), max_rel_err(out, golden_ref), pcc(out, golden_ref))
        outcome = bench.time(v, corr, ceiling)
        outcomes.append(outcome)
        c, r, p = outcome.corr
        if outcome.validated:
            print(f"[bench] VALIDATED {v.name:12s} config={v.config} "
                  f"cosine={c:.6f} rel={r:.3e} pcc={p:.6f} "
                  f"min_ms={outcome.min_ms:.5f} median_ms={outcome.median_ms:.5f}")
        else:
            print(f"[bench] REJECTED  {v.name:12s} config={v.config} "
                  f"cosine={c:.6f} rel={r:.3e} pcc={p:.6f} "
                  f"-> validated:false, NOT timed, excluded")
            print(f"          gate log: {outcome.log}")
    print("=" * 78)

    # Fastest VALIDATED variant (an unvalidated variant is never selected).
    validated = [o for o in outcomes if o.validated]
    if not validated:
        raise SystemExit("error: NO variant passed the correctness gate; no cache entry written")
    best = min(validated, key=lambda o: o.median_ms)
    print(f"[bench] BEST (fastest validated): {best.variant.name} "
          f"config={best.variant.config} median_ms={best.median_ms:.5f}")

    cache_out = Path(args.cache_out) if args.cache_out else _DRIVER_DIR / "tuning_cache.json"
    cache_json = bench.write_cache(args, best, cache_out)
    print(f"[bench] tuning cache written to {cache_out} (validated:true):")
    print(cache_json)


if __name__ == "__main__":
    main()
