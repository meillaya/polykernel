"""End-to-end MLP-block correctness + fusion check (Todo 17 / Wave 3).

Runs the MLP block (examples/mlp_block.mlir op sequence: rmsnorm -> matmul ->
gelu -> matmul -> add residual) through the EXISTING CPU-reference drivers
(tests/kernels mechanism) and asserts EVERY op passes the golden harness
(tests/golden) under contract C, both unfused and fused (the only fusion that
fires on this block is --fuse-rmsnorm-matmul).

SCALE NOTE: the real mlp_block.mlir uses Llama-2-7B dims (a 2048x11008x4096
matmul = ~92 GFLOP under the naive CPU ref) — far too large to EXECUTE in a test.
Correctness is therefore validated on a tractable mini-MLP with the IDENTICAL op
structure; the analytic traffic report (tools/polykernel-report) covers the real
shapes. A separate test confirms the real file fuses as expected (parsed, not
executed).

The residual ``add`` has no CPU-ref kernel (kernels/cpu ships none — it is a
trivial elementwise op fully determined by the rounding contract); the harness
computes it inline (fp32-add + bf16 RNE) and flags it ``ref-only``. The compute
ops (rmsnorm, matmul, gelu, fused_rmsnorm_matmul) are validated against their
real CPU-ref executables.

Standalone harness (negative-test evidence): ``python
tests/e2e/test_mlp_correctness.py --corrupt fused_rmsnorm_matmul`` injects a
broken fused kernel (wrong scale factor), prints the per-op metric table naming
the failing op + cosine/rel/PCC, and exits non-zero.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Make the golden harness + the traffic-report IR parser importable both under
# pytest (tests/e2e/conftest.py also sets this) and when run standalone
# (python tests/e2e/test_mlp_correctness.py), where no conftest is auto-loaded.
for _p in (_PROJECT_ROOT / "tests" / "golden", _PROJECT_ROOT / "tools" / "polykernel-report"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import golden as G  # noqa: E402
from metrics import (  # noqa: E402
    COSINE_THRESHOLD,
    MAX_REL_ERR_THRESHOLD,
    PCC_THRESHOLD,
    cosine,
    max_rel_err,
    pcc,
)
import mlir_traffic  # noqa: E402  (reuse the IR parser for the real-file check)

_MLP_BLOCK = _PROJECT_ROOT / "examples" / "mlp_block.mlir"
_OPT = _PROJECT_ROOT / "build" / "tools" / "polykernel-opt" / "polykernel-opt"
_FUSION_PASSES = ["--fuse-rmsnorm-matmul", "--fuse-matmul-bias-gelu",
                  "--fuse-residual-rmsnorm", "--fuse-softmax-mask"]


@dataclass(frozen=True)
class OpResult:
    variant: str   # "unfused" | "fused" | "cross"
    label: str     # op name (matmul#1/#2 disambiguate the two matmuls)
    kind: str      # "cpu-ref" (real exe) | "ref-only" (no kernel) | "pipeline"
    cosine: float
    max_rel_err: float
    pcc: float
    passed: bool


@dataclass
class E2EResult:
    rows: list[OpResult]

    @property
    def failures(self) -> int:
        return sum(1 for r in self.rows if not r.passed)

    @staticmethod
    def _fmt_row(r: OpResult) -> str:
        status = "PASS" if r.passed else "FAIL"
        return f"{r.variant:9} {r.label:24} {r.kind:9} {r.cosine:11.6f} {r.max_rel_err:13.3e} {r.pcc:11.6f}  {status}"

    def render(self) -> str:
        header = f"{'variant':9} {'op':24} {'kind':9} {'cosine':>11} {'max_rel_err':>13} {'pcc':>11}  status"
        lines = [header, "-" * len(header)]
        lines.extend(self._fmt_row(r) for r in self.rows)
        lines.append(f"\n{self.failures} failed / {len(self.rows)} ops (contract C: cosine>={COSINE_THRESHOLD}, max_rel_err<={MAX_REL_ERR_THRESHOLD:.0e}, pcc>={PCC_THRESHOLD})")
        return "\n".join(lines)


def run_mlp_block(drivers, make_tensor, *, seq: int = 32, d_model: int = 64,
                  ffn: int = 128, eps: float = 1e-5,
                  corrupt: str | None = None) -> E2EResult:
    """Drive the MLP block (unfused + fused) through the CPU refs vs the golden.

    ``corrupt``: name of a fused op whose CPU-ref output is broken (scaled by a
    wrong factor -1.0) to prove the harness detects a bad kernel and fails.
    """
    if corrupt is not None and corrupt != "fused_rmsnorm_matmul":
        raise ValueError(f"unsupported corrupt target: {corrupt!r}")

    x = make_tensor((1, seq, d_model))
    w1 = make_tensor((d_model, ffn))
    w2 = make_tensor((ffn, d_model))

    def check(variant: str, label: str, kind: str, out: np.ndarray, ref: np.ndarray) -> OpResult:
        c, r, p = cosine(out, ref), max_rel_err(out, ref), pcc(out, ref)
        passed = (c >= COSINE_THRESHOLD and r <= MAX_REL_ERR_THRESHOLD and p >= PCC_THRESHOLD)
        return OpResult(variant, label, kind, float(c), float(r), float(p), passed)

    # --- UNFUSED pipeline: CPU-ref drivers vs golden primitives ---------------
    c0 = drivers.rmsnorm(x, eps)
    c1 = drivers.matmul(c0, w1)
    c2 = drivers.gelu(c1)
    c3 = drivers.matmul(c2, w2)
    c4 = drivers.add(c3, x)
    n0 = G.rmsnorm(x, None, eps)
    n1 = G.matmul(n0, w1)
    n2 = G.gelu(n1)
    n3 = G.matmul(n2, w2)
    n4 = G.add(n3, x)

    rows = [
        check("unfused", "rmsnorm", "cpu-ref", c0, n0),
        check("unfused", "matmul#1", "cpu-ref", c1, n1),
        check("unfused", "gelu", "cpu-ref", c2, n2),
        check("unfused", "matmul#2", "cpu-ref", c3, n3),
        check("unfused", "add", "ref-only", c4, n4),
    ]

    # --- FUSED pipeline: fused_rmsnorm_matmul, then gelu/matmul/add -----------
    f0 = drivers.fused_rmsnorm_matmul(x, w1, eps)
    if corrupt == "fused_rmsnorm_matmul":
        f0 = (-f0.astype(np.float32)).astype(f0.dtype)  # broken kernel: wrong factor
    f1 = drivers.gelu(f0)
    f2 = drivers.matmul(f1, w2)
    f3 = drivers.add(f2, x)
    g0 = G.fused_rmsnorm_matmul(x, w1, None, eps)
    g1 = G.gelu(g0)
    g2 = G.matmul(g1, w2)
    g3 = G.add(g2, x)

    rows += [
        check("fused", "fused_rmsnorm_matmul", "cpu-ref", f0, g0),
        check("fused", "gelu", "cpu-ref", f1, g1),
        check("fused", "matmul#2", "cpu-ref", f2, g2),
        check("fused", "add", "ref-only", f3, g3),
        # fusion must preserve semantics: fused block output == unfused block output
        check("cross", "fused==unfused output", "pipeline", f3, c4),
    ]
    return E2EResult(rows)


# --- pytest tests -----------------------------------------------------------


def test_mlp_unfused_all_ops_pass_golden(cpu_drivers, rand_bf16):
    result = run_mlp_block(cpu_drivers, rand_bf16)
    unfused = [r for r in result.rows if r.variant == "unfused"]
    failed = [r for r in unfused if not r.passed]
    assert not failed, f"unfused ops failed golden:\n{result.render()}"


def test_mlp_fused_all_ops_pass_golden(cpu_drivers, rand_bf16):
    result = run_mlp_block(cpu_drivers, rand_bf16)
    fused = [r for r in result.rows if r.variant in ("fused", "cross")]
    failed = [r for r in fused if not r.passed]
    assert not failed, f"fused ops failed golden:\n{result.render()}"


def test_mlp_block_zero_failures_end_to_end(cpu_drivers, rand_bf16):
    result = run_mlp_block(cpu_drivers, rand_bf16)
    assert result.failures == 0, f"expected 0 failures:\n{result.render()}"


def test_mlp_block_real_file_fuses_as_expected():
    """The real examples/mlp_block.mlir fuses (parsed, not executed: too large)."""
    if not _OPT.exists():
        raise FileNotFoundError(f"polykernel-opt not found at {_OPT}; build it first")
    proc = subprocess.run([str(_OPT), str(_MLP_BLOCK), *_FUSION_PASSES],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"polykernel-opt failed:\n{proc.stderr}"
    ops = mlir_traffic.parse_ops(proc.stdout)
    fused = [o for o in ops if o["fused_from"]]
    assert len(fused) == 1, f"expected exactly one fused op, got {fused}"
    assert fused[0]["op"] == "fused_rmsnorm_matmul"
    assert fused[0]["fused_from"] == "rmsnorm_matmul"
    assert fused[0]["eliminated_types"] == ["tensor<1x2048x4096xbf16>"]
    # the eliminated intermediate's global round-trip is the fusion's saving
    saved = sum(2 * mlir_traffic.tensor_bytes(t) for t in fused[0]["eliminated_types"])
    assert saved == 33_554_432  # 2 * 1*2048*4096 * 2 bytes = 32 MiB


def test_negative_broken_fused_kernel_is_detected(cpu_drivers, rand_bf16):
    """A broken fused kernel must surface the failing op + metrics and fail."""
    result = run_mlp_block(cpu_drivers, rand_bf16, corrupt="fused_rmsnorm_matmul")
    assert result.failures >= 1, "broken kernel was NOT detected"
    failing = [r for r in result.rows if not r.passed]
    assert failing[0].label == "fused_rmsnorm_matmul"
    assert failing[0].cosine < COSINE_THRESHOLD
    assert failing[0].max_rel_err > MAX_REL_ERR_THRESHOLD
    text = result.render()
    assert "fused_rmsnorm_matmul" in text and "FAIL" in text


# --- standalone harness (non-zero exit on failure) --------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end MLP correctness harness (exits non-zero on failure).")
    parser.add_argument("--corrupt", metavar="OP", default=None,
                        help="inject a broken kernel for this op (negative test)")
    args = parser.parse_args(argv)

    # Load tests/e2e/conftest.py by absolute path: a bare `import conftest` is
    # ambiguous (tests/golden + tests/kernels each ship a conftest.py on sys.path).
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "e2e_conftest", Path(__file__).resolve().parent / "conftest.py")
    if spec is None or spec.loader is None:
        raise ImportError("cannot load tests/e2e/conftest.py")
    e2e_conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2e_conftest)

    with tempfile.TemporaryDirectory() as td:
        drivers = e2e_conftest.build_cpu_drivers(Path(td))
        result = run_mlp_block(drivers, e2e_conftest.make_rand_bf16(), corrupt=args.corrupt)

    print(result.render())
    if result.failures:
        print(f"\nRESULT: FAIL — {result.failures} op(s) failed golden contract C",
              file=sys.stderr)
        return 1
    print("\nRESULT: PASS — 0 failures (all ops pass golden contract C)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
