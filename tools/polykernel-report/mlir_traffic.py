"""Parse polykernel textual IR into per-op global memory-traffic records.

Single responsibility: turn the printed IR of a ``polykernel.func`` into one
record per compute op carrying its analytic global traffic (bytes read + written)
and any fusion bookkeeping (``polykernel.fused_from`` / ``eliminated_type[_N]`` /
``workspace_bytes``). No policy, no reporting — see traffic_report.py for that.

TRAFFIC MODEL (analytic, not measured):

    traffic(op) = sum(numel(t) * bytes(t) for operand t)     # global reads
                + sum(numel(t) * bytes(t) for result t)      # global writes

Shape-preserving ops (rmsnorm, gelu, the same-shape ``add``) print a single type
shared by every operand and the one result; shape-changing ops (matmul, the fused
ops) print ``operands -> results``. ``split_type_sig`` handles both.
"""

from __future__ import annotations

import re
from typing import TypedDict


class OpRecord(TypedDict):
    """Per-op analytic global-traffic record produced by ``parse_ops``."""

    ssa: str
    op: str
    operands: list[str]
    operand_types: list[str]
    result_types: list[str]
    reads_bytes: int
    writes_bytes: int
    traffic_bytes: int
    fused_from: str | None
    eliminated_types: list[str]
    workspace_bytes: int | None


# Element width in bytes for the dtypes the closed polykernel op set uses.
ELEM_BYTES: dict[str, int] = {
    "bf16": 2, "f16": 2, "f32": 4, "f64": 8,
    "i1": 1, "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "f8e4m3fn": 1, "f8e5m2": 1,
}

_OP_RE = re.compile(r"^\s*(%\w+)\s*=\s*polykernel\.(\w+)\s+(.*)$")
_ELIM_RE = re.compile(r"polykernel\.eliminated_type(?:_\d+)?\s*=\s*(tensor<[^>]+>)")
_FUSED_RE = re.compile(r'polykernel\.fused_from\s*=\s*"([^"]+)"')
_WORKSPACE_RE = re.compile(r"polykernel\.workspace_bytes\s*=\s*(\d+)")
_TENSOR_RE = re.compile(r"^tensor<(.+)>$")


def tensor_bytes(type_str: str) -> int:
    """Global-memory bytes for a static ranked tensor type (numel * elem bytes)."""
    m = _TENSOR_RE.match(type_str.strip())
    if not m:
        raise ValueError(f"not a ranked tensor type: {type_str!r}")
    parts = m.group(1).split("x")
    dtype, dims = parts[-1], parts[:-1]
    if dtype not in ELEM_BYTES:
        raise ValueError(f"unknown element type {dtype!r} in {type_str!r}")
    if any(d == "?" for d in dims):
        raise ValueError(f"dynamic shape has no analytic traffic: {type_str!r}")
    numel = 1
    for d in dims:
        numel *= int(d)
    return numel * ELEM_BYTES[dtype]


def split_type_sig(sig: str, n_operands: int) -> tuple[list[str], list[str]]:
    """Split an op type signature into (operand types, result types)."""
    sig = sig.strip()
    if "->" in sig:
        lhs, rhs = sig.split("->", 1)
        operands = [t.strip() for t in lhs.split(",") if t.strip()]
        results = [t.strip() for t in rhs.split(",") if t.strip()]
        return operands, results
    return [sig] * n_operands, [sig]


def parse_ops(ir: str) -> list[OpRecord]:
    """Parse polykernel compute ops from textual IR into per-op traffic records."""
    ops: list[OpRecord] = []
    for line in ir.splitlines():
        m = _OP_RE.match(line)
        if not m:
            continue  # module / func / return / brace lines carry no compute
        ssa, name, rest = m.group(1), m.group(2), m.group(3)

        if "{" in rest:
            operands_part = rest[: rest.index("{")]
            attrs_part = rest[rest.index("{") + 1 : rest.index("}")]
            type_sig = rest[rest.index("}") + 1 :].split(":", 1)[1].strip()
        else:
            operands_part, type_sig = rest.split(":", 1)
            attrs_part = ""
            type_sig = type_sig.strip()

        operands = [t.strip() for t in operands_part.split(",") if t.strip().startswith("%")]
        op_types, res_types = split_type_sig(type_sig, len(operands))
        reads = sum(tensor_bytes(t) for t in op_types)
        writes = sum(tensor_bytes(t) for t in res_types)
        fused = _FUSED_RE.search(attrs_part)
        workspace = _WORKSPACE_RE.search(attrs_part)
        ops.append({
            "ssa": ssa,
            "op": name,
            "operands": operands,
            "operand_types": op_types,
            "result_types": res_types,
            "reads_bytes": reads,
            "writes_bytes": writes,
            "traffic_bytes": reads + writes,
            "fused_from": fused.group(1) if fused else None,
            "eliminated_types": _ELIM_RE.findall(attrs_part),
            "workspace_bytes": int(workspace.group(1)) if workspace else None,
        })
    return ops
