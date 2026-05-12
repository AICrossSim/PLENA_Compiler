"""Dead-buffer elimination pass.

Walks every op in the HLIRModule (recursing into structured ``for`` op
bodies), collects every buffer name referenced from ``buffer_args``
(strings and ``BufferSlice.parent``) and from ``scalar_args``
(``BufferElement.buffer`` plus any name picked up from a PrimExpr tree).
Buffer names that don't appear in that reachable set get dropped from
``mod.buffers`` — except param buffers, which stay (they're the kernel's
public interface and the HBM staging code assumes they exist).

Why bother: kernels that allocate FP-slot fragments for one algorithm
variant (M_OLD, L_NEW, P_SUM, …) but only end up using a subset still
declare every fragment, and the post-expansion shape-checker can reject
fragments whose shapes don't match the in-use lane mode (see the
``PV_loc shape=(1, 4, 1, 16)`` case). Removing unreachable buffers up
front keeps HLIR honest and avoids spending FPRAM / VRAM on slots no op
will touch.
"""

from __future__ import annotations

from typing import Iterable, Set

from . import hlir as _hlir


def _collect_from_primexpr(expr, out: Set[str]) -> None:
    """Best-effort: walk a PrimExpr tree looking for BufferElement /
    BufferLoad / Var-backed buffer references. We don't import tir here
    so the pass stays usable in pure-HLIR builds; isinstance against the
    BufferElement dataclass and a duck-typed ``.indices``/``.buffer``
    walk is enough for everything to_plena produces."""
    if isinstance(expr, _hlir.BufferElement):
        out.add(expr.buffer)
        for i in expr.indices:
            _collect_from_primexpr(i, out)
        return
    # tir.BufferLoad: .buffer.name + recurse into .indices
    buf_attr = getattr(expr, "buffer", None)
    if buf_attr is not None and hasattr(buf_attr, "name"):
        out.add(str(buf_attr.name))
    indices = getattr(expr, "indices", None)
    if indices is not None:
        for i in indices:
            _collect_from_primexpr(i, out)
    # Generic binop / call recursion via the .a/.b or .args fields tir
    # nodes expose.
    for attr in ("a", "b", "value"):
        sub = getattr(expr, attr, None)
        if sub is not None and not isinstance(sub, (int, float, str, bool)):
            _collect_from_primexpr(sub, out)
    args = getattr(expr, "args", None)
    if args is not None:
        for a in args:
            _collect_from_primexpr(a, out)


def _collect_op_refs(op: _hlir.Op, out: Set[str]) -> None:
    for ba in op.buffer_args:
        if isinstance(ba, str):
            out.add(ba)
        elif isinstance(ba, _hlir.BufferSlice):
            out.add(ba.parent)
    for sa in op.scalar_args:
        _collect_from_primexpr(sa, out)
    if op.body is not None:
        for inner in op.body:
            _collect_op_refs(inner, out)


def _collect_reachable(ops: Iterable[_hlir.Op]) -> Set[str]:
    out: Set[str] = set()
    for op in ops:
        _collect_op_refs(op, out)
    return out


def run(mod: _hlir.HLIRModule) -> _hlir.HLIRModule:
    """Drop buffers that no op references. Param buffers (kernel
    interface) are always kept."""
    reachable = _collect_reachable(mod.ops)
    keep = set(reachable) | set(mod.param_names)
    dropped = [name for name in mod.buffers if name not in keep]
    for name in dropped:
        del mod.buffers[name]
    return mod
