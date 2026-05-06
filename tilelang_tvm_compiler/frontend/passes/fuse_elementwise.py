"""Fuse a parallel-group elementwise op into a single PLENA vector op.

Detects this pattern (post-``annotate_group``)::

    for i in range(N):
        plena.group(N):
            dst[..., i] = lhs[..., i]  OP  rhs[..., i]

(this is what ``T.Parallel(N)`` lowers to once ``annotate_group`` has run)
and rewrites the entire for-loop to a single vector op call::

    plena.v_<op>(lhs.data, rhs.data, dst.data)

Pattern requirements:
  * Outer node is a ``tir.For`` whose body is an ``AttrStmt(plena.group,
    value=N)`` with ``N == for.extent``.
  * The group's body is a single ``BufferStore``.
  * The store's last index is the for-loop's ``loop_var``.
  * The store's value is a supported binary op on two ``BufferLoad``s,
    each with the same lane-var indexing in its last dim.

Supported ops today: ``+`` → ``plena.v_add``. Sub/mul/etc. fall through
unchanged so the kernel still compiles (without fusion); add more by
extending ``_OP_TO_INTRIN``.

Non-matching for-loops are left as-is — this pass is opportunistic, not
mandatory.
"""

from __future__ import annotations

from typing import Optional

import tvm
from tvm import tir

from .annotate_group import GROUP_KEY


# Map from TIR binary-op node type -> plena vector intrinsic name.
# All three lower to ``emit_tile_binary`` in the ISA emitter (the same
# code path) with op ∈ {add, sub, mul}; the only thing that differs is
# the HW opcode (V_ADD_VV / V_SUB_VV / V_MUL_VV).
_OP_TO_INTRIN = {
    tir.Add: "plena.v_add",
    tir.Sub: "plena.v_sub",
    tir.Mul: "plena.v_mul",
}


def _make_call(name: str, args: list) -> tir.Call:
    extern_op = tvm.ir.Op.get("tir.call_extern")
    return tir.Call("handle", extern_op, [tir.StringImm(name), *args])


def _is_lane_var_indexed(load: tir.BufferLoad, lane_var_name: str) -> bool:
    """The buffer load's last index references exactly the lane var
    (no compound expression)."""
    if not load.indices:
        return False
    last = load.indices[-1]
    return isinstance(last, tir.Var) and last.name == lane_var_name


def _try_fuse(for_stmt: tir.For) -> Optional[tir.Stmt]:
    """Return a single Evaluate(call_extern) replacing `for_stmt` if it
    matches the elementwise pattern, else None.

    Two fusion shapes are recognised:
      * Binary op:    ``dst[..., i] = lhs[..., i] OP rhs[..., i]``
                      → ``plena.v_<op>(lhs, rhs, dst)``
      * Constant fill: ``dst[..., i] = const_imm``
                      → ``plena.zero_v(dst)`` when const == 0; other
                       constants fall through (HW lacks a generic fill
                       for now).
    """
    if not isinstance(for_stmt.body, tir.AttrStmt):
        return None
    attr = for_stmt.body
    if attr.attr_key != GROUP_KEY:
        return None
    if not (isinstance(attr.value, tir.IntImm)
            and isinstance(for_stmt.extent, tir.IntImm)
            and int(attr.value.value) == int(for_stmt.extent.value)):
        return None

    body = attr.body
    if not isinstance(body, tir.BufferStore):
        return None

    lane_var_name = for_stmt.loop_var.name

    if not body.indices or not isinstance(body.indices[-1], tir.Var):
        return None
    if body.indices[-1].name != lane_var_name:
        return None

    expr = body.value

    # Constant fill — currently only ``= 0`` lowers (plena.zero_v).
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        if float(expr.value) == 0.0:
            return tir.Evaluate(_make_call("plena.zero_v", [body.buffer.data]))
        return None

    # Binary elementwise — currently only ``+`` (plena.v_add).
    intrin_name = _OP_TO_INTRIN.get(type(expr))
    if intrin_name is None:
        return None
    if not isinstance(expr.a, tir.BufferLoad) or not isinstance(expr.b, tir.BufferLoad):
        return None
    if not _is_lane_var_indexed(expr.a, lane_var_name):
        return None
    if not _is_lane_var_indexed(expr.b, lane_var_name):
        return None

    return tir.Evaluate(_make_call(intrin_name, [
        expr.a.buffer.data,
        expr.b.buffer.data,
        body.buffer.data,
    ]))


def _try_fuse_nested(outer: tir.For) -> Optional[tir.Stmt]:
    """Fold ``for r in T.serial(R): <single-loop fuse target>`` into a
    single whole-buffer ``plena.v_*`` / ``plena.zero_v``.

    Why this is needed: with lane fusion the inner T.Parallel(C) covers
    ``C * lane_count`` elements per outer iteration; running R outer
    iterations means R*C*lane_count total elements touched — which is
    exactly the post-expansion buffer size for the typical
    ``(rows, hlen)`` fragment-shaped buffer in flash-attention kernels.
    The HW ops emitted by ``_try_fuse`` (plena.v_add / plena.zero_v) are
    inherently whole-buffer (no extent / offset args), so a single
    invocation already covers all R*C*lane_count elements. Wrapping it
    in the outer T.serial(R) would re-execute the same whole-buffer op
    R times — semantically wrong (R-fold accumulation for v_add) and
    R× slower. Folding the outer for matches what the user actually
    means without forcing them to write the misleading single-row
    ``dst[0, col] = ...`` workaround.

    Only safe for ops whose HW path genuinely covers the whole buffer
    in one invocation — currently ``plena.zero_v`` and any
    ``plena.v_*``. Other lowerings (matmul, mv, …) are NOT whole-buffer
    and must keep any surrounding for loops.
    """
    if outer.kind != tir.ForKind.SERIAL:
        return None
    inner = outer.body
    if not isinstance(inner, tir.For):
        return None
    inner_fused = _try_fuse(inner)
    if inner_fused is None or not isinstance(inner_fused, tir.Evaluate):
        return None
    v = inner_fused.value
    if not (isinstance(v, tir.Call)
            and getattr(v.op, "name", None) == "tir.call_extern"
            and v.args
            and isinstance(v.args[0], tir.StringImm)):
        return None
    name = v.args[0].value
    if not (name == "plena.zero_v" or name.startswith("plena.v_")):
        return None
    # Outer for is redundant — the inner fused HW op is already
    # whole-buffer. Drop it.
    return inner_fused


def _walk(stmt):
    if isinstance(stmt, tir.For):
        # Try the nested fold first (outer serial + inner T.Parallel
        # both collapse into one whole-buffer op); fall back to the
        # single-loop fold; otherwise recurse.
        replaced = _try_fuse_nested(stmt)
        if replaced is not None:
            return replaced
        replaced = _try_fuse(stmt)
        if replaced is not None:
            return replaced
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body), stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_walk(stmt.block),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=_walk(stmt.body),
            init=stmt.init, alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body))
    return stmt


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    return tir.PrimFunc(
        params=func.params,
        body=_walk(func.body),
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run"]
