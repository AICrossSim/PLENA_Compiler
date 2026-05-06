"""Split a `plena.group` axis into ``outer × lane_count`` when a ``plena.sync``
op inside that group depends on the group's loop variable.

This implements the lane-fusion split the user described as
``group2.id = group1.id % (N/lane_count)`` plus ``group1.id = group0.id``:

    Before:
        for v in range(N):                        # extent N, group axis
            plena.group(N):
                ...
                plena.sync:                       # this op needs lane fusion
                    op(... uses v ...)
                ...

    After (when N > lane_count and N % lane_count == 0):
        for v_outer in range(N / lane_count):
            plena.group(N / lane_count):
                for v_inner in range(lane_count):
                    plena.group(lane_count):      # lane-fusion-eligible
                        ...
                        plena.sync:
                            op(... uses v_outer * lane_count + v_inner ...)
                        ...

The split is *conditional* on:
  * The for-loop body is an immediate ``plena.group`` AttrStmt (i.e. the
    for-loop is a group axis introduced by ``annotate_group``).
  * The body contains at least one ``plena.sync`` AttrStmt.
  * The sync's wrapped op references the for-loop's loop variable
    (so lane fusion across the loop iterations is meaningful).
  * The for-loop extent is a compile-time int divisible by ``lane_count``
    and greater than ``lane_count``.

Groups whose extent already equals ``lane_count`` are left alone — they
are already lane-fusion-eligible. Groups whose extent is less than
``lane_count`` or not a multiple are also left alone (the lowering pass
will either accept partial-lane utilisation or surface an error).

This pass MUST run after ``annotate_sync`` so that the sync markers it
keys off are present.
"""

from __future__ import annotations

from typing import Optional, Set

from tvm import tir

from .annotate_group import GROUP_KEY, _VarSubst
from .annotate_sync import SYNC_KEY, sync_width as _sync_width


class SplitLaneGroupError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Free-var collection inside a stmt (excluding For loop_vars introduced
# below the current scope -- those are not "free" relative to the outer
# for we're considering).
# ---------------------------------------------------------------------------

def _collect_used_vars(stmt) -> Set[str]:
    """Collect the names of every `tir.Var` referenced anywhere in `stmt`,
    excluding names bound by inner `For` loops (since those are local).

    Name-based to be robust against Var-identity churn across passes.
    """
    used: Set[str] = set()
    locally_bound: Set[str] = set()

    def visit(node, bound: Set[str]):
        if isinstance(node, tir.Var):
            if node.name not in bound:
                used.add(node.name)
            return
        if isinstance(node, tir.For):
            new_bound = bound | {node.loop_var.name}
            visit(node.min, bound)
            visit(node.extent, bound)
            visit(node.body, new_bound)
            return
        if isinstance(node, tir.LetStmt):
            visit(node.value, bound)
            visit(node.body, bound | {node.var.name})
            return
        if isinstance(node, tir.SeqStmt):
            for c in node.seq:
                visit(c, bound)
            return
        if isinstance(node, tir.BlockRealize):
            for v in node.iter_values:
                visit(v, bound)
            visit(node.predicate, bound)
            visit(node.block, bound)
            return
        if isinstance(node, tir.Block):
            new_bound = bound | {iv.var.name for iv in node.iter_vars}
            for r in node.reads:
                visit(r.region, bound) if hasattr(r, "region") else None
            visit(node.body, new_bound)
            if node.init is not None:
                visit(node.init, new_bound)
            return
        if isinstance(node, tir.AttrStmt):
            visit(node.value, bound)
            visit(node.body, bound)
            return
        if isinstance(node, tir.Evaluate):
            visit(node.value, bound)
            return
        if isinstance(node, tir.IfThenElse):
            visit(node.condition, bound)
            visit(node.then_case, bound)
            if node.else_case is not None:
                visit(node.else_case, bound)
            return
        if isinstance(node, tir.BufferLoad):
            for i in node.indices:
                visit(i, bound)
            return
        if isinstance(node, tir.BufferStore):
            visit(node.value, bound)
            for i in node.indices:
                visit(i, bound)
            return
        if isinstance(node, tir.Call):
            for a in node.args:
                visit(a, bound)
            return
        # Generic Add/Mul/Sub/etc.
        for child_attr in ("a", "b", "value"):
            child = getattr(node, child_attr, None)
            if child is not None:
                visit(child, bound)

    visit(stmt, locally_bound)
    return used


def _sync_widths_using_var(stmt, var_name: str, default_width: int) -> Set[int]:
    """Return sync widths whose wrapped op references ``var_name``.

    Sync kinds are deliberately ignored here: h2v DMA, h2m DMA and BTMM
    with the same domain/width are compatible and share the same inner
    hardware lane group.
    """
    found: Set[int] = set()

    def visit(s):
        if isinstance(s, tir.AttrStmt) and s.attr_key == SYNC_KEY:
            if var_name in _collect_used_vars(s.body):
                found.add(_sync_width(s.value, default_width))
                return
            # Continue scanning past this sync (siblings may also have syncs)
            visit(s.body)
            return
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
            return
        if isinstance(s, tir.BlockRealize):
            visit(s.block)
            return
        if isinstance(s, tir.Block):
            visit(s.body)
            return
        if isinstance(s, tir.AttrStmt):
            visit(s.body)
            return
        if isinstance(s, tir.For):
            visit(s.body)
            return
        if isinstance(s, tir.LetStmt):
            visit(s.body)
            return
        if isinstance(s, tir.IfThenElse):
            visit(s.then_case)
            if s.else_case is not None:
                visit(s.else_case)
            return

    visit(stmt)
    return found


# ---------------------------------------------------------------------------
# Group AttrStmt rebuild helpers
# ---------------------------------------------------------------------------

def _make_group_attr(extent: int, body: tir.Stmt) -> tir.Stmt:
    return tir.AttrStmt(
        node=tir.IntImm("int32", 0),
        attr_key=GROUP_KEY,
        value=tir.IntImm("int32", int(extent)),
        body=body,
    )


def _split_for(for_stmt: tir.For, lane_count: int) -> tir.Stmt:
    """Replace ``for v: plena.group(N): real_body`` with::

        for v_outer:
          plena.group(N / lane_count):
            for v_inner:
              plena.group(lane_count):
                real_body[v -> v_outer * lane_count + v_inner]
    """
    inner_attr = for_stmt.body
    if not (isinstance(inner_attr, tir.AttrStmt) and inner_attr.attr_key == GROUP_KEY):
        raise SplitLaneGroupError(
            "expected for-loop body to be a plena.group AttrStmt; "
            f"got {type(inner_attr).__name__}"
        )
    N = int(inner_attr.value.value)
    if N % lane_count != 0:
        raise SplitLaneGroupError(
            f"group extent {N} not divisible by lane_count={lane_count}"
        )
    outer_extent = N // lane_count

    v = for_stmt.loop_var
    v_outer = tir.Var(f"{v.name}_o", v.dtype)
    v_inner = tir.Var(f"{v.name}_i", v.dtype)
    new_v_expr = v_outer * tir.IntImm(v.dtype, lane_count) + v_inner

    real_body = inner_attr.body
    real_body = _VarSubst({v: new_v_expr}).run(real_body)

    inner_for = tir.For(
        loop_var=v_inner,
        min=tir.IntImm(v.dtype, 0),
        extent=tir.IntImm(v.dtype, lane_count),
        kind=tir.ForKind.SERIAL,
        body=_make_group_attr(lane_count, real_body),
        thread_binding=None, annotations={},
    )
    outer_for = tir.For(
        loop_var=v_outer,
        min=tir.IntImm(v.dtype, 0),
        extent=tir.IntImm(v.dtype, outer_extent),
        kind=tir.ForKind.SERIAL,
        body=_make_group_attr(outer_extent, inner_for),
        thread_binding=None, annotations={},
    )
    return outer_for


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

def _walk(stmt, default_width: int):
    if isinstance(stmt, tir.For):
        recursed_body = _walk(stmt.body, default_width)
        candidate = tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            recursed_body, stmt.thread_binding, stmt.annotations,
        )
        # Only consider for-loops that are group axes.
        if not (isinstance(recursed_body, tir.AttrStmt)
                and recursed_body.attr_key == GROUP_KEY):
            return candidate
        if not isinstance(stmt.extent, tir.IntImm):
            return candidate
        N = int(stmt.extent.value)
        widths = _sync_widths_using_var(
            recursed_body.body, stmt.loop_var.name, default_width,
        )
        if not widths:
            return candidate
        if len(widths) != 1:
            raise SplitLaneGroupError(
                f"group axis {stmt.loop_var.name!r} has incompatible sync "
                f"widths {sorted(widths)} in one domain; split by sync class "
                f"is not implemented yet"
            )
        width = next(iter(widths))
        if N < width:
            return candidate
        if N % width != 0:
            raise SplitLaneGroupError(
                f"group extent {N} not divisible by sync width {width}"
            )
        if N == width:
            return candidate
        return _split_for(candidate, width)

    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, default_width) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_walk(stmt.block, default_width),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=_walk(stmt.body, default_width),
            init=stmt.init, alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body, default_width),
        )
    return stmt


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run(func: tir.PrimFunc, lane_count: int = 4) -> tir.PrimFunc:
    if lane_count <= 0:
        raise SplitLaneGroupError(f"lane_count must be positive; got {lane_count}")
    new_body = _walk(func.body, lane_count)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "SplitLaneGroupError"]
