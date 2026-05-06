"""Convert tilelang grid bindings and parallel loops into PLENA *groups*.

A *group* is a thread-bundle scope. PLENA hardware is fundamentally
single-threaded; what tilelang expresses as parallel grid axes or
`T.Parallel` iterators becomes, in PLENA-flavoured TIR, a serial for-loop
wrapped in a ``T.attr(0, "plena.group", extent=N)`` AttrStmt. Downstream
passes use this annotation to:

  * fuse per-iteration DMA / BTMM ops at sync points into single multi-
    lane hardware ops (``lower_to_hlir``);
  * expand shared / fragment buffers used inside the group by the group
    extent (``allocate_group_memory``).

Conversions performed:

  * ``AttrStmt(thread_extent, IterVar(blockIdx.*/threadIdx.*), N)``
        →   if N == 1: drop the binding (substitute the var with 0 in
                       the body — degenerate group);
            if N >  1: ``for v in range(N): T.attr(0, "plena.group", N)
                              <body>``.
  * ``For(kind=Parallel)``:
        →   ``for v in range(extent): T.attr(0, "plena.group", extent)
                              <body>``  (kind becomes Serial since the
            hardware doesn't run threads in parallel; the group annotation
            tells the lowering pass that the iterations are
            fusion-eligible).

Invariants on output:

  * No ``AttrStmt(thread_extent, ...)`` remains.
  * No ``tir.For`` has ``ForKind.PARALLEL``.
  * Every group axis is wrapped in exactly one ``plena.group`` AttrStmt
    sitting immediately inside the surrounding ``tir.For``.
"""

from __future__ import annotations

from typing import Dict

import tvm
from tvm import tir


GROUP_KEY = "plena.group"


class GroupAnnotateError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Var substitution helper (extent-1 bindings collapse the var to 0).
# ---------------------------------------------------------------------------

class _VarSubst:
    """Recursively substitute every var occurrence in `sub` with its mapped
    expression. Walks both Stmt and Expr trees."""

    def __init__(self, sub: Dict[tir.Var, tir.PrimExpr]):
        self.sub = sub
        self.sub_by_name = {v.name: e for v, e in sub.items()}

    def _lookup(self, var: tir.Var):
        if var in self.sub:
            return self.sub[var]
        return self.sub_by_name.get(var.name, var)

    def run(self, node):
        return self._visit(node)

    def _visit(self, n):
        if isinstance(n, tir.SeqStmt):
            return tir.SeqStmt([self._visit(c) for c in n.seq])
        if isinstance(n, tir.BlockRealize):
            return tir.BlockRealize(
                iter_values=[self._visit(v) for v in n.iter_values],
                predicate=self._visit(n.predicate),
                block=self._visit(n.block),
            )
        if isinstance(n, tir.Block):
            return tir.Block(
                iter_vars=n.iter_vars, reads=n.reads, writes=n.writes,
                name_hint=n.name_hint, body=self._visit(n.body),
                init=self._visit(n.init) if n.init is not None else None,
                alloc_buffers=n.alloc_buffers,
                match_buffers=n.match_buffers, annotations=n.annotations,
            )
        if isinstance(n, tir.AttrStmt):
            return tir.AttrStmt(n.node, n.attr_key,
                                self._visit(n.value), self._visit(n.body))
        if isinstance(n, tir.For):
            return tir.For(
                n.loop_var, self._visit(n.min), self._visit(n.extent),
                n.kind, self._visit(n.body), n.thread_binding, n.annotations,
            )
        if isinstance(n, tir.Evaluate):
            return tir.Evaluate(self._visit(n.value))
        if isinstance(n, tir.IfThenElse):
            return tir.IfThenElse(
                self._visit(n.condition),
                self._visit(n.then_case),
                self._visit(n.else_case) if n.else_case is not None else None,
            )
        if isinstance(n, tir.LetStmt):
            return tir.LetStmt(n.var, self._visit(n.value), self._visit(n.body))
        if isinstance(n, tir.BufferStore):
            return tir.BufferStore(
                n.buffer, self._visit(n.value),
                [self._visit(i) for i in n.indices],
            )
        if isinstance(n, tir.BufferLoad):
            return tir.BufferLoad(
                n.buffer, [self._visit(i) for i in n.indices],
            )
        if isinstance(n, tir.Call):
            return tir.Call(n.dtype, n.op, [self._visit(a) for a in n.args])
        if isinstance(n, tir.Var):
            return self._lookup(n)
        if isinstance(n, (tir.IntImm, tir.FloatImm, tir.StringImm)):
            return n
        # Generic Add / Mul / etc. — recurse via their `a`, `b`.
        for child_attr in ("a", "b", "value"):
            child = getattr(n, child_attr, None)
            if child is not None:
                # Best-effort generic handling: rebuild the same node type.
                # If this misses an op we will hit it during testing.
                pass
        # Common arithmetic: tir.Add/Sub/Mul/FloorDiv/FloorMod/Min/Max all
        # have (a, b). Reconstruct via the same constructor.
        if hasattr(n, "a") and hasattr(n, "b"):
            return type(n)(self._visit(n.a), self._visit(n.b))
        return n


# ---------------------------------------------------------------------------
# Helpers: thread-binding detection
# ---------------------------------------------------------------------------

_BLOCK_PREFIX = "blockIdx"
_THREAD_PREFIX = "threadIdx"


def _thread_binding_kind(stmt: tir.Stmt) -> Optional[str]:
    """Return ``"block"`` for a blockIdx.* binding, ``"thread"`` for a
    threadIdx.* binding, or None for anything else."""
    if not isinstance(stmt, tir.AttrStmt):
        return None
    if stmt.attr_key != "thread_extent":
        return None
    node = stmt.node
    if not isinstance(node, tir.IterVar):
        return None
    tag = str(node.thread_tag) if node.thread_tag else ""
    if tag.startswith(_BLOCK_PREFIX):
        return "block"
    if tag.startswith(_THREAD_PREFIX):
        return "thread"
    return None


def _wrap_group(loop_var: tir.Var, extent: int, body: tir.Stmt) -> tir.Stmt:
    """Wrap `body` in a serial for-loop and a `plena.group` AttrStmt.

    Layout:    for v in range(extent):
                   T.attr(0, "plena.group", extent):
                       <body>
    """
    inner = tir.AttrStmt(
        node=tir.IntImm("int32", 0),
        attr_key=GROUP_KEY,
        value=tir.IntImm("int32", int(extent)),
        body=body,
    )
    return tir.For(
        loop_var=loop_var,
        min=tir.IntImm(loop_var.dtype, 0),
        extent=tir.IntImm(loop_var.dtype, int(extent)),
        kind=tir.ForKind.SERIAL,
        body=inner,
        thread_binding=None,
        annotations={},
    )


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------

def _walk(stmt: tir.Stmt) -> tir.Stmt:
    binding_kind = _thread_binding_kind(stmt)
    if binding_kind is not None:
        iter_var = stmt.node
        var = iter_var.var
        ext = stmt.value
        if not isinstance(ext, tir.IntImm):
            raise GroupAnnotateError(
                f"thread binding {var.name!r} has non-constant extent {ext!r}; "
                f"groups require compile-time extent"
            )
        ext_val = int(ext.value)
        body = _walk(stmt.body)
        # threadIdx.* on PLENA has no parallel meaning (single-thread HW),
        # so collapse the binding regardless of extent — substitute the
        # var with 0 and drop the wrapper. blockIdx.* extent==1 is also a
        # degenerate (singleton) group; only blockIdx with extent>1 becomes
        # a real group.
        if binding_kind == "thread" or ext_val == 1:
            return _VarSubst({var: tir.IntImm(var.dtype, 0)}).run(body)
        return _wrap_group(var, ext_val, body)

    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body),
        )

    if isinstance(stmt, tir.For):
        new_body = _walk(stmt.body)
        if stmt.kind == tir.ForKind.PARALLEL:
            ext = stmt.extent
            if not isinstance(ext, tir.IntImm):
                raise GroupAnnotateError(
                    f"parallel for {stmt.loop_var.name!r} has non-constant "
                    f"extent {ext!r}; groups require compile-time extent"
                )
            return _wrap_group(stmt.loop_var, int(ext.value), new_body)
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            new_body, stmt.thread_binding, stmt.annotations,
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
    return stmt


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run(func: tir.PrimFunc) -> tir.PrimFunc:
    new_body = _walk(func.body)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "GroupAnnotateError", "GROUP_KEY"]
