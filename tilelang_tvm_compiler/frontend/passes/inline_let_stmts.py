"""Inline every ``tir.LetStmt`` by substituting the bound var into its body.

Why this pass exists
--------------------

When a kernel writes ``e = 2 * i`` and then references ``e`` multiple times,
TVMScript / tilelang's tracer is free to materialize the binding as a
``tir.LetStmt`` (typed ``e: T.int32 = 2 * i``). Several downstream passes
in this compiler walk the IR with hand-rolled visitors that don't enumerate
``tir.LetStmt`` (they fall through to a default ``return stmt`` branch),
which silently skips the body. Symptoms range from "BufferStore not
lowered" (lower_fp_row_patterns) to "unbound tir.Var" at isa-emit time.

Rather than teach every visitor about LetStmt, we run this pass first and
make LetStmts disappear entirely: every ``Let(var, value, body)`` is
replaced by ``substitute(body, {var: value})``. Downstream passes then
only have to handle the canonical Stmt set.

Substitution is recursive: nested LetStmts compose, and a LetStmt whose
``value`` itself references a previously-bound var has its value
substituted too.

This pass is a no-op for kernels without LetStmts.
"""

from __future__ import annotations

from typing import Dict

import tvm
from tvm import tir


class InlineLetStmtsError(RuntimeError):
    pass


def _subst_expr(expr, mapping: Dict[tir.Var, tir.PrimExpr]):
    if expr is None:
        return None
    if isinstance(expr, tir.Var):
        repl = mapping.get(expr)
        return repl if repl is not None else expr
    if isinstance(expr, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return expr
    if isinstance(expr, tir.Cast):
        return tir.Cast(expr.dtype, _subst_expr(expr.value, mapping))
    if isinstance(expr, tir.Call):
        return tir.Call(
            expr.dtype, expr.op,
            [_subst_expr(a, mapping) for a in expr.args],
        )
    if isinstance(expr, tir.BufferLoad):
        return tir.BufferLoad(
            expr.buffer,
            [_subst_expr(i, mapping) for i in expr.indices],
        )
    if isinstance(expr, tir.Select):
        return tir.Select(
            _subst_expr(expr.condition, mapping),
            _subst_expr(expr.true_value, mapping),
            _subst_expr(expr.false_value, mapping),
        )
    if isinstance(expr, tir.Ramp):
        return tir.Ramp(
            _subst_expr(expr.base, mapping),
            _subst_expr(expr.stride, mapping),
            expr.lanes,
        )
    if isinstance(expr, tir.Broadcast):
        return tir.Broadcast(_subst_expr(expr.value, mapping), expr.lanes)
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return type(expr)(
            _subst_expr(expr.a, mapping),
            _subst_expr(expr.b, mapping),
        )
    if hasattr(expr, "value"):
        # Catches tir.Not and friends.
        return type(expr)(_subst_expr(expr.value, mapping))
    return expr


def _walk(stmt, mapping: Dict[tir.Var, tir.PrimExpr]):
    if stmt is None:
        return None
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, mapping) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=[_subst_expr(v, mapping) for v in stmt.iter_values],
            predicate=_subst_expr(stmt.predicate, mapping),
            block=_walk(stmt.block, mapping),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, mapping),
            init=_walk(stmt.init, mapping) if stmt.init is not None else None,
            alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node,
            stmt.attr_key,
            _subst_expr(stmt.value, mapping),
            _walk(stmt.body, mapping),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var,
            _subst_expr(stmt.min, mapping),
            _subst_expr(stmt.extent, mapping),
            stmt.kind,
            _walk(stmt.body, mapping),
            stmt.thread_binding,
            stmt.annotations,
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            _subst_expr(stmt.condition, mapping),
            _walk(stmt.then_case, mapping),
            _walk(stmt.else_case, mapping) if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.LetStmt):
        # Substitute previously-seen vars into the new value, then bind.
        # The original LetStmt is dropped — the body is rewritten with
        # ``var -> value`` and walked.
        new_value = _subst_expr(stmt.value, mapping)
        new_mapping = dict(mapping)
        new_mapping[stmt.var] = new_value
        return _walk(stmt.body, new_mapping)
    if isinstance(stmt, tir.BufferStore):
        return tir.BufferStore(
            stmt.buffer,
            _subst_expr(stmt.value, mapping),
            [_subst_expr(i, mapping) for i in stmt.indices],
        )
    if isinstance(stmt, tir.Evaluate):
        return tir.Evaluate(_subst_expr(stmt.value, mapping))
    if isinstance(stmt, tir.Allocate):
        return tir.Allocate(
            stmt.buffer_var,
            stmt.dtype,
            [_subst_expr(e, mapping) for e in stmt.extents],
            _subst_expr(stmt.condition, mapping),
            _walk(stmt.body, mapping),
            stmt.annotations,
        )
    raise InlineLetStmtsError(
        f"unhandled stmt type {type(stmt).__name__}: {stmt!r}"
    )


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    return tir.PrimFunc(
        params=func.params,
        body=_walk(func.body, {}),
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "InlineLetStmtsError"]
