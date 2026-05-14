"""Decompose compound FPRAM ``BufferStore`` RHS into single-op stores.

Why this pass exists
--------------------

The downstream pass :mod:`lower_fp_row_patterns` only recognises *flat*
single-op assignments on FPRAM fragments::

    OUT_FP[i] = X_FP[i]                       # fp_copy_at
    OUT_FP[i] = X_FP[i] +/- /* Y_FP[i]        # fp_add/sub/mul_at
    OUT_FP[i] = T.exp(X_FP[i])                # fp_exp_at
    OUT_FP[i] = 1 / X_FP[i]                   # fp_reci_at

If a kernel writes a compound expression like::

    OUT_FP[e] = X_FP[e] * C_FP[e] + X_FP[o] * NS_FP[e]

the pattern matcher returns ``None`` and the store falls through unlowered,
producing silently-wrong ISA (the compiler emits an empty ``for`` body).

This pass walks the IR before ``scope_inference`` runs and rewrites such
compound stores into a sequence of single-op stores using auto-allocated
temporary FPRAM fragments (``__tmp_fp_<n>``)::

    __tmp_fp_0[e] = X_FP[e] * C_FP[e]
    __tmp_fp_1[e] = X_FP[o] * NS_FP[e]
    OUT_FP[e]     = __tmp_fp_0[e] + __tmp_fp_1[e]

Each generated temp matches the same shape / dtype / declared-scope
(``local.fragment``) as the original destination, so ``scope_inference``
auto-promotes them to FPRAM (rank-1 fragment used in FP scalar context),
``allocate_group_memory`` auto-expands them to ``(lane_count, ...)``, and
``lower_fp_row_patterns`` lowers each single-op store as usual.

The new buffers are appended to the *enclosing* ``tir.Block``'s
``alloc_buffers`` so they share the same scope as the user-declared
fragments. Each compound store gets its own fresh temps; address allocation
happens later (in HLIR construction) and is not lifetime-aware here, so a
deeply-nested expression may produce more temps than strictly necessary.

This pass is a no-op for stores whose RHS already fits a recognised
single-op pattern.
"""

from __future__ import annotations

from typing import List, Optional

import tvm
from tvm import tir


class LowerCompoundFpStoresError(RuntimeError):
    pass


_BINOPS = (tir.Add, tir.Sub, tir.Mul)


def _is_fragment_buffer(buf: tir.Buffer) -> bool:
    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    return declared == "local.fragment"


_PEEL_BINOPS = (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.Max, tir.Min)


def _peel_cast(expr, target_dtype: str):
    """Recursively strip TVM's fp16↔fp32 widening Casts so the whole
    subtree is rebuilt at ``target_dtype``.

    TVM lowers ``fp16_a op fp16_b`` to
    ``Cast(fp16, Cast(fp32, fp16_a) op Cast(fp32, fp16_b))`` and the
    same widening propagates through nested calls (``T.exp``,
    reciprocal, …). For decomposition we want to see the math as the
    kernel author wrote it — purely at the dst's dtype. This walker
    descends through both layers (outer narrow Cast and inner widen
    Casts) and reconstructs binops / unary calls / leaves at the target
    dtype. Anything it can't normalise is returned unchanged.

    A subtree returned by this function is invariant: it does not
    contain any Cast nodes that change dtype, all literals are at
    ``target_dtype``, and every BufferLoad already at ``target_dtype``
    is exposed as a leaf for ``_is_leaf`` to pick up.
    """

    def _rebuild(e):
        # Drop redundant Cast wrappers regardless of nesting depth.
        if isinstance(e, tir.Cast):
            return _rebuild(e.value)
        if isinstance(e, tir.IntImm):
            return tir.IntImm(target_dtype, int(e.value))
        if isinstance(e, tir.FloatImm):
            return tir.FloatImm(target_dtype, float(e.value))
        if isinstance(e, tir.BufferLoad):
            return e
        cls = type(e)
        if cls in _PEEL_BINOPS:
            return cls(_rebuild(e.a), _rebuild(e.b))
        if isinstance(e, tir.Call) and len(e.args) == 1:
            return tir.Call(target_dtype, e.op, [_rebuild(e.args[0])])
        # Unknown node — bail out by returning as-is. The caller falls
        # back to leaving the store untouched, surfacing the unknown
        # shape downstream rather than silently lowering it wrong.
        return e

    return _rebuild(expr)


def _is_leaf(expr) -> bool:
    """A leaf expression doesn't need decomposition: it can sit directly
    inside the recognised single-op pattern."""
    if isinstance(expr, (tir.BufferLoad, tir.IntImm, tir.FloatImm)):
        return True
    return False


def _is_one(expr) -> bool:
    if isinstance(expr, tir.IntImm):
        return int(expr.value) == 1
    if isinstance(expr, tir.FloatImm):
        return float(expr.value) == 1.0
    return False


def _is_reci_pattern(expr) -> Optional[tir.PrimExpr]:
    """Return the denominator of ``1 / x``, else None."""
    if isinstance(expr, tir.Div) and _is_one(expr.a):
        return expr.b
    return None


def _is_exp_call(expr) -> bool:
    return (
        isinstance(expr, tir.Call)
        and getattr(expr.op, "name", None) == "tir.exp"
        and len(expr.args) == 1
    )


def _is_already_single_op(value) -> bool:
    """True iff `value` already matches a pattern recognised by
    `lower_fp_row_patterns._try_lower_fp_store`."""
    if isinstance(value, tir.BufferLoad):
        return True
    if isinstance(value, _BINOPS):
        return _is_leaf(value.a) and _is_leaf(value.b)
    if _is_exp_call(value):
        return _is_leaf(value.args[0])
    if _is_reci_pattern(value) is not None:
        return _is_leaf(_is_reci_pattern(value))
    return False


class _Ctx:
    """Allocator + accumulator state shared across the recursive walk."""

    def __init__(self) -> None:
        self.next_id = 0
        self.new_buffers: List[tir.Buffer] = []

    def fresh_tmp(self, template: tir.Buffer) -> tir.Buffer:
        name = f"__tmp_fp_{self.next_id}"
        self.next_id += 1
        data = tir.Var(
            name,
            tvm.ir.PointerType(tvm.ir.PrimType(template.dtype), "local.fragment"),
        )
        buf = tir.decl_buffer(
            shape=list(template.shape),
            dtype=template.dtype,
            name=name,
            data=data,
            scope="local.fragment",
        )
        self.new_buffers.append(buf)
        return buf


def _to_leaf(expr, dst: tir.Buffer, indices, pre: List[tir.Stmt],
             ctx: _Ctx) -> tir.PrimExpr:
    """Ensure ``expr`` is a leaf (BufferLoad or constant); if not, evaluate
    it into a fresh fragment and return a BufferLoad of that fragment.

    ``indices`` is reused as the storage index inside the temporary — every
    auto-allocated fragment has the same shape as ``dst`` so it accepts the
    same indexing.
    """
    expr = _peel_cast(expr, str(dst.dtype))
    if _is_leaf(expr):
        return expr
    if isinstance(expr, _BINOPS):
        lhs = _to_leaf(expr.a, dst, indices, pre, ctx)
        rhs = _to_leaf(expr.b, dst, indices, pre, ctx)
        tmp = ctx.fresh_tmp(dst)
        pre.append(tir.BufferStore(tmp, type(expr)(lhs, rhs), list(indices)))
        return tir.BufferLoad(tmp, list(indices))
    if _is_exp_call(expr):
        inner = _to_leaf(expr.args[0], dst, indices, pre, ctx)
        tmp = ctx.fresh_tmp(dst)
        pre.append(tir.BufferStore(
            tmp,
            tir.Call(expr.dtype, expr.op, [inner]),
            list(indices),
        ))
        return tir.BufferLoad(tmp, list(indices))
    denom = _is_reci_pattern(expr)
    if denom is not None:
        inner = _to_leaf(denom, dst, indices, pre, ctx)
        tmp = ctx.fresh_tmp(dst)
        pre.append(tir.BufferStore(
            tmp,
            tir.Div(tir.FloatImm(expr.dtype, 1.0), inner),
            list(indices),
        ))
        return tir.BufferLoad(tmp, list(indices))
    raise LowerCompoundFpStoresError(
        f"unsupported subexpression in compound FP store RHS: "
        f"{type(expr).__name__}: {expr!r}"
    )


def _decompose_store(store: tir.BufferStore, ctx: _Ctx) -> tir.Stmt:
    if not _is_fragment_buffer(store.buffer):
        return store
    if len(store.buffer.shape) != 1:
        # FPRAM fragments are declared rank-1 by convention; anything else is
        # left to the existing passes.
        return store

    pre: List[tir.Stmt] = []
    target_dtype = str(store.buffer.dtype)
    # Peel fp16↔fp32 cast roundtrips so the dispatch below matches the
    # actual op shape regardless of TVM's widening artifacts.
    value = _peel_cast(store.value, target_dtype)

    if _is_already_single_op(value):
        # Rebuild the store so the RHS reflects the peeled form even when
        # no decomposition is required.
        if value is store.value:
            return store
        return tir.BufferStore(store.buffer, value, list(store.indices))

    if isinstance(value, _BINOPS):
        lhs = _to_leaf(value.a, store.buffer, store.indices, pre, ctx)
        rhs = _to_leaf(value.b, store.buffer, store.indices, pre, ctx)
        new_value = type(value)(lhs, rhs)
    elif _is_exp_call(value):
        inner = _to_leaf(value.args[0], store.buffer, store.indices, pre, ctx)
        new_value = tir.Call(value.dtype, value.op, [inner])
    else:
        denom = _is_reci_pattern(value)
        if denom is not None:
            inner = _to_leaf(denom, store.buffer, store.indices, pre, ctx)
            new_value = tir.Div(tir.FloatImm(value.dtype, 1.0), inner)
        else:
            # Unknown shape — leave for downstream to flag.
            return store

    final = tir.BufferStore(store.buffer, new_value, list(store.indices))
    if not pre:
        return final
    return tir.SeqStmt([*pre, final])


def _walk(stmt, ctx: _Ctx):
    if stmt is None:
        return None
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, ctx) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values,
            predicate=stmt.predicate,
            block=_walk(stmt.block, ctx),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, ctx),
            init=_walk(stmt.init, ctx) if stmt.init is not None else None,
            alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body, ctx),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, ctx),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            stmt.condition,
            _walk(stmt.then_case, ctx),
            _walk(stmt.else_case, ctx) if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.LetStmt):
        # inline_let_stmts is supposed to have removed these, but be defensive.
        return tir.LetStmt(stmt.var, stmt.value, _walk(stmt.body, ctx))
    if isinstance(stmt, tir.BufferStore):
        return _decompose_store(stmt, ctx)
    if isinstance(stmt, tir.Evaluate):
        return stmt
    if isinstance(stmt, tir.Allocate):
        return tir.Allocate(
            stmt.buffer_var, stmt.dtype, list(stmt.extents),
            stmt.condition, _walk(stmt.body, ctx), stmt.annotations,
        )
    return stmt


def _inject_alloc_buffers(stmt, new_buffers: List[tir.Buffer]):
    """Append ``new_buffers`` to the alloc_buffers of the *first* tir.Block
    we encounter (the kernel root block under T.Kernel).

    A simple top-down search is fine because there is exactly one root
    block in the kernels we lower; extending the inner scopes wouldn't help
    because every FP fragment needs to be visible across the whole kernel
    body anyway.
    """
    if not new_buffers:
        return stmt
    if isinstance(stmt, tir.SeqStmt):
        out = []
        injected = False
        for c in stmt.seq:
            if injected:
                out.append(c)
            else:
                new_c = _inject_alloc_buffers(c, new_buffers)
                if new_c is not c:
                    injected = True
                out.append(new_c)
        return tir.SeqStmt(out)
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values,
            predicate=stmt.predicate,
            block=_inject_alloc_buffers(stmt.block, new_buffers),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=stmt.body, init=stmt.init,
            alloc_buffers=list(stmt.alloc_buffers) + list(new_buffers),
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _inject_alloc_buffers(stmt.body, new_buffers),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _inject_alloc_buffers(stmt.body, new_buffers),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.LetStmt):
        return tir.LetStmt(stmt.var, stmt.value,
                           _inject_alloc_buffers(stmt.body, new_buffers))
    return stmt  # no Block found in this branch


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    ctx = _Ctx()
    new_body = _walk(func.body, ctx)
    new_body = _inject_alloc_buffers(new_body, ctx.new_buffers)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "LowerCompoundFpStoresError"]
