"""Decompose compound vector ``BufferStore`` chains on VRAM (shared)
buffers into single-op stores, each in its own loop (loop fission).

Why this pass exists
--------------------

Activation kernels (silu_min, gelu_min, …) now do their FP math *directly
on 2D ``shared`` (VRAM) buffers* instead of staging through rank-1 FPRAM
fragments. A row of the kernel body looks like::

    for row in range(rows):
        for i in T.parallel(hlen):
            Y_sh[row, i] = (0.5 * X_sh[row, i]) * (1.0 + tanh_u(...))

After ``inline_let_stmts`` the per-step Python locals (``u``, ``tanh_u``,
…) are inlined, so the store's RHS is one deeply-nested expression.

Two downstream facts force a rewrite:

  * The mid_ir ``fold`` pass only pattern-matches a *single* op per store
    (``a op b``, ``exp(a)``, ``1/a`` with leaf operands). A nested RHS
    falls through and silently lowers to an empty ``for`` body.
  * Even after decomposing the RHS into a sequence of single-op stores
    inside *one* loop body, fold flattens the sequence but the lowering
    cannot have a store read a buffer written by an *earlier store in the
    same loop body*. Each single op must sit in its own loop.

So this pass, run very early (right after ``inline_let_stmts``, before the
rank-1 ``lower_compound_fp_stores`` and ``hoist_float_constants``):

  1. Finds an innermost ``for`` whose body is a single compound BufferStore
     to a non-fragment (VRAM) buffer.
  2. Decomposes the RHS into single-op stores, introducing 2D ``shared``
     temporaries shaped/dtyped like the destination.
  3. Emits each single-op store wrapped in its *own* clone of the loop
     (loop fission), preserving the loop var / extent / kind.
  4. Reuses temporary slots by liveness (linear-scan): a temp whose last
     read precedes another temp's first write shares the same buffer, so
     the live-set size — not the chain length — bounds slot count.

The rank-1 FPRAM path (``lower_compound_fp_stores``) is untouched: this
pass is a no-op for stores to rank-1 ``local.fragment`` buffers and for
stores whose RHS is already a single op.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import tir


class FissionVectorChainsError(RuntimeError):
    pass


_BINOPS = (tir.Add, tir.Sub, tir.Mul)
_PEEL_BINOPS = (tir.Add, tir.Sub, tir.Mul, tir.Div, tir.Max, tir.Min)


# --------------------------------------------------------------------------
# scope / shape predicates
# --------------------------------------------------------------------------
def _scope(buf: tir.Buffer) -> str:
    return buf.scope() if callable(getattr(buf, "scope", None)) else "global"


def _is_fragment_buffer(buf: tir.Buffer) -> bool:
    return _scope(buf) == "local.fragment"


def _is_vram_store_target(buf: tir.Buffer) -> bool:
    """True for the multi-dim VRAM (shared) buffers this pass rewrites.

    Rank-1 ``local.fragment`` (FPRAM scalar) stores are left to
    ``lower_compound_fp_stores``; rank-1 here would also be ambiguous, so
    we only claim rank>=2 shared/global destinations.
    """
    if _is_fragment_buffer(buf):
        return False
    return len(buf.shape) >= 2


# --------------------------------------------------------------------------
# cast peeling — mirror lower_compound_fp_stores so we see author-level math
# --------------------------------------------------------------------------
def _peel_cast(expr, target_dtype: str):
    def _rebuild(e):
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
        return e

    return _rebuild(expr)


def _is_leaf(expr) -> bool:
    return isinstance(expr, (tir.BufferLoad, tir.IntImm, tir.FloatImm))


def _is_one(expr) -> bool:
    if isinstance(expr, tir.IntImm):
        return int(expr.value) == 1
    if isinstance(expr, tir.FloatImm):
        return float(expr.value) == 1.0
    return False


def _reci_denom(expr) -> Optional[tir.PrimExpr]:
    if isinstance(expr, tir.Div) and _is_one(expr.a):
        return expr.b
    return None


def _is_exp_call(expr) -> bool:
    return (
        isinstance(expr, tir.Call)
        and getattr(expr.op, "name", None) == "tir.exp"
        and len(expr.args) == 1
    )


def _is_single_op(value) -> bool:
    if isinstance(value, tir.BufferLoad):
        return True
    if isinstance(value, _BINOPS):
        return _is_leaf(value.a) and _is_leaf(value.b)
    if _is_exp_call(value):
        return _is_leaf(value.args[0])
    if _reci_denom(value) is not None:
        return _is_leaf(_reci_denom(value))
    return False


# --------------------------------------------------------------------------
# temp allocator (logical temps; physical slots assigned later by liveness)
# --------------------------------------------------------------------------
class _Ctx:
    """Accumulates the flat single-op store list for one compound store and
    the logical temps it introduces. Physical slot reuse happens afterward
    in ``_assign_slots`` over the whole list."""

    def __init__(self, template: tir.Buffer) -> None:
        self.template = template          # the VRAM dst — temps mirror it
        self.next_id = 0
        # logical-temp name -> tir.Buffer (one decl per logical temp)
        self.temps: Dict[str, tir.Buffer] = {}
        # ordered flat list of (dst_buffer, value, indices) single-op stores
        self.ops: List[Tuple[tir.Buffer, tir.PrimExpr, List[tir.PrimExpr]]] = []

    def fresh_tmp(self) -> tir.Buffer:
        name = f"__tmp_vec_{self.next_id}"
        self.next_id += 1
        t = self.template
        data = tir.Var(
            name, tvm.ir.PointerType(tvm.ir.PrimType(t.dtype), _scope(t))
        )
        buf = tir.decl_buffer(
            shape=list(t.shape), dtype=t.dtype, name=name, data=data,
            scope=_scope(t),
        )
        self.temps[name] = buf
        return buf

    def emit(self, dst: tir.Buffer, value, indices) -> None:
        self.ops.append((dst, value, list(indices)))

    def emit_copy(self, dst: tir.Buffer, src_load: tir.BufferLoad,
                  indices) -> None:
        """Emit ``dst = src`` as a single-op copy store (BufferLoad RHS)."""
        self.ops.append((dst, src_load, list(indices)))


# Ops that lower to a *masked per-row* HW op (V_EXP_V / V_RECI_V /
# V_*_VF) when row_footprint==1. Their emulator semantics overwrite the
# unselected lane-heads of ``dst`` with ``src1`` — only harmless when
# dst == src1 (in-place). So whenever we emit one of these with a buffer
# source distinct from the dst temp, we first copy src -> dst and rewrite
# the op to read dst, making it in-place.
#
# Pure two-buffer binops (V_*_VV) and the const-only case don't take this
# path, so they are emitted directly.
def _emit_unary_inplace(ctx: _Ctx, dst: tir.Buffer, make_value, inner_leaf,
                        indices) -> None:
    """Emit a per-row unary (exp/reci) as ``dst = inner; dst = f(dst)``
    when ``inner_leaf`` is a BufferLoad of a different buffer; otherwise
    just ``dst = f(inner)``.

    ``make_value(src_load)`` builds the op's RHS from a leaf src."""
    if isinstance(inner_leaf, tir.BufferLoad) and inner_leaf.buffer is not dst:
        ctx.emit_copy(dst, inner_leaf, indices)
        inner_leaf = tir.BufferLoad(dst, list(indices))
    ctx.emit(dst, make_value(inner_leaf), indices)


def _emit_fp_binop_inplace(ctx: _Ctx, dst: tir.Buffer, op_cls,
                           buf_leaf, const_leaf, buf_first, indices) -> None:
    """Emit a one-buffer-one-const binop (lowers to V_*_VF, masked) as
    ``dst = buf; dst = dst <op> const`` so it runs in-place.

    Operand-order subtlety for Sub: the V_SUB_VF lowering is direction-
    sensitive (``src - f1``) but the broadcast-elementwise lowering drops
    the operand order, so a ``const - buf`` (``buf_first=False``) would be
    miscompiled as ``buf - const``. Rewrite ``c - buf`` into the
    order-free equivalent ``buf * (-1) + c`` (mul and add are
    commutative on the V_*_VF path). ``buf - c`` (buf_first) maps cleanly
    to ``src - f1`` and is left as a Sub."""
    if isinstance(buf_leaf, tir.BufferLoad) and buf_leaf.buffer is not dst:
        ctx.emit_copy(dst, buf_leaf, indices)
        buf_leaf = tir.BufferLoad(dst, list(indices))

    if op_cls is tir.Sub and not buf_first:
        # c - buf  ->  (buf * -1) + c
        dtype = str(ctx.template.dtype)
        neg_one = tir.FloatImm(dtype, -1.0)
        ctx.emit(dst, tir.Mul(buf_leaf, neg_one), indices)
        ctx.emit(dst, tir.Add(tir.BufferLoad(dst, list(indices)), const_leaf),
                 indices)
        return

    a, b = (buf_leaf, const_leaf) if buf_first else (const_leaf, buf_leaf)
    ctx.emit(dst, op_cls(a, b), indices)


def _split_binop_operands(lhs, rhs):
    """If exactly one of (lhs, rhs) is a BufferLoad and the other a const
    (FloatImm/IntImm), return (buf_leaf, const_leaf, buf_first); else None.

    This is the V_*_VF (masked fp-scalar) shape. Two BufferLoads -> V_*_VV
    (unmasked, no in-place needed); two consts -> shouldn't occur."""
    l_buf = isinstance(lhs, tir.BufferLoad)
    r_buf = isinstance(rhs, tir.BufferLoad)
    l_const = isinstance(lhs, (tir.FloatImm, tir.IntImm))
    r_const = isinstance(rhs, (tir.FloatImm, tir.IntImm))
    if l_buf and r_const:
        return lhs, rhs, True
    if r_buf and l_const:
        return rhs, lhs, False
    return None


def _to_leaf(expr, indices, ctx: _Ctx) -> tir.PrimExpr:
    """Reduce ``expr`` to a leaf (BufferLoad/const), emitting single-op
    stores into fresh temps for every non-leaf subexpression. Returns a
    BufferLoad of the temp holding the result (or the leaf unchanged)."""
    expr = _peel_cast(expr, str(ctx.template.dtype))
    if _is_leaf(expr):
        return expr
    if isinstance(expr, _BINOPS):
        lhs = _to_leaf(expr.a, indices, ctx)
        rhs = _to_leaf(expr.b, indices, ctx)
        tmp = ctx.fresh_tmp()
        split = _split_binop_operands(lhs, rhs)
        if split is not None:
            buf_leaf, const_leaf, buf_first = split
            _emit_fp_binop_inplace(ctx, tmp, type(expr), buf_leaf,
                                   const_leaf, buf_first, indices)
        else:
            ctx.emit(tmp, type(expr)(lhs, rhs), indices)
        return tir.BufferLoad(tmp, list(indices))
    if _is_exp_call(expr):
        inner = _to_leaf(expr.args[0], indices, ctx)
        tmp = ctx.fresh_tmp()
        _emit_unary_inplace(
            ctx, tmp,
            lambda s: tir.Call(expr.dtype, expr.op, [s]),
            inner, indices,
        )
        return tir.BufferLoad(tmp, list(indices))
    denom = _reci_denom(expr)
    if denom is not None:
        inner = _to_leaf(denom, indices, ctx)
        tmp = ctx.fresh_tmp()
        _emit_unary_inplace(
            ctx, tmp,
            lambda s: tir.Div(tir.FloatImm(expr.dtype, 1.0), s),
            inner, indices,
        )
        return tir.BufferLoad(tmp, list(indices))
    raise FissionVectorChainsError(
        f"unsupported subexpression in compound vector store RHS: "
        f"{type(expr).__name__}: {expr!r}"
    )


def _decompose(store: tir.BufferStore, ctx: _Ctx) -> None:
    """Populate ctx.ops with the flat single-op store list whose final
    entry writes ``store.buffer``. The final store reuses the original dst
    (not a temp)."""
    value = _peel_cast(store.value, str(store.buffer.dtype))
    dst = store.buffer
    idx = store.indices
    if _is_single_op(value):
        # A bare single op may still be a masked per-row op with a
        # distinct buffer src (e.g. final ``Y_sh[r,i] = exp(tmp[r,i])``);
        # route it through the in-place helpers too.
        if isinstance(value, tir.BufferLoad):
            ctx.emit(dst, value, idx)            # copy — V_*_VV-style, fine
            return
        if isinstance(value, _BINOPS):
            split = _split_binop_operands(value.a, value.b)
            if split is not None:
                buf_leaf, const_leaf, buf_first = split
                _emit_fp_binop_inplace(ctx, dst, type(value), buf_leaf,
                                       const_leaf, buf_first, idx)
            else:
                ctx.emit(dst, value, idx)
            return
        if _is_exp_call(value):
            _emit_unary_inplace(
                ctx, dst, lambda s: tir.Call(value.dtype, value.op, [s]),
                value.args[0], idx)
            return
        denom = _reci_denom(value)
        if denom is not None:
            _emit_unary_inplace(
                ctx, dst, lambda s: tir.Div(tir.FloatImm(value.dtype, 1.0), s),
                denom, idx)
            return
        ctx.emit(dst, value, idx)
        return
    if isinstance(value, _BINOPS):
        lhs = _to_leaf(value.a, idx, ctx)
        rhs = _to_leaf(value.b, idx, ctx)
        split = _split_binop_operands(lhs, rhs)
        if split is not None:
            buf_leaf, const_leaf, buf_first = split
            _emit_fp_binop_inplace(ctx, dst, type(value), buf_leaf,
                                   const_leaf, buf_first, idx)
        else:
            ctx.emit(dst, type(value)(lhs, rhs), idx)
    elif _is_exp_call(value):
        inner = _to_leaf(value.args[0], idx, ctx)
        _emit_unary_inplace(
            ctx, dst, lambda s: tir.Call(value.dtype, value.op, [s]),
            inner, idx)
    else:
        denom = _reci_denom(value)
        if denom is None:
            # Unknown shape; leave the store untouched.
            ctx.emit(dst, store.value, idx)
            return
        inner = _to_leaf(denom, idx, ctx)
        _emit_unary_inplace(
            ctx, dst, lambda s: tir.Div(tir.FloatImm(value.dtype, 1.0), s),
            inner, idx)


# --------------------------------------------------------------------------
# liveness-based slot reuse (linear scan over the flat op list)
# --------------------------------------------------------------------------
def _loads_of(value) -> List[tir.Buffer]:
    """Buffers read by ``value`` (only temp buffers matter to the caller)."""
    found: List[tir.Buffer] = []

    def _v(e):
        if isinstance(e, tir.BufferLoad):
            found.append(e.buffer)
            for idx in e.indices:
                _v(idx)
            return
        if isinstance(e, tir.Cast):
            _v(e.value)
            return
        if isinstance(e, tir.Call):
            for a in e.args:
                _v(a)
            return
        if hasattr(e, "a") and hasattr(e, "b"):
            _v(e.a)
            _v(e.b)
            return
        if hasattr(e, "value"):
            _v(e.value)

    _v(value)
    return found


def _assign_slots(
    ops: List[Tuple[tir.Buffer, tir.PrimExpr, List[tir.PrimExpr]]],
    temps: Dict[str, tir.Buffer],
    template: tir.Buffer,
    slot_factory,
) -> Tuple[
    List[Tuple[tir.Buffer, tir.PrimExpr, List[tir.PrimExpr]]],
    List[tir.Buffer],
]:
    """Map each logical temp to a physical slot buffer by liveness.

    A temp's live range is [first write index, last read index] over the
    flat op list. Two temps whose ranges don't overlap share one slot.
    Returns the rewritten op list (temp refs replaced by their slot) and
    the list of physical slot buffers actually used.
    """
    temp_names = set(temps.keys())

    # first-def and last-use index per temp name.
    first_def: Dict[str, int] = {}
    last_use: Dict[str, int] = {}
    for k, (dst, value, _idx) in enumerate(ops):
        if dst.name in temp_names and dst.name not in first_def:
            first_def[dst.name] = k
        for rb in _loads_of(value):
            if rb.name in temp_names:
                last_use[rb.name] = k
    # a temp written but never read still needs a slot for that one step.
    for name in temp_names:
        last_use.setdefault(name, first_def.get(name, 0))

    # linear scan: walk ops in order, free slots whose temp's last_use has
    # passed, allocate a slot at each temp's first_def.
    free: List[tir.Buffer] = []
    slots: List[tir.Buffer] = []
    name_to_slot: Dict[str, tir.Buffer] = {}
    # temps freed *after* processing step k (their last_use == k).
    free_after: Dict[int, List[str]] = {}
    for name, k in last_use.items():
        free_after.setdefault(k, []).append(name)

    for k, (dst, _value, _idx) in enumerate(ops):
        if dst.name in temp_names and dst.name not in name_to_slot:
            if free:
                slot = free.pop()
            else:
                slot = slot_factory(len(slots))
                slots.append(slot)
            name_to_slot[dst.name] = slot
        # free at end of this step (a temp read here is dead afterwards;
        # the dst written here keeps its slot until its own last_use).
        for name in free_after.get(k, []):
            slot = name_to_slot.get(name)
            if slot is not None and slot not in free:
                free.append(slot)

    def _remap_buf(buf: tir.Buffer) -> tir.Buffer:
        return name_to_slot.get(buf.name, buf)

    def _remap_expr(e):
        if isinstance(e, tir.BufferLoad):
            return tir.BufferLoad(_remap_buf(e.buffer),
                                  [_remap_expr(i) for i in e.indices])
        if isinstance(e, tir.Cast):
            return tir.Cast(e.dtype, _remap_expr(e.value))
        if isinstance(e, tir.Call):
            return tir.Call(e.dtype, e.op, [_remap_expr(a) for a in e.args])
        if isinstance(e, tir.Div):
            return tir.Div(_remap_expr(e.a), _remap_expr(e.b))
        if hasattr(e, "a") and hasattr(e, "b"):
            return type(e)(_remap_expr(e.a), _remap_expr(e.b))
        return e

    new_ops = []
    for dst, value, idx in ops:
        new_ops.append((
            _remap_buf(dst),
            _remap_expr(value),
            [_remap_expr(i) for i in idx],
        ))
    return new_ops, slots


# --------------------------------------------------------------------------
# loop fission
# --------------------------------------------------------------------------
def _peel_loop_nest(for_stmt: tir.For):
    """Walk down a chain of nested ``tir.For``s (e.g. ``for row: for i:``)
    until the innermost statement. Return (loop_layers, innermost_stmt),
    where loop_layers is the outer→inner list of For nodes. Stops at the
    first non-For body."""
    layers: List[tir.For] = []
    cur: tir.Stmt = for_stmt
    while isinstance(cur, tir.For):
        layers.append(cur)
        cur = cur.body
    return layers, cur


def _subst_expr_vars(expr, var_map: Dict[tir.Var, tir.Var]):
    """Recursively replace ``tir.Var`` leaves per ``var_map``. Covers the
    expression shapes produced by this pass (BufferLoad indices, binops,
    unary calls, casts, reci div, imms)."""
    if expr is None:
        return None
    if isinstance(expr, tir.Var):
        return var_map.get(expr, expr)
    if isinstance(expr, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return expr
    if isinstance(expr, tir.Cast):
        return tir.Cast(expr.dtype, _subst_expr_vars(expr.value, var_map))
    if isinstance(expr, tir.BufferLoad):
        return tir.BufferLoad(
            expr.buffer,
            [_subst_expr_vars(i, var_map) for i in expr.indices],
        )
    if isinstance(expr, tir.Call):
        return tir.Call(
            expr.dtype, expr.op,
            [_subst_expr_vars(a, var_map) for a in expr.args],
        )
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return type(expr)(_subst_expr_vars(expr.a, var_map),
                          _subst_expr_vars(expr.b, var_map))
    if hasattr(expr, "value"):
        return type(expr)(_subst_expr_vars(expr.value, var_map))
    return expr


def _subst_stmt_vars(stmt: tir.Stmt, var_map: Dict[tir.Var, tir.Var]) -> tir.Stmt:
    """Substitute loop vars inside a fissioned single-op BufferStore."""
    if isinstance(stmt, tir.BufferStore):
        return tir.BufferStore(
            stmt.buffer,
            _subst_expr_vars(stmt.value, var_map),
            [_subst_expr_vars(i, var_map) for i in stmt.indices],
        )
    return stmt


def _wrap_in_nest(layers: List[tir.For], inner: tir.Stmt) -> tir.Stmt:
    """Re-wrap ``inner`` in clones of every loop in ``layers``
    (outer→inner order). Each clone gets a FRESH loop var, and ``inner``
    is rewritten to reference the fresh vars.

    A fresh var per fissioned copy is required: every fissioned op gets
    its own ``for row: for i:`` nest, and a single ``tir.Var`` may only be
    bound by one ``For``. Reusing the original loop vars across all copies
    leaves every var bound by multiple loops, so downstream scope-binding
    (pre_isa_to_mir) sees the var as unbound in all but one — surfacing as
    ``unbound tir.Var 'row'``."""
    # Map each original loop var -> a fresh var, substitute into the body.
    var_map = {f.loop_var: tir.Var(f.loop_var.name, f.loop_var.dtype)
               for f in layers}
    inner = _subst_stmt_vars(inner, var_map)
    stmt = inner
    for f in reversed(layers):
        stmt = tir.For(
            var_map[f.loop_var], f.min, f.extent, f.kind,
            stmt, f.thread_binding, f.annotations,
        )
    return stmt


def _fission_for(for_stmt: tir.For) -> Tuple[Optional[tir.Stmt], List[tir.Buffer]]:
    """If ``for_stmt`` is the OUTERMOST loop of a nest (``for row: for i:
    ... store``) wrapping a single compound VRAM BufferStore, decompose +
    fission it. Each single-op store gets its OWN full clone of the entire
    loop nest:

        for row: for i: a = b              for row: for i: a = b
        for row: for i: a = a + c     ->   for row: for i: a = a + c

    Returns (new_stmt | None, new_slot_buffers); (None, []) when not a
    fission target."""
    layers, inner = _peel_loop_nest(for_stmt)
    if not isinstance(inner, tir.BufferStore):
        return None, []
    store = inner
    if not _is_vram_store_target(store.buffer):
        return None, []

    value = _peel_cast(store.value, str(store.buffer.dtype))
    if _is_single_op(value):
        return None, []  # already legal, nothing to fission

    ctx = _Ctx(store.buffer)
    _decompose(store, ctx)
    if len(ctx.ops) <= 1:
        return None, []

    new_ops, slots = _assign_slots(
        ctx.ops, ctx.temps, store.buffer,
        slot_factory=lambda _n: ctx.fresh_tmp(),
    )

    # Each single-op store gets its own clone of the WHOLE loop nest
    # (every layer: the serial for-row AND the parallel/unroll for-i).
    loops: List[tir.Stmt] = []
    for dst, val, idx in new_ops:
        single = tir.BufferStore(dst, val, idx)
        loops.append(_wrap_in_nest(layers, single))
    return tir.SeqStmt(loops), slots


# --------------------------------------------------------------------------
# multi-statement serial-for fission
# --------------------------------------------------------------------------
def _seq_items(stmt: tir.Stmt) -> List[tir.Stmt]:
    return list(stmt.seq) if isinstance(stmt, tir.SeqStmt) else [stmt]


def _has_parallel_subloop(items: List[tir.Stmt]) -> bool:
    return any(isinstance(s, tir.For) and s.kind == tir.ForKind.PARALLEL
               for s in items)


def _split_multistmt_for(for_stmt: tir.For) -> Optional[tir.Stmt]:
    """Split a serial ``for row`` whose body is a multi-statement SeqStmt
    containing at least one ``T.Parallel`` sub-loop into one ``for row``
    per statement (loop fission), preserving order:

        for row: { for col: SQ=X*X        for row: for col: SQ=X*X
                   VAR_SUM[row] = 0 }  ->  for row: VAR_SUM[row] = 0

    Why: when an integer-tile (buf*buf) parallel sub-loop shares a for-row
    body with a per-row scalar statement, fold can't absorb the outer
    for-row (body isn't a bare single For), so the whole-tile op gets
    re-issued once per row. Splitting lets the whole-tile statement own a
    single-For body (fold absorbs the outer loop) while the per-row
    statement keeps its own for-row.

    Each split copy gets a fresh loop var (a tir.Var may be bound by only
    one For). Returns None when not a split target. Same-row-only deps in
    these kernels make order-preserving fission safe.
    """
    if for_stmt.kind != tir.ForKind.SERIAL:
        return None
    items = _seq_items(for_stmt.body)
    if len(items) <= 1:
        return None
    if not _has_parallel_subloop(items):
        return None
    out: List[tir.Stmt] = []
    for it in items:
        fresh = tir.Var(for_stmt.loop_var.name, for_stmt.loop_var.dtype)
        var_map = {for_stmt.loop_var: fresh}
        body = _subst_any_stmt_vars(it, var_map)
        out.append(tir.For(
            fresh, for_stmt.min, for_stmt.extent, for_stmt.kind,
            body, for_stmt.thread_binding, for_stmt.annotations,
        ))
    return tir.SeqStmt(out)


def _subst_any_stmt_vars(stmt, var_map: Dict[tir.Var, tir.Var]):
    """Substitute loop vars through an arbitrary stmt subtree (For /
    SeqStmt / BufferStore), used when cloning a for-row body during
    multi-statement fission."""
    if stmt is None:
        return None
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, _subst_expr_vars(stmt.min, var_map),
            _subst_expr_vars(stmt.extent, var_map), stmt.kind,
            _subst_any_stmt_vars(stmt.body, var_map),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_subst_any_stmt_vars(c, var_map) for c in stmt.seq])
    if isinstance(stmt, tir.BufferStore):
        return tir.BufferStore(
            stmt.buffer,
            _subst_expr_vars(stmt.value, var_map),
            [_subst_expr_vars(i, var_map) for i in stmt.indices],
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            _subst_expr_vars(stmt.condition, var_map),
            _subst_any_stmt_vars(stmt.then_case, var_map),
            _subst_any_stmt_vars(stmt.else_case, var_map)
            if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.Evaluate):
        return tir.Evaluate(_subst_expr_vars(stmt.value, var_map))
    return stmt


# --------------------------------------------------------------------------
# top-level walk
# --------------------------------------------------------------------------
def _walk(stmt, new_buffers: List[tir.Buffer]):
    if stmt is None:
        return None
    if isinstance(stmt, tir.For):
        # First: split a multi-statement serial for-row body so each
        # statement owns its own for-row (lets fold absorb whole-tile
        # ops, keeps per-row ops on their own loop). Re-walk the result.
        split = _split_multistmt_for(stmt)
        if split is not None:
            return _walk(split, new_buffers)
        fissioned, slots = _fission_for(stmt)
        if fissioned is not None:
            new_buffers.extend(slots)
            return fissioned
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, new_buffers),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([_walk(c, new_buffers) for c in stmt.seq])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values,
            predicate=stmt.predicate,
            block=_walk(stmt.block, new_buffers),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, new_buffers),
            init=_walk(stmt.init, new_buffers) if stmt.init is not None else None,
            alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _walk(stmt.body, new_buffers),
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            stmt.condition,
            _walk(stmt.then_case, new_buffers),
            _walk(stmt.else_case, new_buffers) if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.LetStmt):
        return tir.LetStmt(stmt.var, stmt.value, _walk(stmt.body, new_buffers))
    if isinstance(stmt, tir.Allocate):
        return tir.Allocate(
            stmt.buffer_var, stmt.dtype, list(stmt.extents),
            stmt.condition, _walk(stmt.body, new_buffers), stmt.annotations,
        )
    # BufferStore / Evaluate / leaves: returned unchanged.
    return stmt


def _inject_alloc_buffers(stmt, new_buffers: List[tir.Buffer]):
    """Append ``new_buffers`` to the first tir.Block's alloc_buffers (the
    kernel root block under T.Kernel). Mirrors lower_compound_fp_stores."""
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
                injected = new_c is not c
                out.append(new_c)
        return tir.SeqStmt(out)
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_inject_alloc_buffers(stmt.block, new_buffers),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=stmt.body, init=stmt.init,
            alloc_buffers=list(stmt.alloc_buffers) + list(new_buffers),
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
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
    return stmt


def run(func: tir.PrimFunc) -> tir.PrimFunc:
    new_buffers: List[tir.Buffer] = []
    new_body = _walk(func.body, new_buffers)
    new_body = _inject_alloc_buffers(new_body, new_buffers)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "FissionVectorChainsError"]
