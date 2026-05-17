"""Hoist FP literals out of kernel bodies into ``global.fpram`` buffers.

Why this pass exists
--------------------

A kernel author writes ``a[i,j] = a[i,j] * T.float16(0.0884)``. The
literal is a per-issue scalar; the compiler has no FP-immediate slot in
its scalar-vector ops, so historically the kernel had to declare a
dedicated ``alloc_fragment`` scalar (``SCALE``), wire up a testbench
preload to write the value, and reference ``SCALE[0]``. That's a lot
of boilerplate for one number.

This pass automates that boilerplate. It scans the PrimFunc body, finds
every ``tir.FloatImm`` that appears as a value (RHS of a BufferStore or
inside a Call argument — *not* loop extents / buffer indices, which are
ints anyway), and:

  1. De-duplicates by ``(dtype, value)``.
  2. Synthesises one ``alloc_shared(scope="global.fpram")`` buffer per
     unique entry, shape ``(1,)``, name ``__const_<dtype>_<safevalue>``.
  3. Adds the new buffers to the kernel's ``tilelang_root`` block's
     ``alloc_buffers`` list — i.e. exactly what the author would've
     written by hand.
  4. Rewrites every targeted FloatImm into ``BufferLoad(synth, [0])``.
  5. Stamps ``{"plena.hoisted_constants": {name: value}}`` onto
     ``PrimFunc.attrs`` so to_plena → HLIR can copy the values onto
     ``hlir.Buffer.constant_value`` for the buffer-addrs JSON dump.

Downstream passes (fold, mid_ir, address_alloc, isa_emit) see a plain
``global.fpram`` scalar buffer and lower it normally — exactly the path
the hand-written ``SCALE`` / ``M_INIT`` / ``L_INIT`` buffers in
``flash_decode_min`` already exercise. No other pass needs to know
about hoisting.

What's NOT hoisted
------------------

  * ``T.float16(0)`` as a sole RHS — fold.py already lowers this to a
    multi-lane vector fill (``plena.zero_v``) which is far cheaper than
    an FPRAM round-trip. Hoisting it would be a regression.
  * FloatImms inside a ``T.float16(1.0) / x`` pattern that fold.py
    recognises as ``UnaryOp.RECI`` — the literal is consumed by the
    operator, not stored anywhere.
  * Anything in loop extents / buffer indices (those are IntImm).

The pass keeps a deny-list keyed on parent-expression shape so these
already-handled cases continue to flow through unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import tir


# Attribute key on the rewritten PrimFunc; to_plena reads it to populate
# ``hlir.Buffer.constant_value``.
ATTR_KEY = "plena.hoisted_constants"


def _safe_value_token(value: float) -> str:
    """Render a float so it round-trips into a Python identifier piece.

    Examples: ``0.0884`` → ``0p0884``, ``-10000.0`` → ``neg10000``,
    ``1.5e-3`` → ``0p0015``. We avoid scientific notation in the name
    because ``e`` is ambiguous (could be misread as a hex digit); the
    name is only used for diagnostics anyway.
    """
    if value == int(value):
        s = str(int(value))
    else:
        s = f"{value:.6g}"
    if s.startswith("-"):
        return "neg" + s[1:].replace(".", "p")
    return s.replace(".", "p")


def _dtype_token(dtype: str) -> str:
    """``float16`` → ``f16``, ``float32`` → ``f32``."""
    if dtype.startswith("float"):
        return "f" + dtype[len("float"):]
    return dtype


class _ConstantTable:
    """Dedup by ``(dtype, value)``. Synthesises one ``tir.Buffer`` per
    unique entry and remembers everything needed by the downstream
    rewrite + the to_plena attr handoff."""

    def __init__(self) -> None:
        # (dtype, value) → (buffer, name)
        self._table: Dict[Tuple[str, float], Tuple[tir.Buffer, str]] = {}
        # Insertion order, so generated ASM is reproducible across runs.
        self._order: List[Tuple[str, float]] = []

    def get_or_create(self, dtype: str, value: float) -> tir.Buffer:
        key = (dtype, float(value))
        existing = self._table.get(key)
        if existing is not None:
            return existing[0]
        name = f"__const_{_dtype_token(dtype)}_{_safe_value_token(value)}"
        # Disambiguate if two distinct values somehow rendered to the
        # same name (e.g. 0.000001 vs 0.0000011 both → "1e-06"-ish).
        # Append a counter until unique.
        existing_names = {n for _, n in self._table.values()}
        unique = name
        suffix = 1
        while unique in existing_names:
            suffix += 1
            unique = f"{name}__{suffix}"
        # 0-D scalar buffer. The semantically correct shape — a hoisted
        # constant is a single value, broadcast to whatever rank the
        # consumer needs. fold.py's existing broadcast-prefix matcher
        # handles ``len(src_idx)=0 < len(dst_indices)`` by wrapping the
        # load in a ``Broadcast(broadcast_dims=[0..n-1])`` automatically,
        # so we don't need a special case. address_alloc walks
        # ``num_elements`` which is 1 for any shape with empty tuple
        # (product of empty seq = 1) — single slot, exactly what we
        # want.
        buf = tir.decl_buffer(
            shape=(), dtype=dtype, name=unique, scope="global.fpram",
        )
        self._table[key] = (buf, unique)
        self._order.append(key)
        return buf

    def alloc_buffers(self) -> List[tir.Buffer]:
        return [self._table[k][0] for k in self._order]

    def name_value_map(self) -> Dict[str, float]:
        return {self._table[k][1]: k[1] for k in self._order}


def _is_hoistable_dtype(dtype: str) -> bool:
    """We only hoist fp16 / fp32 / bf16 literals — the only types the
    FPRAM can store. Other dtypes pass through unchanged.
    """
    return dtype in ("float16", "float32", "bfloat16")


# Expressions we deliberately skip — these patterns are absorbed by
# downstream passes more efficiently than an FPRAM round-trip would be.
def _is_skip_expr(expr) -> bool:
    """``T.float16(0)`` as the entire RHS lowers to a multi-lane vector
    fill in fold.py — far cheaper than going through FPRAM. Likewise a
    ``T.float16(1.0) / x`` reciprocal is folded into a unary op. We
    only test the top-level expr; nested cases (0 deep inside a binop)
    still get hoisted, but those don't actually occur in real kernels.
    """
    if isinstance(expr, tir.FloatImm) and float(expr.value) == 0.0:
        return True
    if isinstance(expr, tir.Div):
        a = expr.a
        # Peel TVM's fp16↔fp32 cast roundtrip — matches fold.py:822.
        if isinstance(a, tir.Cast):
            a = a.value
        if isinstance(a, tir.FloatImm) and float(a.value) == 1.0:
            return True
    return False


def _rewrite_expr(expr, table: _ConstantTable, skip_top: bool,
                  broadcast_index=None):
    """Recursively rewrite FloatImms inside ``expr``.

    ``skip_top`` policy:
      * If ``expr`` is a ``FloatImm(0.0)`` and ``skip_top`` is True, we
        return it unchanged — fold.py recognises this as a zero-fill
        store and lowers it to ``plena.zero_v`` (cheaper than going
        through FPRAM).
      * If ``expr`` is ``Div(FloatImm(1.0), x)`` and ``skip_top`` is
        True, we leave the literal ``1.0`` alone (fold.py absorbs it
        into ``UnaryOp.RECI``) but still recurse into ``x`` — a
        denominator can carry its own hoistable literals.
      * Everything else: recurse as normal with ``skip_top=False``.

    ``broadcast_index`` is the leading index expression of the
    enclosing BufferStore's dst — see :func:`_make_synth_load`.
    """
    if expr is None:
        return None
    if skip_top:
        # Zero-fill: leave the literal alone (fold turns it into
        # plena.zero_v with no FPRAM round-trip).
        if isinstance(expr, tir.FloatImm) and float(expr.value) == 0.0:
            return expr
        # Reciprocal: leave the leading 1.0 alone, recurse into the
        # denominator. ``a`` might be wrapped in a Cast(fp32/fp16);
        # peel one layer for the check, but pass the original ``expr.a``
        # through unchanged so we don't accidentally rewrite the dtype.
        if isinstance(expr, tir.Div):
            a = expr.a.value if isinstance(expr.a, tir.Cast) else expr.a
            if isinstance(a, tir.FloatImm) and float(a.value) == 1.0:
                return type(expr)(
                    expr.a,
                    _rewrite_expr(expr.b, table, False, broadcast_index),
                )
        # Any other top-level shape: fall through to the normal walk.
    if isinstance(expr, tir.FloatImm):
        if not _is_hoistable_dtype(str(expr.dtype)):
            return expr
        buf = table.get_or_create(str(expr.dtype), float(expr.value))
        return _make_synth_load(buf, broadcast_index)
    if isinstance(expr, tir.Cast):
        return tir.Cast(
            expr.dtype, _rewrite_expr(expr.value, table, False, broadcast_index),
        )
    if isinstance(expr, tir.Call):
        return tir.Call(
            expr.dtype, expr.op,
            [_rewrite_expr(a, table, False, broadcast_index) for a in expr.args],
        )
    if isinstance(expr, tir.BufferLoad):
        return tir.BufferLoad(
            expr.buffer,
            [_rewrite_expr(i, table, False, broadcast_index) for i in expr.indices],
        )
    if isinstance(expr, tir.Select):
        return tir.Select(
            _rewrite_expr(expr.condition, table, False, broadcast_index),
            _rewrite_expr(expr.true_value, table, False, broadcast_index),
            _rewrite_expr(expr.false_value, table, False, broadcast_index),
        )
    if isinstance(expr, tir.Ramp):
        return tir.Ramp(
            _rewrite_expr(expr.base, table, False, broadcast_index),
            _rewrite_expr(expr.stride, table, False, broadcast_index),
            expr.lanes,
        )
    if isinstance(expr, tir.Broadcast):
        return tir.Broadcast(
            _rewrite_expr(expr.value, table, False, broadcast_index),
            expr.lanes,
        )
    # Generic binary op (Add/Sub/Mul/Div/Max/Min/...).
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return type(expr)(
            _rewrite_expr(expr.a, table, False, broadcast_index),
            _rewrite_expr(expr.b, table, False, broadcast_index),
        )
    # Pass-through for IntImm / Var / StringImm / etc.
    return expr


def _make_synth_load(buf, broadcast_index):
    """Build the ``BufferLoad`` that replaces a hoisted FloatImm.

    The synthesised buffer is 0-D (``shape=()``) in ``global.fpram``
    — semantically a scalar, broadcast to whatever rank the consumer
    needs. We emit a ``BufferLoad`` with *no indices*; fold.py's
    existing broadcast-prefix matcher picks this up as
    ``len(src_idx) = 0 < len(dst_indices)`` and wraps the load in
    ``Broadcast(broadcast_dims=[0..n-1])`` automatically, fanning the
    value across every dst axis with no special-case logic on the
    fold side.

    ``broadcast_index`` is no longer used (the load carries no index
    at all) but is kept in the signature so callers don't break.
    """
    del broadcast_index
    return tir.BufferLoad(buf, [])


def _walk(stmt, table: _ConstantTable, root_block_name: str,
          extra_allocs: List[tir.Buffer]):
    if stmt is None:
        return None
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt(
            [_walk(c, table, root_block_name, extra_allocs) for c in stmt.seq]
        )
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=list(stmt.iter_values),
            predicate=stmt.predicate,
            block=_walk(stmt.block, table, root_block_name, extra_allocs),
        )
    if isinstance(stmt, tir.Block):
        new_body = _walk(stmt.body, table, root_block_name, extra_allocs)
        new_init = _walk(stmt.init, table, root_block_name, extra_allocs) \
            if stmt.init is not None else None
        # Attach the synthesized buffers to the kernel-root Block —
        # tilelang's `T.Kernel(...)` macro emits a Block named
        # ``"tilelang_root"`` that owns every user alloc_buffer. Adding
        # ours there keeps them in the same scoping bracket so all
        # passes treat them like hand-written allocs.
        if stmt.name_hint == root_block_name and extra_allocs:
            new_allocs = list(stmt.alloc_buffers) + list(extra_allocs)
        else:
            new_allocs = stmt.alloc_buffers
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=new_body,
            init=new_init,
            alloc_buffers=new_allocs,
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _walk(stmt.body, table, root_block_name, extra_allocs),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, table, root_block_name, extra_allocs),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            stmt.condition,
            _walk(stmt.then_case, table, root_block_name, extra_allocs),
            _walk(stmt.else_case, table, root_block_name, extra_allocs)
            if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.LetStmt):
        # inline_let_stmts runs before us, so this shouldn't appear in
        # practice — but if it does, we recurse for robustness.
        return tir.LetStmt(
            stmt.var, stmt.value,
            _walk(stmt.body, table, root_block_name, extra_allocs),
        )
    if isinstance(stmt, tir.BufferStore):
        skip_top = _is_skip_expr(stmt.value)
        # Pass the dst's leading index expression to the rewriter so
        # synthesised loads index their (1,)-shape buffer with the
        # same expression. Fold's broadcast-prefix matcher then
        # accepts the load as a scalar broadcast across that axis.
        # ``stmt.indices`` is non-empty in practice for every store
        # we'd rewrite into (a scalar store with no indices wouldn't
        # have a vector RHS to broadcast against).
        broadcast_index = stmt.indices[0] if stmt.indices else None
        return tir.BufferStore(
            stmt.buffer,
            _rewrite_expr(stmt.value, table, skip_top, broadcast_index),
            # Indices stay as-is — they're IntImm / Var / affine expr,
            # never hoistable FloatImms.
            list(stmt.indices),
        )
    if isinstance(stmt, tir.Evaluate):
        return tir.Evaluate(_rewrite_expr(stmt.value, table, False))
    if isinstance(stmt, tir.Allocate):
        return tir.Allocate(
            stmt.buffer_var, stmt.dtype, list(stmt.extents),
            stmt.condition,
            _walk(stmt.body, table, root_block_name, extra_allocs),
            stmt.annotations,
        )
    # Fall through: unknown stmt types pass unchanged so the pass
    # doesn't get in the way of features added to TIR later.
    return stmt


# tilelang names the kernel-body Block ``tilelang_root``. If a future
# tilelang version renames it, callers can pass an override via the
# ``root_block_name`` kwarg on :func:`run`.
DEFAULT_ROOT_BLOCK_NAME = "tilelang_root"


def run(func: tir.PrimFunc,
        *,
        root_block_name: str = DEFAULT_ROOT_BLOCK_NAME) -> tir.PrimFunc:
    """Hoist FP literals to ``global.fpram`` buffers. See module docstring.

    No-op on kernels with no hoistable FloatImms (most kernels today).
    """
    table = _ConstantTable()
    # First sweep: collect + rewrite in one pass (the table is
    # populated as a side effect of ``_rewrite_expr``). Buffers come
    # out in the order they were first encountered.
    new_body = _walk(func.body, table, root_block_name, extra_allocs=[])

    new_allocs = table.alloc_buffers()
    if not new_allocs:
        return func

    # Second sweep: re-walk only to inject the new alloc_buffers into
    # the root block. We can't do it in the first sweep because we
    # don't know what to inject until the first sweep finishes.
    new_body = _walk(new_body, _ConstantTable(), root_block_name, new_allocs)

    # Stash the {name: value} table on PrimFunc.attrs so to_plena can
    # propagate it onto HLIR Buffer.constant_value (for the dump). We
    # rebuild the PrimFunc with the new body / allocs first, then ride
    # ``with_attr`` to set the attr — synthesising a ``DictAttrs``
    # directly is fragile across TVM versions.
    name_value = table.name_value_map()
    new_func = tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )
    return new_func.with_attr(
        ATTR_KEY,
        tvm.runtime.convert({name: float(v) for name, v in name_value.items()}),
    )


__all__ = ["run", "ATTR_KEY", "DEFAULT_ROOT_BLOCK_NAME"]
