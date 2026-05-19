"""Adjacent-loop fusion pass (HLIR post-processing).

Why this pass exists
--------------------

``to_plena`` lowers every per-lane scalar / per-row op independently,
and each one ends up wrapped in its own ``for`` loop. A run of
consecutive per-lane ops therefore produces a run of consecutive
identical loops::

    for by_phase in [0, 4): fp_mul_at(...)
    for by_phase in [0, 4): fp_add_at(...)
    for by_phase in [0, 4): row_mul_fp(...)
    for by_phase in [0, 4): fp_copy_at(...)

That is needless loop overhead — the four loops have the same variable
and the same extent, with nothing (no DMA / btmm / multi-lane op) in
between. They are equivalent to a single loop over the four bodies::

    for by_phase in [0, 4):
        fp_mul_at(...); fp_add_at(...); row_mul_fp(...); fp_copy_at(...)

This pass merges such adjacent loops.

What it does
------------

For every body list in the module (top level and inside every ``for``):

  1. **Recurse first** — fuse inside each ``for`` body bottom-up, so an
     arbitrarily deep nest collapses in one pass.
  2. **Then merge adjacent siblings** — walk the list; whenever two
     neighbouring ops are both ``for`` with the *same loop-variable
     name*, the *same extent*, and the *same init*, concatenate their
     bodies into one loop. Chains of N collapse into one.

Correctness
-----------

  * Per-lane scalar ops carry no cross-lane dependency (each lane owns
    its own FPRAM scalar slot), and same-lane order is preserved
    because body B is appended after body A *inside the same iteration*.
    So merging never reorders dependent work.
  * The two loops use different ``loop_var`` objects that merely share a
    name (``to_plena`` mints a fresh var per loop). After merging we
    keep loop A's var and **substitute** every reference to loop B's var
    in B's body with loop A's var, recursively, through ``scalar_args``
    and ``buffer_args`` PrimExpr / region trees.

This pass is generic: it is not keyed on ``by_phase`` or any particular
loop name — any two adjacent same-shape loops fuse.
"""

from __future__ import annotations

from typing import Any, List

from . import hlir as _hlir


def _loop_meta(op: _hlir.Op):
    """Return ``(loop_var, extent, init)`` for a ``for`` op."""
    a = op.annotations
    return a.get("loop_var"), a.get("extent"), a.get("init", 0)


def _var_name(v) -> Any:
    """A loop var's identifying name. ``to_plena`` stores either a
    ``tir.Var`` (``.name``) or a plain string. Two loops fuse when these
    names match — the underlying objects differ per loop."""
    return getattr(v, "name", v)


# ---------------------------------------------------------------------------
# Variable substitution — rewrite refs to loop B's var with loop A's var
# ---------------------------------------------------------------------------


def _subst_in_value(value: Any, old_name: Any, new_var: Any) -> Any:
    """Recursively replace a loop variable inside a scalar_arg /
    buffer_arg value. Handles tir PrimExpr trees, HLIR BufferElement,
    VramRegion / MramRegion / BufferSlice, plain containers. Anything it
    doesn't recognise is returned untouched.

    Identity is by *name*: a tir.Var (or string) whose name equals
    ``old_name`` becomes ``new_var``.
    """
    # Bare loop var — a tir.Var or a plain string. Match on these
    # EXACT types only: a generic tir PrimExpr has no `.name`, and
    # comparing it with `== old_name` would invoke tir's overloaded
    # `__eq__` (which builds an int32-vs-string equality node and
    # crashes). A compound PrimExpr instead falls through to the
    # `_is_tir_expr` branch below and is handled by tir's substitute.
    if isinstance(value, str):
        return new_var if value == old_name else value
    if _is_tir_var(value):
        return new_var if value.name == old_name else value

    # HLIR BufferElement: substitute inside its index expressions.
    if isinstance(value, _hlir.BufferElement):
        return _hlir.BufferElement(
            buffer=value.buffer,
            indices=tuple(
                _subst_in_value(i, old_name, new_var) for i in value.indices
            ),
        )

    # VramRegion / MramRegion: substitute inside `starts` (extents are
    # ints, never carry a loop var).
    if isinstance(value, _hlir.VramRegion):
        return _hlir.VramRegion(
            parent=value.parent,
            starts=tuple(
                _subst_in_value(s, old_name, new_var) for s in value.starts
            ),
            extents=value.extents,
        )
    if isinstance(value, _hlir.MramRegion):
        return _hlir.MramRegion(
            parent=value.parent,
            starts=tuple(
                _subst_in_value(s, old_name, new_var) for s in value.starts
            ),
            extents=value.extents,
        )

    # tir PrimExpr tree — duck-typed walk over the usual child fields.
    # Rebuilding tir nodes generically is fragile, so we rely on tir's
    # own substitute when the value is a PrimExpr.
    if _is_tir_expr(value):
        return _tir_substitute(value, old_name, new_var)

    # Containers.
    if isinstance(value, list):
        return [_subst_in_value(v, old_name, new_var) for v in value]
    if isinstance(value, tuple):
        return tuple(_subst_in_value(v, old_name, new_var) for v in value)

    return value


def _is_tir_var(value: Any) -> bool:
    """True iff ``value`` is exactly a ``tvm.tir.Var`` — the only tir
    node that carries a ``.name`` and stands for a loop variable.
    Compound PrimExprs (Add / Mul / …) are deliberately excluded."""
    try:
        from tvm import tir as _t
        return isinstance(value, _t.Var)
    except Exception:
        return False


def _is_tir_expr(value: Any) -> bool:
    """True if ``value`` looks like a tvm.tir PrimExpr (has a dtype and
    is not one of our own HLIR dataclasses / a plain scalar)."""
    if isinstance(value, (int, float, str, bool)):
        return False
    return hasattr(value, "dtype") and value.__class__.__module__.startswith(
        "tvm"
    )


def _tir_substitute(expr: Any, old_name: Any, new_var: Any) -> Any:
    """Substitute a variable inside a tir PrimExpr using tir's own
    ``substitute``. Falls back to returning the expr unchanged if tir is
    unavailable or the substitution can't be expressed."""
    try:
        from tvm import tir as _t

        def _mapper(v):
            if getattr(v, "name", None) == old_name:
                return new_var if _is_tir_expr(new_var) else None
            return None

        return _t.stmt_functor.substitute(expr, _mapper) \
            if hasattr(_t, "stmt_functor") else _t.substitute(expr, _mapper)
    except Exception:
        return expr


def _subst_op(op: _hlir.Op, old_name: Any, new_var: Any) -> _hlir.Op:
    """Return a copy of ``op`` with loop var ``old_name`` rewritten to
    ``new_var`` throughout its args and (recursively) its body."""
    new_body = None
    if op.body is not None:
        new_body = [_subst_op(b, old_name, new_var) for b in op.body]
    new_anno = dict(op.annotations)
    # A nested for that happens to reuse the name keeps its own var —
    # but its loop_var object is distinct, so only rewrite if the names
    # collide AND it's not this op redefining it. Safe path: rewrite the
    # stored loop_var only when it is not itself the shadowing var. In
    # practice to_plena never shadows, so a plain rewrite is fine.
    if "loop_var" in new_anno and _var_name(new_anno["loop_var"]) == old_name:
        # This inner loop shadows old_name — stop substitution here for
        # its body by NOT rewriting (its body refers to its own var).
        return _hlir.Op(
            kind=op.kind, buffer_args=list(op.buffer_args),
            scalar_args=list(op.scalar_args), annotations=new_anno,
            body=op.body, buffer_axes=list(op.buffer_axes),
        )
    return _hlir.Op(
        kind=op.kind,
        buffer_args=[_subst_in_value(b, old_name, new_var)
                     for b in op.buffer_args],
        scalar_args=[_subst_in_value(s, old_name, new_var)
                     for s in op.scalar_args],
        annotations=new_anno,
        body=new_body,
        buffer_axes=list(op.buffer_axes),
    )


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _can_fuse(a: _hlir.Op, b: _hlir.Op) -> bool:
    """Two ops fuse iff both are ``for`` loops with matching loop-var
    name, extent and init."""
    if a.kind != "for" or b.kind != "for":
        return False
    a_var, a_ext, a_init = _loop_meta(a)
    b_var, b_ext, b_init = _loop_meta(b)
    return (
        _var_name(a_var) == _var_name(b_var)
        and a_ext == b_ext
        and a_init == b_init
    )


def _fuse_body(ops: List[_hlir.Op], changed: List[bool]) -> List[_hlir.Op]:
    """Fuse adjacent loops in one body list. Recurses into every ``for``
    body first (bottom-up) so nested runs collapse too. ``changed`` is a
    one-element mutable cell flipped to True on any merge."""
    # 1) recurse bottom-up
    recursed: List[_hlir.Op] = []
    for op in ops:
        if op.kind == "for" and op.body is not None:
            recursed.append(_hlir.Op(
                kind=op.kind,
                buffer_args=list(op.buffer_args),
                scalar_args=list(op.scalar_args),
                annotations=dict(op.annotations),
                body=_fuse_body(op.body, changed),
                buffer_axes=list(op.buffer_axes),
            ))
        else:
            recursed.append(op)

    # 2) merge adjacent siblings
    out: List[_hlir.Op] = []
    for op in recursed:
        if out and _can_fuse(out[-1], op):
            prev = out[-1]
            changed[0] = True
            keep_var = _loop_meta(prev)[0]
            drop_name = _var_name(_loop_meta(op)[0])
            # Rewrite op's body to use the kept loop var, then append.
            rebased = [_subst_op(b, drop_name, keep_var) for b in op.body]
            merged_body = list(prev.body) + rebased
            # The merged loop body itself may now expose new adjacent
            # loops — fuse it again so chains fully collapse.
            out[-1] = _hlir.Op(
                kind="for",
                buffer_args=list(prev.buffer_args),
                scalar_args=list(prev.scalar_args),
                annotations=dict(prev.annotations),
                body=_fuse_body(merged_body, changed),
                buffer_axes=list(prev.buffer_axes),
            )
        else:
            out.append(op)
    return out


def run(mod: _hlir.HLIRModule):
    """Merge adjacent same-shape ``for`` loops throughout the module.
    Mutates ``mod.ops`` in place. Returns ``(mod, changed)``; ``changed``
    is True iff at least one merge happened (fixed-point signal)."""
    changed: List[bool] = [False]
    mod.ops = _fuse_body(mod.ops, changed)
    return mod, changed[0]


__all__ = ["run"]
