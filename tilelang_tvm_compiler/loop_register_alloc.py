"""Loop-register allocation pass (HLIR liveness).

Why this pass exists
--------------------

GP registers used to be allocated entirely during ISA emission, from a
single 16-register pool shared by three unrelated consumers:

  * ``_emit_for``        — one ``gp_loop`` (C_LOOP hardware counter) per
                           serial loop, live for the whole loop body.
  * ``expr_materializer``— short-lived temporaries for address algebra.
  * ``emit_*``           — per-instruction scratch.

Mixing long-lived loop registers with short-lived temporaries in one
bare pool meant emit-stage code could not tell which registers it was
allowed to pin / free, and deep loop nests exhausted the pool.

This pass separates the two classes. It walks the HLIR, computes — by
liveness — which GP each serial ``for`` loop's ``gp_loop`` should use,
stamps that onto the ``for`` op, and returns the set of GPs it claimed.
The caller hands that set to the emit-stage ``RegisterAllocator`` as
``gp_reserved``, so emit-stage temporary allocation physically cannot
touch a loop register. See ``doc/LOOP_REGISTER_ALLOC.md``.

Why liveness here is trivial
----------------------------

A loop variable is live exactly over its loop's lexical body. Loop
bodies are *strictly nested* — any two are either nested or disjoint,
never partially overlapping. So "how many loop registers are live at
once" is just the loop-nesting depth, and assigning registers is a
linear walk with a stack: depth-0 loop takes the first reserved GP,
depth-1 the next, and so on. No interference graph, no colouring.

What this pass does NOT touch
-----------------------------

  * ``unroll`` loops — their index is a compile-time constant
    (``_emit_for`` binds it to a ``tir.IntImm``), so they need no
    ``gp_loop`` and no reservation.
  * Loop index storage — the index itself stays in IntRAM
    (``claim_idx_slot``); only the C_LOOP hardware counter is a GP.
  * Emit-stage temporaries — still allocated during emission, just from
    the now-smaller un-reserved pool.
"""

from __future__ import annotations

from typing import Dict, List, Set

from . import hlir as _hlir


class LoopRegisterAllocError(RuntimeError):
    pass


# GP file is 16 registers; gp0 is the constant-zero register. The emit
# stage still needs a workable pool for op temporaries after loop
# registers are reserved — if a nest is so deep that too few GPs are
# left, fail here with a clear message rather than crashing mid-emit.
_GP_TOTAL = 32
_GP0_RESERVED = 1                       # gp0
_MIN_EMIT_POOL = 8                      # heaviest emit_* needs ~7 scratch


def _is_for(op: _hlir.Op) -> bool:
    return op.kind == "for"


def _loop_kind(op: _hlir.Op) -> str:
    return op.annotations.get("loop_kind", "serial")


def _is_serial_for(op: _hlir.Op) -> bool:
    """A serial ``for`` lowers to a hardware C_LOOP and therefore needs
    a ``gp_loop`` register. An ``unroll`` ``for`` does not."""
    return _is_for(op) and _loop_kind(op) not in ("unroll", "unrolled")


def _max_serial_depth(ops: List[_hlir.Op]) -> int:
    """Deepest chain of *serial* ``for`` nesting in a body list."""
    best = 0
    for op in ops:
        if op.body is None:
            continue
        inner = _max_serial_depth(op.body)
        if _is_serial_for(op):
            inner += 1
        best = max(best, inner)
    return best


def _assign(ops: List[_hlir.Op], depth: int, reserved: List[int]) -> None:
    """Walk ``ops``; stamp each serial ``for`` with the ``gp_loop`` GP
    for its nesting depth. ``reserved`` is the depth-indexed list of GP
    numbers (reserved[d] == the GP used by a serial loop at depth d)."""
    for op in ops:
        if _is_serial_for(op):
            # ``depth`` can never exceed what _max_serial_depth measured
            # — both count "+1 only on a serial for" identically — so
            # this index is always valid. Assert rather than risk a bare
            # IndexError if the two ever drift apart.
            assert depth < len(reserved), (
                f"loop depth {depth} exceeds reserved register count "
                f"{len(reserved)} — _max_serial_depth / _assign drifted"
            )
            gp_loop = reserved[depth]
            op.annotations["loop_gp"] = gp_loop
            if op.body is not None:
                _assign(op.body, depth + 1, reserved)
        else:
            # Non-serial-for (unroll for, leaf op): depth unchanged.
            if op.body is not None:
                _assign(op.body, depth, reserved)


def run(mod: _hlir.HLIRModule) -> Set[int]:
    """Assign a ``gp_loop`` GP to every serial ``for`` in ``mod`` and
    stamp it onto the op's ``annotations['loop_gp']``.

    Returns the set of GP numbers reserved for loop counters — the
    caller passes this to the emit-stage ``RegisterAllocator`` as
    ``gp_reserved`` so temporary allocation cannot collide with a loop
    register. Mutates ``mod`` in place.
    """
    depth = _max_serial_depth(mod.ops)
    if depth == 0:
        return set()

    # Reserve from the TOP of the GP file (gp15, gp14, …) so the
    # low-numbered registers — which emit-stage code and dumps tend to
    # use first — stay with the temporary pool.
    reserved_list = [(_GP_TOTAL - 1) - d for d in range(depth)]
    reserved = set(reserved_list)

    free_for_emit = _GP_TOTAL - _GP0_RESERVED - len(reserved)
    if free_for_emit < _MIN_EMIT_POOL:
        raise LoopRegisterAllocError(
            f"serial loop nesting depth {depth} reserves {len(reserved)} "
            f"GP(s); only {free_for_emit} left for emit-stage temporaries "
            f"(need >= {_MIN_EMIT_POOL}). Convert an outer loop to "
            f"T.unroll(...) — an unrolled loop needs no gp_loop."
        )

    _assign(mod.ops, 0, reserved_list)
    return reserved


__all__ = ["run", "LoopRegisterAllocError"]
