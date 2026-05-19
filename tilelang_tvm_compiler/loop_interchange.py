"""Loop interchange pass (HLIR post-processing).

Why this pass exists
--------------------

``to_plena`` lowers per-lane work as ``for X { for C { ... } }`` where
``C`` is the cluster (per-lane) axis and ``X`` is some enclosing loop
(typically ``for row``). A sibling cluster loop lowered separately —
e.g. ``for C { matmul }`` — sits next to ``for X``, NOT next to the
inner ``for C``, so :mod:`fuse_adjacent_loops` cannot reach the two.

Interchanging the nest::

    for X { for C { body } }   ->   for C { for X { body } }

lifts the cluster loop to the same level as its sibling, after which
the fusion pass merges them. Run the two passes alternately to a fixed
point (see :func:`pipeline`) so a chain collapses fully.

Legality
--------

This pass interchanges **only** when the inner loop is a cluster axis
(tagged ``is_cluster_axis`` by ``to_plena``) and the outer loop is not.
A cluster axis is per-lane: every lane owns its own buffer slots, so
there is no cross-iteration dependency between a cluster axis and any
non-cluster axis. That single structural condition *is* the legality
proof — no dependency analysis is needed.

Scope
-----

Conservative on purpose: only the clean case is handled — an outer
``for`` whose body is *exactly one* statement, that statement being a
cluster ``for``. A mixed outer body (cluster loop interleaved with
other ops) would need those other ops dragged inside the cluster loop,
which is not always legal; that case is left untouched.
"""

from __future__ import annotations

from typing import List, Tuple

from . import hlir as _hlir


def _is_for(op: _hlir.Op) -> bool:
    return op.kind == "for"


def _is_cluster_for(op: _hlir.Op) -> bool:
    return _is_for(op) and bool(op.annotations.get("is_cluster_axis"))


def _clone_for(op: _hlir.Op, body: List[_hlir.Op]) -> _hlir.Op:
    """A copy of a ``for`` op with a different body, annotations and all
    loop metadata preserved."""
    return _hlir.Op(
        kind="for",
        buffer_args=list(op.buffer_args),
        scalar_args=list(op.scalar_args),
        annotations=dict(op.annotations),
        body=body,
        buffer_axes=list(op.buffer_axes),
    )


def _interchange_body(ops: List[_hlir.Op]) -> Tuple[List[_hlir.Op], bool]:
    """Interchange eligible nests in one body list. Recurses first so a
    deep nest is handled bottom-up. Returns ``(new_ops, changed)``."""
    changed = False

    # 1) recurse into every for body first.
    recursed: List[_hlir.Op] = []
    for op in ops:
        if _is_for(op) and op.body is not None:
            new_body, sub_changed = _interchange_body(op.body)
            changed = changed or sub_changed
            recursed.append(_clone_for(op, new_body))
        else:
            recursed.append(op)

    # 2) at this level, interchange ``for X { for C {...} }``.
    out: List[_hlir.Op] = []
    for op in recursed:
        if (_is_for(op)
                and not _is_cluster_for(op)
                and op.body is not None
                and len(op.body) == 1
                and _is_cluster_for(op.body[0])):
            outer = op            # for X
            inner = op.body[0]    # for C  (cluster)
            # for X { for C { body } }  ->  for C { for X { body } }
            new_inner = _clone_for(outer, list(inner.body))   # for X { body }
            new_outer = _clone_for(inner, [new_inner])        # for C { for X }
            out.append(new_outer)
            changed = True
        else:
            out.append(op)
    return out, changed


def run(mod: _hlir.HLIRModule) -> Tuple[_hlir.HLIRModule, bool]:
    """Interchange cluster-inner loops outward throughout the module.
    Returns ``(mod, changed)``; ``changed`` is False once no nest is
    eligible (the fixed-point signal). Mutates ``mod.ops`` in place."""
    new_ops, changed = _interchange_body(mod.ops)
    mod.ops = new_ops
    return mod, changed


__all__ = ["run"]
