"""pass_0_infer_lane_axis: pick the lane axis from a raw PrimFunc.

Why this pass exists
--------------------

Kernel authors used to declare the lane axis explicitly via
``T.func_attr({"plena.lane_axis": "by"})``. That's annoying boilerplate
the compiler can deduce by looking at how each grid var is *used* in
the kernel body — not at its extent.

The judgment principle: a lane axis is a blockIdx grid var that
appears as a **bare** index into some buffer access (e.g. ``T.copy(
Q_hbm[0, q_block*rows, by, 0], Q_sh)`` — ``by`` sits at index slot
2 directly, naked). A grid var that only appears wrapped in
arithmetic (``q_block * rows`` for an offset computation) is acting
as an outer control loop, not as a per-lane indexing dim.

Algorithm:

  * Walk every ``AttrStmt(thread_extent, IterVar(thread_tag="blockIdx.*"))``
    to enumerate grid vars + their extents.
  * Walk every ``BufferLoad`` and every ``tl.tileop.region`` extern
    call. For each grid var, check if it appears as a **bare**
    index slot somewhere (``BufferLoad.indices[i] is the same Var``,
    not a compound expression containing it).
  * Lane candidates = grid vars that appear bare AT LEAST ONCE,
    AND whose extent is divisible by LANE.
  * If the user manually set ``plena.lane_axis``, respect it.
  * If 0 candidates → leave func.attrs as-is; cluster_guard will
    skip the cluster pipeline.
  * If 1 candidate → pick it.
  * If 2+ candidates → ambiguous; raise InferLaneAxisError and ask
    the kernel author to disambiguate via ``T.func_attr``.

Runs BEFORE pass_1_fold — it operates on raw TIR.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import tvm
from tvm import tir


# Same constant as cluster_guard.MLEN; we re-derive LANE here so this
# pass doesn't have to depend on the mid_ir scope vocabulary.
_DEFAULT_LANE = 4

_LANE_AXIS_FUNC_ATTR = "plena.lane_axis"


class InferLaneAxisError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


def _collect_block_idx_bindings(func: tir.PrimFunc
                                ) -> List[Tuple[str, int]]:
    """Walk the body, collect ``(var_name, extent)`` for every
    ``thread_extent`` AttrStmt whose IterVar is bound to ``blockIdx.*``
    and has a static integer extent."""
    out: List[Tuple[str, int]] = []

    def visit(stmt) -> None:
        if stmt is None:
            return
        if isinstance(stmt, tir.AttrStmt):
            if (stmt.attr_key == "thread_extent"
                    and isinstance(stmt.node, tir.IterVar)
                    and stmt.node.thread_tag is not None
                    and stmt.node.thread_tag.startswith("blockIdx")
                    and isinstance(stmt.value, tir.IntImm)):
                out.append((stmt.node.var.name, int(stmt.value.value)))
            visit(stmt.body)
            return
        if isinstance(stmt, tir.SeqStmt):
            for c in stmt.seq:
                visit(c)
            return
        if isinstance(stmt, tir.BlockRealize):
            visit(stmt.block.body)
            if stmt.block.init is not None:
                visit(stmt.block.init)
            return
        if isinstance(stmt, (tir.For, tir.LetStmt, tir.Allocate)):
            visit(stmt.body)
            return
        if isinstance(stmt, tir.IfThenElse):
            visit(stmt.then_case)
            if stmt.else_case is not None:
                visit(stmt.else_case)
            return

    visit(func.body)
    return out


def _existing_lane_axis(func: tir.PrimFunc) -> Optional[str]:
    if func.attrs is None:
        return None
    if _LANE_AXIS_FUNC_ATTR not in func.attrs:
        return None
    raw = func.attrs[_LANE_AXIS_FUNC_ATTR]
    if isinstance(raw, tir.StringImm):
        return str(raw.value)
    return str(raw)


# ---------------------------------------------------------------------------
# Bare-index detection
# ---------------------------------------------------------------------------


def _collect_bare_index_var_names(func: tir.PrimFunc) -> set:
    """Return the set of var names that appear as a *bare* index slot
    in some BufferLoad anywhere in the body.

    "Bare" means: ``BufferLoad.indices[i]`` is exactly a ``tir.Var``,
    not a compound expression. ``q_block * 64`` doesn't qualify;
    ``by`` does.
    """
    found: set = set()
    from tvm.tir import stmt_functor

    def visit(node) -> None:
        if isinstance(node, tir.BufferLoad):
            for idx in node.indices:
                if isinstance(idx, tir.Var):
                    found.add(idx.name)
        # Region-extern calls (tl.tileop.region(BufferLoad, ...)) are
        # already covered by the BufferLoad inside their first arg —
        # post_order_visit walks down into args.

    stmt_functor.post_order_visit(func.body, visit)
    return found


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: tir.PrimFunc, lane: int = _DEFAULT_LANE) -> tir.PrimFunc:
    """Return ``func`` with ``plena.lane_axis`` set on attrs.

    Picks the unique grid var that:
      * is bound to ``blockIdx.*`` with a static integer extent,
      * has extent divisible by ``lane``,
      * appears as a bare index slot in some BufferLoad.

    Manual override (``T.func_attr({"plena.lane_axis": ...})``) wins
    over the auto-pick. Zero candidates → no attr (cluster_guard
    later skips). Multiple candidates → ambiguous, raises.
    """
    if _existing_lane_axis(func) is not None:
        return func

    grid_bindings = _collect_block_idx_bindings(func)
    bare_names = _collect_bare_index_var_names(func)

    candidates = [
        (name, ext) for (name, ext) in grid_bindings
        if ext % lane == 0 and name in bare_names
    ]

    if not candidates:
        return func
    if len(candidates) == 1:
        return func.with_attr(_LANE_AXIS_FUNC_ATTR, candidates[0][0])

    raise InferLaneAxisError(
        f"ambiguous lane axis: more than one grid var qualifies as a "
        f"lane candidate ({[n for n, _ in candidates]!r}). Disambiguate "
        f"by writing T.func_attr({{'plena.lane_axis': '<name>'}}) in "
        f"the kernel."
    )


__all__ = ["run", "InferLaneAxisError"]
