"""pass_4_async: wrap can_async ops in Async regions.

Why this pass exists
--------------------

After ``split`` + ``distribute_cluster``, every cluster body holds a
mix of:

  * ``can_async=True`` ops (Dma, Gemm[btmm], pure Elementwise) — each
    lowers to a single multi-lane HW instruction (M_BTMM / H_LOAD_V /
    V_ADD ...). These get wrapped one-per-Async (strict 'one async
    one op' rule from SPMD_REWRITE.md); the next pass picks the
    physical buffer view per Async, and pass_5 then fuses each
    Async into a MultiLaneOp.

  * ``can_async=False`` ops (Reduce, Elementwise containing a
    Broadcast src) — these lower to per-row HW instructions
    (row_reduce_max_at, row_sub_fp_at, ...) that need a fresh fp
    scalar address per lane. They stay in the cluster body bare;
    pass_8 emits ``for lane in range(cluster)`` around them.

What this pass does NOT touch
-----------------------------

  * BufferRef indices / view_perm — that's the next pass.
    The buffer-rank vs ref-rank mismatch introduced by ``split``
    persists past this pass; it gets resolved by the view pass.
  * Buffer shapes — those were set by pass_3.
  * MultiLaneOp synthesis — pass_5.
  * Anything outside a cluster body — Fors, RawStore, grid headers
    etc. are walked structurally but not rewritten.
"""

from __future__ import annotations

from typing import List

from ..cluster_guard import should_skip_cluster
from ..ir import (
    Dma, Gemm, Elementwise, Reduce,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


class AsyncWrapError(RuntimeError):
    pass


class _IdCounter:
    """Module-global counter for generating fresh Async scope IDs."""
    def __init__(self) -> None:
        self.next = 0

    def fresh(self) -> int:
        n = self.next
        self.next += 1
        return n


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk(stmt: Stmt, in_cluster: bool, ids: _IdCounter) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            if stmt.parent_grid_axis_name is None:
                raise AsyncWrapError(
                    f"cluster {stmt.axis_name!r} has no parent_grid_axis_name; "
                    f"split should have set it"
                )
            new_body = [_walk(s, in_cluster=True, ids=ids) for s in stmt.body]
            new_body = _wrap_async_runs(new_body, ids)
            return ParallelAxis(
                axis_name=stmt.axis_name,
                extent=stmt.extent,
                body=new_body,
                kind=stmt.kind,
                thread_tag=stmt.thread_tag,
                parent_grid_axis_name=stmt.parent_grid_axis_name,
                original_axis_name=stmt.original_axis_name,
                axis_var=stmt.axis_var,
                original_axis_var=stmt.original_axis_var,
            )
        # grid / logical_grid: pass through, but if we're already inside
        # a cluster the leaf-op bodies here still need async wrapping.
        new_body = [_walk(s, in_cluster=in_cluster, ids=ids) for s in stmt.body]
        if in_cluster:
            new_body = _wrap_async_runs(new_body, ids)
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=new_body,
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
            original_axis_name=stmt.original_axis_name,
            axis_var=stmt.axis_var,
            original_axis_var=stmt.original_axis_var,
        )
    if isinstance(stmt, For):
        new_body = [_walk(s, in_cluster=in_cluster, ids=ids) for s in stmt.body]
        if in_cluster:
            new_body = _wrap_async_runs(new_body, ids)
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=new_body,
            kind=stmt.kind,
            loop_var_var=stmt.loop_var_var,
        )
    if isinstance(stmt, Async):
        # Already wrapped; preserve and recurse (idempotency).
        return Async(
            body=[_walk(s, in_cluster=in_cluster, ids=ids) for s in stmt.body],
            scope_id=stmt.scope_id,
        )
    # Leaf op or MultiLaneOp: pass through unchanged. The async wrap
    # decision is made one level up by _wrap_async_runs (which only
    # fires inside cluster bodies).
    return stmt


def _wrap_async_runs(stmts: List[Stmt], ids: _IdCounter) -> List[Stmt]:
    """For each ``can_async=True`` leaf op in the cluster body, wrap it
    in its own Async region (strict one-async-one-op). can_async=False
    ops stay unwrapped. Already-wrapped Async / MultiLaneOp / nested
    structure stays as-is."""
    out: List[Stmt] = []
    for s in stmts:
        if _is_async_eligible(s):
            out.append(Async(body=[s], scope_id=ids.fresh()))
        else:
            out.append(s)
    return out


def _is_async_eligible(s: Stmt) -> bool:
    """True if ``s`` is a leaf op with ``can_async=True``."""
    return isinstance(s, (Dma, Gemm, Elementwise, Reduce)) \
        and getattr(s, "can_async", False)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Wrap can_async ops in Async regions. Index/view rewriting is a
    separate downstream pass.

    No-op if ``should_skip_cluster(func)`` — without cluster fusion
    there's no notion of multi-lane dispatch to mark."""
    if should_skip_cluster(func):
        return func
    ids = _IdCounter()
    new_body = [_walk(s, in_cluster=False, ids=ids) for s in func.body]
    return MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=new_body,
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "AsyncWrapError"]
