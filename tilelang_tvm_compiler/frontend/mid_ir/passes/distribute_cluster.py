"""pass_3b_distribute_cluster: push CLUSTER axes inside enclosing
unroll/pipeline For loops.

Why this pass exists
--------------------

After ``split``, the IR can have a CLUSTER ParallelAxis whose body
contains a ``For(kind="unroll")``::

    cluster c [...]:
        for kh in unroll(KH):
            op_a
            op_b

When pass_4_async wraps the can_async ops in Async regions, an Async
that lives outside the unroll loop logically spans **all KH iters**:

    cluster c [...]:
        async { dma_a, dma_b }                  ← single Async covers ALL iters
        for kh in unroll(KH):
            ...

That's wrong for the HW: each unroll iter should fire its own Async
dispatch (own DMA, own gemm, etc.) and complete independently. Mixing
"one Async / KH iters" semantics with the per-iter HW model creates
ambiguity about when the Async actually waits.

This pass rewrites the nesting so each unroll iter has its OWN cluster
body — pass_4 then naturally produces one Async per iter::

    for kh in unroll(KH):
        cluster c [...]:           ← cluster repeats inside each iter
            op_a
            op_b

Mixed cluster bodies
--------------------

When a cluster body holds an unroll For interleaved with other ops,
the cluster splits into multiple cluster instances around each unroll
For::

    cluster c [...]:
        op_pre
        for kh in unroll(KH):
            op_inner
        op_post
        ↓
    cluster c [...]:
        op_pre
    for kh in unroll(KH):
        cluster c [...]:
            op_inner
    cluster c [...]:
        op_post

This preserves the original execution order and keeps every op inside
some cluster body (so pass_4 can still see lane fusion context for it).

What this pass does NOT touch
-----------------------------

  * ``For(kind="serial")`` — sequential loops; cluster sits OUTSIDE
    just fine. Sequential iters don't have the "concurrent dispatch"
    problem unroll has.
  * Nested clusters (cluster inside cluster) — kernels don't produce
    them today; if one shows up the inner one passes through.
  * Async / MultiLaneOp — neither exists yet at this point in the
    pipeline (pass_4 hasn't run).
"""

from __future__ import annotations

from typing import List

from ..cluster_guard import should_skip_cluster
from ..ir import (
    BufferDef, BufferRef,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


class DistributeClusterError(RuntimeError):
    pass


def _is_unroll_for(s: Stmt) -> bool:
    return isinstance(s, For) and s.kind == "unroll"


def _clone_cluster_with_body(template: ParallelAxis,
                             body: List[Stmt]) -> ParallelAxis:
    """Make a fresh CLUSTER axis with the same axis_name / extent /
    parent_grid_axis_name as ``template`` but a different body."""
    return ParallelAxis(
        axis_name=template.axis_name,
        extent=template.extent,
        body=body,
        kind=ParallelKind.CLUSTER,
        thread_tag=template.thread_tag,         # always None for CLUSTER, but copy anyway
        parent_grid_axis_name=template.parent_grid_axis_name,
        original_axis_name=template.original_axis_name,
        axis_var=template.axis_var,
        original_axis_var=template.original_axis_var,
    )


# ---------------------------------------------------------------------------
# Cluster distribution
# ---------------------------------------------------------------------------


def _distribute_one_cluster(cluster: ParallelAxis) -> List[Stmt]:
    """Take a single CLUSTER axis whose body may contain unroll Fors,
    return a list of stmts equivalent to the original but with each
    unroll For lifted out and the cluster pushed inside.

    See module docstring for the rewrite rule.
    """
    out: List[Stmt] = []
    pending: List[Stmt] = []   # ops accumulated since last unroll-for boundary

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            out.append(_clone_cluster_with_body(cluster, pending))
            pending = []

    for child in cluster.body:
        if _is_unroll_for(child):
            # Boundary: emit any pending ops as a cluster, then emit
            # the unroll For with cluster pushed into its body.
            flush_pending()
            inner_body = _walk_stmts(child.body)
            new_for = For(
                loop_var=child.loop_var,
                extent=child.extent,
                body=[_clone_cluster_with_body(cluster, inner_body)],
                kind=child.kind,
                loop_var_var=child.loop_var_var,
            )
            out.append(new_for)
        else:
            # Recurse into nested structure first, then accumulate.
            pending.append(_walk_stmt(child))

    flush_pending()
    return out


def _walk_stmt(stmt: Stmt) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            distributed = _distribute_one_cluster(stmt)
            # If nothing changed (no unroll for inside), wrap back.
            # Otherwise the cluster has been distributed across
            # multiple stmts; we can't return a list from this point —
            # callers expect one stmt. Wrap multiple in a synthetic
            # cluster… no, better: surface a SeqStmt-like via the
            # parent. Easiest path: parent walker collects stmt lists,
            # not single stmts. See _walk_stmts below.
            if (len(distributed) == 1
                    and isinstance(distributed[0], ParallelAxis)
                    and distributed[0] is not stmt):
                return distributed[0]
            # Single-stmt no-change case
            if (len(distributed) == 1
                    and isinstance(distributed[0], ParallelAxis)
                    and distributed[0].axis_name == stmt.axis_name
                    and distributed[0].body == stmt.body):
                return stmt
            # Multi-stmt distribution — caller has to flatten via _walk_stmts.
            # Mark by returning a "marker" sentinel? We avoid that by
            # not exposing single-stmt callers in this pass — _walk_stmts
            # is the canonical entrypoint for body lists.
            # Fallthrough for safety: re-wrap into a plain cluster.
            return _clone_cluster_with_body(stmt, distributed) if len(distributed) > 1 else distributed[0]
        # Non-CLUSTER ParallelAxis: walk body recursively, no rewrite.
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=_walk_stmts(stmt.body),
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
            original_axis_name=stmt.original_axis_name,
            axis_var=stmt.axis_var,
            original_axis_var=stmt.original_axis_var,
        )
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=_walk_stmts(stmt.body),
            kind=stmt.kind,
            loop_var_var=stmt.loop_var_var,
        )
    if isinstance(stmt, Async):
        return Async(body=_walk_stmts(stmt.body), scope_id=stmt.scope_id)
    if isinstance(stmt, MultiLaneOp):
        return MultiLaneOp(
            inner=_walk_stmt(stmt.inner),
            cluster_axis_names=list(stmt.cluster_axis_names),
            cluster_axis_vars=list(stmt.cluster_axis_vars),
            dim_map=dict(stmt.dim_map),
        )
    # Leaf op: pass through unchanged.
    return stmt


def _walk_stmts(stmts: List[Stmt]) -> List[Stmt]:
    """Walk a body list. CLUSTER axes that distribute into multiple
    stmts are flattened in here (the only place that handles a 1→N
    rewrite cleanly)."""
    out: List[Stmt] = []
    for s in stmts:
        if (isinstance(s, ParallelAxis)
                and s.kind == ParallelKind.CLUSTER
                and any(_is_unroll_for(c) for c in s.body)):
            # Recurse into the cluster body's children first so any
            # nested clusters / fors are handled, then distribute.
            child = ParallelAxis(
                axis_name=s.axis_name,
                extent=s.extent,
                body=_walk_stmts(s.body),
                kind=s.kind,
                thread_tag=s.thread_tag,
                parent_grid_axis_name=s.parent_grid_axis_name,
                original_axis_name=s.original_axis_name,
                axis_var=s.axis_var,
                original_axis_var=s.original_axis_var,
            )
            out.extend(_distribute_one_cluster(child))
        else:
            out.append(_walk_stmt(s))
    return out


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Push every CLUSTER axis inside any unroll For it currently wraps
    around. ``serial`` Fors and non-CLUSTER ParallelAxes are left
    alone.

    No-op if ``should_skip_cluster(func)`` — there's no cluster to
    distribute."""
    if should_skip_cluster(func):
        return func
    return MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=_walk_stmts(func.body),
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "DistributeClusterError"]
