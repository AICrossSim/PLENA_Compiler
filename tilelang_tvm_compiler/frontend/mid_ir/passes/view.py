"""pass_4b_view: assign view_perm to every BufferRef in cluster scope;
substitute the lane axis var on HBM refs; check global view consistency.

Why this pass exists
--------------------

After ``split``, on-chip buffers got a LANE outer dim (physical
shape: ``(LANE, ..., D)``), but BufferRef.indices weren't updated —
they still address the pre-grow rank. After ``async_wrap``, can_async
ops are wrapped in Async regions but indices are still untouched.

This pass does three things, all op-local + read-only on structure:

  1. **Assign view_perm to each non-global BufferRef** by looking up
     a static rule table keyed on (op-kind, ref-position).
       * BTMM output (S_loc) and per-head LHS (S_loc as P @ V LHS) →
         BHSD = lane stays at physical dim 0 (identity perm).
       * Everything else → BSHD = lane permuted to logical dim 1
         (perm ``[1, 0, ..., N-1]``).
  2. **Prepend the cluster phase index** to each non-global ref's
     indices, growing the ref from rank N to rank N+1 to match the
     buffer's new physical rank. The prepended slot is ``cluster.axis_name``
     (e.g. ``"by_phase"``). NB: this addresses the *physical* dim 0,
     not the logical view dim 0 — view_perm is applied on top.
  3. **Substitute the lane axis var on HBM refs** — any IndexExpr
     equal to the original lane axis name (e.g. ``"by"``) becomes
     the composite ``cluster_phase + grid_number * cluster_count``.
     HBM buffers are NOT rank-grown (they're global) and don't get
     view_perm.

Global view consistency
-----------------------

After per-ref assignment, the pass walks every BufferRef once more
and checks: for each buffer, all view_perms must be identical. If
two ops disagree on a buffer's view, raise ``ViewConflictError`` —
the kernel author has to refactor (no auto-reshape today).

What this pass does NOT touch
-----------------------------

  * RawStore (lives outside cluster contexts; left alone)
  * For / Async / ParallelAxis structure
  * Buffer shapes
  * Op kind / marker / can_async
  * Anything outside a cluster body
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..cluster_guard import should_skip_cluster, MLEN
from ..ir import (
    BufferDef, BufferRef, Slice,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


class ViewError(RuntimeError):
    pass


class ViewConflictError(ViewError):
    pass


# ---------------------------------------------------------------------------
# Roles → view kind
# ---------------------------------------------------------------------------

# Two view shapes for now (rank 3, lane in dim 0 physically):
#   BHSD: lane stays at logical dim 0 (identity perm)
#   BSHD: lane swaps with the next dim → perm [1, 0, 2, ...]


def _identity_perm(rank: int) -> List[int]:
    return list(range(rank))


def _bshd_perm(rank: int) -> List[int]:
    """Swap logical dim 0 (lane) and logical dim 1 (S). Other dims
    untouched. Requires rank >= 2."""
    if rank < 2:
        raise ViewError(f"BSHD requires rank >= 2, got {rank}")
    return [1, 0] + list(range(2, rank))


# ---------------------------------------------------------------------------
# Cluster context (active while we're inside a cluster body)
# ---------------------------------------------------------------------------


@dataclass
class _ClusterCtx:
    phase_name: str            # "by_phase"
    number_name: str            # "by_number"
    cluster_count: int
    original_axis_name: str    # "by" (for HBM lane-var substitution)


def _strip_number_suffix(name: str) -> str:
    if name.endswith("_number"):
        return name[: -len("_number")]
    return name


# ---------------------------------------------------------------------------
# Index rewriting (HBM lane-var substitution)
# ---------------------------------------------------------------------------


def _subst_lane_var(idx, ctx: _ClusterCtx):
    """Recursively rewrite an IndexExpr: any string == original lane
    axis name (e.g. ``"by"``) becomes the composite expression."""
    if isinstance(idx, str) and idx == ctx.original_axis_name:
        return {
            "op": "add",
            "args": [
                ctx.phase_name,
                {"op": "mul", "args": [ctx.number_name, ctx.cluster_count]},
            ],
        }
    if isinstance(idx, dict):
        return {
            "op": idx.get("op"),
            "args": [_subst_lane_var(a, ctx) for a in idx.get("args", [])],
        }
    return idx


# ---------------------------------------------------------------------------
# Per-ref rewrite
# ---------------------------------------------------------------------------


def _rewrite_global_ref(ref: BufferRef, ctx: _ClusterCtx) -> BufferRef:
    """User-declared global tensor.

      * Bare ``global`` (HBM-resident params): substitute the lane var
        in indices so kernel-side ``[..., by, ...]`` accesses the
        correct head slice.
      * ``global.vram`` / ``global.mram`` / ``global.fpram`` (on-chip
        global caches): leave indices verbatim — the kernel author
        indexes them with raw scope-local coordinates; no lane-axis
        substitution and no rank growth.

    Either way, ``view_perm`` stays as the user's declaration."""
    if ref.buffer.scope.startswith("global."):
        return ref
    return BufferRef(
        buffer=ref.buffer,
        indices=[_subst_lane_var(i, ctx) for i in ref.indices],
        view_perm=ref.view_perm,
    )


def _forced_view_kind(buf: BufferDef) -> Optional[str]:
    """Decide a view_kind that the buffer's physical shape demands,
    overriding whatever the op-role table proposes.

    Rules keyed on the innermost dim D (= ``shape[-1]``), with ``shape``
    already post-split (LANE prepended):

      * rank == 2 (``[LANE, rows]`` — no H/D axis at all)  → identity
        perm regardless of op role.  Row-only state buffers (M_OLD,
        P_SUM, etc.) live here.
      * D % MLEN == 0  → no force; op-role choice is fine (S dim spans
        a full lane width, so head-pack vs head-stack are equivalent).
      * 0 < D < MLEN   → must be BSHD; the innermost dim is sub-lane
        (typically hlen) so heads have to be col-packed.
    """
    rank = len(buf.shape)
    if rank <= 2:
        return "IDENTITY"
    d = buf.shape[-1]
    if isinstance(d, int):
        if d % MLEN == 0:
            return None
        if 0 < d < MLEN:
            return "BSHD"
    return None


def _rewrite_lane_ref(ref: BufferRef, ctx: _ClusterCtx,
                      view_kind: str) -> BufferRef:
    """Non-global ref: prepend cluster phase to indices, set view_perm.

    ``view_kind`` is "BHSD" (identity) or "BSHD" (swap dim 0/1) from the
    op-role table. The buffer's physical shape may override this (e.g.
    row-only buffers force identity, sub-MLEN D forces BSHD) — see
    ``_forced_view_kind``.
    """
    new_indices = [ctx.phase_name] + list(ref.indices)
    rank = len(new_indices)
    forced = _forced_view_kind(ref.buffer)
    effective = forced if forced is not None else view_kind
    if effective == "BHSD" or effective == "IDENTITY":
        perm = _identity_perm(rank)
    elif effective == "BSHD":
        perm = _bshd_perm(rank)
    else:
        raise ViewError(f"unknown view_kind {effective!r}")
    return BufferRef(
        buffer=ref.buffer,
        indices=new_indices,
        view_perm=perm,
    )


def _rewrite_ref(ref: BufferRef, ctx: _ClusterCtx,
                 view_kind: str) -> BufferRef:
    # ``global`` / ``global.vram`` / ``global.mram`` / ``global.fpram``:
    # user-declared global tensors keep their declared rank verbatim.
    # No phase axis prepended, no view_perm, no lane-axis substitution
    # of any kind (the kernel author indexes them with raw scope-local
    # coordinates).
    if ref.buffer.scope == "global" or ref.buffer.scope.startswith("global."):
        return _rewrite_global_ref(ref, ctx)
    return _rewrite_lane_ref(ref, ctx, view_kind)


def _rewrite_src(src, ctx: _ClusterCtx, view_kind: str):
    """Wrap-aware rewrite: Broadcast carries a BufferRef inside +
    broadcast_dims that point into dst's logical rank. Since dst rank
    grows by 1 (the prepended phase index), broadcast_dims need to
    shift by 1 too — but only if the original dim index was at or
    after dim 0 in dst's logical (post-prepend, post-view) shape.
    Simpler: bump every dim by 1."""
    if isinstance(src, Broadcast):
        new_dims = [d + 1 for d in src.broadcast_dims]
        return Broadcast(
            src=_rewrite_ref(src.src, ctx, view_kind),
            broadcast_dims=new_dims,
        )
    return _rewrite_ref(src, ctx, view_kind)


# ---------------------------------------------------------------------------
# BHSD buffer set — pre-scan
# ---------------------------------------------------------------------------


def _collect_bhsd_buffers(stmts) -> set:
    """Return the set of buffer names that MUST be BHSD because some
    op produces or consumes them in BHSD form.

    Today the BHSD-anchored ops are:
      * Gemm[btmm].c   — BTMM output buffer
      * Gemm[overwrite].a — per-head matmul LHS

    Once a buffer is anchored to BHSD by one of those ops, every other
    op touching the same buffer (Reduce, Elementwise w/ Broadcast,
    even pure Elementwise) must also use BHSD on that buffer to keep
    the global view consistent.
    """
    out: set = set()

    def visit(s):
        if isinstance(s, Gemm):
            if s.kind == "btmm" and s.c.buffer.scope != "global":
                out.add(s.c.buffer.name)
            if s.kind == "overwrite" and s.a.buffer.scope != "global":
                out.add(s.a.buffer.name)
        if isinstance(s, (ParallelAxis, For, Async)):
            for c in s.body:
                visit(c)
            return
        if isinstance(s, MultiLaneOp):
            visit(s.inner)
            return

    for s in stmts:
        visit(s)
    return out


# ---------------------------------------------------------------------------
# Op rewrite — picks per-ref view_kind from a static rule table
# ---------------------------------------------------------------------------


# Convention: every non-global ref is BSHD by default, except the
# specific (op, position) pairs listed here (which want BHSD).
_BHSD_POSITIONS = {
    # BTMM output
    ("Gemm[btmm]", "c"),
    # per-head matmul LHS
    ("Gemm[overwrite]", "a"),
}


def _gemm_kind_key(op: Gemm) -> str:
    return f"Gemm[{op.kind}]"


def _view_kind_for(op_key: str, position: str) -> str:
    return "BHSD" if (op_key, position) in _BHSD_POSITIONS else "BSHD"


def _rewrite_op(op, ctx: _ClusterCtx, bhsd_buffers: set):
    if isinstance(op, Dma):
        return Dma(
            src=_rewrite_ref(op.src, ctx, _view_kind_for("Dma", "src")),
            dst=_rewrite_ref(op.dst, ctx, _view_kind_for("Dma", "dst")),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Gemm):
        key = _gemm_kind_key(op)
        return Gemm(
            a=_rewrite_ref(op.a, ctx, _view_kind_for(key, "a")),
            b=_rewrite_ref(op.b, ctx, _view_kind_for(key, "b")),
            c=_rewrite_ref(op.c, ctx, _view_kind_for(key, "c")),
            transpose_a=op.transpose_a,
            transpose_b=op.transpose_b,
            kind=op.kind,
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Elementwise):
        # View follows the dst buffer's anchor: if dst was anchored to
        # BHSD by a BTMM output (or per-head LHS), elementwise must
        # honor that. Otherwise default to BSHD — both pure ew (v_add)
        # and broadcast ew (row_*_fp_at) accept BSHD freely.
        view = "BHSD" if op.dst.buffer.name in bhsd_buffers else "BSHD"
        return Elementwise(
            dst=_rewrite_ref(op.dst, ctx, view),
            srcs=[_rewrite_src(s, ctx, view) for s in op.srcs],
            op=op.op,
            axis=op.axis,
            size=op.size,
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Reduce):
        # Reduce src is what determines layout (the row-reducible
        # buffer). If src is BHSD-anchored (BTMM output) → BHSD; else
        # BSHD.
        view = "BHSD" if op.src.buffer.name in bhsd_buffers else "BSHD"
        return Reduce(
            dst=_rewrite_ref(op.dst, ctx, view),
            src=_rewrite_ref(op.src, ctx, view),
            op=op.op,
            axis=op.axis,
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, RawStore):
        # RawStore is opaque; don't rewrite. (And it shouldn't appear
        # inside a cluster anyway; pass_4 doesn't wrap it.)
        return op
    raise ViewError(f"unhandled op type {type(op).__name__}")


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk(stmt: Stmt, ctx: Optional[_ClusterCtx], bhsd_buffers: set) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            if stmt.parent_grid_axis_name is None:
                raise ViewError(
                    f"cluster {stmt.axis_name!r} has no parent_grid_axis_name"
                )
            new_ctx = _ClusterCtx(
                phase_name=stmt.axis_name,
                number_name=stmt.parent_grid_axis_name,
                cluster_count=stmt.extent,
                original_axis_name=_strip_number_suffix(
                    stmt.parent_grid_axis_name),
            )
            return ParallelAxis(
                axis_name=stmt.axis_name,
                extent=stmt.extent,
                body=[_walk(s, new_ctx, bhsd_buffers) for s in stmt.body],
                kind=stmt.kind,
                thread_tag=stmt.thread_tag,
                parent_grid_axis_name=stmt.parent_grid_axis_name,
            )
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=[_walk(s, ctx, bhsd_buffers) for s in stmt.body],
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
        )
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=[_walk(s, ctx, bhsd_buffers) for s in stmt.body],
            kind=stmt.kind,
        )
    if isinstance(stmt, Async):
        return Async(
            body=[_walk(s, ctx, bhsd_buffers) for s in stmt.body],
            scope_id=stmt.scope_id,
        )
    if isinstance(stmt, MultiLaneOp):
        return stmt
    # Leaf op.
    if ctx is None:
        return stmt
    return _rewrite_op(stmt, ctx, bhsd_buffers)


# ---------------------------------------------------------------------------
# Global view consistency check
# ---------------------------------------------------------------------------


def _collect_views(stmt: Stmt,
                   table: Dict[str, List[Tuple[Optional[List[int]], str]]]
                   ) -> None:
    """Walk; for every BufferRef record (view_perm, op_label) under the
    buffer's name. Only refs to non-global buffers tracked."""
    def visit_ref(ref: BufferRef, label: str) -> None:
        if ref.buffer.scope == "global":
            return
        table.setdefault(ref.buffer.name, []).append(
            (tuple(ref.view_perm) if ref.view_perm is not None else None, label)
        )

    def visit_src(src, label: str) -> None:
        if isinstance(src, Broadcast):
            visit_ref(src.src, label)
        else:
            visit_ref(src, label)

    if isinstance(stmt, Dma):
        visit_ref(stmt.src, "Dma.src")
        visit_ref(stmt.dst, "Dma.dst")
    elif isinstance(stmt, Gemm):
        visit_ref(stmt.a, f"Gemm[{stmt.kind}].a")
        visit_ref(stmt.b, f"Gemm[{stmt.kind}].b")
        visit_ref(stmt.c, f"Gemm[{stmt.kind}].c")
    elif isinstance(stmt, Elementwise):
        visit_ref(stmt.dst, "Elementwise.dst")
        for i, s in enumerate(stmt.srcs):
            visit_src(s, f"Elementwise.src[{i}]")
    elif isinstance(stmt, Reduce):
        visit_ref(stmt.dst, "Reduce.dst")
        visit_ref(stmt.src, "Reduce.src")
    elif isinstance(stmt, ParallelAxis):
        for s in stmt.body:
            _collect_views(s, table)
    elif isinstance(stmt, For):
        for s in stmt.body:
            _collect_views(s, table)
    elif isinstance(stmt, Async):
        for s in stmt.body:
            _collect_views(s, table)
    elif isinstance(stmt, MultiLaneOp):
        _collect_views(stmt.inner, table)
    # RawStore not inspected — its refs are opaque to view rules.


def _check_global_consistency(func: MidFunc) -> None:
    table: Dict[str, List[Tuple[Optional[List[int]], str]]] = {}
    for s in func.body:
        _collect_views(s, table)
    for buf_name, entries in table.items():
        # Drop entries with None perm (they came from outside cluster
        # — never rewritten by this pass; not relevant to consistency
        # within cluster scope).
        in_cluster = [(p, l) for (p, l) in entries if p is not None]
        if not in_cluster:
            continue
        first_perm, first_label = in_cluster[0]
        for perm, label in in_cluster[1:]:
            if perm != first_perm:
                raise ViewConflictError(
                    f"buffer {buf_name!r} has conflicting view perms: "
                    f"{list(first_perm)} (from {first_label}) vs "
                    f"{list(perm)} (from {label}). "
                    f"Refactor the kernel to use two separate buffers, "
                    f"or change the op's role table."
                )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Assign view_perm + align ranks + substitute HBM lane var.
    Errors on globally-inconsistent views."""
    if should_skip_cluster(func):
        return func
    bhsd_buffers = _collect_bhsd_buffers(func.body)
    new_body = [_walk(s, ctx=None, bhsd_buffers=bhsd_buffers) for s in func.body]
    new_func = MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=new_body,
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )
    _check_global_consistency(new_func)
    return new_func


__all__ = ["run", "ViewError", "ViewConflictError"]
