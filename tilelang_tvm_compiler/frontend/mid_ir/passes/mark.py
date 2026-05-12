"""pass_2_mark: tag op sites with their lane-fusion role marker.

Why this pass exists
--------------------

After ``fold``, the body is a clean sequence of mid_ir op nodes
(``Dma`` / ``Gemm`` / ``Elementwise`` / ``Reduce`` / ``RawStore``) plus
structure (``For`` for any preserved loops). But there's no per-op
hint about *which* of those sites care about lane fusion.

This pass walks the function and sets ``.marker`` on each op according
to a small fixed table:

    Dma                                → Marker.DMA
    Gemm(kind="btmm")                  → Marker.BTMM
    Gemm(kind="overwrite" / other)     → no marker (per-head, runs inside the lane loop)
    Elementwise                        → Marker.LANE_OP
    Reduce                             → Marker.LANE_OP
    RawStore                           → no marker (pass_3/4 leave it alone)

    Broadcast                          → no marker — it's not a top-level
                                          stmt. Broadcast appears only as
                                          an entry inside ``Elementwise.srcs``
                                          (e.g. ``S[r,c] - M_CURR[r]`` folds
                                          to ``Elementwise(S, [S,
                                          Broadcast(M_CURR, dims=[1])], SUB)``).
                                          The enclosing Elementwise already
                                          carries Marker.LANE_OP, which
                                          covers the whole expression incl.
                                          its broadcast srcs.

That's the entire pass. It does NOT decide which buffers are lane-aware
(pass_3 does that). It does NOT split the grid (pass_3 does that
either). It does NOT wrap anything in Async (pass_4 does that). It
just sets a per-op flag the later passes consult.

Why no LANE_OP exclusion rules
------------------------------

It's tempting to skip Elementwise that operates on already-known
"non-lane" buffers (e.g. an FP scalar update on a buffer the kernel
will keep at full extent). But:

  * we don't know "non-lane" yet — that's pass_3's call
  * conservative marking is safe — pass_4 only wraps marked ops in
    Async, but the wrapping is harmless if the underlying buffers
    happen to not need cluster expansion
  * the wrapping decision lives in pass_4 anyway, where it has access
    to pass_3's grown-buffer info

So mark stays dumb-and-uniform.

Output
------

Returns a new MidFunc with the same shape, but with ``.marker`` set on
every Dma / Gemm[btmm] / Elementwise / Reduce node. The pass is
idempotent — calling it twice yields the same markers.
"""

from __future__ import annotations

from typing import List

from ..ir import (
    Dma, Gemm, Elementwise, Reduce, RawStore, For, Async, MultiLaneOp,
    Broadcast, Marker, MidFunc, Stmt,
    ParallelAxis,
)


class MarkError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Per-node marker assignment
# ---------------------------------------------------------------------------


def _mark_dma(op: Dma) -> Dma:
    # DMA is always a single multi-lane HW instruction.
    return Dma(src=op.src, dst=op.dst, marker=Marker.DMA, can_async=True)


def _mark_gemm(op: Gemm) -> Gemm:
    # btmm: one multi-lane M_BTMM instruction → async.
    # overwrite (per-head): one matmul per lane, runs inside the lane
    # loop → not async.
    is_btmm = op.kind == "btmm"
    return Gemm(
        a=op.a, b=op.b, c=op.c,
        transpose_a=op.transpose_a, transpose_b=op.transpose_b,
        kind=op.kind,
        marker=Marker.BTMM if is_btmm else None,
        can_async=is_btmm,
    )


def _has_broadcast_src(op: Elementwise) -> bool:
    return any(isinstance(s, Broadcast) for s in op.srcs)


def _mark_elementwise(op: Elementwise) -> Elementwise:
    # ``can_async`` marks ops that lower to a single heavyweight HW
    # instruction the control thread can fire and walk past — DMA,
    # systolic matmul, and vector-engine ops over a full tile. Only
    # those genuinely benefit from async dispatch; tagging every
    # scalar / per-row op as async just clutters the IR.
    #
    # Eligible elementwise lowerings:
    #   * VRAM dst, no broadcast src → tile-wide ``V_*_VV`` /
    #     ``V_EXP_V`` / etc. — async-eligible.
    # Excluded:
    #   * FPRAM-scalar dst (``fragment`` allocated at rank 1) → lowers
    #     to a sequence of ``S_*_FP`` per slot. Per-row, fp scalar
    #     instruction, not async-eligible.
    #   * Any elementwise with a Broadcast src (``S[r,c] - M_CURR[r]``)
    #     → lowers to ``row_*_fp`` per row, also not async-eligible.
    is_fpram_dst = (
        op.dst.buffer.scope in ("fragment", "local.fragment", "fragment.fpram")
        and len(op.dst.buffer.shape) == 1
    )
    can_async = (
        not _has_broadcast_src(op)
        and not is_fpram_dst
    )
    return Elementwise(
        dst=op.dst, srcs=op.srcs, op=op.op, axis=op.axis, size=op.size,
        marker=Marker.LANE_OP,
        can_async=can_async,
    )


def _mark_reduce(op: Reduce) -> Reduce:
    # Reduce on PLENA is ``row_reduce_max_at`` / ``row_reduce_sum_at`` —
    # per-row, never async.
    return Reduce(
        dst=op.dst, src=op.src, op=op.op, axis=op.axis,
        marker=Marker.LANE_OP,
        can_async=False,
    )


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk(stmt: Stmt) -> Stmt:
    if isinstance(stmt, Dma):
        return _mark_dma(stmt)
    if isinstance(stmt, Gemm):
        return _mark_gemm(stmt)
    if isinstance(stmt, Elementwise):
        return _mark_elementwise(stmt)
    if isinstance(stmt, Reduce):
        return _mark_reduce(stmt)
    if isinstance(stmt, RawStore):
        return stmt  # pass-through; never gets a marker
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=[_walk(s) for s in stmt.body],
            kind=stmt.kind,
        )
    if isinstance(stmt, ParallelAxis):
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=[_walk(s) for s in stmt.body],
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
        )
    if isinstance(stmt, Async):
        # mark runs before pass_4_async, so we don't expect Async here.
        # But if a caller runs mark twice (idempotency), preserve the
        # wrapper and re-mark its body.
        return Async(body=[_walk(s) for s in stmt.body], scope_id=stmt.scope_id)
    if isinstance(stmt, MultiLaneOp):
        # Likewise: shouldn't show up before pass_6, but be defensive.
        return MultiLaneOp(
            inner=_walk(stmt.inner),
            cluster_axis_names=list(stmt.cluster_axis_names),
            dim_map=dict(stmt.dim_map),
        )
    raise MarkError(f"unhandled stmt type {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Set ``.marker`` on every Dma / Gemm[btmm] / Elementwise /
    Reduce in ``func.body``. Returns a new MidFunc; original is not
    mutated."""
    new_body: List[Stmt] = [_walk(s) for s in func.body]
    return MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=new_body,
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "MarkError"]
