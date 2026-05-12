"""pass_3_split: split lane-axis blockIdx into (number, phase); grow
non-global buffers by one cluster outer dim.

What this pass does
-------------------

Two structural changes, both purely additive:

  1. **Split the lane-axis blockIdx ParallelAxis** into a (number, phase)
     pair, both still parallel axes (NOT for loops):

         parallel by in 0..head_count [blockIdx.y, BLOCK_IDX]
             body
        ↓
         parallel by_number in 0..head_count/cluster_count [blockIdx.y, BLOCK_IDX]
             parallel by_phase in 0..cluster_count [CLUSTER]
                 body

     ``by_number`` keeps the BLOCK_IDX kind + ``blockIdx.*`` thread_tag.
     The HW grid dim shrinks but blockIdx stays a HW grid axis.

     ``by_phase`` is the new CLUSTER axis — ``cluster_count`` lanes
     execute the body in lockstep. Mid-IR keeps multi-thread semantics
     here; pass_8_to_plena is the only place we ever flatten parallel
     to a serial for.

  2. **Grow every non-global buffer by one outermost dim** of size
     ``cluster_count``:

         BufferDef(name="Q_sh", shape=[64, 16], scope="shared")
        ↓
         BufferDef(name="Q_sh", shape=[4, 64, 16], scope="shared")

     Every BufferDef referenced from this point forward — params with
     ``scope != "global"`` (rare), allocs, and the ``buffer`` field of
     every BufferRef — is replaced by the grown version.

     The pass does **not** touch ``BufferRef.indices``: an existing
     ``Q_sh[r, c]`` (rank 2) now references a rank-3 buffer with
     mismatched index rank. That's intentional. ``pass_4_async`` /
     ``pass_5_loop`` introduce ``by_phase`` and prepend it to indices
     when wrapping ops in Async regions.

What this pass DOES NOT do
--------------------------

  * No async wrapping (pass_4)
  * No cluster loop introduction inside the body (pass_5)
  * No layout permute / reshape (pass_7)
  * No Stmt-rewriting except for the outer-loop split

Inputs / outputs
----------------

Reads ``MidFunc.lane_axes`` (set by fold from the kernel's
``T.func_attr({"plena.lane_axis": ...})``). Writes
``MidFunc.cluster_counts`` (one entry per lane axis; defaults to LANE).

LANE defaults to 4 (= MLEN/btmm_hlen for the current target). Callable
override via the ``cluster_counts`` argument to ``run`` for tests /
non-default targets.

Multi-axis lane fusion is supported by passing multiple entries in
``lane_axes`` — each gets its own split + matching cluster_count entry.
The split happens outside-in: for ``lane_axes=["q_block", "by"]`` the
final body shape is ``For(q_block_number) → For(q_block_phase) →
For(by_number) → For(by_phase) → ...``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..cluster_guard import should_skip_cluster
from ..ir import (
    BufferDef, BufferRef, Slice,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


_DEFAULT_LANE = 4   # MLEN / btmm_hlen for the current target


class SplitError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Buffer growth
# ---------------------------------------------------------------------------


def _grow_buffer(buf: BufferDef, cluster: int) -> BufferDef:
    """Return a new BufferDef with shape (cluster,) + buf.shape.

    The prepended cluster dim is marked with ``cluster_dim=0`` so any
    later permutation (view / burn_view) can track which axis is the
    lane axis without re-deriving it from shape values.

    Preserves the kernel-author's logical rank in the scope string when
    it carries semantics: a ``fragment`` allocated with a 1D shape is
    per-lane scalar state (M_OLD, P_SUM etc.) and must end up in FPRAM
    even after the lane dim is prepended (post-grow rank is 2). We
    rename its scope to ``"fragment.fpram"`` so ``to_plena._map_scope``
    can route it without having to re-derive the original rank.
    """
    new_scope = buf.scope
    if buf.scope in ("fragment", "local.fragment") and len(buf.shape) == 1:
        new_scope = "fragment.fpram"
    return BufferDef(
        name=buf.name,
        shape=[cluster] + list(buf.shape),
        dtype=buf.dtype,
        scope=new_scope,
        cluster_dim=0,
    )


def _is_lane_aware_buffer(buf: BufferDef) -> bool:
    """Anything not user-declared global gets cluster-grown. ``"global"``
    is HBM and ``"global.vram"`` / ``"global.mram"`` / ``"global.fpram"``
    are on-chip user-managed caches — both keep their as-written shape."""
    return not (buf.scope == "global" or buf.scope.startswith("global."))


# ---------------------------------------------------------------------------
# Buffer remapping helpers
# ---------------------------------------------------------------------------


@dataclass
class _Ctx:
    cluster_counts: List[int]
    lane_axes: List[str]
    # name -> grown BufferDef. Only contains entries for buffers that
    # were grown (lane-aware). Other buffers are left alone.
    grown: Dict[str, BufferDef]


def _swap_buf(buf: BufferDef, ctx: _Ctx) -> BufferDef:
    return ctx.grown.get(buf.name, buf)


def _swap_ref(ref: BufferRef, ctx: _Ctx) -> BufferRef:
    """Replace the Buffer in a BufferRef. Indices are NOT touched —
    a rank-2 ref into a now-rank-3 buffer is intentional; pass_4
    fixes the rank mismatch by prepending ``by_phase`` when it wraps
    an op in Async."""
    new_buf = _swap_buf(ref.buffer, ctx)
    if new_buf is ref.buffer:
        return ref
    return BufferRef(buffer=new_buf, indices=list(ref.indices))


def _swap_src(src, ctx: _Ctx):
    if isinstance(src, Broadcast):
        return Broadcast(
            src=_swap_ref(src.src, ctx),
            broadcast_dims=list(src.broadcast_dims),
        )
    return _swap_ref(src, ctx)


# ---------------------------------------------------------------------------
# Stmt walker — only swaps BufferRefs; structure unchanged
# ---------------------------------------------------------------------------


def _walk_stmt(stmt: Stmt, ctx: _Ctx) -> Stmt:
    if isinstance(stmt, Dma):
        return Dma(
            src=_swap_ref(stmt.src, ctx),
            dst=_swap_ref(stmt.dst, ctx),
            marker=stmt.marker,
            can_async=stmt.can_async,
        )
    if isinstance(stmt, Gemm):
        return Gemm(
            a=_swap_ref(stmt.a, ctx),
            b=_swap_ref(stmt.b, ctx),
            c=_swap_ref(stmt.c, ctx),
            transpose_a=stmt.transpose_a,
            transpose_b=stmt.transpose_b,
            kind=stmt.kind,
            marker=stmt.marker,
            can_async=stmt.can_async,
        )
    if isinstance(stmt, Elementwise):
        return Elementwise(
            dst=_swap_ref(stmt.dst, ctx),
            srcs=[_swap_src(s, ctx) for s in stmt.srcs],
            op=stmt.op,
            axis=stmt.axis,
            size=stmt.size,
            marker=stmt.marker,
            can_async=stmt.can_async,
        )
    if isinstance(stmt, Reduce):
        return Reduce(
            dst=_swap_ref(stmt.dst, ctx),
            src=_swap_ref(stmt.src, ctx),
            op=stmt.op,
            axis=stmt.axis,
            marker=stmt.marker,
            can_async=stmt.can_async,
        )
    if isinstance(stmt, RawStore):
        return RawStore(
            dst=_swap_ref(stmt.dst, ctx),
            value=stmt.value,
        )
    if isinstance(stmt, ParallelAxis):
        return _split_or_walk_parallel(stmt, ctx)
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=[_walk_stmt(s, ctx) for s in stmt.body],
            kind=stmt.kind,
        )
    if isinstance(stmt, Async):
        return Async(
            body=[_walk_stmt(s, ctx) for s in stmt.body],
            scope_id=stmt.scope_id,
        )
    if isinstance(stmt, MultiLaneOp):
        return MultiLaneOp(
            inner=_walk_stmt(stmt.inner, ctx),
            cluster_axis_names=list(stmt.cluster_axis_names),
            dim_map=dict(stmt.dim_map),
        )
    raise SplitError(f"unhandled stmt type {type(stmt).__name__}")


def _split_or_walk_parallel(stmt: ParallelAxis, ctx: _Ctx) -> Stmt:
    """If ``stmt`` is a (BLOCK_IDX | LOGICAL_GRID) axis whose name is in
    ``lane_axes``, split it into (number axis, CLUSTER phase axis).
    The number axis preserves the source kind (BLOCK_IDX stays
    BLOCK_IDX, LOGICAL_GRID stays LOGICAL_GRID); the phase axis
    becomes CLUSTER and back-references the number axis name via
    ``parent_grid_axis_name``."""
    splittable_kinds = (ParallelKind.BLOCK_IDX, ParallelKind.LOGICAL_GRID)
    if stmt.kind in splittable_kinds and stmt.axis_name in ctx.lane_axes:
        idx = ctx.lane_axes.index(stmt.axis_name)
        cluster = ctx.cluster_counts[idx]
        if stmt.extent % cluster != 0:
            raise SplitError(
                f"lane axis {stmt.axis_name!r} extent={stmt.extent} is "
                f"not a multiple of cluster_count={cluster}"
            )
        outer_extent = stmt.extent // cluster
        inner_body = [_walk_stmt(s, ctx) for s in stmt.body]
        number_name = f"{stmt.axis_name}_number"
        phase_axis = ParallelAxis(
            axis_name=f"{stmt.axis_name}_phase",
            extent=cluster,
            body=inner_body,
            kind=ParallelKind.CLUSTER,
            thread_tag=None,
            parent_grid_axis_name=number_name,
        )
        number_axis = ParallelAxis(
            axis_name=number_name,
            extent=outer_extent,
            body=[phase_axis],
            kind=stmt.kind,                 # BLOCK_IDX or LOGICAL_GRID
            thread_tag=stmt.thread_tag,     # only set for BLOCK_IDX
            parent_grid_axis_name=None,
        )
        return number_axis

    # Not a lane axis: pass through, recurse into body.
    return ParallelAxis(
        axis_name=stmt.axis_name,
        extent=stmt.extent,
        body=[_walk_stmt(s, ctx) for s in stmt.body],
        kind=stmt.kind,
        thread_tag=stmt.thread_tag,
        parent_grid_axis_name=stmt.parent_grid_axis_name,
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc,
        cluster_counts: Optional[List[int]] = None) -> MidFunc:
    """Split each declared lane-axis blockIdx into (number, phase) and
    grow every non-global buffer by one outer cluster dim.

    ``cluster_counts`` defaults to ``[_DEFAULT_LANE] * len(lane_axes)``.
    Pass an explicit list to override (e.g. for non-default targets or
    multi-axis cluster fusion with different per-axis sizes).

    No-op if ``should_skip_cluster(func)`` (kernel didn't declare any
    lane axis, OR every on-chip buffer's last dim already covers a
    full HW vector).
    """
    if should_skip_cluster(func):
        return func
    if cluster_counts is None:
        cluster_counts = [_DEFAULT_LANE] * len(func.lane_axes)
    if len(cluster_counts) != len(func.lane_axes):
        raise SplitError(
            f"cluster_counts has {len(cluster_counts)} entries but "
            f"lane_axes has {len(func.lane_axes)}; must match"
        )

    # Build the grown-buffer map up front.
    # NOTE: with multiple lane axes we currently apply the **product**
    # of all cluster sizes as a single outermost dim. This matches
    # the only multi-axis case we've seen on paper. If a kernel needs
    # a different layout (e.g. separate dims per axis) the BufferDef
    # growth would change shape; pass_7_perm decides physical placement
    # afterwards.
    cluster_total = 1
    for c in cluster_counts:
        cluster_total *= c

    grown: Dict[str, BufferDef] = {}
    for buf in list(func.params) + list(func.allocs):
        if _is_lane_aware_buffer(buf) and buf.name not in grown:
            grown[buf.name] = _grow_buffer(buf, cluster_total)

    ctx = _Ctx(
        cluster_counts=list(cluster_counts),
        lane_axes=list(func.lane_axes),
        grown=grown,
    )

    new_body: List[Stmt] = [_walk_stmt(s, ctx) for s in func.body]
    new_params = [_swap_buf(b, ctx) for b in func.params]
    new_allocs = [_swap_buf(b, ctx) for b in func.allocs]

    return MidFunc(
        name=func.name,
        params=new_params,
        allocs=new_allocs,
        body=new_body,
        lane_axes=list(func.lane_axes),
        cluster_counts=list(cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "SplitError"]
