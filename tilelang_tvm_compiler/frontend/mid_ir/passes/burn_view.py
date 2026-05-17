"""pass_5b_burn_view: bake each BufferRef.view_perm into the buffer's
physical shape and into every ref's index tuple.

Why this pass exists
--------------------

In the cluster pipeline (pass_4b), each non-global BufferRef carries
a ``view_perm`` describing the op-local view of an underlying physical
buffer. The view is "soft" — the BufferDef.shape is the storage shape
(BHSD shell with lane at physical dim 0), but ops see it permuted
(BSHD with lane at logical dim 1, etc).

By the time we lower to HLIR, the soft view has to become hard:

  * Buffer.shape needs to be a single physical layout (HLIR has no
    notion of per-ref view).
  * Indices that referenced the buffer have to match the new physical
    dim order.

This pass does that bake. For each lane-aware buffer:

  1. Collect all refs of that buffer; gather their view_perm.
  2. Verify they're all identical (pass_4b's consistency check
     guarantees this; we re-verify for safety).
  3. If non-identity: permute Buffer.shape and every ref's indices
     via that perm.
  4. Reset every ref's view_perm to None — the perm is now baked.

After this pass: ``view_perm == None`` on every ref. HLIR lowering
treats Buffer.shape as authoritative.

What's left untouched
---------------------

  * HBM (global) buffers — view_perm was always None on those.
  * Buffers in a MidFunc skipped by cluster_guard.
  * BufferDef objects in MidFunc.params / .allocs — replaced with
    the permuted version. ``BufferRef.buffer`` references get
    swapped to point at the permuted def.
  * RawStore.dst — opaque, no view rewriting.
  * broadcast_dims — index *into* dst's logical shape; since the
    bake just relabels physical dims to match the existing logical
    view, broadcast_dims stay the same (they always referred to the
    logical view).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..cluster_guard import should_skip_cluster
from ..ir import (
    AxisRole, AxisInfo,
    BufferDef, BufferRef, Slice,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


class BurnViewError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Phase 1: collect per-buffer view_perm
# ---------------------------------------------------------------------------


def _identity_perm(rank: int) -> List[int]:
    return list(range(rank))


def _is_identity(perm: List[int]) -> bool:
    return perm == _identity_perm(len(perm))


def _collect_views(stmt: Stmt, table: Dict[str, List[Tuple[int, ...]]]) -> None:
    """Walk; for every BufferRef record its view_perm under buffer name.
    Skips global buffers and refs with view_perm=None."""

    def visit_ref(ref: BufferRef) -> None:
        # User-declared globals (HBM + on-chip ``global.*`` caches)
        # keep their as-written shape — burn_view never touches them.
        if ref.buffer.scope == "global" or ref.buffer.scope.startswith("global."):
            return
        if ref.view_perm is None:
            return
        table.setdefault(ref.buffer.name, []).append(tuple(ref.view_perm))

    def visit_src(src) -> None:
        if isinstance(src, Broadcast):
            visit_ref(src.src)
        else:
            visit_ref(src)

    if isinstance(stmt, Dma):
        visit_ref(stmt.src)
        visit_ref(stmt.dst)
    elif isinstance(stmt, Gemm):
        visit_ref(stmt.a)
        visit_ref(stmt.b)
        visit_ref(stmt.c)
    elif isinstance(stmt, Elementwise):
        visit_ref(stmt.dst)
        for s in stmt.srcs:
            visit_src(s)
    elif isinstance(stmt, Reduce):
        visit_ref(stmt.dst)
        visit_ref(stmt.src)
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
    # RawStore: opaque, skip.


def _agreed_perms(table: Dict[str, List[Tuple[int, ...]]]
                  ) -> Dict[str, Tuple[int, ...]]:
    """For each buffer, verify all collected perms agree. Returns
    name → single perm. Raises on mismatch."""
    out: Dict[str, Tuple[int, ...]] = {}
    for name, perms in table.items():
        first = perms[0]
        for p in perms[1:]:
            if p != first:
                raise BurnViewError(
                    f"buffer {name!r} has inconsistent view perms after "
                    f"pass_4b: {set(perms)}. pass_4b should have caught this."
                )
        out[name] = first
    return out


# ---------------------------------------------------------------------------
# Phase 2: build permuted BufferDefs
# ---------------------------------------------------------------------------


def _permute_buffer(buf: BufferDef, perm: Tuple[int, ...]) -> BufferDef:
    if len(perm) != len(buf.shape):
        raise BurnViewError(
            f"buffer {buf.name!r} rank {len(buf.shape)} doesn't match "
            f"perm rank {len(perm)}"
        )
    # Track the cluster axis through the permutation: it lands at the
    # new position whose ``perm[i]`` equals the old cluster_dim.
    new_cluster: Optional[int] = None
    if buf.cluster_dim is not None:
        for new_i, old_i in enumerate(perm):
            if old_i == buf.cluster_dim:
                new_cluster = new_i
                break
    return BufferDef(
        name=buf.name,
        shape=[buf.shape[i] for i in perm],
        dtype=buf.dtype,
        scope=buf.scope,
        cluster_dim=new_cluster,
    )


def _build_permuted_defs(func: MidFunc,
                         perms: Dict[str, Tuple[int, ...]]) -> Dict[str, BufferDef]:
    """Return name → permuted BufferDef for every lane-aware buffer that
    needs a non-identity perm. Identity perms still build a fresh def
    for uniformity (so callers swap to a single canonical def)."""
    out: Dict[str, BufferDef] = {}
    for buf in list(func.params) + list(func.allocs):
        if buf.scope == "global" or buf.scope.startswith("global."):
            continue
        if buf.name not in perms:
            # No ref carried a view (e.g. unused buffer). Leave alone.
            continue
        out[buf.name] = _permute_buffer(buf, perms[buf.name])
    return out


# ---------------------------------------------------------------------------
# Phase 3: rewrite refs (indices + buffer pointer + clear view_perm)
# ---------------------------------------------------------------------------


def _rewrite_ref(ref: BufferRef,
                 new_defs: Dict[str, BufferDef]) -> BufferRef:
    if ref.buffer.scope == "global" or ref.buffer.scope.startswith("global."):
        return ref
    new_def = new_defs.get(ref.buffer.name)
    if new_def is None:
        # Buffer not in the table: nothing to bake.
        return ref
    perm = ref.view_perm
    if perm is None:
        # View was never set (e.g. ref outside cluster). Just swap to
        # the new def to keep the buffer-pointer single-source-of-truth.
        return BufferRef(buffer=new_def, indices=list(ref.indices))
    new_indices = [ref.indices[i] for i in perm]
    return BufferRef(
        buffer=new_def,
        indices=new_indices,
        view_perm=None,    # baked
    )


def _permute_axes(axes: List[AxisInfo], ref: BufferRef) -> List[AxisInfo]:
    """Apply the same view_perm to the per-axis info that burn_view
    applies to ref.indices. ``ref.view_perm`` is the perm; ``axes``
    is in pre-permute (view-pass) order.

    Returns the post-permute list. Length must equal ``len(ref.indices)``;
    we accept and pass through a zero-length / mismatched list as
    a transitional courtesy (so passes not yet axes-aware don't crash
    the bake).
    """
    perm = ref.view_perm
    if perm is None or not axes:
        return list(axes)
    if len(axes) != len(perm):
        # Length mismatch usually indicates a producer pass hasn't been
        # updated yet (axes still empty). Don't permute; downstream
        # consumers will fall back to legacy guess and we'll catch the
        # gap during verification.
        return list(axes)
    return [axes[i] for i in perm]


def _rewrite_src(src, new_defs):
    if isinstance(src, Broadcast):
        return Broadcast(
            src=_rewrite_ref(src.src, new_defs),
            broadcast_dims=list(src.broadcast_dims),
        )
    return _rewrite_ref(src, new_defs)


def _rewrite_op(op, new_defs):
    if isinstance(op, Dma):
        return Dma(
            src=_rewrite_ref(op.src, new_defs),
            dst=_rewrite_ref(op.dst, new_defs),
            src_axes=_permute_axes(op.src_axes, op.src),
            dst_axes=_permute_axes(op.dst_axes, op.dst),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Gemm):
        return Gemm(
            a=_rewrite_ref(op.a, new_defs),
            b=_rewrite_ref(op.b, new_defs),
            c=_rewrite_ref(op.c, new_defs),
            transpose_a=op.transpose_a,
            transpose_b=op.transpose_b,
            kind=op.kind,
            a_axes=_permute_axes(op.a_axes, op.a),
            b_axes=_permute_axes(op.b_axes, op.b),
            c_axes=_permute_axes(op.c_axes, op.c),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Elementwise):
        # Permute per-src axes using each src's own view_perm.
        new_src_axes = []
        for s, sa in zip(op.srcs, op.src_axes or [[]] * len(op.srcs)):
            inner = s.src if isinstance(s, Broadcast) else s
            new_src_axes.append(_permute_axes(sa, inner))
        return Elementwise(
            dst=_rewrite_ref(op.dst, new_defs),
            srcs=[_rewrite_src(s, new_defs) for s in op.srcs],
            op=op.op,
            dst_axes=_permute_axes(op.dst_axes, op.dst),
            src_axes=new_src_axes,
            axis=op.axis,
            size=op.size,
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Reduce):
        return Reduce(
            dst=_rewrite_ref(op.dst, new_defs),
            src=_rewrite_ref(op.src, new_defs),
            op=op.op,
            axis=op.axis,
            dst_axes=_permute_axes(op.dst_axes, op.dst),
            src_axes=_permute_axes(op.src_axes, op.src),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, RawStore):
        return RawStore(
            dst=_rewrite_ref(op.dst, new_defs),
            value=op.value,
        )
    raise BurnViewError(f"unhandled op type {type(op).__name__}")


def _walk(stmt: Stmt, new_defs: Dict[str, BufferDef]) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=[_walk(s, new_defs) for s in stmt.body],
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
            body=[_walk(s, new_defs) for s in stmt.body],
            kind=stmt.kind,
            loop_var_var=stmt.loop_var_var,
        )
    if isinstance(stmt, Async):
        return Async(
            body=[_walk(s, new_defs) for s in stmt.body],
            scope_id=stmt.scope_id,
        )
    if isinstance(stmt, MultiLaneOp):
        return MultiLaneOp(
            inner=_rewrite_op(stmt.inner, new_defs),
            cluster_axis_names=list(stmt.cluster_axis_names),
            cluster_axis_vars=list(stmt.cluster_axis_vars),
            dim_map=dict(stmt.dim_map),
        )
    return _rewrite_op(stmt, new_defs)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Bake view_perm into Buffer shapes + ref indices."""
    if should_skip_cluster(func):
        return func

    # Phase 1+2: gather views, verify, build new defs.
    table: Dict[str, List[Tuple[int, ...]]] = {}
    for s in func.body:
        _collect_views(s, table)
    perms = _agreed_perms(table)
    new_defs = _build_permuted_defs(func, perms)

    if not new_defs:
        # Nothing to bake (e.g. all views were identity / no view set).
        return func

    # Phase 3: rewrite body + replace BufferDefs in params/allocs.
    new_body = [_walk(s, new_defs) for s in func.body]
    new_params = [new_defs.get(b.name, b) for b in func.params]
    new_allocs = [new_defs.get(b.name, b) for b in func.allocs]

    return MidFunc(
        name=func.name,
        params=new_params,
        allocs=new_allocs,
        body=new_body,
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "BurnViewError"]
