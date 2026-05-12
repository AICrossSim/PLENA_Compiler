"""pass_5_fuse: collapse each Async region into a single MultiLaneOp.

Why this pass exists
--------------------

After ``async_wrap`` + ``view``, each cluster body holds:

  * ``Async(body=[one_can_async_op])`` — one async per op (strict)
  * bare ``can_async=False`` ops (Reduce, Elementwise w/ Broadcast)
  * possibly nested For / cluster / grid

We collapse each Async into one ``MultiLaneOp`` that:

  * carries the underlying op as ``inner``
  * names the enclosing cluster axes via ``cluster_axis_names``
    (outermost-to-innermost order)
  * exposes ``dim_map`` — for each lane-aware (non-global) buffer the
    op references, the list of physical dims that correspond to each
    cluster axis. Today every buffer's cluster dim is physical dim 0
    (pass_4b prepends the phase index there), so dim_map values are
    always ``[0]``. Multi-axis cluster fusion would put extra entries
    in this list.

What stays untouched
--------------------

  * Bare can_async=False ops in the cluster body — these are per-row
    ops that lower to ``for lane in range(cluster)`` at pass_8.
    pass_5 doesn't wrap them; they keep BufferRefs with view_perm.
  * RawStore, For, ParallelAxis structure
  * Buffer shapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..cluster_guard import should_skip_cluster
from ..ir import (
    BufferRef, Broadcast,
    Dma, Gemm, Elementwise, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt,
)


class FuseError(RuntimeError):
    pass


@dataclass
class _ClusterAxis:
    """One enclosing cluster axis as seen by fuse.

    ``phase_name`` is the name of the cluster-phase ParallelAxis (e.g.
    ``"by_phase"``); ``number_name`` is its sibling grid number axis
    (``"by_number"``); ``count`` is the cluster width (lane count).
    ``original_name`` is the user-visible lane axis (e.g. ``"by"``) —
    derived from ``phase_name`` by stripping the ``"_phase"`` suffix,
    matching pass_3_split's naming convention.

    Used by ``_collapse_lane_axis`` to recognise both:
      * Per-lane indices written as ``add(by_phase, mul(by_number, 4))``
        (produced by pass_4b_view for non-global buffers).
      * Bare-string ``"by"`` (kept verbatim for global / global.* refs
        whose indices are never rewritten by view).
    Both forms collapse to ``ranged_slice(mul(by_number, 4), 4)`` so
    multi-lane sync ops read the full cluster's chunk in one go.
    """
    phase_name: str
    number_name: str
    count: int
    original_name: str


# ---------------------------------------------------------------------------
# Cluster stack — track enclosing cluster axes outermost → innermost
# ---------------------------------------------------------------------------


def _collect_op_refs(op) -> List[BufferRef]:
    """Return every BufferRef the op directly references. Used to
    build dim_map."""
    refs: List[BufferRef] = []
    if isinstance(op, Dma):
        refs.extend([op.src, op.dst])
    elif isinstance(op, Gemm):
        refs.extend([op.a, op.b, op.c])
    elif isinstance(op, Elementwise):
        refs.append(op.dst)
        for s in op.srcs:
            if isinstance(s, Broadcast):
                refs.append(s.src)
            else:
                refs.append(s)
    elif isinstance(op, Reduce):
        refs.extend([op.dst, op.src])
    elif isinstance(op, RawStore):
        refs.append(op.dst)
    return refs


def _build_dim_map(op, cluster_axis_names: List[str]) -> Dict[str, List[int]]:
    """For each non-global buffer the op touches, record the physical
    dims that map to each cluster axis (in cluster_axis_names order).

    Today every cluster axis lands at physical dim 0 (pass_4b
    prepends the phase index at index position 0). Multi-axis
    cluster nests would prepend multiple times — outermost cluster
    at dim 0, next at dim 1, etc. The list reflects that ordering.
    """
    n_axes = len(cluster_axis_names)
    out: Dict[str, List[int]] = {}
    for ref in op.list_refs() if hasattr(op, "list_refs") else _collect_op_refs(op):
        if ref.buffer.scope == "global":
            continue
        # The convention: outermost cluster phase is at physical dim 0,
        # next inner at dim 1, ... etc. So dim_map[name] = [0, 1, ...,
        # n_axes-1] in cluster_axis_names' order.
        out[ref.buffer.name] = list(range(n_axes))
    return out


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


def _walk(stmt: Stmt, cluster_stack: List[_ClusterAxis]) -> Stmt:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            if stmt.parent_grid_axis_name is None:
                raise FuseError(
                    f"cluster axis {stmt.axis_name!r} missing "
                    f"parent_grid_axis_name; pass_3_split should have set it"
                )
            # Derive the user-visible original axis name from
            # ``phase_name``: pass_3_split names the cluster phase as
            # ``"{original}_phase"`` and the grid number as
            # ``"{original}_number"``.
            phase = stmt.axis_name
            original = phase[:-len("_phase")] if phase.endswith("_phase") else phase
            new_stack = cluster_stack + [_ClusterAxis(
                phase_name=phase,
                number_name=stmt.parent_grid_axis_name,
                count=stmt.extent,
                original_name=original,
            )]
            return ParallelAxis(
                axis_name=stmt.axis_name,
                extent=stmt.extent,
                body=[_walk(s, new_stack) for s in stmt.body],
                kind=stmt.kind,
                thread_tag=stmt.thread_tag,
                parent_grid_axis_name=stmt.parent_grid_axis_name,
            )
        return ParallelAxis(
            axis_name=stmt.axis_name,
            extent=stmt.extent,
            body=[_walk(s, cluster_stack) for s in stmt.body],
            kind=stmt.kind,
            thread_tag=stmt.thread_tag,
            parent_grid_axis_name=stmt.parent_grid_axis_name,
        )
    if isinstance(stmt, For):
        return For(
            loop_var=stmt.loop_var,
            extent=stmt.extent,
            body=[_walk(s, cluster_stack) for s in stmt.body],
            kind=stmt.kind,
        )
    if isinstance(stmt, Async):
        return _fuse_async(stmt, cluster_stack)
    if isinstance(stmt, MultiLaneOp):
        # Already fused (idempotency). Recurse into inner just in case
        # — though typically inner is a leaf.
        return stmt
    # Leaf: pass through.
    return stmt


def _collapse_lane_axis(idx, axes: List[_ClusterAxis]):
    """Fold a per-lane index expression back into a cluster-wide
    ``ranged_slice``.

    pass_4b_view turns the original lane var (e.g. ``"by"``) into
    ``add(phase, mul(number, count))`` — that's the correct per-lane
    expression for op kinds that fire once per lane (Reduce, Broadcast
    Elementwise). For multi-lane ops (Async-wrapped DMA / btmm / pure
    Elementwise) the cluster fires the op exactly once across all
    lanes, so the same axis position should describe a span of
    ``count`` consecutive lane indices starting at the cluster's base
    — encoded as ``ranged_slice(mul(number, count), count)``.

    Match the exact shape produced by ``_subst_lane_var``:
        ``{"op": "add", "args": [phase_name_str,
                                 {"op": "mul", "args":
                                  [number_name_str, count_int]}]}``
    OR a bare ``"by"`` string (kept on global / global.* refs whose
    indices view skipped). Anything else is left alone.
    """
    if isinstance(idx, str):
        for ax in axes:
            if idx == ax.original_name:
                return {
                    "op": "ranged_slice",
                    "args": [
                        {"op": "mul", "args": [ax.number_name, ax.count]},
                        ax.count,
                    ],
                }
        return idx
    if not isinstance(idx, dict):
        return idx
    if idx.get("op") == "add":
        args = idx.get("args", [])
        if len(args) == 2 and isinstance(args[0], str):
            phase = args[0]
            inner = args[1]
            if (isinstance(inner, dict) and inner.get("op") == "mul"):
                m_args = inner.get("args", [])
                if (len(m_args) == 2 and isinstance(m_args[0], str)
                        and isinstance(m_args[1], int)):
                    number, count = m_args[0], m_args[1]
                    for ax in axes:
                        if (ax.phase_name == phase
                                and ax.number_name == number
                                and ax.count == count):
                            return {
                                "op": "ranged_slice",
                                "args": [
                                    {"op": "mul", "args": [number, count]},
                                    count,
                                ],
                            }
    # Recurse into children — the lane composite may live deep inside
    # a compound (e.g. mul(by_expr, stride)).
    return {
        "op": idx.get("op"),
        "args": [_collapse_lane_axis(a, axes) for a in idx.get("args", [])],
    }


def _collapse_ref(ref: BufferRef, axes: List[_ClusterAxis]) -> BufferRef:
    """Apply ``_collapse_lane_axis`` to every index of a user-declared
    global ref (``global`` HBM or ``global.vram`` / ``global.mram`` /
    ``global.fpram`` on-chip caches).

    Non-global refs already had their lane axis baked into the
    prepended phase by pass_4b_view, so we only rewrite globals. For
    on-chip globals (``global.*``), the kernel author still indexes
    them with the un-split logical lane var ``by`` — fuse_pass widens
    that to a cluster-wide ``ranged_slice`` so the multi-lane sync
    wrap reads the full cluster's chunk (by_phase drops out)."""
    if not (ref.buffer.scope == "global"
            or ref.buffer.scope.startswith("global.")):
        return ref
    return BufferRef(
        buffer=ref.buffer,
        indices=[_collapse_lane_axis(i, axes) for i in ref.indices],
        view_perm=ref.view_perm,
    )


def _collapse_src(src, axes: List[_ClusterAxis]):
    if isinstance(src, Broadcast):
        return Broadcast(
            src=_collapse_ref(src.src, axes),
            broadcast_dims=list(src.broadcast_dims),
        )
    return _collapse_ref(src, axes)


def _collapse_lane_in_op(op, axes: List[_ClusterAxis]):
    """Rebuild ``op`` with HBM refs widened to cluster-wide ranged
    slices. Only HBM refs are touched; on-chip refs are unchanged."""
    if isinstance(op, Dma):
        return Dma(
            src=_collapse_ref(op.src, axes),
            dst=_collapse_ref(op.dst, axes),
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Gemm):
        return Gemm(
            a=_collapse_ref(op.a, axes),
            b=_collapse_ref(op.b, axes),
            c=_collapse_ref(op.c, axes),
            transpose_a=op.transpose_a,
            transpose_b=op.transpose_b,
            kind=op.kind,
            marker=op.marker,
            can_async=op.can_async,
        )
    if isinstance(op, Elementwise):
        return Elementwise(
            dst=_collapse_ref(op.dst, axes),
            srcs=[_collapse_src(s, axes) for s in op.srcs],
            op=op.op,
            axis=op.axis,
            size=op.size,
            marker=op.marker,
            can_async=op.can_async,
        )
    return op


def _fuse_async(stmt: Async, cluster_stack: List[_ClusterAxis]) -> Stmt:
    """One Async → one MultiLaneOp. Async body must hold exactly one
    op (the strict one-async-one-op invariant from pass_4).

    During fusion we also collapse the per-lane index expressions
    inside the inner op's HBM refs back into cluster-wide ranged
    slices, since the resulting MultiLaneOp fires once across all
    lanes (not once per lane).
    """
    if not cluster_stack:
        raise FuseError(
            f"Async #{stmt.scope_id} found outside any cluster — "
            f"this shouldn't happen if pass_4_async ran first"
        )
    if len(stmt.body) != 1:
        raise FuseError(
            f"Async #{stmt.scope_id} must hold exactly one op "
            f"(got {len(stmt.body)}); pass_4 enforces one-async-one-op"
        )
    inner = stmt.body[0]
    if isinstance(inner, (Async, MultiLaneOp, ParallelAxis, For)):
        raise FuseError(
            f"Async #{stmt.scope_id} body must be a leaf op, got "
            f"{type(inner).__name__}"
        )
    inner = _collapse_lane_in_op(inner, cluster_stack)
    axis_names = [ax.phase_name for ax in cluster_stack]
    return MultiLaneOp(
        inner=inner,
        cluster_axis_names=axis_names,
        dim_map=_build_dim_map(inner, axis_names),
    )


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc) -> MidFunc:
    """Collapse Async regions into MultiLaneOp nodes."""
    if should_skip_cluster(func):
        return func
    return MidFunc(
        name=func.name,
        params=list(func.params),
        allocs=list(func.allocs),
        body=[_walk(s, cluster_stack=[]) for s in func.body],
        lane_axes=list(func.lane_axes),
        cluster_counts=list(func.cluster_counts),
        attrs=dict(func.attrs),
    )


__all__ = ["run", "FuseError"]
