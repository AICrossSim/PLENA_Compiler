"""pass_6_to_plena: lower MidFunc → HLIRModule.

This is the only pass that exits the mid-IR domain. Everything before
it stays in mid_ir-native dataclasses; this one walks the (now-baked)
mid_ir and produces the HLIR Buffer/Op/HLIRModule that the legacy
backend (AddressAllocationPass + ISAEmitterPass) consumes unchanged.

What gets lowered
-----------------

* Buffers
    BufferDef.scope                  →  HLIR Buffer.scope
        "global"                          → _scope.HBM
        "shared" / "shared.dyn"           → _scope.VRAM
        "fragment" / "local.fragment"     → _scope.VRAM (rank ≥ 2)
                                          or _scope.FPRAM (rank == 1)
        "global.<phys>"                    → strip prefix
        already a physical scope          → identity
    BufferDef.shape / .dtype          → identity copy
    BufferDef.name                    → identity

* Op nodes
    MultiLaneOp(inner=Dma)            →  Op(kind="dma_h2v" / "dma_v2h"
                                                  / "dma_h2m",
                                            buffer_args=[src_or_slice, dst_or_slice],
                                            scalar_args=[lane_count])
    MultiLaneOp(inner=Gemm[btmm])     →  Op(kind="btmm",
                                            buffer_args=[a, b, c],
                                            scalar_args=[group_heads])
    MultiLaneOp(inner=Elementwise pure) →  Op(kind="v_add" / "v_sub" /
                                                "v_mul" / "v_exp" /
                                                "v_zero" / ...)
                                                (1-D vector op; multi-row
                                                ops are wrapped in an
                                                explicit for-row in HLIR)
    Gemm(kind="overwrite") [bare in cluster] →
                                          for-loop over lane that wraps
                                          one Op(kind="matmul"|"mv") per iter
    Reduce [bare]                     →  for lane: for row: Op("row_reduce_<op>_at")
    Elementwise(broadcast) [bare]     →  for lane: for row: Op("row_<op>_fp_at")
    ParallelAxis(BLOCK_IDX|LOGICAL_GRID) →  Op(kind="for", body=[...])
    ParallelAxis(CLUSTER)             →  unwrapped (its body becomes
                                          flat ops in the enclosing scope;
                                          cluster info already burned into
                                          MultiLaneOp scalar_args / dim_map)
    For(serial / unroll)              →  Op(kind="for") + loop_kind annotation
    RawStore                          →  *not handled here yet*; raises

Auto-dump
---------

Pass 6 also writes a human-readable mid_ir snapshot to disk before
lowering, per the convention used for HLIR (``<name>.hlir.txt``).
File name: ``<name>.midir.txt`` under the supplied ``build_dir``.
Dumping is opt-in via ``build_dir`` argument; pass None to skip
(handy for tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tvm import tir as _tir

from .... import hlir as _hlir
from .... import scope as _scope
from ..cluster_guard import should_skip_cluster
from ..ir import (
    BinOp, UnaryOp, ReduceOp,
    AxisRole, AxisInfo,
    BufferDef, BufferRef, Slice, VarRef,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt, format_func,
)


class ToPlenaError(RuntimeError):
    pass


def _make_loop_var(name: str) -> _tir.Var:
    """Build a tir.Var for use as an HLIR ``for`` loop_var annotation
    from a bare name. Used when no mid_ir VarRef carries the identity
    (e.g. synthetic for-rows that to_plena introduces itself).

    Shares ``_VAR_CACHE`` keyed by name so two calls with the same name
    produce the same tir.Var object (the ISA pass keys ``symbol_table``
    by identity).

    Prefer ``_axis_loop_var(stmt)`` / ``_for_loop_var(stmt)`` when
    lowering a mid_ir ParallelAxis / For: those reuse the identity
    captured during fold so inner BufferRef indices keyed off the same
    var resolve to the same tir.Var object.
    """
    return _get_var(name)


def _axis_loop_var(axis: ParallelAxis) -> _tir.Var:
    """HLIR for-loop_var for ``axis``. Routed through the name cache so
    every reference to this axis (via ``_render_idx_as_primexpr`` on
    matching VarRefs) resolves to the same ``tir.Var`` object the ISA
    pass binds in its ``symbol_table``."""
    return _make_loop_var(axis.axis_name)


def _for_loop_var(for_stmt: For) -> _tir.Var:
    """HLIR for-loop_var for a mid_ir ``For``. Same name-cache routing
    as :func:`_axis_loop_var`."""
    return _make_loop_var(for_stmt.loop_var)


# ---------------------------------------------------------------------------
# Scope mapping
# ---------------------------------------------------------------------------


def _map_scope(scope: str, rank: int,
               override: Optional[str] = None) -> str:
    """mid_ir scope string → HLIR scope string.

    ``override`` is set by the use-driven inference (e.g. a buffer used
    as a Gemm B operand needs MRAM regardless of its declared shared/
    fragment scope). Override wins over the rank-based default.
    """
    if scope == "global":
        return _scope.HBM
    if scope.startswith("global."):
        return _scope.physical_scope(scope)
    if override is not None:
        return override
    if scope in ("shared", "shared.dyn"):
        return _scope.VRAM
    if scope == "fragment.fpram":
        # split pass marks per-lane scalar-state fragments (M_OLD, L_NEW
        # etc., originally rank-1) with this scope so we route them to
        # FPRAM even after the lane dim prepend bumped rank to 2.
        return _scope.FPRAM
    if scope in ("fragment", "local.fragment"):
        # Rank-1 fragments are FPRAM scalar-state (M_OLD, L_NEW etc.);
        # higher-rank fragments stay in VRAM (S_loc, PV_loc).
        return _scope.FPRAM if rank == 1 else _scope.VRAM
    if scope in _scope.PHYSICAL_SCOPES:
        return scope
    raise ToPlenaError(f"unknown mid_ir scope {scope!r}")


# ---------------------------------------------------------------------------
# Buffer construction
# ---------------------------------------------------------------------------


def _pad_to_4d_shape(
    shape: Tuple[int, ...], heads_at_h: bool = False,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Pad a 1D / 2D shape up to canonical 4D for downstream uniformity.

    Distinct from cluster expansion: this is a pure rank-normalisation
    step. It carries no lane / cluster semantics — the inserted axes
    are extent-1 placeholders so that address_alloc / isa_emit see one
    rank everywhere and don't need rank-conditional branches.

    Returns ``(new_4d_shape, inserted_positions)``. Positions index
    into the OUTPUT 4D shape; callers must pad every same-rank
    reference (VramRegion starts / extents) at exactly these
    positions, with ``start=0`` / ``extent=1``.

    Default rule (on-chip scratch):
      * 1D ``(n,)``    -> ``(1, 1, 1, n)``    inserts at (0, 1, 2)
      * 2D ``(a, b)``  -> ``(1, a, 1, b)``    inserts at (0, 2)
      * 4D             -> unchanged           inserts == ()

    ``heads_at_h=True`` (author-pinned ``global.vram``/``global.mram``
    tensor caches whose first axis is the head dim):
      * 2D ``(a, b)``  -> ``(1, 1, a, b)``    inserts at (0, 1)
    """
    rank = len(shape)
    if rank == 4:
        return tuple(int(d) for d in shape), ()
    if rank == 2:
        a, b = int(shape[0]), int(shape[1])
        if heads_at_h:
            return (1, 1, a, b), (0, 1)
        return (1, a, 1, b), (0, 2)
    if rank == 1:
        n = int(shape[0])
        return (1, 1, 1, n), (0, 1, 2)
    raise ToPlenaError(
        f"_pad_to_4d_shape: only 1D/2D/4D supported; got "
        f"rank-{rank} shape={tuple(shape)}"
    )


def _make_hlir_buffer(
    buf: BufferDef,
    override: Optional[str] = None,
    lane_count: Optional[int] = None,
    mode: Optional[str] = None,
    kernel_layout: str = "BSHD",
) -> Tuple[_hlir.Buffer, Tuple[int, ...]]:
    """Build an HLIR ``Buffer`` from a mid_ir ``BufferDef``.

    Returns ``(buffer, inserted_positions)`` — positions in the OUTPUT
    4D shape that ``_pad_to_4d_shape`` synthesised (extent-1). Empty
    tuple when no padding happened (cluster-expanded or already 4D).

    Two routes, both producing 4D on VRAM/MRAM:

      * Cluster-fusion route (``mode != None`` and ``lane_count >= 1``)
        — ``_expand_buffer_shape_with_cluster`` picks BSHD axes per
        lane mode, carries a ``cluster_dim`` to track the lane axis.
      * Pad-to-4D route (``mode == None``, cluster skipped) — pure
        rank normalisation; no cluster semantics, ``cluster_dim``
        stays as whatever the BufferDef had (typically None).

    HBM (``global*``) and FPRAM buffers keep their author-declared
    rank. FPRAM is scalar-addressed (no tile layout); HBM keeps its
    kernel-surface shape for parent-stride math.
    """
    # Resolve the destination physical scope first so the pad-to-4D
    # decision uses the same source of truth the HLIR ``Buffer.scope``
    # field ends up with — rather than guessing from the spelling of
    # the mid_ir scope string (``"local.fragment"`` vs
    # ``"fragment.fpram"`` vs ``"global.fpram"``).
    physical = _map_scope(buf.scope, len(buf.shape), override)
    is_global = _is_global_scope(buf.scope)
    inserts: Tuple[int, ...] = ()
    # Cluster expand only applies to allocatable on-chip buffers. HBM
    # (``"global"``) and author-pinned on-chip globals
    # (``"global.vram"`` / ``"global.mram"`` / ``"global.fpram"``)
    # keep the shape the kernel author wrote — they're explicit
    # tensor-cache regions, not lane-aware scratch — so neither
    # cluster grow nor lane-mode expand may touch them.
    if mode is not None and not is_global and lane_count is not None:
        shape_list, cluster_dim = _expand_buffer_shape_with_cluster(buf, lane_count, mode)
        shape = tuple(shape_list)
    else:
        shape = tuple(int(d) for d in buf.shape)
        cluster_dim = buf.cluster_dim
        # Pad-to-4D for on-chip VRAM / MRAM. Author-pinned globals
        # (``global.vram`` / ``global.mram``) also get padded so every
        # downstream pass sees rank-4, but their pad rule puts heads at
        # H (slot 2) — matches how kernels actually use these tensor
        # caches (head-major (head_count, hlen)). Crucially we DO NOT
        # stamp a ``cluster_dim`` on globals: they aren't cluster-
        # expanded, so a cluster tag would confuse the sync-wrap
        # iterator. HBM (plain ``"global"``) keeps its author rank for
        # parent-stride math; FPRAM is scalar-addressed.
        is_onchip_global = is_global and physical in (_scope.VRAM, _scope.MRAM)
        if physical in (_scope.VRAM, _scope.MRAM) and len(shape) != 4:
            if is_onchip_global:
                shape, inserts = _pad_to_4d_shape(shape, heads_at_h=True)
            elif not is_global:
                shape, inserts = _pad_to_4d_shape(shape)
    # ``plena.layout`` describes the HBM-side physical layout of the
    # kernel's tensor params. On-chip buffers (VRAM/MRAM/FPRAM allocs,
    # and on-chip pad-to-4D synthetic axes) keep the default BSHD —
    # tile_layout machinery interprets their (B,S,H,D) by position.
    # Stamping NCHW onto e.g. ``in_stage`` would mis-assign its synthetic
    # 4D axes (axis-1 isn't a channel dim, it's the row dim by
    # construction) and inflate ``h_groups`` to the row extent.
    buf_layout = kernel_layout if is_global else "BSHD"
    is_pinned = is_global and physical in (_scope.VRAM, _scope.MRAM)
    return (
        _hlir.Buffer(
            name=buf.name,
            scope=physical,
            shape=shape,
            dtype=buf.dtype,
            cluster_dim=cluster_dim,
            layout=buf_layout,
            is_pinned_global=is_pinned,
        ),
        inserts,
    )


# ---------------------------------------------------------------------------
# Use-driven scope overrides
# ---------------------------------------------------------------------------


# Lane-expansion modes for the 4D-BSHD rewrite below. Strings match the
# graph_passes/expand_buffers vocabulary so anyone reading both layers
# sees the same names.
_MODE_COL_PACK = "col_pack"   # H carries lane: (1, S, lane, D_narrow)
_MODE_ROW_STACK = "row_stack" # B carries lane: (lane, S, 1, MLEN)
_MODE_FP_LANE = "fp_lane"     # FPRAM: (lane, N)
_MODE_BSHD_LIFT = "bshd_lift"  # No lane fusion: (1, S, 1, D)


def _is_global_scope(scope: str) -> bool:
    """True for any author-declared global buffer.

    Matches ``"global"`` (HBM) plus the ``global.<phys>`` family
    (``global.vram`` / ``global.mram`` / ``global.fpram``) used by
    kernels to pin a buffer to a specific on-chip cache without
    letting downstream cluster / lane logic re-expand its shape.

    Centralises the check so every pass that needs to skip globals
    uses the same predicate (the bug from flash_decode_min was
    ``_infer_lane_modes`` checking ``scope == "global"`` only, which
    miscategorised ``global.vram`` as lane-aware).
    """
    return scope == "global" or scope.startswith("global.")


def _infer_lane_modes(func: MidFunc) -> Dict[str, str]:
    """For every non-global buffer, decide its lane-expansion mode by
    inspecting how mid_ir ops use it. Mirrors
    ``graph_passes.allocate_group_memory`` but works on mid_ir nodes.

    Priority (first match wins) — ``ROW_STACK`` takes precedence over
    ``COL_PACK`` if a buffer is used as a BTMM dst somewhere.
    """
    modes: Dict[str, str] = {}

    def record(name: str, mode: str) -> None:
        prev = modes.get(name)
        if prev is None:
            modes[name] = mode
            return
        if prev == _MODE_ROW_STACK or mode == _MODE_ROW_STACK:
            modes[name] = _MODE_ROW_STACK
            return
        if prev == _MODE_FP_LANE or mode == _MODE_FP_LANE:
            modes[name] = _MODE_FP_LANE

    def visit_op(op) -> None:
        if isinstance(op, Gemm):
            if op.kind == "btmm":
                if not _is_global_scope(op.a.buffer.scope):
                    record(op.a.buffer.name, _MODE_COL_PACK)
                if not _is_global_scope(op.b.buffer.scope):
                    record(op.b.buffer.name, _MODE_COL_PACK)
                if not _is_global_scope(op.c.buffer.scope):
                    record(op.c.buffer.name, _MODE_ROW_STACK)
            else:
                if not _is_global_scope(op.a.buffer.scope):
                    record(op.a.buffer.name, _MODE_ROW_STACK)
                if not _is_global_scope(op.b.buffer.scope):
                    record(op.b.buffer.name, _MODE_COL_PACK)
                if not _is_global_scope(op.c.buffer.scope):
                    record(op.c.buffer.name, _MODE_COL_PACK)
            return
        if isinstance(op, Dma):
            for ref in (op.src, op.dst):
                if _is_global_scope(ref.buffer.scope):
                    continue
                if ref.buffer.scope == "fragment.fpram":
                    record(ref.buffer.name, _MODE_FP_LANE)
                else:
                    record(ref.buffer.name, _MODE_COL_PACK)
            return
        if isinstance(op, (Elementwise, Reduce)):
            refs = []
            if isinstance(op, Elementwise):
                refs.append(op.dst)
                for s in op.srcs:
                    refs.append(s.src if isinstance(s, Broadcast) else s)
            else:
                refs.extend([op.dst, op.src])
            for ref in refs:
                if _is_global_scope(ref.buffer.scope):
                    continue
                if ref.buffer.scope == "fragment.fpram":
                    record(ref.buffer.name, _MODE_FP_LANE)
                else:
                    record(ref.buffer.name, _MODE_COL_PACK)
            return

    def visit_stmt(s) -> None:
        if isinstance(s, (ParallelAxis, For, Async)):
            for c in s.body:
                visit_stmt(c)
            return
        if isinstance(s, MultiLaneOp):
            visit_op(s.inner)
            return
        visit_op(s)

    for s in func.body:
        visit_stmt(s)

    # Anything left without a mode (allocated but unused by any
    # tracked op) gets the no-lane-fusion catch-all so it still ends
    # up as 4D BSHD downstream.
    for buf in func.allocs:
        if _is_global_scope(buf.scope):
            continue
        if buf.name not in modes:
            modes[buf.name] = (
                _MODE_FP_LANE if buf.scope == "fragment.fpram"
                else _MODE_BSHD_LIFT
            )
    return modes


def _expand_buffer_shape_with_cluster(
    buf: BufferDef, lane_count: int, mode: str,
) -> Tuple[List[int], Optional[int]]:
    """Reshape to canonical 4D BSHD (or 2D for FPRAM) and report which
    axis holds the cluster (lane) dim in the new shape.

    Modes:
      * COL_PACK  → ``(1, rows, lane, D_narrow)`` — cluster_dim = 2
      * ROW_STACK → ``(lane, rows, 1, MLEN)``    — cluster_dim = 0
      * FP_LANE   → ``(lane, N)``                — cluster_dim = 0
      * BSHD_LIFT → ``(1, rows, 1, D)``          — no cluster axis

    Mid-IR pre-expansion shapes (per ``view``/``burn_view`` placement
    of the lane axis recorded in ``buf.cluster_dim``):
      COL_PACK  : ``(rows=shape[0], lane=shape[1], D=shape[2])``
      ROW_STACK : ``(lane=shape[0], rows=shape[1], D=shape[2])``
      BSHD_LIFT : ``(?, rows=shape[1], D=shape[2])``
    """
    if mode == _MODE_FP_LANE:
        if len(buf.shape) != 2:
            raise ToPlenaError(
                f"{buf.name!r}: FP_LANE expansion expects post-grow rank 2 "
                f"(lane, N); got shape={list(buf.shape)}"
            )
        return [int(buf.shape[0]), int(buf.shape[1])], 0
    if len(buf.shape) != 3:
        raise ToPlenaError(
            f"{buf.name!r}: 2D-lane expansion expects post-grow rank 3; "
            f"got shape={list(buf.shape)} mode={mode}"
        )
    if mode == _MODE_ROW_STACK:
        rows = int(buf.shape[1])
        last = int(buf.shape[2])
        return [int(lane_count), rows, 1, last], 0
    if mode == _MODE_COL_PACK:
        # view leaves lane at axis 1; rows is the leading axis.
        rows = int(buf.shape[0])
        last = int(buf.shape[2])
        return [1, rows, int(lane_count), last], 2
    if mode == _MODE_BSHD_LIFT:
        # No lane axis in scope; rows still sits at axis 1 per view.
        rows = int(buf.shape[1])
        last = int(buf.shape[2])
        return [1, rows, 1, last], None
    raise ToPlenaError(f"unknown lane mode {mode!r} for {buf.name!r}")


# Where each cluster mode lands the lane axis in the post-expansion
# 4D BSHD shape. Used to drive ref rewriting from rank 3 to rank 4
# without enumerating per-mode permutations.
#
# This must stay consistent with the new ``cluster_dim`` value
# returned by :func:`_expand_buffer_shape_with_cluster`.
_CLUSTER_MODE_NEW_LANE_DIM: Dict[str, Optional[int]] = {
    _MODE_COL_PACK:  2,    # lane → H
    _MODE_ROW_STACK: 0,    # lane → B
    _MODE_BSHD_LIFT: None, # no lane in scope
}


def _rewrite_ref_for_cluster_mode(
    starts_or_extents: Tuple[Any, ...],
    mode: str,
    old_cluster_dim: Optional[int],
    new_shape: Tuple[int, ...],
    *,
    is_extent: bool,
) -> Tuple[Any, ...]:
    """Map a rank-3 ref to its rank-4 BSHD equivalent.

    Inputs:
      * ``starts_or_extents`` — the rank-3 source tuple from mid_ir.
        For starts the lane-axis slot has already been zeroed under
        the sync-wrap convention (so the value at
        ``old_cluster_dim`` is ``0``, not a Var).
      * ``old_cluster_dim`` — where lane sat in the rank-3 source
        (from ``buf.cluster_dim``).
      * ``new_shape`` — the post-expansion 4D buffer shape; used to
        recover the lane extent (BSHD H or B dim) under sync wrap.

    Sync-wrap semantics on the lane axis (matches
    ``_ref_per_dim_starts`` zero-ing the phase):
      * lane-axis start  -> ``0``
      * lane-axis extent -> ``new_shape[new_lane_dim]`` (the full
                             lane span, not the per-lane ``1``)

    Non-lane source axes keep their relative order and fill the
    remaining non-lane output slots; the leftover output slot is
    inserted with ``0`` / ``1``.
    """
    if mode == _MODE_FP_LANE:
        return starts_or_extents
    if len(starts_or_extents) != 3:
        raise ToPlenaError(
            f"cluster mode {mode!r} expects rank-3 ref tuple, got "
            f"{tuple(starts_or_extents)}"
        )
    if mode not in _CLUSTER_MODE_NEW_LANE_DIM:
        raise ToPlenaError(f"no ref-rewrite rule for cluster mode {mode!r}")
    new_lane_dim = _CLUSTER_MODE_NEW_LANE_DIM[mode]
    fill: Any = 1 if is_extent else 0

    if mode == _MODE_BSHD_LIFT or old_cluster_dim is None or new_lane_dim is None:
        # No lane axis to relocate. mid_ir source is rank-3
        # ``(?, S, D)`` and we lift it to ``(?, S, 1, D)`` —
        # i.e. insert an extent-1 H at position 2.
        a, b, c = starts_or_extents
        return (a, b, fill, c)

    # Anchor non-lane axes to canonical BSHD positions:
    #   * source last axis (always D in mid_ir convention)  → out[3] (D)
    #   * source first non-lane axis (S in mid_ir convention) → out[1] (S)
    # The remaining BSHD slot is whichever of B (0) / H (2) is NOT the
    # lane position; it stays as ``fill``.
    non_lane_sources = [i for i in range(3) if i != old_cluster_dim]
    if len(non_lane_sources) != 2:
        raise ToPlenaError(
            f"unexpected non-lane source count {len(non_lane_sources)}"
        )
    s_src, d_src = non_lane_sources[0], non_lane_sources[1]

    out: List[Any] = [fill] * 4
    # Lane slot: sync-wrap convention (start=0, extent=lane span).
    if is_extent:
        out[new_lane_dim] = int(new_shape[new_lane_dim])
    else:
        out[new_lane_dim] = 0
    out[1] = starts_or_extents[s_src]   # S
    out[3] = starts_or_extents[d_src]   # D
    return tuple(out)


def _expand_buffer_shape(
    buf: BufferDef, lane_count: int, mode: str,
) -> List[int]:
    """Back-compat shape-only wrapper. New code should use
    ``_expand_buffer_shape_with_cluster``."""
    shape, _ = _expand_buffer_shape_with_cluster(buf, lane_count, mode)
    return shape


def _infer_scope_overrides(func: MidFunc) -> Dict[str, str]:
    """Scan body; figure out per-buffer scope overrides driven by op
    usage.

    Today the only override: any buffer used as a Gemm B operand
    (BTMM RHS or per-head matmul RHS) has to live in MRAM. PLENA's
    BTMM/MM hardware reads RHS from MRAM only.

    This propagates through DMA destinations: ``dma_h2v`` lowering is
    chosen by ``_dma_kind_from_scopes`` from the dst's scope, so once
    the dst buffer is tagged MRAM, ``dma_h2m`` will be picked
    automatically.
    """
    overrides: Dict[str, str] = {}

    def visit_op(op) -> None:
        if isinstance(op, Gemm):
            # B operand → MRAM, regardless of how shared/fragment
            # would otherwise default it. Never overrides author-pinned
            # globals (HBM / global.vram / global.mram / global.fpram):
            # those carry an explicit user-side placement contract.
            if not _is_global_scope(op.b.buffer.scope):
                overrides[op.b.buffer.name] = _scope.MRAM

    def visit_stmt(s) -> None:
        if isinstance(s, (ParallelAxis, For, Async)):
            for c in s.body:
                visit_stmt(c)
            return
        if isinstance(s, MultiLaneOp):
            visit_op(s.inner)
            return
        visit_op(s)

    for s in func.body:
        visit_stmt(s)
    return overrides


# ---------------------------------------------------------------------------
# Index expression rendering
# ---------------------------------------------------------------------------


def _render_idx(idx) -> Any:
    """Convert mid_ir IndexExpr into an HLIR-friendly form. Slice maps
    to 0 (whole-axis access starts at 0). Compound dict expressions are
    flattened to a string for the legacy ExprMaterializer to handle —
    pre-existing convention is to pass non-static offsets as PrimExpr
    or string in scalar_args, but here for HBM we only need int-or-str."""
    if isinstance(idx, Slice):
        return 0
    if isinstance(idx, int):
        return int(idx)
    if isinstance(idx, VarRef):
        return idx.name
    if isinstance(idx, str):
        return idx
    if isinstance(idx, dict):
        op = idx.get("op", "?")
        args = idx.get("args", [])
        # ranged_slice carries (start_expr, extent_int): for the HLIR
        # BufferSlice 'starts' field we want the start expression; the
        # extent is recovered separately by _ref_extents.
        if op == "ranged_slice":
            return _render_idx(args[0])
        rendered = [_render_idx(a) for a in args]
        # Render compound exprs as readable Python syntax for now.
        # Legacy ExprMaterializer is what actually materializes them.
        if op == "add":
            return f"({rendered[0]} + {rendered[1]})"
        if op == "sub":
            return f"({rendered[0]} - {rendered[1]})"
        if op == "mul":
            return f"({rendered[0]} * {rendered[1]})"
        if op == "fdiv":
            return f"({rendered[0]} // {rendered[1]})"
        if op == "fmod":
            return f"({rendered[0]} % {rendered[1]})"
        return f"<expr {op} {rendered}>"
    return idx


def _ref_extents(ref: BufferRef) -> Tuple[int, ...]:
    """Extents of the slice this ref describes — full-axis Slices give
    the buffer's full size on that dim; a ranged_slice compound carries
    its own extent; everything else (single scalar index) is 1."""
    out: List[int] = []
    for i, dim in enumerate(ref.buffer.shape):
        idx = ref.indices[i]
        if isinstance(idx, Slice):
            out.append(int(dim))
        elif isinstance(idx, dict) and idx.get("op") == "ranged_slice":
            out.append(int(idx["args"][1]))
        else:
            out.append(1)
    return tuple(out)


def _is_whole_buffer_ref(ref: BufferRef) -> bool:
    """All indices are Slice (no narrowing).

    The view pass prepends a cluster-phase index (a bare string) on
    every non-global ref that went through cluster fusion, and
    burn_view may permute it to a non-zero position. For DMA / btmm /
    pure-elementwise ops that fire across all lanes (wrapped in
    MultiLaneOp) the phase index is a whole-lane-axis access —
    equivalent to Slice for whole-buffer purposes.

    Use the buffer's own ``cluster_dim`` (set by ``split._grow_buffer``
    and propagated through view / burn_view) as the source of truth:
    a bare-string index sitting at that physical position is the
    phase shorthand. When ``cluster_dim`` is None (cluster was
    skipped — kernel has no lane axis, or every on-chip buffer was
    already mlen-wide), no axis is treated as a phase shorthand, and
    we fall back to "all-Slice only". This avoids silently swallowing
    ordinary loop-var narrowings (e.g. ``oc`` in
    ``Output[:, oc, :, :]``) when no cluster is in scope.
    """
    if not ref.indices:
        return True
    if all(isinstance(i, Slice) for i in ref.indices):
        return True
    cdim = getattr(ref.buffer, "cluster_dim", None)
    if cdim is None or not (0 <= cdim < len(ref.indices)):
        return False
    # The cluster-dim slot must be a VarRef phase shorthand; every
    # other slot must be a Slice (real narrowing on a non-phase axis
    # disqualifies whole-buffer treatment).
    cluster_idx = ref.indices[cdim]
    if not isinstance(cluster_idx, VarRef):
        return False
    for i, idx in enumerate(ref.indices):
        if i == cdim:
            continue
        if not isinstance(idx, Slice):
            return False
    return True


# ---------------------------------------------------------------------------
# Op-arg construction
# ---------------------------------------------------------------------------


_INT32 = "int32"

# Cache (name → tir.Var) so HLIR ``for`` ops constructed by to_plena
# (loop_var on synthetic for-rows etc.) and any name-keyed lookups
# resolve to the same Python object. ISA pass identifies bindings by
# object identity in its symbol_table.
#
# This cache services only ``_make_loop_var(name)`` — paths that
# synthesise *new* loop vars (e.g. ``"row0"``, the HLIR for created
# from a mid_ir For). Index expressions from mid_ir come through
# VarRef and bypass this cache entirely (``_render_idx_as_primexpr``
# unwraps the wrapped ``tir.Var`` directly).
_VAR_CACHE: Dict[str, "_tir.Var"] = {}

# Module-global lane modes table, populated by ``run`` at the start of
# each compile and read by per-op lowering helpers. Cleared by ``run``.
_LANE_MODES: Dict[str, str] = {}

# Per-buffer pad-to-4D insert positions. Populated by ``run`` after
# ``_make_hlir_buffer``. Used by ``_axes_for_ref`` to align mid_ir
# per-op axes (rank == mid_ir buffer rank) with the post-pad HLIR
# shape so ``hlir.Op.buffer_axes`` agrees with ``buf.shape``.
_PAD_INSERTS: Dict[str, Tuple[int, ...]] = {}

# Per-buffer cluster-expansion records ``(mode, mid_ir_cluster_dim,
# new_4d_shape)``. The pair to ``_PAD_INSERTS``: any on-chip buffer
# that wasn't pad-to-4D'd was instead grown by a cluster expansion
# (col_pack / row_stack / bshd_lift / fp_lane), and that's recorded
# here so axes-aware helpers can translate mid_ir-side (rank-3) axes
# tables onto the post-expansion 4D HLIR shape.
_CLUSTER_MODES: Dict[
    str, Tuple[str, Optional[int], Tuple[int, ...]],
] = {}

# Per-buffer HLIR-side ``buffer_axes`` (post pad-to-4D / cluster-expand).
# Each entry is ``Tuple[(role_name, extent), ...]`` aligned with the
# HLIR buffer's shape — one role tag per physical dim. Populated by
# ``run`` once the per-buffer mode is known; read by every lowering
# helper that needs to stamp ``buffer_axes`` on its emitted hlir.Op.
_BUFFER_HLIR_AXES: Dict[str, Tuple[Tuple[str, int], ...]] = {}


def _axes_of(buf_name: str) -> Optional[Tuple[Tuple[str, int], ...]]:
    """Look up the per-dim role tuple for ``buf_name`` from the
    ``_BUFFER_HLIR_AXES`` table populated at ``run`` start. Returns
    ``None`` for unknown buffers (e.g. transient names that never
    landed in the HLIR buffer table) so callers can store ``None`` in
    ``buffer_axes`` without crashing."""
    return _BUFFER_HLIR_AXES.get(buf_name)


def _hlir_axes_for_buffer(buf: "_hlir.Buffer") -> Tuple[Tuple[str, int], ...]:
    """Synthesise the ``buffer_axes`` tuple for ``buf`` from its
    post-expansion shape + ``cluster_dim``.

    Role assignment per physical dim:
      * the innermost dim (axis ``rank-1``) is the SIMD / D / N axis
        — tagged ``"simd"``.
      * the cluster dim (``buf.cluster_dim``, if set) is the lane
        axis — tagged ``"cluster"``.
      * every other dim is a row-fanout axis — tagged ``"batch"``.
        Both the leading B=1 placeholder (pad-to-4D) and the real S
        (rows) end up as ``"batch"``; downstream callers can pick the
        non-degenerate one via ``extent != 1`` if needed, but most
        row-at consumers just want "which dim is rows" and that's the
        ``"batch"`` dim with the largest extent.
    """
    shape = [int(d) for d in buf.shape]
    rank = len(shape)
    if rank == 0:
        return ()
    d_axis = rank - 1
    cdim = buf.cluster_dim
    out: List[Tuple[str, int]] = []
    for i in range(rank):
        if i == d_axis:
            role = "simd"
        elif cdim is not None and i == cdim:
            role = "cluster"
        else:
            role = "batch"
        out.append((role, shape[i]))
    return tuple(out)


# Logical lane var (identity-keyed) → (phase VarRef, number VarRef, count).
# Populated by ``run()`` by scanning the body for CLUSTER ParallelAxes
# that carry ``original_axis_var`` matching one of ``func.lane_axes``.
# ``_render_idx_as_primexpr`` consults this to expand a bare lane-var
# index (e.g. ``by``) into ``by_phase + by_number * lane_count`` so the
# ISA materializer only sees axes bound by enclosing HLIR for-ops.
_LANE_AXIS_INFO: "Dict[VarRef, Tuple[VarRef, VarRef, int]]" = {}


# Hardware vector lane width (MLEN). Set by ``run()`` from the compile
# target, consumed by lowerings that emit per-row HW vector ops where
# the row stride is the HW vlen rather than anything derivable from
# the buffer's logical shape (e.g. vram→vram copy lowers each iteration
# to one V_ADD_VF that strides by mlen).
_HW_MLEN: int = 0


def _get_var(name: str) -> "_tir.Var":
    v = _VAR_CACHE.get(name)
    if v is None:
        v = _tir.Var(name, _INT32)
        _VAR_CACHE[name] = v
    return v


def _populate_lane_axis_info(
    func: MidFunc,
    lane_axes: List[str],
    cluster_counts: List[int],
) -> None:
    """Walk ``func.body`` looking for each named lane axis's pair of
    (CLUSTER phase ParallelAxis, sibling number ParallelAxis). Record
    ``original_var -> (phase_var, number_var, count)`` in
    ``_LANE_AXIS_INFO`` for later VarRef-keyed lookup in
    ``_render_idx_as_primexpr``.

    The matching CLUSTER axis is the one whose
    ``original_axis_name == lane_axes[i]`` and whose
    ``parent_grid_axis_name`` names a sibling axis with the same
    ``original_axis_name``. Both axes are produced by split as a
    ``number -> phase`` nest.
    """
    # First gather every ParallelAxis in the body, keyed by axis_name.
    axes_by_name: Dict[str, ParallelAxis] = {}

    def collect(s) -> None:
        if isinstance(s, ParallelAxis):
            axes_by_name[s.axis_name] = s
            for c in s.body:
                collect(c)
        elif isinstance(s, (For, Async)):
            for c in s.body:
                collect(c)
        elif isinstance(s, MultiLaneOp):
            # Inner is a leaf op; no nested ParallelAxis.
            return

    for s in func.body:
        collect(s)

    for axis_name, count in zip(lane_axes, cluster_counts):
        phase_axis = None
        for ax in axes_by_name.values():
            if (ax.kind == ParallelKind.CLUSTER
                    and ax.original_axis_name == axis_name):
                phase_axis = ax
                break
        if phase_axis is None:
            # Either cluster was skipped for this kernel, or the kernel
            # has no CLUSTER for this lane. Nothing to expand.
            continue
        number_axis = axes_by_name.get(phase_axis.parent_grid_axis_name)
        if number_axis is None:
            raise ToPlenaError(
                f"lane axis {axis_name!r}: CLUSTER "
                f"{phase_axis.axis_name!r} references unknown number "
                f"axis {phase_axis.parent_grid_axis_name!r}"
            )
        if (phase_axis.axis_var is None
                or phase_axis.original_axis_var is None
                or number_axis.axis_var is None):
            raise ToPlenaError(
                f"lane axis {axis_name!r}: identity (axis_var) fields "
                f"missing on CLUSTER {phase_axis.axis_name!r} or "
                f"number axis {number_axis.axis_name!r}. Split should "
                f"have populated them."
            )
        _LANE_AXIS_INFO[phase_axis.original_axis_var] = (
            phase_axis.axis_var,
            number_axis.axis_var,
            int(count),
        )


def _render_idx_as_primexpr(idx):
    """Like ``_render_idx`` but returns a value suitable for
    ``hlir.BufferSlice.starts``: ints stay ints; VarRefs become a
    ``tir.Var`` (or, for logical lane vars, the split-form composite);
    compound dicts become real ``tir.PrimExpr`` trees so the ISA pass's
    ``_build_slice_offset_expr`` can multiply them by a stride directly.

    NOTE: VarRefs are unwrapped via the *name cache* (``_get_var``) —
    not via ``idx.var`` directly. The downstream ISA materialiser binds
    by ``tir.Var`` *identity* through its ``symbol_table``, and the
    matching HLIR ``for`` loop_var is also minted from the same name
    cache. Routing through the cache here makes the two halves resolve
    to the same Python object, so ``symbol_table[var]`` finds the
    binding. The earlier in-pipeline identity discipline (provided by
    ``VarRef.same_as``) is independent of this rendering step.
    """
    if isinstance(idx, Slice):
        return 0
    if isinstance(idx, int):
        return int(idx)
    if isinstance(idx, VarRef):
        # Logical lane vars (e.g. the user-written ``by``) get expanded
        # to their split form ``by_phase + by_number * lane_count`` so
        # the ISA layer, which only binds the split axes, can
        # materialise the index. The lookup is by VarRef identity.
        info = _LANE_AXIS_INFO.get(idx)
        if info is not None:
            phase, number, count = info
            return _get_var(phase.name) + _get_var(number.name) * _tir.IntImm(_INT32, count)
        return _get_var(idx.name)
    if isinstance(idx, dict):
        op = idx.get("op", "?")
        args = idx.get("args", [])
        if op == "ranged_slice":
            # For HBM starts: the slice begins at args[0] (the cluster
            # base expression). The extent (args[1]) is recovered by
            # _ref_extents elsewhere.
            return _render_idx_as_primexpr(args[0])
        rendered = [_render_idx_as_primexpr(a) for a in args]
        if op == "add":
            return rendered[0] + rendered[1]
        if op == "sub":
            return rendered[0] - rendered[1]
        if op == "mul":
            return rendered[0] * rendered[1]
        if op == "fdiv":
            return _tir.floordiv(rendered[0], rendered[1])
        if op == "fmod":
            return _tir.floormod(rendered[0], rendered[1])
        raise ToPlenaError(
            f"unhandled compound index op {op!r} in _render_idx_as_primexpr"
        )
    return idx


def _make_buffer_arg(ref: BufferRef) -> Union[str, _hlir.BufferSlice]:
    """Build an HLIR buffer_arg from a mid_ir BufferRef. Whole-buffer
    refs become bare buffer-name strings; partial refs become BufferSlice."""
    if _is_whole_buffer_ref(ref):
        return ref.buffer.name
    starts = tuple(_render_idx_as_primexpr(i) for i in ref.indices)
    extents = _ref_extents(ref)
    return _hlir.BufferSlice(
        parent=ref.buffer.name,
        starts=starts,
        extents=extents,
    )


# ---------------------------------------------------------------------------
# Op lowering
# ---------------------------------------------------------------------------


_BINOP_TO_INTRIN = {
    BinOp.ADD: "v_add",
    BinOp.SUB: "v_sub",
    BinOp.MUL: "v_mul",
}


_UNARY_TO_INTRIN = {
    UnaryOp.EXP: "v_exp",
    UnaryOp.RECI: "v_reci",
    UnaryOp.SQRT: "v_sqrt",
    UnaryOp.COPY: "copy_v_to_v",
}


_REDUCE_TO_ROW_AT = {
    ReduceOp.MAX: "row_reduce_max_at",
    ReduceOp.SUM: "row_reduce_sum_at",
}


_ROW_FP_BINOP_TO_INTRIN = {
    # Per-row VRAM × FPRAM-scalar op. One HLIR op = one HW instruction
    # over a single row; ``_lower_bare_broadcast_elementwise`` wraps
    # this in a ``for row`` so multi-row callers don't need to.
    BinOp.ADD: "row_add_fp",
    BinOp.SUB: "row_sub_fp",
    BinOp.MUL: "row_mul_fp",
}


# FPRAM-resident per-lane scalar Elementwise: lower to a ``for row:
# fp_<op>_at`` loop. Used for things like ``M_OLD[row] = M_INIT[row]``
# where every operand is a rank-1 FPRAM buffer.
_FP_AT_BINOP_TO_INTRIN = {
    BinOp.ADD: "fp_add_at",
    BinOp.SUB: "fp_sub_at",
    BinOp.MUL: "fp_mul_at",
    BinOp.MAX: "fp_max_at",
}

_FP_AT_UNARY_TO_INTRIN = {
    UnaryOp.COPY: "fp_copy_at",
    UnaryOp.EXP: "fp_exp_at",
    UnaryOp.RECI: "fp_reci_at",
    UnaryOp.SQRT: "fp_sqrt_at",
}


def _dma_kind_from_scopes(src_scope: str, dst_scope: str) -> str:
    """Pick the dma_* op kind from the (src, dst) scope pair. Mirrors
    the legacy intrinsics registry: H↔V/M only, V↔H whole-buffer."""
    if src_scope == _scope.HBM and dst_scope == _scope.VRAM:
        return "dma_h2v"
    if src_scope == _scope.HBM and dst_scope == _scope.MRAM:
        return "dma_h2m"
    if src_scope == _scope.VRAM and dst_scope == _scope.HBM:
        return "dma_v2h"
    raise ToPlenaError(
        f"unsupported DMA src→dst: {src_scope}→{dst_scope}"
    )


def _dma_kind_slice_variant(base: str) -> str:
    """``dma_h2v`` → ``dma_h2v_slice`` etc. Used when one of the refs
    isn't whole-buffer."""
    return f"{base}_slice"


def _lower_multi_lane_dma(op: Dma, lane_count: int,
                          buf_name_to_hlir: Dict[str, _hlir.Buffer],
                          cluster_axis_name: Optional[str] = None,
                          cluster_axis_var: "Optional[VarRef]" = None,
                          ) -> _hlir.Op:
    src_scope = buf_name_to_hlir[op.src.buffer.name].scope
    dst_scope = buf_name_to_hlir[op.dst.buffer.name].scope
    # VRAM → VRAM "copy" isn't a HW DMA — emit ``copy_v_to_v``
    # (V_ADD_VF dst, src, f0=0) instead.
    if src_scope == _scope.VRAM and dst_scope == _scope.VRAM:
        return _lower_vram_to_vram_copy(
            op, buf_name_to_hlir, cluster_axis_name=cluster_axis_name,
            cluster_axis_var=cluster_axis_var,
        )
    # VRAM ↔ FPRAM: not a DMA either — single S_MAP_FP_V / S_MAP_V_FP
    # per mlen-wide row. tilelang authors write these as T.copy and
    # rely on us to route them to the right HW path.
    if src_scope == _scope.VRAM and dst_scope == _scope.FPRAM:
        return _lower_v_fp_transfer(
            op, "v_to_fp", buf_name_to_hlir, cluster_axis_name,
            cluster_axis_var=cluster_axis_var,
        )
    if src_scope == _scope.FPRAM and dst_scope == _scope.VRAM:
        return _lower_v_fp_transfer(
            op, "fp_to_v", buf_name_to_hlir, cluster_axis_name,
            cluster_axis_var=cluster_axis_var,
        )
    base = _dma_kind_from_scopes(src_scope, dst_scope)
    src_arg = _make_buffer_arg(op.src)
    dst_arg = _make_buffer_arg(op.dst)
    has_slice = isinstance(src_arg, _hlir.BufferSlice) \
        or isinstance(dst_arg, _hlir.BufferSlice)
    kind = _dma_kind_slice_variant(base) if has_slice else base
    return _hlir.Op(
        kind=kind,
        buffer_args=[src_arg, dst_arg],
        scalar_args=[lane_count],
        annotations={"source": "MultiLaneOp(Dma)"},
    )


def _ref_flat_offset(ref: BufferRef,
                     phase_var_zero: "Optional[VarRef]" = None) -> _tir.PrimExpr:
    """Compute ``ref``'s starting element offset in row-major flat layout.

    Iterates buffer.shape backwards accumulating stride; concrete indices
    contribute ``idx * stride``. ``Slice`` is whole-axis (start = 0),
    contributes nothing. ``ranged_slice(start_expr, extent)`` contributes
    ``start_expr * stride``. When ``phase_var_zero`` is set (the cluster
    phase axis's VarRef), VarRef occurrences equal to it (by identity)
    are treated as 0 — mirrors the ``_is_whole_buffer_ref`` convention
    for sync-wrap multi-lane ops where the phase index just marks
    "this op covers every lane in lockstep"."""
    offset: _tir.PrimExpr = _tir.IntImm(_INT32, 0)
    stride = 1
    for dim, idx in zip(reversed(ref.buffer.shape), reversed(ref.indices)):
        if isinstance(idx, Slice):
            pass  # whole-axis access — start is 0
        elif (phase_var_zero is not None
                and isinstance(idx, VarRef) and idx == phase_var_zero):
            pass  # cluster phase axis under sync wrap — contributes 0
        elif isinstance(idx, dict) and idx.get("op") == "ranged_slice":
            start_expr = _render_idx_as_primexpr(idx["args"][0])
            scaled = start_expr if stride == 1 else _tir.Mul(
                start_expr, _tir.IntImm(_INT32, int(stride)),
            )
            offset = scaled if (
                isinstance(offset, _tir.IntImm) and int(offset.value) == 0
            ) else _tir.Add(offset, scaled)
        else:
            term = _render_idx_as_primexpr(idx)
            if isinstance(term, _tir.IntImm) and int(term.value) == 0:
                pass
            else:
                scaled = term if stride == 1 else _tir.Mul(
                    term, _tir.IntImm(_INT32, int(stride)),
                )
                offset = scaled if (
                    isinstance(offset, _tir.IntImm) and int(offset.value) == 0
                ) else _tir.Add(offset, scaled)
        stride *= int(dim)
    return offset


def _lower_vram_to_vram_copy(op: Dma,
                             buf_name_to_hlir: Dict[str, _hlir.Buffer],
                             cluster_axis_name: Optional[str] = None,
                             cluster_axis_var: "Optional[VarRef]" = None,
                             ) -> _hlir.Op:
    """``T.copy(vram_src, vram_dst)`` → region-schema ``copy_v_to_v``.

    Emits one VramRegion per side, at each buffer's PRE-expansion
    (native) rank using the ref's own indices:
        * Slice / phase-var      -> start=0, extent=full
        * ranged_slice(s, ext)   -> start=s, extent=ext
        * VarRef / concrete idx  -> start=expr, extent=1

    The post-walk ``_rewrite_refs_to_4d`` pass then lifts each region
    to 4D using the buffer's own ``_PAD_INSERTS`` or ``_CLUSTER_MODES``
    entry — which is exactly the right thing for cluster-asymmetric
    pairs (one side cluster-expanded, the other a pinned ``global.vram``
    that only got pad-to-4D). Each side gets lifted on its own terms;
    the lifted 4D extents end up matching because mid_ir guarantees
    the logical region is the same on both sides (it's a sync-wrap
    multi-lane copy).
    """
    def _ref_region(ref: BufferRef) -> _hlir.VramRegion:
        starts: List[Any] = []
        extents: List[int] = []
        for dim, idx in zip(ref.buffer.shape, ref.indices):
            if isinstance(idx, Slice):
                starts.append(_tir.IntImm(_INT32, 0))
                extents.append(int(dim))
            elif (cluster_axis_var is not None
                    and isinstance(idx, VarRef) and idx == cluster_axis_var):
                starts.append(_tir.IntImm(_INT32, 0))
                extents.append(int(dim))
            elif isinstance(idx, dict) and idx.get("op") == "ranged_slice":
                s_expr = _render_idx_as_primexpr(idx["args"][0])
                ext = int(idx["args"][1])
                starts.append(s_expr)
                extents.append(ext)
            else:
                starts.append(_render_idx_as_primexpr(idx))
                extents.append(1)
        return _hlir.VramRegion(
            parent=ref.buffer.name,
            starts=tuple(starts), extents=tuple(extents),
        )

    src_region = _ref_region(op.src)
    dst_region = _ref_region(op.dst)
    for_specs: List[Tuple[Any, int]] = []
    leaf = _hlir.Op(
        kind="copy_v_to_v",
        buffer_args=[src_region, dst_region],
        scalar_args=[],
        annotations={"source": "vram→vram copy"},
    )
    if not for_specs:
        return leaf
    body: List[_hlir.Op] = [leaf]
    for v, ext in reversed(for_specs):
        body = [_hlir.make_for_op(loop_var=v, extent=int(ext), body=body)]
    return body[0]


def _is_zero_imm(expr) -> bool:
    return isinstance(expr, _tir.IntImm) and int(expr.value) == 0


def _ref_per_dim_starts(
    ref: BufferRef, phase_var_zero: "Optional[VarRef]" = None,
) -> Tuple[Any, ...]:
    """Per-dim start indices for a BufferRef, mirroring ``_ref_extents``.

    Slice → 0 (whole axis); ranged_slice → start_expr (rendered);
    anything else → the index rendered as a PrimExpr. The
    ``phase_var_zero`` convention matches ``_ref_flat_offset`` — a
    VarRef equal (by identity) to ``phase_var_zero`` (the
    cluster-phase axis VarRef) is treated as 0 under sync wrap.
    """
    out: List[Any] = []
    for idx in ref.indices:
        if isinstance(idx, Slice):
            out.append(0)
        elif (phase_var_zero is not None
                and isinstance(idx, VarRef) and idx == phase_var_zero):
            out.append(0)
        elif isinstance(idx, dict) and idx.get("op") == "ranged_slice":
            out.append(_render_idx_as_primexpr(idx["args"][0]))
        else:
            out.append(_render_idx_as_primexpr(idx))
    return tuple(out)


def _lower_v_fp_transfer(
    op: Dma,
    direction: str,                          # "v_to_fp" or "fp_to_v"
    buf_name_to_hlir: Dict[str, _hlir.Buffer],
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
) -> _hlir.Op:
    """``T.copy(vram, fpram)`` / ``T.copy(fpram, vram)`` → one HLIR slice
    op carrying the full logical region.

    Splitting the logical transfer into HW-MLEN-wide ``S_MAP_*_FP/V``
    issues (and computing per-issue physical VRAM offsets through the
    parent's 7D tile layout) is the ISA emitter's job — HLIR stays at
    the logical-region level.

    HLIR ops emitted:
      * ``v_fp_transfer_slice_v_to_fp``
            buffer_args=[VramRegion]  scalars=[fp_addr]
      * ``v_fp_transfer_slice_fp_to_v``
            buffer_args=[VramRegion]  scalars=[fp_addr]
    """
    if direction == "v_to_fp":
        vram_ref, fp_ref = op.src, op.dst
        kind = "v_fp_transfer_slice_v_to_fp"
    else:
        vram_ref, fp_ref = op.dst, op.src
        kind = "v_fp_transfer_slice_fp_to_v"
    vram_buf = buf_name_to_hlir[vram_ref.buffer.name]
    fp_buf = buf_name_to_hlir[fp_ref.buffer.name]

    starts = _ref_per_dim_starts(vram_ref, phase_var_zero=cluster_axis_var)
    extents = _ref_extents(vram_ref)
    region = _hlir.VramRegion(
        parent=vram_buf.name,
        starts=starts,
        extents=extents,
    )

    fp_indices = _zero_cluster_axis_in_fp_indices(fp_ref, cluster_axis_name)
    fp_addr = _hlir.BufferElement(buffer=fp_buf.name, indices=fp_indices)

    return _hlir.Op(
        kind=kind,
        buffer_args=[region],
        scalar_args=[fp_addr],
        annotations={"source": f"T.copy vram↔fp ({direction})"},
    )


def _zero_cluster_axis_in_fp_indices(
    ref: BufferRef, cluster_axis_name: Optional[str],
) -> "tuple":
    """Render ``ref.indices`` to PrimExprs, replacing the cluster-phase
    axis with 0 when the enclosing MultiLaneOp covers all lanes in one
    issue. ``ref.buffer.cluster_dim`` (set by split / propagated by
    burn_view) tells us which physical axis carries the phase index.
    """
    indices = list(ref.indices)
    cluster_dim = getattr(ref.buffer, "cluster_dim", None)
    if (cluster_axis_name is not None
            and cluster_dim is not None
            and 0 <= cluster_dim < len(indices)):
        indices[cluster_dim] = 0
    return tuple(_render_idx_as_primexpr(i) for i in indices)


def _ref_touch_count(ref: BufferRef) -> int:
    """How many elements does ``ref`` actually touch?

    Slice → buffer dim; ranged_slice → its declared extent; concrete
    index → 1. Multiplied across all axes."""
    count = 1
    for dim, idx in zip(ref.buffer.shape, ref.indices):
        if isinstance(idx, Slice):
            count *= int(dim)
        elif isinstance(idx, dict) and idx.get("op") == "ranged_slice":
            count *= int(idx["args"][1])
        # concrete index: contributes 1
    return count


def _lower_multi_lane_btmm(op: Gemm, lane_count: int,
                           buf_name_to_hlir: Optional[Dict[str, _hlir.Buffer]] = None,
                           ) -> _hlir.Op:
    """MultiLaneOp(Gemm) → region-schema btmm / btmv op.

    Like the bare per-head gemm path, but the cluster axis is folded
    into a single HW multi-lane issue (no ``for lane`` synthesised).
    Region.starts on the lane axis are 0 with extent == lane_count
    so the emitter knows it fires across every lane natively.
    """
    rows = _logical_rows_from_buf(op.a)
    kind = "btmv" if rows == 1 else "btmm"
    if buf_name_to_hlir is None:
        raise ToPlenaError(
            f"multi-lane Gemm[{kind}]: buf_name_to_hlir is required for "
            f"region-schema lowering"
        )
    a_buf = buf_name_to_hlir[op.a.buffer.name]
    b_buf = buf_name_to_hlir[op.b.buffer.name]
    c_buf = buf_name_to_hlir[op.c.buffer.name]

    # Region: lane_axis_name=None ⇒ start=0 on every axis (whole-buffer
    # region). The emitter sees the cluster axis covered by its full
    # lane_count extent and issues a single multi-lane instruction.
    a_region = _gemm_full_region(op.a, a_buf, lane_axis_name=None)
    b_region = _gemm_full_region(op.b, b_buf, lane_axis_name=None)
    c_region = _gemm_full_region(op.c, c_buf, lane_axis_name=None)
    a_roles = _align_dim_roles_to_4d(op.a.buffer.name, op.a_axes)
    b_roles = _align_dim_roles_to_4d(op.b.buffer.name, op.b_axes)
    c_roles = _align_dim_roles_to_4d(op.c.buffer.name, op.c_axes)

    return _hlir.Op(
        kind=kind,
        buffer_args=[a_region, b_region, c_region],
        scalar_args=[a_roles, b_roles, c_roles],
        annotations={"source": f"MultiLaneOp(Gemm[{kind}])"},
    )


def _find_role(axes: List[AxisInfo], role: AxisRole
                ) -> Tuple[Optional[int], Optional[AxisInfo]]:
    """Return ``(dim_index, AxisInfo)`` of the first entry tagged
    ``role``, or ``(None, None)`` if not found."""
    for i, a in enumerate(axes):
        if a.role == role:
            return i, a
    return None, None


def _has_cluster_role(axes: Optional[List[AxisInfo]]) -> bool:
    """True if any dim in ``axes`` is tagged ``CLUSTER``."""
    if not axes:
        return False
    return any(a.role == AxisRole.CLUSTER for a in axes)


def _axes_to_hlir_tuple(
    axes: List[AxisInfo],
    inserts: Tuple[int, ...] = (),
) -> Tuple[Tuple[str, int], ...]:
    """Convert mid_ir per-axis ``AxisInfo`` list into the
    ``Tuple[(role_name, extent), ...]`` form ``hlir.Op.buffer_axes``
    expects.

    ``inserts`` mirrors the per-buffer pad-to-4D rule: each position
    in this list adds a ``("batch", 1)`` placeholder dim so the
    returned tuple aligns with the post-pad HLIR shape. Positions
    follow the same convention as ``_pad_tuple_at`` — ascending
    indices in OUTPUT coords.
    """
    out: List[Tuple[str, int]] = [
        (a.role.value, int(a.extent)) for a in axes
    ]
    for pos in inserts:
        out.insert(pos, ("batch", 1))
    return tuple(out)


def _split_axes_by_role(axes: List[AxisInfo]):
    """Group an op's axes table into ``(batch_extents, inner_extent)``.

    ``batch_extents`` are the BATCH-role extents in axis order — each
    one becomes an outer ``for`` loop wrapped around the leaf op.
    ``inner_extent`` is the product of every CLUSTER + SIMD axis —
    that is, the contiguous 1-D vector each leaf op processes.

    REDUCE / BROADCAST axes are op-specific and not handled here (only
    Elementwise / Dma / Reduce-leaf paths use this helper).
    """
    batch_extents: List[int] = []
    inner = 1
    for a in axes:
        if a.role == AxisRole.BATCH:
            batch_extents.append(int(a.extent))
        elif a.role in (AxisRole.SIMD, AxisRole.CLUSTER):
            inner *= int(a.extent)
        else:
            raise ToPlenaError(
                f"_split_axes_by_role: unsupported role {a.role!r} in axes "
                f"{axes!r}"
            )
    return batch_extents, inner


def _lower_multi_lane_elementwise(
    op: Elementwise, lane_count: int,
    buf_name_to_hlir: Optional[Dict[str, _hlir.Buffer]] = None,
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
) -> _hlir.Op:
    """Pure elementwise (no Broadcast srcs) → ``v_add`` / ``v_exp`` /
    ``v_zero`` / etc., wrapped in explicit ``for`` loops for each
    BATCH axis.

    Reads the per-axis ``dst_axes`` table directly — every BATCH axis
    becomes an outer ``for``; the contiguous CLUSTER + SIMD extents
    multiply into the leaf ``v_*`` op's ``n_elem``. No buffer-shape
    guessing, no row-geometry heuristics.

    Falls back to the FPRAM and per-row unary paths when the dst scope
    or axes shape demand them.
    """
    if (buf_name_to_hlir is not None
            and buf_name_to_hlir[op.dst.buffer.name].scope == _scope.FPRAM):
        return _lower_bare_fp_scalar_elementwise(
            op, lane_count, cluster_axis_name,
            cluster_axis_var=cluster_axis_var,
        )
    # COPY has a multi-lane ``copy_v_to_v`` intrin (handled below) and
    # no per-row variant — keep it on the v_copy path even when the
    # dst's row footprint is 1 (an artifact of fold absorbing
    # T.Parallel rather than a request for a row_<op> emission).
    if (op.op in _UNARY_TO_INTRIN
            and op.op != UnaryOp.COPY
            and len(op.srcs) == 1
            and _row_footprint(op.dst) == 1):
        return _lower_per_row_unary(
            op, cluster_axis_name, cluster_axis_var=cluster_axis_var,
        )
    if op.op in _BINOP_TO_INTRIN:
        kind = _BINOP_TO_INTRIN[op.op]
    elif op.op in _UNARY_TO_INTRIN:
        kind = _UNARY_TO_INTRIN[op.op]
        if op.op == UnaryOp.COPY and not op.srcs:
            kind = "v_zero"
    else:
        raise ToPlenaError(f"unsupported elementwise op {op.op!r}")
    buffer_args: List[Any] = []
    for s in op.srcs:
        if isinstance(s, Broadcast):
            raise ToPlenaError(
                f"MultiLaneOp Elementwise with Broadcast src — pass_2_mark "
                f"should have set can_async=False"
            )
        buffer_args.append(s.buffer.name)
    buffer_args.append(op.dst.buffer.name)

    # Read fan-out structure straight off the axes table.
    if not op.dst_axes:
        raise ToPlenaError(
            f"Elementwise on {op.dst.buffer.name!r} has empty dst_axes — "
            f"fold/split/view must populate the axes table for every op "
            f"so lower can stop guessing geometry from buffer shape."
        )
    # Binop family (v_add / v_sub / v_mul) uses the new logical-coord
    # schema: scalar_args = [idx0, idx1, idx2] picks one mlen-wide row
    # per non-SIMD axis of the dst's 4D shape (B, S, H), and the D axis
    # is implicit — the emitter walks all d_tiles itself. Extent-1 axes
    # (e.g. the pad-to-4D B=1 placeholder) collapse to ``IntImm(0)``
    # rather than wrapping ``for _ in range(1)``.
    #
    # Unary / copy family (v_exp / v_reci / v_sqrt / v_zero /
    # copy_v_to_v) still uses the legacy flat-offset schema until its
    # emitters are migrated; that path stays on the older
    # ``[off, off, n_elem]`` form below.
    # Region schema (unified for v_add/v_sub/v_mul/v_exp/v_reci/v_sqrt/v_zero):
    # each buffer_arg becomes a VramRegion with 4D BSHD (starts,
    # extents). The HLIR axes table is the source of truth — it's
    # computed from the post-pad-to-4D buffer shape and is always 4
    # entries in canonical (B, S, H, D) order.
    #
    # For each non-SIMD axis (slot 0..2):
    #   * extent == 1                -> start=0, extent=1 (no for, no idx)
    #   * cluster axis (lane span)   -> start=0, extent=full
    #         (whole packed-head group lives in one mlen-row; emitter
    #          folds this axis out of the walk via parent.cluster_dim)
    #   * other batch axis ext > 1   -> fresh row var, outer for-op,
    #         region.start=var, region.extent=1
    # Slot 3 (SIMD/D) is always start=0, extent=D_full.
    if kind in ("v_add", "v_sub", "v_mul",
                "v_exp", "v_reci", "v_sqrt",
                "v_zero", "copy_v_to_v"):
        dst_hlir_axes = _axes_of(op.dst.buffer.name)
        if dst_hlir_axes is None or len(dst_hlir_axes) != 4:
            raise ToPlenaError(
                f"v_* lowering: dst {op.dst.buffer.name!r} has no 4D "
                f"hlir axes table; got {dst_hlir_axes!r}"
            )
        starts: List[Any] = []
        extents: List[int] = []
        for_specs: List[Tuple[Any, int]] = []
        for slot, (role_name, extent) in enumerate(dst_hlir_axes[:3]):
            if int(extent) == 1:
                starts.append(_tir.IntImm(_INT32, 0))
                extents.append(1)
            elif role_name == "cluster":
                starts.append(_tir.IntImm(_INT32, 0))
                extents.append(int(extent))
            else:
                v = _fresh_var(f"row{slot}")
                starts.append(v)
                extents.append(1)
                for_specs.append((v, int(extent)))
        starts.append(_tir.IntImm(_INT32, 0))
        extents.append(int(dst_hlir_axes[3][1]))

        region_args = [
            _hlir.VramRegion(
                parent=name, starts=tuple(starts), extents=tuple(extents),
            )
            for name in buffer_args
        ]
        leaf = _hlir.Op(
            kind=kind,
            buffer_args=region_args,
            scalar_args=[],
            annotations={"source": f"MultiLaneOp(Elementwise {op.op.value})"},
        )
        if not for_specs:
            return leaf
        body: List[_hlir.Op] = [leaf]
        for v, ext in reversed(for_specs):
            body = [_hlir.make_for_op(loop_var=v, extent=ext, body=body)]
        return body[0]

    # Legacy flat-offset fallback (nothing currently routes here; kept
    # in case a future v_* kind shows up before the migration cleanup).
    batch_extents, inner_extent = _split_axes_by_role(op.dst_axes)

    def _build_leaf_flat(row_offset):
        off = row_offset if row_offset is not None else _tir.IntImm(_INT32, 0)
        raise ToPlenaError(f"unhandled v_* kind {kind!r}")

    if not batch_extents:
        return _build_leaf_flat(row_offset=None)

    strides: List[int] = []
    running = inner_extent
    for ext in reversed(batch_extents):
        strides.append(running)
        running *= int(ext)
    strides.reverse()
    row_vars = [_fresh_var(f"row{i}") for i in range(len(batch_extents))]
    total_offset = None
    for v, st in zip(row_vars, strides):
        term = _tir.Mul(v, _tir.IntImm(_INT32, st)) if st != 1 else v
        total_offset = term if total_offset is None else _tir.Add(total_offset, term)
    body: List[_hlir.Op] = [_build_leaf_flat(row_offset=total_offset)]
    for v, ext in reversed(list(zip(row_vars, batch_extents))):
        body = [_hlir.make_for_op(loop_var=v, extent=int(ext), body=body)]
    return body[0]


_UNARY_TO_ROW_INTRIN = {
    UnaryOp.EXP: "row_exp",
}


def _lower_per_row_unary(
    op: Elementwise,
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
) -> _hlir.Op:
    """Per-row unary VRAM op → single-row ``row_<op>`` leaf.

    Contract matches ``_emit_row_scalar_op_at`` with no fp operand:
        buffer_args = [vram_src, vram_dst]
        scalar_args = [row_var, lane_var]
    Caller's ``for row`` (kernel-written, preserved by mid_ir) is
    rendered by the walker.
    """
    if op.op not in _UNARY_TO_ROW_INTRIN:
        raise ToPlenaError(
            f"per-row unary op {op.op!r} has no row_<op> intrinsic"
        )
    intrin = _UNARY_TO_ROW_INTRIN[op.op]
    (src,) = op.srcs
    if isinstance(src, Broadcast):
        raise ToPlenaError(
            "per-row unary expects a direct BufferRef src, got Broadcast"
        )
    # Pick row_var from the dst's BATCH axis index (mid_ir source of
    # truth via op.dst_axes). The kernel-written ``for row`` /
    # ``for oh`` / ... wraps this op; the dst index at that axis is
    # exactly the bound loop var we need.
    row_axis = _row_axis_index_from_axes(
        op.dst_axes, ctx=f"per-row unary {intrin} dst",
    )
    row_var = _render_idx_as_primexpr(op.dst.indices[row_axis])
    if cluster_axis_name is not None:
        lane_var = _make_loop_var(cluster_axis_name)
    elif cluster_axis_var is not None:
        lane_var = _make_loop_var(cluster_axis_var.name)
    else:
        lane_var = _fresh_var("lane")
    src_region = _build_row_at_region(
        src.buffer.name, row_var=row_var, lane_var=lane_var,
        ctx=f"per-row unary {intrin} src",
    )
    dst_region = _build_row_at_region(
        op.dst.buffer.name, row_var=row_var, lane_var=lane_var,
        ctx=f"per-row unary {intrin} dst",
    )
    return _hlir.Op(
        kind=intrin,
        buffer_args=[src_region, dst_region],
        scalar_args=[],
        annotations={"source": f"per-row Elementwise[{op.op.value}]"},
    )


def _lower_multi_lane(mlo: MultiLaneOp,
                      buf_name_to_hlir: Dict[str, _hlir.Buffer],
                      lane_count: int) -> _hlir.Op:
    """Single MultiLaneOp → single HLIR Op with lane_count scalar.

    ``lane_count`` is the enclosing cluster's extent (e.g. 4 for a
    typical MLEN/hlen split). It controls both (a) how many lanes the
    HW-side multi-lane op spans (passed via scalar_args[0]) and (b)
    the synthetic ``for lane`` extent when the inner op falls back to
    a per-lane FPRAM template.
    """
    axis_name = mlo.cluster_axis_names[0] if mlo.cluster_axis_names else None
    axis_var = mlo.cluster_axis_vars[0] if mlo.cluster_axis_vars else None
    inner = mlo.inner
    if isinstance(inner, Dma):
        return _lower_multi_lane_dma(
            inner, lane_count, buf_name_to_hlir,
            cluster_axis_name=axis_name,
            cluster_axis_var=axis_var,
        )
    if isinstance(inner, Gemm):
        return _lower_multi_lane_btmm(inner, lane_count, buf_name_to_hlir)
    if isinstance(inner, Elementwise):
        return _lower_multi_lane_elementwise(
            inner, lane_count, buf_name_to_hlir, axis_name,
            cluster_axis_var=axis_var,
        )
    raise ToPlenaError(
        f"unsupported MultiLaneOp inner: {type(inner).__name__}"
    )


def _lane_loop_var(cluster_axis_name: Optional[str],
                   cluster_axis_var: "Optional[VarRef]" = None) -> _tir.Var:
    """Pick a loop_var for the synthetic ``for lane`` that wraps a
    bare op inside a cluster. Prefer the cluster axis's VarRef (so the
    loop_var identity matches the in-buffer phase var); fall back to
    the name-cached var, or finally ``"lane"`` for bare ops outside
    any cluster."""
    if cluster_axis_var is not None:
        return cluster_axis_var.var
    if cluster_axis_name is not None:
        return _make_loop_var(cluster_axis_name)
    return _make_loop_var("lane")


def _fresh_var(name: str) -> _tir.Var:
    """Allocate a brand-new ``tir.Var`` that is NOT cached. Use for
    synthetic loops (row/col templates) whose Var must not collide
    with an enclosing same-named loop in the kernel."""
    return _tir.Var(name, _INT32)


def _per_lane_stride(buf: _hlir.Buffer, mode: str) -> int:
    """Stride (in elements) between consecutive lanes for a buffer.

    Reads ``buf.cluster_dim`` directly: the lane stride is the product
    of every shape axis strictly to the right of the cluster dim.
    Earlier versions had a ``mode``-keyed fallback that hard-coded
    ``shape[1] * shape[2] * shape[3]`` etc.; that legacy path masked
    a missing ``cluster_dim``. Any buffer reaching codegen without a
    ``cluster_dim`` is a real bug now — raise loudly instead of
    silently miscomputing.
    """
    shape = [int(d) for d in buf.shape]
    if buf.cluster_dim is None:
        raise ToPlenaError(
            f"_per_lane_stride: buffer {buf.name!r} (mode={mode!r}) has no "
            f"cluster_dim; split / view passes must have populated it before "
            f"codegen. shape={shape}"
        )
    stride = 1
    for axis in range(buf.cluster_dim + 1, len(shape)):
        stride *= shape[axis]
    return stride


_GEMM_ROLE_TO_LABEL: Dict[AxisRole, str] = {
    AxisRole.GEMM_M: "M",
    AxisRole.GEMM_K: "K",
    AxisRole.GEMM_N: "N",
}


def _axes_to_dim_roles(axes: List[AxisInfo]) -> Tuple[str, ...]:
    """Project a mid_ir per-axis ``AxisInfo`` table onto the gemm
    dim-role labels emitters consume.

    Only the matmul-specific roles (M / K / N) keep a distinct label;
    everything else (BATCH, CLUSTER, SIMD, BROADCAST, ...) collapses
    to ``"_"`` — the emitter only cares about M/K/N positions to drive
    instruction selection (M_MM vs M_TMM, M_MV vs M_BTMV) and to look
    up extents; other axes contribute via region.starts/extents in
    the usual way (lane idx, batch fan-out).
    """
    return tuple(
        _GEMM_ROLE_TO_LABEL.get(a.role, "_") for a in axes
    )


def _align_dim_roles_to_4d(buf_name: str,
                           mid_axes: List[AxisInfo]) -> Tuple[str, ...]:
    """Align a mid_ir per-axis roles table onto the HLIR buffer's
    post-expansion 4D shape, returning a 4-tuple of dim-role labels
    ("M"/"K"/"N"/"_") suitable for the gemm Region+roles schema.

    Two cases:

    * ``_PAD_INSERTS`` entry → the buffer was rank-padded to 4D by
      inserting extent-1 axes at recorded positions. We pad the
      roles list at the same positions with ``"_"``.

    * ``_CLUSTER_MODES`` entry → the buffer was cluster-expanded.
      We mirror the exact axis placement that
      ``_rewrite_ref_for_cluster_mode`` does for starts/extents,
      so roles end up at the right *physical* axis after the
      lane → BSHD anchoring (row_stack puts lane at axis 0;
      col_pack puts it at axis 2). The mid_ir axis at the source
      cluster_dim collapses to ``"_"`` (cluster never carries a
      gemm role); the leftover BSHD slot is also ``"_"``.

    Returns a 4-tuple. If the mid_ir axes are already rank-4
    (HBM buffer, or a no-op) returns the projection directly.
    """
    base = _axes_to_dim_roles(mid_axes)
    if len(base) == 4:
        return base
    if buf_name in _PAD_INSERTS:
        inserts = _PAD_INSERTS[buf_name]
        out = list(base)
        for pos in inserts:
            out.insert(pos, "_")
        if len(out) != 4:
            raise ToPlenaError(
                f"_align_dim_roles_to_4d: pad applied to {buf_name!r} but "
                f"result rank {len(out)} != 4 (mid_axes={mid_axes!r}, "
                f"inserts={inserts!r})"
            )
        return tuple(out)
    if buf_name in _CLUSTER_MODES:
        mode, old_cluster_dim, new_shape = _CLUSTER_MODES[buf_name]
        if mode == _MODE_FP_LANE:
            return base
        if len(base) != 3:
            raise ToPlenaError(
                f"_align_dim_roles_to_4d: cluster-expanded {buf_name!r} "
                f"expects rank-3 mid axes, got {len(base)} ({mid_axes!r})"
            )
        if mode == _MODE_BSHD_LIFT or old_cluster_dim is None:
            # mid_ir (?, S, D) -> BSHD (?, S, 1, D)
            return (base[0], base[1], "_", base[2])
        new_lane_dim = _CLUSTER_MODE_NEW_LANE_DIM[mode]
        if new_lane_dim is None:
            return (base[0], base[1], "_", base[2])
        # Same anchoring as _rewrite_ref_for_cluster_mode:
        #   * non-lane sources keep order: first non-lane mid axis -> out[1] (S),
        #     second non-lane mid axis -> out[3] (D)
        #   * lane slot at new_lane_dim is "_" (cluster never carries a gemm role)
        #   * remaining BSHD slot is "_"
        non_lane_sources = [i for i in range(3) if i != old_cluster_dim]
        if len(non_lane_sources) != 2:
            raise ToPlenaError(
                f"_align_dim_roles_to_4d: unexpected non-lane source count "
                f"{len(non_lane_sources)} for {buf_name!r}"
            )
        s_src, d_src = non_lane_sources
        out = ["_"] * 4
        out[1] = base[s_src]
        out[3] = base[d_src]
        # new_lane_dim already "_" from initialisation; leftover slot too.
        return tuple(out)
    # Unknown buffer (or no rank change needed): pad with leading "_"
    # placeholders defensively.
    deficit = 4 - len(base)
    if deficit < 0:
        raise ToPlenaError(
            f"_align_dim_roles_to_4d: {buf_name!r} mid axes rank "
            f"{len(base)} > 4 with no pad/cluster record"
        )
    return tuple(["_"] * deficit) + base


def _make_onchip_region(
    name: str,
    buf: "_hlir.Buffer",
    starts: Tuple[Any, ...],
    extents: Tuple[int, ...],
):
    """Build a Vram/MramRegion based on buffer scope."""
    if buf.scope == _scope.MRAM:
        return _hlir.MramRegion(parent=name, starts=starts, extents=extents)
    return _hlir.VramRegion(parent=name, starts=starts, extents=extents)


def _gemm_full_region(
    ref: "BufferRef",
    buf: "_hlir.Buffer",
    *,
    lane_axis_name: Optional[str] = None,
) -> "Any":
    """Build a 4D Vram/MramRegion that covers the *whole* buffer.

    ``starts`` are all zero (or ``lane_var`` on the cluster axis when
    the caller's gemm sits inside a CLUSTER and the buffer's
    ``cluster_dim`` marks that axis); ``extents`` are the buffer's
    full per-dim extents from the mid_ir-side shape. The emitter walks
    M_tiles / K_tiles / N from those extents using the parallel
    ``dim_roles`` tuple, so this region uniformly describes the gemm
    workspace whether it's BTMM-shaped (single mlen-tile per axis) or
    multi-tile linear.

    Cluster handling: when ``lane_axis_name`` is provided (the gemm
    is wrapped in a CLUSTER → ``for lane:`` in mid_ir), each per-lane
    issue lives in ``starts[cluster_dim] = lane_var`` with the lane
    axis's extent reduced to 1 (a single lane per leaf op). Without
    a cluster context, every start is 0.
    """
    cluster_dim = getattr(buf, "cluster_dim", None)
    shape = [int(d) for d in buf.shape]
    starts: List[Any] = []
    extents: List[int] = []
    for i, dim_extent in enumerate(shape):
        if (cluster_dim is not None
                and i == cluster_dim
                and lane_axis_name is not None):
            starts.append(_make_loop_var(lane_axis_name))
            extents.append(1)
        else:
            starts.append(_tir.IntImm(_INT32, 0))
            extents.append(int(dim_extent))
    return _make_onchip_region(
        ref.buffer.name, buf, tuple(starts), tuple(extents),
    )


def _lower_bare_per_head_gemm(
    op: Gemm,
    cluster_extent: Optional[int],
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
    buf_name_to_hlir: Optional[Dict[str, _hlir.Buffer]] = None,
    lane_modes: Optional[Dict[str, str]] = None,
) -> _hlir.Op:
    """Bare (non-async) per-head matmul / mv → region-based HLIR op.

    New region schema:
        buffer_args = [a_region, b_region, c_region]    # Vram/MramRegion
        scalar_args = [a_dim_roles, b_dim_roles, c_dim_roles]
            each is a 4-tuple of "M"/"K"/"N"/"_" labels aligned with
            the parent buffer's 4D physical shape.

    Region.starts encodes the per-lane gemm position: when this op
    is wrapped in a CLUSTER (``cluster_axis_name`` set), the
    cluster-marked axis of each region gets ``lane_var`` at start
    (extent=1 → one lane per leaf op). The emitter walks the lane
    axis using its tile_layout stride; the mid_ir layer just hands
    over the logical (b, s, h, d) position, without doing any
    physical stride math.

    transpose_b is dropped as a flag — b's dim_roles tuple encodes
    "K before N" (K-inner, standard) vs "N before K" (transpose_b);
    the emitter decides M_MM vs M_TMM from that ordering.

    Falls back to ``kind="mv"`` (instead of "matmul") when the LHS
    has only one M-row (decode-style P @ V).
    """
    a_buf = buf_name_to_hlir[op.a.buffer.name] if buf_name_to_hlir else None
    b_buf = buf_name_to_hlir[op.b.buffer.name] if buf_name_to_hlir else None
    c_buf = buf_name_to_hlir[op.c.buffer.name] if buf_name_to_hlir else None
    if a_buf is None or b_buf is None or c_buf is None:
        raise ToPlenaError(
            f"per-head Gemm: missing HLIR buffer for one of a/b/c "
            f"(a={op.a.buffer.name!r}, b={op.b.buffer.name!r}, "
            f"c={op.c.buffer.name!r})"
        )

    # LHS rows == 1 → matrix-vector. Read off ``op.a_axes`` GEMM_M
    # extent (authoritative); fall back to buffer shape only if axes
    # are missing.
    a_M_extent: Optional[int] = None
    if op.a_axes:
        _, a_m_info = _find_role(op.a_axes, AxisRole.GEMM_M)
        if a_m_info is not None:
            a_M_extent = int(a_m_info.extent)
    if a_M_extent is not None:
        use_mv = a_M_extent == 1
    else:
        use_mv = _logical_rows_from_buf(op.a) == 1

    inside_cluster = (
        cluster_extent is not None and cluster_axis_name is not None
    )
    lane_axis_name = cluster_axis_name if inside_cluster else None

    a_region = _gemm_full_region(op.a, a_buf, lane_axis_name=lane_axis_name)
    b_region = _gemm_full_region(op.b, b_buf, lane_axis_name=lane_axis_name)
    c_region = _gemm_full_region(op.c, c_buf, lane_axis_name=lane_axis_name)
    a_roles = _align_dim_roles_to_4d(op.a.buffer.name, op.a_axes)
    b_roles = _align_dim_roles_to_4d(op.b.buffer.name, op.b_axes)
    c_roles = _align_dim_roles_to_4d(op.c.buffer.name, op.c_axes)

    annotations: Dict[str, Any] = {
        "source": (
            "per-head Gemm(rows=1) inside cluster" if (use_mv and inside_cluster)
            else "per-head Gemm(rows=1)"            if use_mv
            else "per-head Gemm(overwrite) inside cluster" if inside_cluster
            else "per-head Gemm(overwrite)"
        ),
    }

    return _hlir.Op(
        kind="mv" if use_mv else "matmul",
        buffer_args=[a_region, b_region, c_region],
        scalar_args=[a_roles, b_roles, c_roles],
        annotations=annotations,
    )


def _row_axis_index_of_buf(name: str, shape, cluster_dim: Optional[int]) -> int:
    """Index of the logical row axis in a buffer's shape.

    Skips the innermost axis (column / D / hlen) and the cluster axis
    (lane); the first remaining axis is rows. ``cluster_dim`` is the
    explicit marker propagated from pass_3_split.

    NOTE: this still derives the rows axis from buffer shape + cluster
    dim rather than the op's per-axis ``axes`` table. Equivalent in
    every shape layout we ship today, but the cleanest replacement is
    ``next(i for i, a in enumerate(op_axes) if a.role == AxisRole.BATCH)``
    once every caller has an op handle in scope. Left as-is to limit
    blast radius — flag if a future buffer layout pushes rows past a
    non-cluster, non-innermost axis the per-axis table sees but the
    cluster_dim heuristic doesn't."""
    shape = [int(d) for d in shape]
    if len(shape) < 2:
        raise ToPlenaError(
            f"buffer {name!r} rank {len(shape)} has no row axis"
        )
    inner = len(shape) - 1
    for axis in range(len(shape)):
        if axis == inner:
            continue
        if cluster_dim is not None and axis == cluster_dim:
            continue
        return axis
    raise ToPlenaError(
        f"buffer {name!r} shape={shape} cluster_dim={cluster_dim}: "
        "no row axis available (only inner + cluster?)"
    )


def _row_axis_index(ref: BufferRef) -> int:
    """Row axis index for a mid_ir BufferRef. Delegates to the
    buffer-only form below."""
    return _row_axis_index_of_buf(
        ref.buffer.name, ref.buffer.shape, ref.buffer.cluster_dim,
    )


def _row_axis_index_from_axes(
    axes: List[AxisInfo],
    *,
    ctx: str,
) -> int:
    """Pick the rows axis from an op's per-axis role table.

    The rows axis is the (last) ``BATCH`` axis with the largest
    extent. SIMD / CLUSTER / REDUCE / BROADCAST / GEMM_* are
    skipped. Caller passes an ``op.dst_axes`` / ``op.src_axes[i]``
    so the answer comes from mid_ir's authoritative role table,
    not from buffer shape + cluster_dim heuristics.
    """
    rows_axis = -1
    rows_extent = -1
    for i, a in enumerate(axes):
        if a.role != AxisRole.BATCH:
            continue
        if int(a.extent) > rows_extent:
            rows_extent = int(a.extent)
            rows_axis = i
    if rows_axis < 0:
        raise ToPlenaError(
            f"{ctx}: no BATCH axis in axes {axes!r}; cannot locate "
            f"rows dimension"
        )
    return rows_axis


def _row_footprint(ref: BufferRef) -> int:
    """How many rows does this op-local ref actually touch?

    ``Slice`` on the row axis → full buffer row extent (op covers the
    whole row stack). Concrete index (var / int) → 1 (op acts on a
    single row picked by the enclosing for-row).
    """
    axis = _row_axis_index(ref)
    idx = ref.indices[axis]
    if isinstance(idx, Slice):
        return int(ref.buffer.shape[axis])
    if isinstance(idx, dict) and idx.get("op") == "ranged_slice":
        return int(idx["args"][1])
    return 1


def _logical_rows_from_buf(ref: BufferRef) -> int:
    """Recover the kernel's logical row count from a mid_ir BufferRef.

    Uses the explicit ``cluster_dim`` marker on the BufferDef to skip
    the lane axis; the rows axis is the remaining non-innermost dim."""
    shape = [int(d) for d in ref.buffer.shape]
    if len(shape) < 2:
        return 1
    try:
        return int(shape[_row_axis_index(ref)])
    except ToPlenaError:
        # Defensive fallback (shouldn't trigger in practice).
        return shape[-2]


def _render_ref_with_role_axes(
    ref: "BufferRef",
    axes: List[AxisInfo],
    *,
    row_var,
    lane_var,
    ctx: str,
) -> Tuple[Any, ...]:
    """Render a mid_ir ``BufferRef`` as an HLIR index tuple, using its
    per-axis role table to resolve any ``Slice`` ("the op covers this
    whole axis") into the right loop var.

    Used for Reduce dst, whose axes (per mid_ir) are exactly the dst
    fragment's surviving dims after the REDUCE collapse — so each axis
    is either:
      * ``CLUSTER`` -> ``lane_var`` (per-lane fan-out)
      * ``BATCH``   -> ``row_var`` (per-row fan-out)

    User-pinned indices (``MEAN_SUM[r]`` with concrete ``r``) bypass
    the Slice rule and are rendered as-is — kernels that explicitly
    thread a loop var still work.

    SIMD / REDUCE roles never appear on a Reduce dst by construction
    (REDUCE is collapsed; SIMD is what made the src vectorisable);
    seeing either here means an upstream pass produced inconsistent
    axes, so we raise.
    """
    if len(ref.indices) != len(axes):
        raise ToPlenaError(
            f"{ctx}: rank mismatch — ref {ref.buffer.name!r} has "
            f"{len(ref.indices)} indices but axes table has {len(axes)} "
            f"entries (indices={list(ref.indices)!r}, axes={axes!r})"
        )
    out: List[Any] = []
    for axis_idx, (raw_idx, axis) in enumerate(zip(ref.indices, axes)):
        if not isinstance(raw_idx, Slice):
            out.append(_render_idx_as_primexpr(raw_idx))
            continue
        if axis.role == AxisRole.CLUSTER:
            out.append(lane_var)
        elif axis.role == AxisRole.BATCH:
            out.append(row_var)
        else:
            raise ToPlenaError(
                f"{ctx}: axis {axis_idx} on ref {ref.buffer.name!r} is a "
                f"Slice with role {axis.role!r}; expected CLUSTER or "
                f"BATCH (SIMD / REDUCE shouldn't survive on a reduce dst)."
            )
    return tuple(out)


def _build_row_at_region(
    buf_name: str,
    *,
    row_var,
    lane_var,
    ctx: str,
) -> "_hlir.VramRegion":
    """Build a ``VramRegion`` for a single-row row_*_at op.

    ``row_*_at`` ops touch exactly one logical row (one (b, s, h)
    cell), so non-D extents are always 1. The starts are placed per
    axis ROLE so the buffer's actual physical layout is respected:
      * SIMD axis (innermost, always D) -> start = 0, extent = D_full
      * CLUSTER axis (lane carrier)     -> start = lane_var, extent = 1
      * BATCH axis with the largest extent (the "rows" axis)
                                         -> start = row_var, extent = 1
      * any other BATCH axis (extent-1 placeholders, e.g. pad-to-4D
        B=1 or row_stack-spare H=1)     -> start = 0, extent = 1

    This is needed because different lane-fusion modes put the
    cluster axis at different physical positions:
      * col_pack: (B=1, S, lane=H, D_narrow)  cluster_dim=2
      * row_stack: (lane=B, S, H=1, MLEN)     cluster_dim=0
    A schema that hard-codes ``starts=(0, row, lane, 0)`` would
    misplace the lane var in row_stack mode and corrupt all reads.
    """
    axes = _axes_of(buf_name)
    if axes is None or len(axes) != 4:
        raise ToPlenaError(
            f"{ctx}: buffer {buf_name!r} has no 4D hlir axes; got {axes!r}"
        )
    starts: List[Any] = []
    extents: List[int] = []
    rows_slot: Optional[int] = None
    rows_extent: int = -1
    # First pass: find the rows axis (largest batch).
    for i, (role, extent) in enumerate(axes):
        if role == "batch" and int(extent) > rows_extent:
            rows_extent = int(extent)
            rows_slot = i
    for i, (role, extent) in enumerate(axes):
        if role == "simd":
            starts.append(_tir.IntImm(_INT32, 0))
            extents.append(int(extent))
        elif role == "cluster":
            starts.append(lane_var)
            extents.append(1)
        elif i == rows_slot and int(extent) > 1:
            starts.append(row_var)
            extents.append(1)
        else:
            # Degenerate batch placeholder (extent 1 or smaller batch).
            starts.append(_tir.IntImm(_INT32, 0))
            extents.append(1)
    return _hlir.VramRegion(
        parent=buf_name,
        starts=tuple(starts),
        extents=tuple(extents),
    )


def _lower_bare_reduce(op: Reduce,
                       cluster_extent: Optional[int],
                       cluster_axis_name: Optional[str] = None,
                       cluster_axis_var: "Optional[VarRef]" = None,
                       ) -> _hlir.Op:
    """Bare reduce → single-row ``row_reduce_*_at`` leaf op,
    optionally wrapped in a synthesised ``for row``.

    Contract: one HLIR leaf op = one HW instruction. Decision uses
    the src ref's row footprint (reduce collapses the inner axis;
    iterating over rows yields one HW reduce per row):

      * Slice on the row axis → reduce covers every row; synthesise
        ``for row``.
      * Concrete index → reduce acts on a single row picked by the
        kernel's outer ``for row``, which the walker renders.

    The enclosing CLUSTER ``ParallelAxis`` (lowered to ``for lane``
    by the walker) binds the lane var.
    """
    if op.op not in _REDUCE_TO_ROW_AT:
        raise ToPlenaError(f"unsupported reduce op {op.op!r}")
    intrin = _REDUCE_TO_ROW_AT[op.op]
    # Lane axis only exists when a cluster wraps the op; otherwise
    # there is no lane dim and the head index in VRAM addressing is 0.
    if cluster_axis_name is not None:
        lane_var: _tir.PrimExpr = _make_loop_var(cluster_axis_name)
    else:
        lane_var = _tir.IntImm(_INT32, 0)
    row_footprint = _row_footprint(op.src)
    if row_footprint > 1:
        row_var: _tir.PrimExpr = _fresh_var("row")
        wrap_rows = row_footprint
    else:
        # Single-row reduce: pick row var from the src's row-axis
        # index via the op's per-axis role table (mid_ir's
        # authoritative info). The enclosing kernel-written for
        # already bound the index; we just thread its VarRef /
        # IntImm through the rendered tree.
        row_axis = _row_axis_index_from_axes(
            op.src_axes, ctx=f"reduce[{op.op.value}] src",
        )
        row_var = _render_idx_as_primexpr(op.src.indices[row_axis])
        wrap_rows = None
    fp_addr = _hlir.BufferElement(
        buffer=op.dst.buffer.name,
        indices=_render_ref_with_role_axes(
            op.dst, op.dst_axes,
            row_var=row_var, lane_var=lane_var,
            ctx=f"reduce[{op.op.value}] dst",
        ),
    )
    # Region schema: starts placed per axis ROLE (cluster slot gets
    # the lane var, the largest non-cluster batch slot gets the row
    # var, everything else is 0). One mlen-row covers all of D.
    src_region = _build_row_at_region(
        op.src.buffer.name, row_var=row_var, lane_var=lane_var,
        ctx="reduce src",
    )
    leaf = _hlir.Op(
        kind=intrin,
        buffer_args=[src_region],
        scalar_args=[fp_addr],
        annotations={"source": f"bare Reduce[{op.op.value}]"},
    )
    if wrap_rows is None:
        return leaf
    return _hlir.make_for_op(loop_var=row_var, extent=wrap_rows, body=[leaf])


def _lower_bare_broadcast_elementwise(
    op: Elementwise,
    cluster_extent: Optional[int],
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
) -> _hlir.Op:
    """Bare elementwise with a per-row FPRAM Broadcast src
    → single-row ``row_<op>_fp`` leaf op.

    Contract: one HLIR leaf op = one HW V_*_VF instruction over one
    row. Whether we wrap the leaf in a synthesised ``for row`` is
    decided by the op's row-axis footprint on its dst ref:

      * Slice (op covers the whole row stack)  → fold absorbed the
        kernel's outer ``for row``; we re-synthesise the loop here so
        the HW still gets one issue per row.
      * Concrete index (var / int)             → the kernel's outer
        ``for row`` is still in mid_ir as a ``For`` and the walker
        will render it; we just emit the leaf.
    """
    if op.op not in _ROW_FP_BINOP_TO_INTRIN:
        raise ToPlenaError(
            f"unsupported broadcast Elementwise op {op.op!r}"
        )
    intrin = _ROW_FP_BINOP_TO_INTRIN[op.op]
    bcast_src = None
    bcast_axes = None
    direct_src = None
    for s, s_axes in zip(op.srcs, op.src_axes):
        if isinstance(s, Broadcast):
            bcast_src = s
            bcast_axes = s_axes
        else:
            direct_src = s
    if bcast_src is None or direct_src is None:
        raise ToPlenaError(
            "broadcast Elementwise expected one BufferRef + one Broadcast src"
        )
    row_footprint = _row_footprint(op.dst)
    if row_footprint > 1:
        row_var = _fresh_var("row")
        wrap_rows = row_footprint
    else:
        # Single-row leaf: pick the row var from the dst's row-axis
        # index via the op's per-axis role table (mid_ir source of
        # truth). The enclosing kernel-written for already bound it
        # — int IntImm or VarRef rendered with name-cached identity
        # so symbol_table lookups match.
        row_axis = _row_axis_index_from_axes(
            op.dst_axes, ctx=f"row_*_fp[{op.op.value}] dst",
        )
        row_var = _render_idx_as_primexpr(op.dst.indices[row_axis])
        wrap_rows = None
    # Lane axis only exists when a cluster wraps the op; otherwise the
    # FP buffer has no lane dim and the VRAM dst's head index is 0.
    if cluster_axis_name is not None:
        lane_var: _tir.PrimExpr = _make_loop_var(cluster_axis_name)
    else:
        lane_var = _tir.IntImm(_INT32, 0)
    # Resolve fp_addr indices via the mid_ir src_axes role table —
    # same approach as ``_lower_bare_reduce``: a Slice on a CLUSTER
    # axis gets ``lane_var``, on a BATCH axis ``row_var``; concrete
    # indices (the kernel pinned them) are rendered as-is. This
    # replaces the old packed / non-packed branch that hard-coded
    # ``(lane_var, row_var)`` on packed-head and silently rendered
    # Slice -> 0 on non-packed paths.
    fp_addr = _hlir.BufferElement(
        buffer=bcast_src.src.buffer.name,
        indices=_render_ref_with_role_axes(
            bcast_src.src, bcast_axes,
            row_var=row_var, lane_var=lane_var,
            ctx=f"row_*_fp[{op.op.value}] bcast src",
        ),
    )
    src_region = _build_row_at_region(
        direct_src.buffer.name, row_var=row_var, lane_var=lane_var,
        ctx="row_*_fp src",
    )
    dst_region = _build_row_at_region(
        op.dst.buffer.name, row_var=row_var, lane_var=lane_var,
        ctx="row_*_fp dst",
    )
    leaf = _hlir.Op(
        kind=intrin,
        buffer_args=[src_region, dst_region],
        scalar_args=[fp_addr],
        annotations={"source": f"bare Elementwise[{op.op.value}] w/ broadcast"},
    )
    if wrap_rows is None:
        return leaf
    return _hlir.make_for_op(loop_var=row_var, extent=wrap_rows, body=[leaf])


def _fp_buffer_element_from_ref(ref: BufferRef) -> _hlir.BufferElement:
    """Build an HLIR ``BufferElement`` from a mid_ir ``BufferRef`` whose
    indices already address a single scalar position (i.e. the enclosing
    mid_ir loops bind every index var). All indices are passed through
    ``_render_idx_as_primexpr`` so loop-var strings become the same
    ``tir.Var`` objects the surrounding HLIR ``for`` ops use."""
    return _hlir.BufferElement(
        buffer=ref.buffer.name,
        indices=tuple(_render_idx_as_primexpr(i) for i in ref.indices),
    )


def _lower_bare_fp_scalar_elementwise(
    op: Elementwise,
    cluster_extent: Optional[int],
    cluster_axis_name: Optional[str] = None,
    cluster_axis_var: "Optional[VarRef]" = None,
) -> _hlir.Op:
    """Bare elementwise on FPRAM rank-1 per-lane state → ``for lane:
    fp_<op>_at(<addr exprs>)``.

    Two shapes show up here:

      * ``M_OLD[row] = M_INIT[row]`` — single-scalar update, already
        nested inside a kernel-written ``for row`` (rendered to a HLIR
        for op by the walker). Emit a single ``fp_<op>_at`` leaf.

      * ``for m in T.Parallel(MLEN): A_sh_acc[m] = ...`` — fold
        absorbed the parallel loop into ``axis=-1, size=N``. FPRAM has
        no vector op, so we re-emit a ``for`` over ``size`` issuing one
        ``S_*_FP`` per element. The loop var name comes from the
        mid_ir Elementwise's dst index (preserving Var identity with
        the rendered indices below).
    """
    if op.op in _FP_AT_BINOP_TO_INTRIN:
        intrin = _FP_AT_BINOP_TO_INTRIN[op.op]
    elif op.op in _FP_AT_UNARY_TO_INTRIN:
        intrin = _FP_AT_UNARY_TO_INTRIN[op.op]
        # COPY with srcs=[] is fold's zero-fill sentinel — route to the
        # FPRAM-side zero op so the emitter sees only the dst address.
        if op.op == UnaryOp.COPY and not op.srcs:
            intrin = "fp_zero_at"
    else:
        raise ToPlenaError(
            f"unsupported FPRAM Elementwise op {op.op!r}"
        )
    # _emit_fp_scalar_op_at signature: for unary/copy, scalar_args =
    # (src_addr, dst_addr); for binary, (lhs_addr, rhs_addr, dst_addr).
    src_elements = [_fp_buffer_element_from_ref(s) for s in op.srcs]
    dst_element = _fp_buffer_element_from_ref(op.dst)
    leaf = _hlir.Op(
        kind=intrin,
        buffer_args=[],
        scalar_args=src_elements + [dst_element],
        annotations={"source": f"bare FPRAM Elementwise[{op.op.value}]"},
    )
    # Unroll the SIMD axis when fold absorbed a T.Parallel into the op:
    # FPRAM has no vector ISA, so emit one S_*_FP per element.
    if op.axis == -1 and op.size > 1:
        # Identify the SIMD-axis loop var: the VarRef idx in dst that
        # isn't the cluster phase axis. With a cluster, FP_LANE expansion
        # gives dst rank 2 — (lane_var, row_var); without a cluster it's
        # rank 1 — (row_var,). Skip the cluster axis (if present) and the
        # remaining single VarRef is the SIMD axis.
        candidates = [
            i for i in op.dst.indices
            if isinstance(i, VarRef)
            and (cluster_axis_name is None or i.name != cluster_axis_name)
        ]
        if len(candidates) != 1:
            raise ToPlenaError(
                f"FPRAM elementwise with axis=-1 size={op.size} expects "
                f"exactly one non-cluster VarRef in dst; got dst "
                f"{op.dst.buffer.name!r} indices {list(op.dst.indices)!r} "
                f"(cluster_axis={cluster_axis_name!r})"
            )
        idx = candidates[0]
        loop_var = _make_loop_var(idx.name)
        return _hlir.make_for_op(loop_var=loop_var, extent=op.size, body=[leaf])
    return leaf


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


_MULTI_LANE_OP_KINDS = frozenset({
    "dma_h2v", "dma_h2m", "dma_v2h",
    "dma_h2v_slice", "dma_h2m_slice", "dma_v2h_slice",
    "btmm", "btmv",
    "v_add", "v_sub", "v_mul", "v_exp", "v_reci",
    "v_sqrt", "v_zero",
    # vram→vram copy: one V_ADD_VF (f0=0) spans a full MLEN-wide row;
    # under sync wrap the by_phase index is already folded to 0, so the
    # op covers all cluster lanes in one issue — don't re-wrap in a
    # synthetic for-lane loop.
    "copy_v_to_v",
    # vram↔fpram transfer: the ``v_fp_transfer_slice_*`` ops carry
    # the whole logical region; isa_emit splits it into per-MLEN
    # S_MAP_FP_V / S_MAP_V_FP issues. Each issue natively spans all
    # cluster lanes (sync wrap zeroes the cluster phase axis on both
    # sides), so the op is multi-lane in hardware and must not be
    # re-wrapped in a synthetic for-lane loop.
    "v_fp_transfer_slice_v_to_fp", "v_fp_transfer_slice_fp_to_v",
})


def _op_fires_once_across_lanes(op: _hlir.Op) -> bool:
    """True if ``op`` is a single HW instruction that already spans
    every lane natively (multi-lane in hardware). Such ops must NOT
    be wrapped in a synthetic ``for lane`` loop — that would re-issue
    the same multi-lane instruction lane_count times."""
    return op.kind in _MULTI_LANE_OP_KINDS


def _wrap_per_lane_ops_with_for_lane(
    body: List[_hlir.Op],
    lane_var: "_tir.Var",
    lane_extent: int,
) -> List[_hlir.Op]:
    """Emit the body of a CLUSTER ParallelAxis. Each per-lane op gets
    its OWN ``for lane in cluster_extent`` wrapper so the lane axis is
    threaded one instruction at a time (matching the kernel's program
    order). Multi-lane HW ops stay un-wrapped: they fire once and
    cover every lane natively.

    Structural ops (``for`` nodes — typically the kernel's ``for row``
    or ``for kv_block``) are recursed into: their body is rewritten
    with the same rule so per-lane ops inside still get their own
    for-lane wrapper.
    """
    out: List[_hlir.Op] = []
    for op in body:
        if op.kind == "for":
            # Recurse: inner body may contain per-lane ops that need
            # their own per-op for-lane wrapper.
            inner = _wrap_per_lane_ops_with_for_lane(
                op.body or [], lane_var, lane_extent,
            )
            out.append(_hlir.Op(
                kind="for",
                buffer_args=list(op.buffer_args),
                scalar_args=list(op.scalar_args),
                annotations=dict(op.annotations),
                body=inner,
            ))
        elif _op_fires_once_across_lanes(op):
            out.append(op)
        else:
            out.append(_hlir.make_for_op(
                loop_var=lane_var, extent=lane_extent, body=[op],
            ))
    return out


def _walk_stmts(stmts: List[Stmt],
                buf_name_to_hlir: Dict[str, _hlir.Buffer],
                cluster_extent: Optional[int],
                cluster_axis_name: Optional[str] = None,
                cluster_axis_var: "Optional[VarRef]" = None,
                ) -> List[_hlir.Op]:
    out: List[_hlir.Op] = []
    for s in stmts:
        out.extend(_walk_stmt(s, buf_name_to_hlir, cluster_extent,
                              cluster_axis_name, cluster_axis_var))
    return out


def _walk_stmt(stmt: Stmt,
               buf_name_to_hlir: Dict[str, _hlir.Buffer],
               cluster_extent: Optional[int],
               cluster_axis_name: Optional[str] = None,
               cluster_axis_var: "Optional[VarRef]" = None,
               ) -> List[_hlir.Op]:
    if isinstance(stmt, ParallelAxis):
        if stmt.kind == ParallelKind.CLUSTER:
            # CLUSTER: each per-lane op gets its own ``for lane in
            # cluster_extent`` wrapper (matching the kernel's program
            # order); multi-lane HW ops (DMA / BTMM / V-machine whole-
            # tile) stay un-wrapped because they fire across all lanes
            # natively. We recurse into structural ``for`` nodes so
            # per-lane ops nested inside (e.g. inside a kernel
            # ``for row``) still get a per-op for-lane wrapper.
            body = _walk_stmts(stmt.body, buf_name_to_hlir, stmt.extent,
                               stmt.axis_name, stmt.axis_var)
            lane_var = _axis_loop_var(stmt)
            return _wrap_per_lane_ops_with_for_lane(
                body, lane_var, stmt.extent,
            )
        # threadIdx.* axes are a GPU abstraction: PLENA HW has no
        # thread-level dispatch, every instruction is implicitly
        # broadcast across the SIMD width. Unwrap them so the body
        # runs once, not threads-many times.
        if (stmt.thread_tag is not None
                and stmt.thread_tag.startswith("threadIdx.")):
            return _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                               cluster_axis_name, cluster_axis_var)
        # BLOCK_IDX or LOGICAL_GRID → flatten to a serial for.
        body = _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name, cluster_axis_var)
        return [_hlir.make_for_op(
            loop_var=_axis_loop_var(stmt),
            extent=stmt.extent, body=body,
        )]
    if isinstance(stmt, For):
        body = _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name, cluster_axis_var)
        for_op = _hlir.make_for_op(
            _for_loop_var(stmt), stmt.extent, body=body,
        )
        for_op.annotations["loop_kind"] = stmt.kind
        return [for_op]
    if isinstance(stmt, Async):
        # By pass_5 every Async should be MultiLaneOp; if it lingers,
        # walk through.
        return _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name, cluster_axis_var)
    if isinstance(stmt, MultiLaneOp):
        # The actual cluster extent comes from the enclosing CLUSTER
        # ParallelAxis (forwarded as ``cluster_extent``). Pass it down
        # so per-op lowering helpers can use it directly — both for
        # the HW-side lane_count scalar and for any synthetic
        # ``for lane`` they wrap around per-lane FPRAM ops.
        actual_lane = cluster_extent or 1
        return [_lower_multi_lane(stmt, buf_name_to_hlir, actual_lane)]
    if isinstance(stmt, Dma):
        # Bare Dma — shouldn't normally happen post-pipeline, but
        # support it as a single-lane DMA.
        return [_lower_multi_lane_dma(stmt, 1, buf_name_to_hlir)]
    if isinstance(stmt, Gemm):
        if stmt.kind == "btmm":
            # Shouldn't be bare; treat as single-lane btmm.
            return [_lower_multi_lane_btmm(stmt, 1, buf_name_to_hlir)]
        return [_lower_bare_per_head_gemm(
            stmt, cluster_extent, cluster_axis_name,
            cluster_axis_var=cluster_axis_var,
            buf_name_to_hlir=buf_name_to_hlir, lane_modes=_LANE_MODES,
        )]
    if isinstance(stmt, Reduce):
        return [_lower_bare_reduce(stmt, cluster_extent, cluster_axis_name,
                                   cluster_axis_var=cluster_axis_var)]
    if isinstance(stmt, Elementwise):
        has_broadcast = any(isinstance(s, Broadcast) for s in stmt.srcs)
        if has_broadcast:
            return [_lower_bare_broadcast_elementwise(
                stmt, cluster_extent, cluster_axis_name,
                cluster_axis_var=cluster_axis_var,
            )]
        dst_scope = buf_name_to_hlir[stmt.dst.buffer.name].scope
        if dst_scope == _scope.FPRAM:
            # Per-lane FPRAM scalar update (M_OLD[row] = M_INIT[row]
            # etc.). Lower to a ``for lane: fp_<op>_at`` loop — the
            # ``row`` loop is the enclosing mid_ir For (already a HLIR
            # for op by now).
            return [_lower_bare_fp_scalar_elementwise(
                stmt, cluster_extent, cluster_axis_name,
                cluster_axis_var=cluster_axis_var,
            )]
        # Pure elementwise that wasn't wrapped (shouldn't happen if
        # pass_4 ran). Treat as single-lane multi_lane.
        return [_lower_multi_lane_elementwise(stmt, cluster_extent or 1, buf_name_to_hlir)]
    if isinstance(stmt, RawStore):
        raise ToPlenaError(
            f"RawStore lowering is not implemented yet — fold did not "
            f"recognise the op pattern. dst={stmt.dst.buffer.name}"
            f"{list(stmt.dst.indices)} := {stmt.value!r}"
        )
    raise ToPlenaError(f"unhandled mid_ir stmt {type(stmt).__name__}")


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run(func: MidFunc,
        build_dir: Optional[Path] = None,
        mlen: int = 64) -> _hlir.HLIRModule:
    """Lower a MidFunc to HLIRModule.

    ``mlen`` is the hardware vector lane width (V_*_V row width). It is
    stashed in ``_HW_MLEN`` so lowerings that emit per-row HW vector
    ops (vram→vram copy etc.) can stride by it without having to
    reverse-engineer it from buffer shapes.

    If ``build_dir`` is given, write a ``<func.name>.midir.txt`` snapshot
    there before lowering — useful for diff against the legacy pipeline
    or for post-mortem when HLIR looks wrong.
    """
    global _HW_MLEN
    _HW_MLEN = int(mlen)
    _VAR_CACHE.clear()
    _LANE_MODES.clear()
    _LANE_AXIS_INFO.clear()
    _BUFFER_HLIR_AXES.clear()
    # Register the split-axis form for each logical lane axis. mid_ir
    # carries ``lane_axes`` (one per cluster) and ``cluster_counts``;
    # the original lane var (as a ``VarRef``) appears in BufferRef
    # indices as the un-split logical view (kept on global refs whose
    # view pass skipped), and must expand to ``phase_var + number_var
    # * count`` for ISA materialisation.
    #
    # Look up each pair (original VarRef, phase VarRef, number VarRef,
    # count) by walking the body for matching CLUSTER ParallelAxes.
    # Split records ``original_axis_var`` on both the CLUSTER (phase)
    # and its enclosing number axis, with parent_grid_axis_name on the
    # CLUSTER pointing at the number axis by name.
    _populate_lane_axis_info(
        func,
        lane_axes=getattr(func, "lane_axes", []) or [],
        cluster_counts=getattr(func, "cluster_counts", []) or [],
    )
    if build_dir is not None:
        build_dir = Path(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)
        dump_path = build_dir / f"{func.name}.midir.txt"
        dump_path.write_text(format_func(func))

    # Use-driven scope overrides (e.g. Gemm B operands → MRAM).
    overrides = _infer_scope_overrides(func)
    # Lane-expansion modes for each non-global buffer (COL_PACK /
    # ROW_STACK / FP_LANE / BSHD_LIFT). Used to reshape buffers and
    # fold ref indices into canonical 4D BSHD form.
    #
    # If the kernel didn't go through cluster fusion (no lane axes, or
    # every non-global buffer already covers a full MLEN-wide vector),
    # ``split`` and friends were no-ops and buffers still have their
    # as-written shapes. Lane expansion in that case would force an
    # extra cluster dim onto already-correct shapes and crash
    # ``_expand_buffer_shape_with_cluster``'s rank check. Skip it.
    cluster_skipped = should_skip_cluster(func)
    if cluster_skipped:
        lane_modes: Dict[str, str] = {}
        lane_count = 1
    else:
        lane_modes = _infer_lane_modes(func)
        lane_count = func.cluster_counts[0] if func.cluster_counts else 1
    _LANE_MODES.update(lane_modes)

    # Build buffer table. Track per-buffer ref-rewrite recipe so refs
    # in the op stream can be transformed into the buffer's post-
    # expansion 4D coordinate system after the walk.
    #
    # Two sources of rank growth, both needing to keep refs in sync
    # with their buffers:
    #   * pad-to-4D path: 1D/2D author-declared on-chip buffers got
    #     extent-1 axes inserted; refs need the same fills (start=0,
    #     extent=1) at those positions.
    #   * cluster-expand path: rank-3 ``(LANE, S, D)`` from
    #     ``split._grow_buffer`` got permuted to 4D BSHD; refs are
    #     still rank 3 (view pass prepended phase) and need the
    #     cluster-mode-specific permutation in
    #     ``_CLUSTER_REF_REWRITE``.
    # Kernel-wide layout hint (``T.func_attr({"plena.layout": "NCHW"})``)
    # stamped onto every HLIR Buffer so downstream stride math picks the
    # right axis. Default ``BSHD`` matches what the rest of the pipeline
    # assumes when no hint is provided.
    kernel_layout = "BSHD"
    if func.attrs is not None and "plena.layout" in func.attrs:
        kernel_layout = str(func.attrs["plena.layout"])

    buf_name_to_hlir: Dict[str, _hlir.Buffer] = {}
    pad_inserts: Dict[str, Tuple[int, ...]] = {}
    cluster_modes_for_refs: Dict[
        str, Tuple[str, Optional[int], Tuple[int, ...]],
    ] = {}
    # Reset module-global mid_ir-axes ↔ hlir-axes alignment tables so a
    # fresh compile doesn't see stale entries from a previous one.
    _PAD_INSERTS.clear()
    _CLUSTER_MODES.clear()
    for buf in list(func.params) + list(func.allocs):
        if buf.name in buf_name_to_hlir:
            continue
        hlir_buf, inserts = _make_hlir_buffer(
            buf,
            override=overrides.get(buf.name),
            lane_count=lane_count,
            mode=lane_modes.get(buf.name),
            kernel_layout=kernel_layout,
        )
        buf_name_to_hlir[buf.name] = hlir_buf
        # Stamp HLIR-side per-dim role tags for every buffer, derived
        # directly from the post-expansion shape + cluster_dim. Ops
        # that need to identify dims by role (row_*_at family) read
        # this through ``_BUFFER_HLIR_AXES``.
        _BUFFER_HLIR_AXES[buf.name] = _hlir_axes_for_buffer(hlir_buf)
        if inserts:
            pad_inserts[buf.name] = inserts
            _PAD_INSERTS[buf.name] = inserts
        else:
            # Cluster-expand route: track ``(mode, mid_ir_cluster_dim,
            # new_4d_shape)`` so the walker can apply
            # ``_rewrite_ref_for_cluster_mode``. ``cluster_dim`` on the
            # mid_ir BufferDef tells us where the lane axis sits in
            # the rank-3 source (set by ``split._grow_buffer`` and
            # tracked through view/burn_view); the new 4D shape is
            # needed to recover the lane extent under sync-wrap.
            mode = lane_modes.get(buf.name)
            if (mode is not None
                    and mode != _MODE_FP_LANE
                    and not _is_global_scope(buf.scope)):
                cluster_modes_for_refs[buf.name] = (
                    mode, buf.cluster_dim, tuple(hlir_buf.shape),
                )
                _CLUSTER_MODES[buf.name] = (
                    mode, buf.cluster_dim, tuple(hlir_buf.shape),
                )

    # Copy hoisted-constant values onto the matching HLIR buffers so
    # ``--dump-buffer-addrs`` can emit them and the test harness can
    # auto-preload. The pre-pass (frontend/passes/hoist_float_constants.py)
    # stashes ``{buffer_name: value}`` on PrimFunc.attrs; mid_ir passes
    # carry that attr forward to ``func.attrs`` unchanged.
    if func.attrs is not None and "plena.hoisted_constants" in func.attrs:
        for name, value in func.attrs["plena.hoisted_constants"].items():
            hlir_buf = buf_name_to_hlir.get(str(name))
            if hlir_buf is not None:
                hlir_buf.constant_value = float(value)

    # Walk the body.
    ops = _walk_stmts(func.body, buf_name_to_hlir, cluster_extent=None)

    # Synchronise refs with their buffers' post-expansion 4D rank.
    # One walker, two rewrite mechanisms. In-place on op.buffer_args.
    if pad_inserts or cluster_modes_for_refs:
        _rewrite_refs_to_4d(ops, pad_inserts, cluster_modes_for_refs)

    return _hlir.HLIRModule(
        name=func.name,
        buffers=buf_name_to_hlir,
        ops=ops,
        param_names=[b.name for b in func.params],
    )


def _pad_tuple_at(values, inserts: Tuple[int, ...], fill):
    """Insert ``fill`` at each position in ``inserts`` (positions in
    OUTPUT coords, ascending). E.g. inserting (0, 2) into ('a', 'b')
    with fill 0 yields (0, 'a', 0, 'b')."""
    out = list(values)
    for pos in inserts:
        out.insert(pos, fill)
    return tuple(out)


def _rewrite_refs_to_4d(
    ops,
    pad_inserts: Dict[str, Tuple[int, ...]],
    cluster_modes: Dict[str, Tuple[str, Optional[int], Tuple[int, ...]]],
) -> None:
    """Bring every ``VramRegion`` ref in line with its buffer's post-
    expansion 4D rank, in place.

    Picks the rewrite per buffer: pad-to-4D buffers get extent-1
    fills inserted at recorded positions; cluster-expanded buffers
    use the lane-position-driven rewrite in
    :func:`_rewrite_ref_for_cluster_mode`. BufferSlice (HBM-parent
    slices) is not affected — HBM never gets rank-expanded. Recurses
    into structured ops' bodies.
    """
    for op in ops:
        new_bargs = []
        for a in op.buffer_args:
            if isinstance(a, (_hlir.VramRegion, _hlir.MramRegion)):
                # Idempotent guard: lower paths that already emit 4D
                # regions don't need any extra padding here.
                if len(a.starts) == 4 and len(a.extents) == 4:
                    new_bargs.append(a)
                    continue
                ctor = type(a)
                if a.parent in pad_inserts:
                    inserts = pad_inserts[a.parent]
                    a = ctor(
                        parent=a.parent,
                        starts=_pad_tuple_at(a.starts, inserts, 0),
                        extents=_pad_tuple_at(a.extents, inserts, 1),
                    )
                elif a.parent in cluster_modes:
                    mode, old_cluster_dim, new_shape = cluster_modes[a.parent]
                    a = ctor(
                        parent=a.parent,
                        starts=_rewrite_ref_for_cluster_mode(
                            tuple(a.starts), mode, old_cluster_dim,
                            new_shape, is_extent=False,
                        ),
                        extents=_rewrite_ref_for_cluster_mode(
                            tuple(a.extents), mode, old_cluster_dim,
                            new_shape, is_extent=True,
                        ),
                    )
            new_bargs.append(a)
        op.buffer_args = new_bargs
        if op.body:
            _rewrite_refs_to_4d(op.body, pad_inserts, cluster_modes)


__all__ = ["run", "ToPlenaError"]
