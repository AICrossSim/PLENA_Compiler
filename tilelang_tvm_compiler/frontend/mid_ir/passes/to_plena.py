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
    MultiLaneOp(inner=Elementwise pure) →  Op(kind="tile_add" / "tile_sub" /
                                                "tile_mul" / "tile_exp" /
                                                "tile_zero" / ...)
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
from ..ir import (
    BinOp, UnaryOp, ReduceOp,
    BufferDef, BufferRef, Slice,
    Dma, Gemm, Elementwise, Broadcast, Reduce, RawStore,
    For, Async, MultiLaneOp,
    ParallelAxis, ParallelKind,
    MidFunc, Stmt, format_func,
)


class ToPlenaError(RuntimeError):
    pass


def _make_loop_var(name: str) -> _tir.Var:
    """Build a tir.Var for use as an HLIR ``for`` loop_var annotation.

    Shares ``_VAR_CACHE`` with index-expression rendering so for-ops
    and the indices that reference them resolve to the same Python
    object (the ISA pass keys ``symbol_table`` by identity).
    """
    return _get_var(name)


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


def _make_hlir_buffer(
    buf: BufferDef,
    override: Optional[str] = None,
    lane_count: Optional[int] = None,
    mode: Optional[str] = None,
) -> _hlir.Buffer:
    is_global = buf.scope == "global" or buf.scope.startswith("global.")
    if mode is not None and not is_global and lane_count is not None:
        shape, cluster_dim = _expand_buffer_shape_with_cluster(buf, lane_count, mode)
        shape = tuple(shape)
    else:
        shape = tuple(int(d) for d in buf.shape)
        cluster_dim = buf.cluster_dim
    return _hlir.Buffer(
        name=buf.name,
        scope=_map_scope(buf.scope, len(buf.shape), override),
        shape=shape,
        dtype=buf.dtype,
        cluster_dim=cluster_dim,
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
                if op.a.buffer.scope != "global":
                    record(op.a.buffer.name, _MODE_COL_PACK)
                if op.b.buffer.scope != "global":
                    record(op.b.buffer.name, _MODE_COL_PACK)
                if op.c.buffer.scope != "global":
                    record(op.c.buffer.name, _MODE_ROW_STACK)
            else:
                if op.a.buffer.scope != "global":
                    record(op.a.buffer.name, _MODE_ROW_STACK)
                if op.b.buffer.scope != "global":
                    record(op.b.buffer.name, _MODE_COL_PACK)
                if op.c.buffer.scope != "global":
                    record(op.c.buffer.name, _MODE_COL_PACK)
            return
        if isinstance(op, Dma):
            for ref in (op.src, op.dst):
                if ref.buffer.scope == "global":
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
                if ref.buffer.scope == "global":
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
        if buf.scope == "global":
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
    rows = int(buf.shape[0])
    last = int(buf.shape[2])
    if mode == _MODE_COL_PACK:
        return [1, rows, int(lane_count), last], 2
    if mode == _MODE_BSHD_LIFT:
        return [1, rows, 1, last], None
    raise ToPlenaError(f"unknown lane mode {mode!r} for {buf.name!r}")


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
            # would otherwise default it.
            if op.b.buffer.scope != "global":
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
    every non-global ref. burn_view may permute it to a non-zero
    position. For DMA / btmm / pure-elementwise ops that fire across
    all lanes (wrapped in MultiLaneOp) the phase index is a
    whole-lane-axis access — equivalent to Slice for whole-buffer
    purposes. Recognise: at most one non-Slice index, and that index
    is a bare string (the cluster phase var name).
    """
    if not ref.indices:
        return True
    if all(isinstance(i, Slice) for i in ref.indices):
        return True
    non_slice = [i for i in ref.indices if not isinstance(i, Slice)]
    if len(non_slice) == 1 and isinstance(non_slice[0], str):
        return True
    return False


# ---------------------------------------------------------------------------
# Op-arg construction
# ---------------------------------------------------------------------------


_INT32 = "int32"

# Cache (name → tir.Var) so multiple ranged_slice / compound rewrites
# referring to the same loop var produce the *same* Var object — ISA
# pass identifies bindings by object identity in its symbol_table.
_VAR_CACHE: Dict[str, "_tir.Var"] = {}

# Module-global lane modes table, populated by ``run`` at the start of
# each compile and read by per-op lowering helpers. Cleared by ``run``.
_LANE_MODES: Dict[str, str] = {}


# Logical lane axis (e.g. ``"by"``) → (phase_name, number_name, count).
# Populated by ``run()`` from ``func.lane_axes`` + ``func.cluster_counts``.
# ``_render_idx_as_primexpr`` consults this to expand a bare ``by``
# reference into ``by_phase + by_number * lane_count`` so the ISA
# materializer sees only the split axes it has bound.
_LANE_AXIS_INFO: Dict[str, "tuple[str, str, int]"] = {}


def _get_var(name: str) -> "_tir.Var":
    v = _VAR_CACHE.get(name)
    if v is None:
        v = _tir.Var(name, _INT32)
        _VAR_CACHE[name] = v
    return v


def _render_idx_as_primexpr(idx):
    """Like ``_render_idx`` but returns a value suitable for
    ``hlir.BufferSlice.starts``: ints stay ints; bare var names become
    ``tir.Var``; compound dicts become real ``tir.PrimExpr`` trees so
    the ISA pass's ``_build_slice_offset_expr`` can multiply them by a
    stride directly."""
    if isinstance(idx, Slice):
        return 0
    if isinstance(idx, int):
        return int(idx)
    if isinstance(idx, str):
        # Logical lane axes (e.g. ``"by"``) get expanded to their split
        # form ``by_phase + by_number * lane_count`` so the ISA layer,
        # which only binds the split axes, can materialise the index.
        info = _LANE_AXIS_INFO.get(idx)
        if info is not None:
            phase, number, count = info
            return _get_var(phase) + _get_var(number) * _tir.IntImm(_INT32, count)
        return _get_var(idx)
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
    BinOp.ADD: "tile_add",
    BinOp.SUB: "tile_sub",
    BinOp.MUL: "tile_mul",
}


_UNARY_TO_INTRIN = {
    UnaryOp.EXP: "tile_exp",
    UnaryOp.RECI: "tile_reci",
    UnaryOp.SQRT: "tile_sqrt",
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
                          ) -> _hlir.Op:
    src_scope = buf_name_to_hlir[op.src.buffer.name].scope
    dst_scope = buf_name_to_hlir[op.dst.buffer.name].scope
    # VRAM → VRAM "copy" isn't a HW DMA — emit ``copy_v_to_v``
    # (V_ADD_VF dst, src, f0=0) instead.
    if src_scope == _scope.VRAM and dst_scope == _scope.VRAM:
        return _lower_vram_to_vram_copy(
            op, buf_name_to_hlir, cluster_axis_name=cluster_axis_name,
        )
    # VRAM ↔ FPRAM: not a DMA either — single S_MAP_FP_V / S_MAP_V_FP
    # per mlen-wide row. tilelang authors write these as T.copy and
    # rely on us to route them to the right HW path.
    if src_scope == _scope.VRAM and dst_scope == _scope.FPRAM:
        return _lower_v_fp_transfer(
            op, "v_to_fp", buf_name_to_hlir, cluster_axis_name,
        )
    if src_scope == _scope.FPRAM and dst_scope == _scope.VRAM:
        return _lower_v_fp_transfer(
            op, "fp_to_v", buf_name_to_hlir, cluster_axis_name,
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
                     phase_var_zero: Optional[str] = None) -> _tir.PrimExpr:
    """Compute ``ref``'s starting element offset in row-major flat layout.

    Iterates buffer.shape backwards accumulating stride; concrete indices
    contribute ``idx * stride``. ``Slice`` is whole-axis (start = 0),
    contributes nothing. ``ranged_slice(start_expr, extent)`` contributes
    ``start_expr * stride``. When ``phase_var_zero`` is set (the cluster
    phase axis name, e.g. ``"by_phase"``), bare-string occurrences of
    that name are treated as 0 — mirrors the ``_is_whole_buffer_ref``
    convention for sync-wrap multi-lane ops where the phase index just
    marks "this op covers every lane in lockstep"."""
    offset: _tir.PrimExpr = _tir.IntImm(_INT32, 0)
    stride = 1
    for dim, idx in zip(reversed(ref.buffer.shape), reversed(ref.indices)):
        if isinstance(idx, Slice):
            pass  # whole-axis access — start is 0
        elif phase_var_zero is not None and isinstance(idx, str) and idx == phase_var_zero:
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
                             ) -> _hlir.Op:
    """``T.copy(vram_src, vram_dst)`` (per-by_o slice or whole-buffer).

    Each ``copy_v_to_v`` HW emit handles ONE MLEN-wide row. If the copy
    spans multiple rows we wrap in ``for row``. Offset is computed from
    each ref's mid_ir indices. When invoked inside a sync wrap (the
    enclosing ``MultiLaneOp`` covers all cluster lanes in one HW op),
    the cluster phase axis (e.g. ``"by_phase"``) is treated as 0 in
    offset math — same convention ``_is_whole_buffer_ref`` uses.
    """
    src_buf = buf_name_to_hlir[op.src.buffer.name]
    dst_buf = buf_name_to_hlir[op.dst.buffer.name]
    mlen = max(int(d) for d in src_buf.shape[-1:])  # innermost = mlen-aligned
    # How many mlen-rows does this copy cover? Use the smaller of src/dst
    # element counts the slice actually touches. With a single concrete
    # index the slice is buffer_elem_count / shape[0], etc. — for our use
    # case (single by_o slice) the copy is one row; for whole-buffer it's
    # buf_elements / mlen.
    src_elem = _ref_touch_count(op.src)
    dst_elem = _ref_touch_count(op.dst)
    n_elem = min(src_elem, dst_elem)
    if n_elem % mlen != 0:
        raise ToPlenaError(
            f"vram→vram copy element count {n_elem} not a multiple of "
            f"MLEN {mlen}: src={op.src.buffer.name!r} dst={op.dst.buffer.name!r}"
        )
    n_rows = n_elem // mlen
    src_off_base = _ref_flat_offset(op.src, phase_var_zero=cluster_axis_name)
    dst_off_base = _ref_flat_offset(op.dst, phase_var_zero=cluster_axis_name)
    if n_rows == 1:
        return _hlir.Op(
            kind="copy_v_to_v",
            buffer_args=[op.src.buffer.name, op.dst.buffer.name],
            scalar_args=[src_off_base, dst_off_base],
            annotations={"source": "vram→vram copy"},
        )
    row_var = _fresh_var("row")
    row_stride = _tir.Mul(row_var, _tir.IntImm(_INT32, mlen))
    src_off = _tir.Add(src_off_base, row_stride) if (
        not (isinstance(src_off_base, _tir.IntImm) and int(src_off_base.value) == 0)
    ) else row_stride
    dst_off = _tir.Add(dst_off_base, row_stride) if (
        not (isinstance(dst_off_base, _tir.IntImm) and int(dst_off_base.value) == 0)
    ) else row_stride
    leaf = _hlir.Op(
        kind="copy_v_to_v",
        buffer_args=[op.src.buffer.name, op.dst.buffer.name],
        scalar_args=[src_off, dst_off],
        annotations={"source": "vram→vram copy"},
    )
    return _hlir.make_for_op(loop_var=row_var, extent=n_rows, body=[leaf])


def _lower_v_fp_transfer(
    op: Dma,
    direction: str,                          # "v_to_fp" or "fp_to_v"
    buf_name_to_hlir: Dict[str, _hlir.Buffer],
    cluster_axis_name: Optional[str] = None,
) -> _hlir.Op:
    """``T.copy(vram, fpram)`` / ``T.copy(fpram, vram)`` → single
    ``S_MAP_*_FP/V`` per mlen-wide row.

    HLIR ops emitted:
      * ``row_load_v_to_fp``  buffer_args=[vram]  scalars=[vram_offset, fp_addr]
      * ``row_store_fp_to_v`` buffer_args=[vram]  scalars=[fp_addr, vram_offset]

    Wrapped in ``for row { ... }`` when the copy spans multiple
    mlen-rows. Sync wrap collapses the cluster phase axis to 0 the
    same way ``copy_v_to_v`` does.
    """
    if direction == "v_to_fp":
        vram_ref, fp_ref = op.src, op.dst
        kind = "row_load_v_to_fp"
    else:
        vram_ref, fp_ref = op.dst, op.src
        kind = "row_store_fp_to_v"
    vram_buf = buf_name_to_hlir[vram_ref.buffer.name]
    fp_buf = buf_name_to_hlir[fp_ref.buffer.name]
    mlen = int(vram_buf.shape[-1])
    src_elem = _ref_touch_count(op.src)
    dst_elem = _ref_touch_count(op.dst)
    n_elem = min(src_elem, dst_elem)
    if n_elem % mlen != 0:
        raise ToPlenaError(
            f"v↔fp transfer element count {n_elem} not a multiple of "
            f"MLEN {mlen}: src={op.src.buffer.name!r} dst={op.dst.buffer.name!r}"
        )
    n_rows = n_elem // mlen
    vram_off_base = _ref_flat_offset(vram_ref, phase_var_zero=cluster_axis_name)
    # The HW instruction (S_MAP_FP_V / S_MAP_V_FP) transfers VLEN=MLEN
    # contiguous fp slots in one issue across all cluster lanes. Under
    # sync wrap the cluster-phase index on the FP ref must be treated
    # as 0 — same convention vram_off_base uses. cluster_dim records
    # exactly which physical axis carries the phase index.
    fp_indices = _zero_cluster_axis_in_fp_indices(
        fp_ref, cluster_axis_name,
    )
    fp_addr = _hlir.BufferElement(
        buffer=fp_buf.name,
        indices=fp_indices,
    )

    def _make_leaf(vram_off, fp_addr_arg):
        if direction == "v_to_fp":
            scalar_args = [vram_off, fp_addr_arg]
        else:
            scalar_args = [fp_addr_arg, vram_off]
        return _hlir.Op(
            kind=kind,
            buffer_args=[vram_buf.name],
            scalar_args=scalar_args,
            annotations={"source": f"T.copy vram↔fp ({direction})"},
        )

    if n_rows == 1:
        return _make_leaf(vram_off_base, fp_addr)
    row_var = _fresh_var("row")
    row_stride = _tir.Mul(row_var, _tir.IntImm(_INT32, mlen))
    vram_off = (
        row_stride if (isinstance(vram_off_base, _tir.IntImm)
                       and int(vram_off_base.value) == 0)
        else _tir.Add(vram_off_base, row_stride)
    )
    # FPRAM advances by mlen elements per row too.
    fp_addr_stepped = _hlir.BufferElement(
        buffer=fp_buf.name,
        indices=fp_indices,
    )  # NB: fp ref indices stay; row_stride lives in vram offset only.
    leaf = _make_leaf(vram_off, fp_addr_stepped)
    return _hlir.make_for_op(loop_var=row_var, extent=n_rows, body=[leaf])


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


def _lower_multi_lane_btmm(op: Gemm, lane_count: int) -> _hlir.Op:
    # Dispatch BTMV (decode-style, LHS rows == 1) vs BTMM (rows > 1) on
    # the LHS row footprint. Both fire across all lanes in one HW issue;
    # BTMV reads a single q-row, BTMM reads MLEN q-rows.
    rows = _logical_rows_from_buf(op.a)
    kind = "btmv" if rows == 1 else "btmm"
    return _hlir.Op(
        kind=kind,
        buffer_args=[
            _make_buffer_arg(op.a),
            _make_buffer_arg(op.b),
            _make_buffer_arg(op.c),
        ],
        scalar_args=[lane_count],
        annotations={"source": f"MultiLaneOp(Gemm[{kind}])",
                     "transpose_b": op.transpose_b},
    )


def _lower_multi_lane_elementwise(
    op: Elementwise, lane_count: int,
    buf_name_to_hlir: Optional[Dict[str, _hlir.Buffer]] = None,
    cluster_axis_name: Optional[str] = None,
) -> _hlir.Op:
    """Pure elementwise (no Broadcast srcs) → tile_add / tile_exp /
    row_exp / etc.

    Routing is decided by the dst ref's row-axis footprint:

      * Slice (op covers the whole row stack) → whole-tile intrinsic
        (``tile_exp`` / ``tile_add`` / ...). The HW op fires once
        across all on-chip rows.
      * Concrete var/int → single-row intrinsic (``row_exp`` for
        unary; binary elementwise on whole-row VRAM stays at MLEN
        width so ``tile_add`` etc. still applies). The enclosing
        kernel-written ``for row`` is rendered by the walker.

    If the dst lives in FPRAM (rank-1 per-lane state), redirect to the
    ``for lane: for row: fp_<op>_at`` template — ``copy_v_to_v`` etc.
    don't apply to scalar FP slots.
    """
    if (buf_name_to_hlir is not None
            and buf_name_to_hlir[op.dst.buffer.name].scope == _scope.FPRAM):
        return _lower_bare_fp_scalar_elementwise(op, lane_count, cluster_axis_name)
    # Per-row VRAM unary path: emit one ``row_<op>`` per row instead of
    # a whole-tile ``tile_<op>``. Required when the dst's row axis is
    # a concrete index — meaning an enclosing ``for row`` already
    # iterates and the HW op must only touch one row each issue.
    if (op.op in _UNARY_TO_INTRIN
            and len(op.srcs) == 1
            and _row_footprint(op.dst) == 1):
        return _lower_per_row_unary(op, cluster_axis_name)
    if op.op in _BINOP_TO_INTRIN:
        kind = _BINOP_TO_INTRIN[op.op]
    elif op.op in _UNARY_TO_INTRIN:
        kind = _UNARY_TO_INTRIN[op.op]
        # Special: COPY with srcs=[] is the zero-fill sentinel from fold.
        if op.op == UnaryOp.COPY and not op.srcs:
            kind = "tile_zero"
    else:
        raise ToPlenaError(f"unsupported elementwise op {op.op!r}")
    buffer_args: List[Any] = []
    for s in op.srcs:
        if isinstance(s, Broadcast):
            # MultiLaneOp Elementwise shouldn't carry Broadcast — those
            # are can_async=False and stay bare.
            raise ToPlenaError(
                f"MultiLaneOp Elementwise with Broadcast src — pass_2_mark "
                f"should have set can_async=False"
            )
        buffer_args.append(s.buffer.name)
    buffer_args.append(op.dst.buffer.name)
    return _hlir.Op(
        kind=kind,
        buffer_args=buffer_args,
        scalar_args=[lane_count],
        annotations={"source": f"MultiLaneOp(Elementwise {op.op.value})"},
    )


_UNARY_TO_ROW_INTRIN = {
    UnaryOp.EXP: "row_exp",
}


def _lower_per_row_unary(
    op: Elementwise,
    cluster_axis_name: Optional[str] = None,
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
    row_var = _make_loop_var("row")
    lane_var = (_make_loop_var(cluster_axis_name)
                if cluster_axis_name else _fresh_var("lane"))
    return _hlir.Op(
        kind=intrin,
        buffer_args=[src.buffer.name, op.dst.buffer.name],
        scalar_args=[row_var, lane_var],
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
    inner = mlo.inner
    if isinstance(inner, Dma):
        return _lower_multi_lane_dma(
            inner, lane_count, buf_name_to_hlir,
            cluster_axis_name=axis_name,
        )
    if isinstance(inner, Gemm):
        return _lower_multi_lane_btmm(inner, lane_count)
    if isinstance(inner, Elementwise):
        return _lower_multi_lane_elementwise(
            inner, lane_count, buf_name_to_hlir, axis_name,
        )
    raise ToPlenaError(
        f"unsupported MultiLaneOp inner: {type(inner).__name__}"
    )


def _lane_loop_var(cluster_axis_name: Optional[str]) -> _tir.Var:
    """Pick a loop_var for the synthetic ``for lane`` that wraps a
    bare op inside a cluster. Prefer the actual cluster axis name
    (``by_phase`` — same identity view pass used in on-chip index
    expressions); fall back to ``"lane"`` for bare ops emitted
    outside any cluster (synthetic, sibling-only)."""
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

    Computed directly from ``buf.cluster_dim`` and the post-cluster
    dims: it's the product of every shape axis to the right of the
    cluster dim. ``mode`` is now just a fallback for buffers without
    a tracked ``cluster_dim`` (legacy / pre-cluster-dim paths)."""
    shape = [int(d) for d in buf.shape]
    if buf.cluster_dim is not None:
        stride = 1
        for axis in range(buf.cluster_dim + 1, len(shape)):
            stride *= shape[axis]
        return stride
    # Legacy fallback paths (kept for safety; new buffers always carry
    # cluster_dim).
    if mode == _MODE_ROW_STACK:
        return shape[1] * shape[2] * shape[3]
    if mode == _MODE_COL_PACK:
        return shape[3]
    if mode == _MODE_FP_LANE:
        return shape[1]
    return 0


def _lower_bare_per_head_gemm(
    op: Gemm,
    cluster_extent: Optional[int],
    cluster_axis_name: Optional[str] = None,
    buf_name_to_hlir: Optional[Dict[str, _hlir.Buffer]] = None,
    lane_modes: Optional[Dict[str, str]] = None,
) -> _hlir.Op:
    """Bare (non-async) per-head matmul → ``for lane: plena.matmul(...)``.

    Builds the 7 scalar args plena.matmul expects:
      ``(M_tiles, K_tiles, N, lhs_offset, rhs_offset, dst_offset,
        dst_row_stride)``
    Per-lane offsets are ``lane_var * per_lane_stride(buf, mode)``.
    """
    a_buf = buf_name_to_hlir[op.a.buffer.name] if buf_name_to_hlir else None
    b_buf = buf_name_to_hlir[op.b.buffer.name] if buf_name_to_hlir else None
    c_buf = buf_name_to_hlir[op.c.buffer.name] if buf_name_to_hlir else None
    a_mode = lane_modes.get(op.a.buffer.name) if lane_modes else None
    b_mode = lane_modes.get(op.b.buffer.name) if lane_modes else None
    c_mode = lane_modes.get(op.c.buffer.name) if lane_modes else None

    # M, K, N from the 4D BSHD shapes:
    #   lhs ROW_STACK (lane, S=M, 1, K==MLEN) → M_tiles = S / MLEN,
    #                                            K_tiles = K / MLEN
    #   rhs COL_PACK  (1, K, lane, D=N_narrow) → N = D_narrow
    M_tiles = 1
    K_tiles = 1
    N = 1
    if c_buf is not None and len(c_buf.shape) == 4:
        N = int(c_buf.shape[3])

    # dst_row_stride = elements between consecutive logical rows.
    # For canonical 4D BSHD ``(B, S, H, D)`` the S step in flat memory
    # is ``H * D`` (everything to the right of the rows axis). Smaller
    # ranks fall back to the innermost dim alone.
    dst_row_stride = N
    if c_buf is not None and len(c_buf.shape) >= 2:
        cshape = [int(d) for d in c_buf.shape]
        dst_row_stride = cshape[-2] * cshape[-1] if len(cshape) >= 2 else cshape[-1]

    # LHS rows == 1 → matrix-vector (M_MV / M_MV_WO) instead of M_MM.
    # ``plena.mv`` takes only 3 offsets (no M_tiles / K_tiles / N /
    # row_stride). Decode-style P @ V uses this when S_loc is a single
    # query token.
    lhs_rows = _logical_rows_from_buf(op.a)
    use_mv = lhs_rows == 1

    if cluster_extent is None or cluster_axis_name is None:
        # Outside any cluster: zero offsets, single op.
        if use_mv:
            return _hlir.Op(
                kind="mv",
                buffer_args=[op.a.buffer.name, op.b.buffer.name, op.c.buffer.name],
                scalar_args=[0, 0, 0],
            )
        return _hlir.Op(
            kind="matmul",
            buffer_args=[op.a.buffer.name, op.b.buffer.name, op.c.buffer.name],
            scalar_args=[M_tiles, K_tiles, N, 0, 0, 0, dst_row_stride],
        )
    # Inside a cluster: leaf op only. The enclosing CLUSTER -> for_lane
    # the walker emits binds ``lane_var`` for us. Per-lane offsets
    # ``lane * stride_for_buffer`` are computed against that var.
    lane_var = _make_loop_var(cluster_axis_name)
    a_stride = _per_lane_stride(a_buf, a_mode) if a_buf is not None else 0
    b_stride = _per_lane_stride(b_buf, b_mode) if b_buf is not None else 0
    c_stride = _per_lane_stride(c_buf, c_mode) if c_buf is not None else 0
    a_off = lane_var * _tir.IntImm(_INT32, a_stride) if a_stride else 0
    b_off = lane_var * _tir.IntImm(_INT32, b_stride) if b_stride else 0
    c_off = lane_var * _tir.IntImm(_INT32, c_stride) if c_stride else 0
    if use_mv:
        return _hlir.Op(
            kind="mv",
            buffer_args=[op.a.buffer.name, op.b.buffer.name, op.c.buffer.name],
            scalar_args=[a_off, b_off, c_off],
            annotations={"source": "per-head Gemm(rows=1) inside cluster"},
        )
    return _hlir.Op(
        kind="matmul",
        buffer_args=[op.a.buffer.name, op.b.buffer.name, op.c.buffer.name],
        scalar_args=[M_tiles, K_tiles, N, a_off, b_off, c_off, dst_row_stride],
        annotations={"source": "per-head Gemm(overwrite) inside cluster"},
    )


def _row_axis_index_of_buf(name: str, shape, cluster_dim: Optional[int]) -> int:
    """Index of the logical row axis in a buffer's shape.

    Skips the innermost axis (column / D / hlen) and the cluster axis
    (lane); the first remaining axis is rows. ``cluster_dim`` is the
    explicit marker propagated from pass_3_split."""
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


def _lower_bare_reduce(op: Reduce,
                       cluster_extent: Optional[int],
                       cluster_axis_name: Optional[str] = None) -> _hlir.Op:
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
    lane_var = (_make_loop_var(cluster_axis_name)
                if cluster_axis_name else _fresh_var("lane"))
    row_footprint = _row_footprint(op.src)
    if row_footprint > 1:
        row_var: _tir.PrimExpr = _fresh_var("row")
        wrap_rows = row_footprint
    elif row_footprint == 1:
        # Single-row reduce — no for-row needed, row index is literally 0.
        row_var = _tir.IntImm(_INT32, 0)
        wrap_rows = None
    else:
        row_var = _make_loop_var("row")
        wrap_rows = None
    fp_addr = _hlir.BufferElement(
        buffer=op.dst.buffer.name,
        indices=(lane_var, row_var),
    )
    leaf = _hlir.Op(
        kind=intrin,
        buffer_args=[op.src.buffer.name],
        scalar_args=[fp_addr, row_var, lane_var],
        annotations={"source": f"bare Reduce[{op.op.value}]"},
    )
    if wrap_rows is None:
        return leaf
    return _hlir.make_for_op(loop_var=row_var, extent=wrap_rows, body=[leaf])


def _lower_bare_broadcast_elementwise(
    op: Elementwise,
    cluster_extent: Optional[int],
    cluster_axis_name: Optional[str] = None,
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
    bcast_src = next((s for s in op.srcs if isinstance(s, Broadcast)), None)
    direct_src = next((s for s in op.srcs if not isinstance(s, Broadcast)), None)
    if bcast_src is None or direct_src is None:
        raise ToPlenaError(
            "broadcast Elementwise expected one BufferRef + one Broadcast src"
        )
    row_footprint = _row_footprint(op.dst)
    if row_footprint > 1:
        row_var = _fresh_var("row")
        wrap_rows = row_footprint
    else:
        # Reuse the kernel's row Var so the ISA materializer sees the
        # same identity the enclosing HLIR for-op binds.
        row_var = _make_loop_var("row")
        wrap_rows = None
    lane_var = (_make_loop_var(cluster_axis_name)
                if cluster_axis_name else _fresh_var("lane"))
    fp_addr = _hlir.BufferElement(
        buffer=bcast_src.src.buffer.name,
        indices=(lane_var, row_var),
    )
    leaf = _hlir.Op(
        kind=intrin,
        buffer_args=[direct_src.buffer.name, op.dst.buffer.name],
        scalar_args=[fp_addr, row_var, lane_var],
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
) -> _hlir.Op:
    """Bare elementwise on FPRAM rank-1 per-lane state → ``for lane:
    fp_<op>_at(<addr exprs>)``.

    The mid_ir Elementwise here came from kernel code like
    ``M_OLD[row] = M_INIT[row]`` already nested inside a ``for row``
    (rendered to a HLIR for op by the walker). The cluster axis is
    unwrapped at this point, so we re-emit ``for lane:`` here using
    the cluster's own axis name (``by_phase``) — keeping Var identity
    consistent with the indices view pass put into on-chip refs.
    """
    if op.op in _FP_AT_BINOP_TO_INTRIN:
        intrin = _FP_AT_BINOP_TO_INTRIN[op.op]
    elif op.op in _FP_AT_UNARY_TO_INTRIN:
        intrin = _FP_AT_UNARY_TO_INTRIN[op.op]
    else:
        raise ToPlenaError(
            f"unsupported FPRAM Elementwise op {op.op!r}"
        )
    # _emit_fp_scalar_op_at signature: for unary/copy, scalar_args =
    # (src_addr, dst_addr); for binary, (lhs_addr, rhs_addr, dst_addr).
    src_elements = [_fp_buffer_element_from_ref(s) for s in op.srcs]
    dst_element = _fp_buffer_element_from_ref(op.dst)
    return _hlir.Op(
        kind=intrin,
        buffer_args=[],
        scalar_args=src_elements + [dst_element],
        annotations={"source": f"bare FPRAM Elementwise[{op.op.value}]"},
    )


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------


_MULTI_LANE_OP_KINDS = frozenset({
    "dma_h2v", "dma_h2m", "dma_v2h",
    "dma_h2v_slice", "dma_h2m_slice", "dma_v2h_slice",
    "btmm", "btmv",
    "tile_add", "tile_sub", "tile_mul", "tile_exp", "tile_reci",
    "tile_sqrt", "tile_zero",
    # vram→vram copy: one V_ADD_VF (f0=0) spans a full MLEN-wide row;
    # under sync wrap the by_phase index is already folded to 0, so the
    # op covers all cluster lanes in one issue — don't re-wrap in a
    # synthetic for-lane loop.
    "copy_v_to_v",
    # vram↔fpram transfer: one S_MAP_FP_V / S_MAP_V_FP transfers VLEN=MLEN
    # contiguous slots in one issue (a full MLEN-wide vram row, spanning
    # all cluster lanes natively). Sync wrap zeroes the cluster phase
    # axis on both sides — same one-issue-covers-all-lanes contract.
    "row_load_v_to_fp", "row_store_fp_to_v",
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
                cluster_axis_name: Optional[str] = None) -> List[_hlir.Op]:
    out: List[_hlir.Op] = []
    for s in stmts:
        out.extend(_walk_stmt(s, buf_name_to_hlir, cluster_extent,
                              cluster_axis_name))
    return out


def _walk_stmt(stmt: Stmt,
               buf_name_to_hlir: Dict[str, _hlir.Buffer],
               cluster_extent: Optional[int],
               cluster_axis_name: Optional[str] = None) -> List[_hlir.Op]:
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
                               stmt.axis_name)
            lane_var = _make_loop_var(stmt.axis_name)
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
                               cluster_axis_name)
        # BLOCK_IDX or LOGICAL_GRID → flatten to a serial for.
        body = _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name)
        return [_hlir.make_for_op(
            loop_var=_make_loop_var(stmt.axis_name),
            extent=stmt.extent, body=body,
        )]
    if isinstance(stmt, For):
        body = _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name)
        for_op = _hlir.make_for_op(
            _make_loop_var(stmt.loop_var), stmt.extent, body=body,
        )
        for_op.annotations["loop_kind"] = stmt.kind
        return [for_op]
    if isinstance(stmt, Async):
        # By pass_5 every Async should be MultiLaneOp; if it lingers,
        # walk through.
        return _walk_stmts(stmt.body, buf_name_to_hlir, cluster_extent,
                           cluster_axis_name)
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
            return [_lower_multi_lane_btmm(stmt, 1)]
        return [_lower_bare_per_head_gemm(
            stmt, cluster_extent, cluster_axis_name,
            buf_name_to_hlir=buf_name_to_hlir, lane_modes=_LANE_MODES,
        )]
    if isinstance(stmt, Reduce):
        return [_lower_bare_reduce(stmt, cluster_extent, cluster_axis_name)]
    if isinstance(stmt, Elementwise):
        has_broadcast = any(isinstance(s, Broadcast) for s in stmt.srcs)
        if has_broadcast:
            return [_lower_bare_broadcast_elementwise(stmt, cluster_extent,
                                                     cluster_axis_name)]
        dst_scope = buf_name_to_hlir[stmt.dst.buffer.name].scope
        if dst_scope == _scope.FPRAM:
            # Per-lane FPRAM scalar update (M_OLD[row] = M_INIT[row]
            # etc.). Lower to a ``for lane: fp_<op>_at`` loop — the
            # ``row`` loop is the enclosing mid_ir For (already a HLIR
            # for op by now).
            return [_lower_bare_fp_scalar_elementwise(stmt, cluster_extent,
                                                     cluster_axis_name)]
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
        build_dir: Optional[Path] = None) -> _hlir.HLIRModule:
    """Lower a MidFunc to HLIRModule.

    If ``build_dir`` is given, write a ``<func.name>.midir.txt`` snapshot
    there before lowering — useful for diff against the legacy pipeline
    or for post-mortem when HLIR looks wrong.
    """
    _VAR_CACHE.clear()
    _LANE_MODES.clear()
    _LANE_AXIS_INFO.clear()
    # Register the split-axis form for each logical lane axis. mid_ir
    # carries ``lane_axes`` (one per cluster) and ``cluster_counts``;
    # each name there appears in BufferRef indices as the un-split
    # logical view, and must expand to ``<name>_phase + <name>_number
    # * count`` for ISA materialisation.
    for axis_name, count in zip(getattr(func, "lane_axes", []) or [],
                                 getattr(func, "cluster_counts", []) or []):
        _LANE_AXIS_INFO[axis_name] = (
            f"{axis_name}_phase", f"{axis_name}_number", int(count),
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
    lane_modes = _infer_lane_modes(func)
    _LANE_MODES.update(lane_modes)
    lane_count = func.cluster_counts[0] if func.cluster_counts else 1

    # Build buffer table.
    buf_name_to_hlir: Dict[str, _hlir.Buffer] = {}
    for buf in list(func.params) + list(func.allocs):
        if buf.name in buf_name_to_hlir:
            continue
        buf_name_to_hlir[buf.name] = _make_hlir_buffer(
            buf,
            override=overrides.get(buf.name),
            lane_count=lane_count,
            mode=lane_modes.get(buf.name),
        )

    # Walk the body.
    ops = _walk_stmts(func.body, buf_name_to_hlir, cluster_extent=None)

    return _hlir.HLIRModule(
        name=func.name,
        buffers=buf_name_to_hlir,
        ops=ops,
        param_names=[b.name for b in func.params],
    )


__all__ = ["run", "ToPlenaError"]
