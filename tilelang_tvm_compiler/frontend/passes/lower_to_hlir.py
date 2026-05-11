"""Helpers used by the graph back end (`graph_pipeline.py`) to lower
individual tile-DSL ops to ``plena.*`` extern calls.

This module used to host a top-level `run()` walker that wove tile→plena
translation together with lane-fusion segmentation in one recursive
stmt rewrite. That walker has been replaced by `graph_pipeline.run`,
which operates on the lifted block IR and treats lane-fusion segmentation
as a list partition rather than a stmt rewrite. What remains here are
the per-op lowering helpers that `graph_pipeline` calls:

  * ``_lower_copy(call, scopes, ...)`` — translate ``tl.tileop.copy`` to
    ``plena.dma_h2v_slice`` / ``dma_h2m_slice`` / ``dma_v2h_slice`` /
    ``copy_v_to_v`` / ``row_load_v_to_fp`` / ``row_store_fp_to_v``,
    folding the lane var into a multi-lane DMA when ``in_sync`` is set.
  * ``_lower_gemm(call, scopes, kind, ...)`` — translate
    ``tl.tileop.gemm_py`` to ``plena.matmul`` (kind=overwrite) or
    ``plena.btmm`` / ``plena.btmv`` (kind=btmm), with auto-injected
    per-lane offsets.
  * ``_rewrite_buffer_scopes(stmt, scopes)`` — replace declared
    ``shared.dyn`` / ``local.fragment`` scopes on alloc'd buffers with
    the resolved PLENA scopes (vram / mram / fpram / global.*).

Pre-conditions: ``annotate_gemm_kind``, ``annotate_group``,
``annotate_sync``, ``split_lane_groups``, ``scope_inference``,
``allocate_group_memory``, ``fuse_elementwise``, and ``lift_to_blocks``
have all run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import tir

from .graph_passes.scope_inference import BufferScopeMap
from ... import scope as _scope
from ...hlir import LAYOUT_AXES, TileLayout, make_tile_layout


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"


class LowerToHLIRError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Tile-aware layout helpers — see hlir.TileLayout for the 7D physical
# layout that VRAM/MRAM buffers use when their (B, S, H, D) overflows
# one inner tile. These helpers compute the (s_tile, s_inner, ...)
# decomposition and the resulting flat physical offset using only
# shift+sub TIR ops (PLENA has no integer divide and no bitwise AND, but
# expr_materializer lowers ``tir.shift_right`` / ``tir.shift_left`` to
# the corresponding ``S_SR(L)I_INT`` / ``S_SLLI_INT`` instructions, and
# ``x % 2^k`` is materialized as ``x - (x >> k) << k``).
#
# Simplifying assumption (per kernel-author feedback): all // and %
# divisors are powers of two. That covers MLEN, HLEN, LANE_COUNT, and
# the per-tile strides we generate, which is enough for the conv /
# attention / decode kernels we have today.
# ---------------------------------------------------------------------------


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _log2_pow2(n: int) -> int:
    """log2 of a strictly positive power of two."""
    if not _is_pow2(n):
        raise LowerToHLIRError(f"expected power of 2, got {n}")
    return n.bit_length() - 1


def _shr(expr: tir.PrimExpr, amount: int) -> tir.PrimExpr:
    """``expr >> amount`` (TIR ``tir.shift_right`` Call)."""
    if amount == 0:
        return expr
    return tir.Call(expr.dtype, tir.op.Op.get("tir.shift_right"),
                    [expr, tir.IntImm(expr.dtype, amount)])


def _shl(expr: tir.PrimExpr, amount: int) -> tir.PrimExpr:
    """``expr << amount`` (TIR ``tir.shift_left`` Call)."""
    if amount == 0:
        return expr
    return tir.Call(expr.dtype, tir.op.Op.get("tir.shift_left"),
                    [expr, tir.IntImm(expr.dtype, amount)])


def _try_tile_layout_for_buf(
    buf: tir.Buffer, *, mlen: int, hlen: int, buf_layout: str = "BSHD",
) -> Optional[TileLayout]:
    """Compute a TileLayout for ``buf`` if its 4D shape needs multi-tile
    storage. Returns ``None`` for non-4D shapes or shapes that fit one
    inner tile (caller falls back to the existing row-major path).

    ``buf_layout`` names how to interpret the 4D shape's axes. Default
    ``"BSHD"`` matches the original convention: axes[1] is the row dim,
    axes[2] is the channel dim. ``"NCHW"`` swaps those two — axes[1] is
    channel, axes[2] is row. The downstream TileLayout / 7D physical
    layout always works in canonical BSHD terms; this function's only
    job is to permute axes before handing them off.
    """
    shape = tuple(int(s) for s in buf.shape)
    if len(shape) != 4:
        return None
    return make_tile_layout(
        shape=shape, layout=buf_layout, mlen=mlen, hlen=hlen,
    )


def _flatten_starts_tiled(
    layout: TileLayout, starts, *, mlen: int, buf_layout: str = "BSHD",
) -> tir.PrimExpr:
    """Compute the physical flat offset of ``starts`` in a tile-laid-out
    buffer. ``starts`` is a 4D index tuple (4 PrimExprs / ints). The 7D
    physical layout is the same regardless of source layout — we just
    permute ``starts`` to canonical (b, s, h, d) order via
    ``LAYOUT_AXES[buf_layout]`` before the offset math.

    All // and % use power-of-2 divisors (``mlen``, ``layout.lane_count``,
    ``layout.d_inner``), and every stride below is a power of 2 too in
    the cases we support. Each piece is one shift-left / shift-right /
    add / sub TIR op.
    """
    if len(starts) != 4:
        raise LowerToHLIRError(
            f"_flatten_starts_tiled expects 4D starts; got {len(starts)}-D"
        )
    if buf_layout not in LAYOUT_AXES:
        raise LowerToHLIRError(
            f"unknown buf_layout {buf_layout!r}; known: {sorted(LAYOUT_AXES)}"
        )
    bi, ri, ci, di = LAYOUT_AXES[buf_layout]
    b_start = starts[bi]
    s_start = starts[ri]   # row-tile dim
    h_start = starts[ci]   # channel-group / lane dim
    d_start = starts[di]   # col-tile dim

    # Decompose s and d via shift-right (// MLEN) and shift-left+sub
    # (% MLEN = x - (x >> log2_mlen) << log2_mlen).
    log2_mlen = _log2_pow2(mlen)
    s_tile = _shr(s_start, log2_mlen)
    s_inner = tir.Sub(s_start, _shl(s_tile, log2_mlen))
    d_tile = _shr(d_start, log2_mlen)
    d_inner = tir.Sub(d_start, _shl(d_tile, log2_mlen))

    # H dim splits into (h_grp, lane) only when LANE_COUNT > 1.
    if layout.lane_count > 1:
        log2_lane = _log2_pow2(layout.lane_count)
        h_grp = _shr(h_start, log2_lane)
        lane = tir.Sub(h_start, _shl(h_grp, log2_lane))
    else:
        h_grp = h_start
        lane = tir.IntImm(b_start.dtype, 0)

    # Per-axis strides in the 7D physical layout (must all be pow2).
    # 7D layout order: (D_TILES, S_TILES, H_GROUPS, B, MLEN, LANE_COUNT, D_INNER).
    # Each stride is the total elem count of everything inner-of it:
    #   inner_d          = D_INNER
    #   inner_lane       = LANE_COUNT * D_INNER
    #   inner_s          = MLEN * inner_lane          (one inner tile = inner-of B)
    #   b_stride         = inner_s                    (B is inner-of H_GROUPS)
    #   inner_b          = logical_b * inner_s        (volume of B axis)
    #   h_grp_stride     = inner_b
    #   s_tile_stride    = h_groups * inner_b
    #   d_tile_stride    = s_tiles  * s_tile_stride
    inner_d = layout.d_inner
    inner_lane = layout.lane_count * inner_d
    inner_s = mlen * inner_lane
    b_stride = inner_s
    inner_b = layout.logical_b * inner_s
    h_grp_stride = inner_b
    s_tile_stride = layout.h_groups * inner_b
    d_tile_stride = layout.s_tiles * s_tile_stride

    offset: tir.PrimExpr = tir.IntImm(b_start.dtype, 0)
    if layout.d_tiles > 1:
        offset = tir.Add(offset, _shl(d_tile, _log2_pow2(d_tile_stride)))
    if layout.s_tiles > 1:
        offset = tir.Add(offset, _shl(s_tile, _log2_pow2(s_tile_stride)))
    if layout.h_groups > 1:
        offset = tir.Add(offset, _shl(h_grp, _log2_pow2(h_grp_stride)))
    if layout.logical_b > 1:
        offset = tir.Add(offset, _shl(b_start, _log2_pow2(b_stride)))
    if mlen > 1:
        offset = tir.Add(offset, _shl(s_inner, _log2_pow2(inner_lane)))
    if layout.lane_count > 1:
        offset = tir.Add(offset, _shl(lane, _log2_pow2(inner_d)))
    offset = tir.Add(offset, d_inner)
    return offset


# ---------------------------------------------------------------------------
# Buffer scope rewrite
# ---------------------------------------------------------------------------

def _rebuild_buffer_with_scope(buf: tir.Buffer, new_scope: str) -> tir.Buffer:
    """Return a fresh Buffer mirroring `buf` but in `new_scope`.

    The shape is preserved as-is — isa_pass's ``_logical_2d`` handles
    arbitrary ranks by flattening into a (rows, cols) view.
    """
    new_data = tir.Var(buf.data.name, tvm.ir.PointerType(
        tvm.ir.PrimType(buf.dtype), new_scope,
    ))
    return tir.decl_buffer(
        shape=list(buf.shape),
        dtype=buf.dtype,
        name=buf.name,
        data=new_data,
        scope=new_scope,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _region_components(call: tir.Call):
    """T.region(buf[start_idx, ...], access_mode, *extents) ->
       (buffer, starts, extents)."""
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        raise LowerToHLIRError(f"expected {_TILEOP_REGION}, got {call!r}")
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        raise LowerToHLIRError(
            f"region arg[0] must be BufferLoad, got {type(load).__name__}"
        )
    starts = list(load.indices)
    extents = list(call.args[2:])
    if len(starts) != len(extents):
        diff = len(starts) - len(extents)
        if diff > 0:
            extents = [tir.IntImm("int32", 1)] * diff + extents
        else:
            raise LowerToHLIRError(
                f"region rank mismatch: {len(starts)} starts vs {len(extents)} extents"
            )
    return load.buffer, starts, extents


def _make_call_extern(name: str, args: list) -> tir.Call:
    extern_op = tvm.ir.Op.get("tir.call_extern")
    return tir.Call("handle", extern_op, [tir.StringImm(name), *args])


def _evaluate(call: tir.Call) -> tir.Evaluate:
    return tir.Evaluate(call)


def _substitute_var(expr, var_name: str, replacement) -> object:
    """Walk an Expr and replace every Var named `var_name` with `replacement`.
    Best-effort generic walker."""
    if isinstance(expr, tir.Var):
        if expr.name == var_name:
            return replacement
        return expr
    if isinstance(expr, tir.IntImm) or isinstance(expr, tir.FloatImm):
        return expr
    if isinstance(expr, tir.Call):
        return tir.Call(expr.dtype, expr.op,
                        [_substitute_var(a, var_name, replacement) for a in expr.args])
    if isinstance(expr, tir.BufferLoad):
        return tir.BufferLoad(expr.buffer,
                              [_substitute_var(i, var_name, replacement) for i in expr.indices])
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return type(expr)(
            _substitute_var(expr.a, var_name, replacement),
            _substitute_var(expr.b, var_name, replacement),
        )
    return expr


def _expr_uses_var(expr, var_name: str) -> bool:
    if isinstance(expr, tir.Var):
        return expr.name == var_name
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        return False
    if isinstance(expr, tir.Call):
        return any(_expr_uses_var(a, var_name) for a in expr.args)
    if isinstance(expr, tir.BufferLoad):
        return any(_expr_uses_var(i, var_name) for i in expr.indices)
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return _expr_uses_var(expr.a, var_name) or _expr_uses_var(expr.b, var_name)
    return False


def _expr_has_any_var(expr) -> bool:
    if isinstance(expr, tir.Var):
        return True
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        return False
    if isinstance(expr, tir.Call):
        return any(_expr_has_any_var(a) for a in expr.args)
    if isinstance(expr, tir.BufferLoad):
        return any(_expr_has_any_var(i) for i in expr.indices)
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return _expr_has_any_var(expr.a) or _expr_has_any_var(expr.b)
    return False


def _zero_like(expr):
    dtype = getattr(expr, "dtype", "int32")
    return tir.IntImm(dtype, 0)


def _project_expr_to_var(expr, var_name: str):
    """Keep the part of ``expr`` that belongs to ``var_name``.

    After head-domain splitting, logical head expressions look like
    ``by_o * width + by_i``. HBM DMAs need the full logical expression, but
    local-tile offsets for per-lane ops (currently manual ``plena.matmul``)
    must use only the inner hardware lane ``by_i``. Terms that depend on
    other vars are dropped; pure constants are preserved.
    """
    if isinstance(expr, tir.Var):
        return expr if expr.name == var_name else _zero_like(expr)
    if isinstance(expr, (tir.IntImm, tir.FloatImm)):
        return expr
    if isinstance(expr, tir.Add):
        a = _project_expr_to_var(expr.a, var_name)
        b = _project_expr_to_var(expr.b, var_name)
        if _const_int(a) == 0:
            return b
        if _const_int(b) == 0:
            return a
        return tir.Add(a, b)
    if isinstance(expr, tir.Sub):
        a = _project_expr_to_var(expr.a, var_name)
        b = _project_expr_to_var(expr.b, var_name)
        if _const_int(b) == 0:
            return a
        return tir.Sub(a, b)
    if isinstance(expr, tir.Mul):
        a_uses = _expr_uses_var(expr.a, var_name)
        b_uses = _expr_uses_var(expr.b, var_name)
        if not a_uses and not b_uses:
            return expr if not _expr_has_any_var(expr) else _zero_like(expr)
        if a_uses and not b_uses:
            other = expr.b if not _expr_has_any_var(expr.b) else tir.IntImm("int32", 1)
            return tir.Mul(_project_expr_to_var(expr.a, var_name), other)
        if b_uses and not a_uses:
            other = expr.a if not _expr_has_any_var(expr.a) else tir.IntImm("int32", 1)
            return tir.Mul(other, _project_expr_to_var(expr.b, var_name))
        return tir.Mul(
            _project_expr_to_var(expr.a, var_name),
            _project_expr_to_var(expr.b, var_name),
        )
    return expr if not _expr_has_any_var(expr) else _zero_like(expr)


def _project_matmul_offsets_to_lane(stmt: tir.Evaluate,
                                    lane_var: Optional[str]) -> tir.Evaluate:
    if lane_var is None:
        return stmt
    v = stmt.value
    if not (isinstance(v, tir.Call)
            and getattr(v.op, "name", None) == "tir.call_extern"
            and v.args
            and isinstance(v.args[0], tir.StringImm)):
        return stmt
    name = v.args[0].value
    # Per-extern offset positions in the call_extern arg list. Each per-lane
    # local-tile op has trailing scalar offsets that must be projected from
    # the full head index ``by`` down to just the inner-lane ``by_i``;
    # otherwise a head_count > lane_count kernel walks past the per-tile
    # MLEN bound and trips the HW assertion.
    OFFSET_POSITIONS = {
        # plena.matmul: [0]name [1:4]bufs [4:7]M/K/N [7:10]offsets [10]stride
        "plena.matmul": (7, 8, 9),
        # plena.mv:     [0]name [1:4]bufs [4:7]offsets
        "plena.mv":     (4, 5, 6),
    }
    positions = OFFSET_POSITIONS.get(name)
    if positions is None:
        return stmt
    args = list(v.args)
    for idx in positions:
        if idx < len(args):
            args[idx] = _project_expr_to_var(args[idx], lane_var)
    return tir.Evaluate(tir.Call(v.dtype, v.op, args))


# ---------------------------------------------------------------------------
# Op lowering
# ---------------------------------------------------------------------------

def _flatten_starts(buf: tir.Buffer, starts) -> tir.PrimExpr:
    """Linearize ``starts`` over ``buf``'s row-major strides (post-expansion).

    Used by VRAM↔FPRAM lowering to convert n-D buffer-relative indices into
    a single flat element offset that materializes into a gp register at
    isa-emit time.
    """
    shape = [int(s) for s in buf.shape]
    if len(starts) != len(shape):
        raise LowerToHLIRError(
            f"_flatten_starts rank mismatch on {buf.name!r}: "
            f"{len(starts)} starts vs {len(shape)} dims"
        )
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    offset: tir.PrimExpr = tir.IntImm("int32", 0)
    for s, stride in zip(starts, strides):
        term = s if stride == 1 else tir.Mul(s, tir.IntImm("int32", stride))
        offset = tir.Add(offset, term)
    return offset


def _lower_row_v_fp_copy(*, vram_buf, vram_starts, fp_buf, fp_starts,
                         direction: str, lane_var: Optional[str],
                         in_sync: bool,
                         target_mlen: int,
                         target_hlen: int,
                         target_layout: str = "BSHD") -> tir.Stmt:
    """Lower one ``T.copy`` between VRAM and FPRAM to a row-wide MAP transfer.

    The HW op (S_MAP_V_FP / S_MAP_FP_V) moves VLEN=MLEN elements per
    invocation, naturally serving all lanes at once. Lane fusion is
    therefore implicit — when in_sync, we just substitute lane_var to 0
    in both index sides; we do NOT multiply any extent (HW op size is
    fixed).

    Tile-aware VRAM offset: same rule as ``_lower_v_to_v_copy`` — when
    the VRAM buffer's 4D BSHD shape overflows one inner tile, use the
    7D physical-layout offset (``_flatten_starts_tiled``) instead of
    the row-major ``_flatten_starts``. The S_MAP_V_FP / S_MAP_FP_V
    instruction itself still wants the resulting flat offset to be
    MLEN-aligned (it copies VLEN=MLEN at a time); the tiled-layout
    offset is naturally MLEN-aligned for ``d_inner == 0`` access
    patterns (which is what tile-row-aligned reads use).
    """
    if in_sync and lane_var is not None:
        zero = tir.IntImm("int32", 0)
        vram_starts = [_substitute_var(s, lane_var, zero) for s in vram_starts]
        fp_starts = [_substitute_var(s, lane_var, zero) for s in fp_starts]

    vram_layout = _try_tile_layout_for_buf(
        vram_buf, mlen=target_mlen, hlen=target_hlen, buf_layout=target_layout,
    )
    if vram_layout is not None:
        vram_offset_expr = _flatten_starts_tiled(
            vram_layout, vram_starts, mlen=target_mlen,
            buf_layout=target_layout,
        )
    else:
        vram_offset_expr = _flatten_starts(vram_buf, vram_starts)
    # Pass fp side as a BufferLoad so isa_pass._resolve_fp_scalar_addr_arg
    # can fold in the fragment's allocated FPRAM base address (same path
    # used by the plena.fp_*_at family).
    fp_addr_expr = tir.BufferLoad(fp_buf, list(fp_starts))

    if direction == "v_to_fp":
        intrin = "plena.row_load_v_to_fp"
        args = [vram_buf.data, vram_offset_expr, fp_addr_expr]
    elif direction == "fp_to_v":
        intrin = "plena.row_store_fp_to_v"
        args = [fp_addr_expr, vram_buf.data, vram_offset_expr]
    else:
        raise LowerToHLIRError(f"unknown direction {direction!r}")

    return _evaluate(_make_call_extern(intrin, args))


def _lower_v_to_v_copy(*, src_buf, src_starts, dst_buf, dst_starts,
                       lane_var: Optional[str], in_sync: bool,
                       target_mlen: int, target_hlen: int,
                       target_layout: str = "BSHD") -> tir.Stmt:
    """Lower a vram→vram T.copy to one V_ADD_VF row transfer.

    Lane fusion handling mirrors _lower_row_v_fp_copy: when in_sync, the
    lane_var is substituted to 0 in both index sides (the HW V_ADD_VF
    processes one full MLEN-wide vector per call, naturally covering all
    lanes — no extent multiplication needed).

    Tile-aware offset: if either side's buffer has a 4D BSHD shape that
    overflows one inner tile (see ``hlir.TileLayout``), the flat offset
    is computed via the 7D physical layout — using shift+sub TIR ops
    (PLENA has no integer divide and no AND, but expr_materializer
    lowers ``tir.shift_left/right`` to ``S_S(L|R)LI_INT`` and ``x % 2^k``
    becomes ``x - (x >> k) << k``). Otherwise fall back to the
    row-major ``_flatten_starts``.
    """
    if in_sync and lane_var is not None:
        zero = tir.IntImm("int32", 0)
        src_starts = [_substitute_var(s, lane_var, zero) for s in src_starts]
        dst_starts = [_substitute_var(s, lane_var, zero) for s in dst_starts]

    src_layout = _try_tile_layout_for_buf(
        src_buf, mlen=target_mlen, hlen=target_hlen, buf_layout=target_layout,
    )
    dst_layout = _try_tile_layout_for_buf(
        dst_buf, mlen=target_mlen, hlen=target_hlen, buf_layout=target_layout,
    )

    if src_layout is not None:
        src_offset_expr = _flatten_starts_tiled(
            src_layout, src_starts, mlen=target_mlen,
            buf_layout=target_layout,
        )
    else:
        src_offset_expr = _flatten_starts(src_buf, src_starts)

    if dst_layout is not None:
        dst_offset_expr = _flatten_starts_tiled(
            dst_layout, dst_starts, mlen=target_mlen,
            buf_layout=target_layout,
        )
    else:
        dst_offset_expr = _flatten_starts(dst_buf, dst_starts)

    return _evaluate(_make_call_extern(
        "plena.copy_v_to_v",
        [src_buf.data, src_offset_expr, dst_buf.data, dst_offset_expr],
    ))


def _lower_copy(call: tir.Call,
                scopes: BufferScopeMap,
                lane_count: int,
                lane_var: Optional[str],
                in_sync: bool,
                *,
                target_mlen: int,
                target_hlen: int,
                target_layout: str = "BSHD") -> tir.Stmt:
    """Lower a tl.tileop.copy to plena.dma_h2v_slice / dma_h2m_slice /
    dma_v2h_slice. When `in_sync` is True and `lane_var` is set, substitute
    the lane var to 0 and multiply the lane-position extent by lane_count
    to fold all per-lane iterations into one multi-lane DMA."""
    src_buf, src_starts, _src_exts = _region_components(call.args[0])
    dst_buf, dst_starts, _dst_exts = _region_components(call.args[1])
    # Collapse `global.<phys>` to `<phys>` for routing — a DMA into a
    # `global.vram` buffer takes the same plena.dma_h2v_slice path as
    # one into a regular `vram` buffer; the user-declared global flag
    # only suppressed lane-fusion expansion (already handled upstream).
    src_scope = _scope.physical_scope(scopes.get(src_buf.name) or "")
    dst_scope = _scope.physical_scope(scopes.get(dst_buf.name) or "")

    if src_scope == "hbm" and dst_scope in ("vram", "mram"):
        intrin = "plena.dma_h2v_slice" if dst_scope == "vram" else "plena.dma_h2m_slice"
        # Use HBM-side starts; derive per-dim extents from HBM shape.
        hbm_buf, hbm_starts = src_buf, src_starts
        local_buf = dst_buf
    elif src_scope == "vram" and dst_scope == "hbm":
        intrin = "plena.dma_v2h_slice"
        hbm_buf, hbm_starts = dst_buf, dst_starts
        local_buf = src_buf
    elif src_scope == "vram" and dst_scope == "fpram":
        return _lower_row_v_fp_copy(
            vram_buf=src_buf, vram_starts=src_starts,
            fp_buf=dst_buf, fp_starts=dst_starts,
            direction="v_to_fp",
            lane_var=lane_var, in_sync=in_sync,
            target_mlen=target_mlen, target_hlen=target_hlen,
            target_layout=target_layout,
        )
    elif src_scope == "fpram" and dst_scope == "vram":
        return _lower_row_v_fp_copy(
            vram_buf=dst_buf, vram_starts=dst_starts,
            fp_buf=src_buf, fp_starts=src_starts,
            direction="fp_to_v",
            lane_var=lane_var, in_sync=in_sync,
            target_mlen=target_mlen, target_hlen=target_hlen,
            target_layout=target_layout,
        )
    elif src_scope == "vram" and dst_scope == "vram":
        # In-VRAM copy ("tensor cache" path). Lowers to one V_ADD_VF row
        # per call (see plena.copy_v_to_v intrinsic). Lane fusion is
        # implicit at the HW level — V_ADD_VF processes one MLEN-wide
        # vector regardless of how many lanes' data it covers.
        return _lower_v_to_v_copy(
            src_buf=src_buf, src_starts=src_starts,
            dst_buf=dst_buf, dst_starts=dst_starts,
            lane_var=lane_var, in_sync=in_sync,
            target_mlen=target_mlen, target_hlen=target_hlen,
            target_layout=target_layout,
        )
    else:
        raise LowerToHLIRError(
            f"unsupported copy direction {src_scope}->{dst_scope}"
        )

    local_size = 1
    for s in local_buf.shape:
        local_size *= int(s)

    # Detect whether the lane-var actually drives an HBM dim — only then
    # is the DMA "lane-fused" (one multi-lane HW op). When sync is on but
    # the lane var doesn't appear in any start, the copy is per-lane
    # replicated and treated as a regular DMA.
    lane_dim = None
    if in_sync and lane_var is not None:
        for i, s in enumerate(hbm_starts):
            if _expr_uses_var(s, lane_var):
                lane_dim = i
                break

    if lane_dim is not None:
        if local_size % lane_count != 0:
            raise LowerToHLIRError(
                f"lane-fused DMA on {hbm_buf.name!r} requires local size "
                f"({local_size}) divisible by lane_count ({lane_count})"
            )
        target = local_size // lane_count
        per_dim_exts = _derive_per_dim_extents(
            hbm_buf, hbm_starts, target, lane_var=lane_var,
        )
        new_starts = [_substitute_var(s, lane_var, tir.IntImm("int32", 0))
                      for s in hbm_starts]
        new_extents = list(per_dim_exts)
        new_extents[lane_dim] = tir.IntImm(
            "int32", int(new_extents[lane_dim].value) * lane_count,
        )
        _validate_extent_size(new_extents, local_buf, hbm_buf.name,
                              msg_prefix="(lane-fused) ")
        return _evaluate(_make_call_extern(intrin, [
            src_buf.data, dst_buf.data, len(new_starts),
            *new_starts, *new_extents,
        ]))

    per_dim_exts = _derive_per_dim_extents(hbm_buf, hbm_starts, local_size)
    _validate_extent_size(per_dim_exts, local_buf, hbm_buf.name)
    return _evaluate(_make_call_extern(intrin, [
        src_buf.data, dst_buf.data, len(hbm_starts),
        *hbm_starts, *per_dim_exts,
    ]))


def _derive_per_dim_extents(hbm_buf, starts, target_size: int,
                            lane_var: Optional[str] = None) -> List[tir.IntImm]:
    """Derive per-dim DMA extents whose product equals ``target_size``.

    For each dim:
      * If the start references a loop var, the dim's extent is the
        affine coefficient (the var's stride along this dim, typically 1).
      * Else (static 0): extents are filled greedily from the innermost
        dim outward, taking the full shape as long as the cumulative
        product still divides ``target_size``; otherwise 1.
    """
    if len(starts) != len(hbm_buf.shape):
        raise LowerToHLIRError(
            f"start indices ({len(starts)}) and hbm shape ({len(hbm_buf.shape)}) "
            f"rank mismatch on {hbm_buf.name!r}"
        )

    extents: List[Optional[int]] = [None] * len(starts)
    var_product = 1
    for dim_idx, start in enumerate(starts):
        if _const_int(start) is not None:
            continue
        if lane_var is not None and _expr_uses_var(start, lane_var):
            coeff = _affine_coeff_of_var(start, lane_var)
        else:
            coeff = _affine_coeff(start)
        if coeff is None:
            raise LowerToHLIRError(
                f"non-affine start expression on {hbm_buf.name!r} dim {dim_idx}: {start!r}"
            )
        extents[dim_idx] = coeff
        var_product *= coeff

    if target_size % var_product != 0:
        raise LowerToHLIRError(
            f"target_size {target_size} not divisible by var-stride product "
            f"{var_product} on {hbm_buf.name!r}"
        )
    quota = target_size // var_product

    # Greedy fill of static-0 dims, innermost first.
    for dim_idx in reversed(range(len(starts))):
        if extents[dim_idx] is not None:
            continue
        start = starts[dim_idx]
        if _const_int(start) != 0:
            raise LowerToHLIRError(
                f"non-zero constant start ({start}) on {hbm_buf.name!r} "
                f"dim {dim_idx} not supported"
            )
        shape_i = int(hbm_buf.shape[dim_idx])
        if shape_i == 1:
            extents[dim_idx] = 1
            continue
        if quota >= shape_i and quota % shape_i == 0:
            extents[dim_idx] = shape_i
            quota //= shape_i
        else:
            extents[dim_idx] = 1

    if quota != 1:
        raise LowerToHLIRError(
            f"could not derive extents matching target_size on "
            f"{hbm_buf.name!r}: leftover quota {quota}"
        )
    return [tir.IntImm("int32", e) for e in extents]


def _const_int(expr) -> Optional[int]:
    """Best-effort integer constant evaluator for simple TIR expressions."""
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    if isinstance(expr, tir.Add):
        a = _const_int(expr.a)
        b = _const_int(expr.b)
        return None if a is None or b is None else a + b
    if isinstance(expr, tir.Sub):
        a = _const_int(expr.a)
        b = _const_int(expr.b)
        return None if a is None or b is None else a - b
    if isinstance(expr, tir.Mul):
        a = _const_int(expr.a)
        b = _const_int(expr.b)
        return None if a is None or b is None else a * b
    return None


def _validate_extent_size(extents, local_buf, hbm_name, msg_prefix=""):
    prod_ext = 1
    for e in extents:
        prod_ext *= int(e.value)
    prod_local = 1
    for s in local_buf.shape:
        prod_local *= int(s)
    if prod_ext != prod_local:
        raise LowerToHLIRError(
            f"{msg_prefix}derived extents {[int(e.value) for e in extents]} "
            f"(product {prod_ext}) don't match local {local_buf.name!r} "
            f"size {prod_local}"
        )


def _affine_coeff(expr) -> Optional[int]:
    """Best-effort: detect `c * var` or `var * c` or `var` (coeff=1) or
    `c1 * var + c2`. Returns the coefficient of the (single) var or None
    if not affine in a single var."""
    if isinstance(expr, tir.Var):
        return 1
    if isinstance(expr, tir.IntImm):
        return 0
    if isinstance(expr, tir.Mul):
        if isinstance(expr.a, tir.Var) and isinstance(expr.b, tir.IntImm):
            return int(expr.b.value)
        if isinstance(expr.b, tir.Var) and isinstance(expr.a, tir.IntImm):
            return int(expr.a.value)
        return None
    if isinstance(expr, tir.Add):
        ca = _affine_coeff(expr.a)
        cb = _affine_coeff(expr.b)
        if ca is None or cb is None:
            return None
        return ca + cb if ca > 0 or cb > 0 else max(ca, cb)
    return None


def _affine_coeff_of_var(expr, var_name: str) -> Optional[int]:
    """Return the coefficient of ``var_name`` in a simple affine expr.

    Other vars are treated as part of the base address. This is what split
    head fusion needs for expressions like ``by_o * 4 + by_i``: the DMA
    lane extent is driven by ``by_i`` only, not by the outer logical head
    tile.
    """
    if isinstance(expr, tir.Var):
        return 1 if expr.name == var_name else 0
    if isinstance(expr, tir.IntImm):
        return 0
    if isinstance(expr, tir.Add):
        ca = _affine_coeff_of_var(expr.a, var_name)
        cb = _affine_coeff_of_var(expr.b, var_name)
        if ca is None or cb is None:
            return None
        return ca + cb
    if isinstance(expr, tir.Sub):
        ca = _affine_coeff_of_var(expr.a, var_name)
        cb = _affine_coeff_of_var(expr.b, var_name)
        if ca is None or cb is None:
            return None
        return ca - cb
    if isinstance(expr, tir.Mul):
        if isinstance(expr.a, tir.IntImm):
            cb = _affine_coeff_of_var(expr.b, var_name)
            return None if cb is None else int(expr.a.value) * cb
        if isinstance(expr.b, tir.IntImm):
            ca = _affine_coeff_of_var(expr.a, var_name)
            return None if ca is None else int(expr.b.value) * ca
        return None
    return None


def _auto_lane_offset(buf: tir.Buffer,
                      lane_var: Optional[str],
                      lane_count: int) -> tir.PrimExpr:
    """Find the lane axis of ``buf`` (the dimension whose extent equals
    ``lane_count``) and return ``lane_var * stride_of_that_axis`` as a
    PrimExpr.

    Used when a ``T.gemm`` (kind=mv / overwrite) is written WITHOUT explicit
    slicing — the lowering infers per-lane offsets from buffer shape so
    the kernel author never has to deal with post-expansion shapes or
    lane-aware indexing. Returns ``IntImm(0)`` when there is no detectable
    lane axis or no lane_var in scope (e.g. a non-lane-fused gemm)."""
    if lane_var is None:
        return tir.IntImm("int32", 0)
    shape = []
    for s in buf.shape:
        try:
            shape.append(int(s))
        except (TypeError, ValueError):
            return tir.IntImm("int32", 0)
    if lane_count not in shape:
        return tir.IntImm("int32", 0)
    lane_dim = shape.index(lane_count)
    stride = 1
    for d in shape[lane_dim + 1:]:
        stride *= d
    if stride == 0:
        return tir.IntImm("int32", 0)
    return tir.Mul(tir.Var(lane_var, "int32"), tir.IntImm("int32", stride))


def _resolve_offset(buf: tir.Buffer,
                    starts,
                    lane_var: Optional[str],
                    lane_count: int) -> tir.PrimExpr:
    """Pick the right offset expression for a gemm operand:
    * If author wrote slicing (any non-zero / non-trivial start), fold the
      starts via ``_flatten_starts`` (subject to the existing lane
      projection downstream).
    * Otherwise (whole-buffer gemm), auto-inject ``lane_var * stride`` so
      the per-lane HW op naturally addresses lane[lane_var]'s slice.
    """
    has_explicit_slicing = any(
        not (isinstance(s, tir.IntImm) and int(s.value) == 0)
        for s in starts
    )
    if has_explicit_slicing:
        return _flatten_starts(buf, starts)
    return _auto_lane_offset(buf, lane_var, lane_count)


def _lower_gemm(call: tir.Call,
                scopes: BufferScopeMap,
                kind: str,
                lane_count: int,
                target_mlen: int,
                target_hlen: int,
                lane_var: Optional[str] = None) -> tir.Stmt:
    """Lower tl.tileop.gemm_py based on its `kind` annotation."""
    a_buf, a_starts, _a_exts = _region_components(call.args[0])
    b_buf, b_starts, _b_exts = _region_components(call.args[1])
    c_buf, c_starts, c_exts = _region_components(call.args[2])

    # `global.<phys>` operands satisfy the gemm scope rule the same as
    # plain `<phys>` — the user-declared global flag only affects
    # lane-fusion expansion, not which physical RAM the operand sits in.
    a_scope = _scope.physical_scope(scopes.get(a_buf.name) or "")
    b_scope = _scope.physical_scope(scopes.get(b_buf.name) or "")
    c_scope = _scope.physical_scope(scopes.get(c_buf.name) or "")
    if (a_scope, b_scope, c_scope) != ("vram", "mram", "vram"):
        raise LowerToHLIRError(
            f"gemm operand scopes must be (vram, mram, vram); got "
            f"({a_scope}, {b_scope}, {c_scope})"
        )

    if kind == "btmm":
        # Shape-based dispatch between matrix-matrix (BTMM) and
        # matrix-vector (BTMV). The user signals "this is a GEMV" by
        # declaring the LHS shared buffer with rows-dim == 1
        # (T.alloc_shared((1, hlen), ...)). After allocate_group_memory's
        # column-pack expansion, the buffer is 4-D (1, rows, lane_count,
        # last); rows=1 marks the BTMV path. Pre-expansion 2-D shape is
        # also accepted in case this pass runs before expansion.
        if len(a_buf.shape) == 4:
            rows_dim = int(a_buf.shape[1])
        elif len(a_buf.shape) == 2:
            rows_dim = int(a_buf.shape[0])
        else:
            rows_dim = -1  # unknown layout, default to BTMM
        intrin = "plena.btmv" if rows_dim == 1 else "plena.btmm"
        return _evaluate(_make_call_extern(
            intrin,
            [a_buf.data, b_buf.data, c_buf.data, lane_count],
        ))

    if kind == "overwrite":
        # Per-buffer flat element offsets. Two sources:
        #   * Author wrote slicing → fold starts into offsets via
        #     _flatten_starts (then run through lane projection below).
        #   * Author wrote whole-buffer T.gemm → auto-inject
        #     ``lane_var * stride_of_lane_axis`` so the kernel never
        #     has to know about post-expansion shapes or lane indexing.
        a_off = _resolve_offset(a_buf, a_starts, lane_var, lane_count)
        b_off = _resolve_offset(b_buf, b_starts, lane_var, lane_count)
        c_off = _resolve_offset(c_buf, c_starts, lane_var, lane_count)

        # Shape-based dispatch between matrix-matrix (plena.matmul, M_MM
        # path) and matrix-vector (plena.mv, M_MV path), mirroring how
        # the btmm kind picks btmm vs btmv. Looks at the first non-lane
        # dim of the LHS post-expansion: if rows == 1, it's a GEMV.
        rows_dim = _lhs_rows_dim(a_buf, lane_count)
        if rows_dim == 1:
            # plena.mv only takes the three offsets — no M_tiles / K_tiles /
            # row_stride. The M_MV/M_MV_WO HW path always processes one
            # MLEN-wide LHS row × blen-tile slices of the matrix per call.
            stmt = _evaluate(_make_call_extern(
                "plena.mv",
                [a_buf.data, b_buf.data, c_buf.data, a_off, b_off, c_off],
            ))
        else:
            c_inner_ext = int(c_exts[-1].value) if c_exts else int(c_buf.shape[-1])
            N = c_inner_ext
            row_stride = _dst_row_stride(c_buf, lane_count)
            stmt = _evaluate(_make_call_extern(
                "plena.matmul",
                [
                    a_buf.data, b_buf.data, c_buf.data,
                    tir.IntImm("int32", 1),    # M_tiles
                    tir.IntImm("int32", 1),    # K_tiles
                    tir.IntImm("int32", N),
                    a_off, b_off, c_off,
                    tir.IntImm("int32", row_stride),
                ],
            ))
        # Apply the same lane projection used for already-lowered plena.*
        # extern calls. Sliced offsets that contain the full kernel grid
        # var (e.g. ``by * MLEN``) get replaced with their inner-lane part,
        # mirroring the path kernel-author-written extern calls take.
        return _project_matmul_offsets_to_lane(stmt, lane_var)

    if kind == "add":
        # Reserved interface (PIPELINE_ARCHITECTURE.md § 5.4): the plan
        # is for the user to pre-allocate a scratch buffer and pass it
        # via ``T.attr(scratch.data, "plena.gemm_scratch", 0)`` around
        # the gemm; the lowering would then emit
        # ``plena.matmul → scratch`` followed by
        # ``plena.v_add(C, scratch, C)``. Not implemented yet — for now
        # write the two ops manually:
        #     T.gemm(A, B, scratch)              # KIND=overwrite (default)
        #     for r in T.serial(rows):
        #         for c in T.Parallel(C):
        #             dst[r, c] = dst[r, c] + scratch[r, c]
        # (the latter folds to plena.v_add via fuse_elementwise).
        raise NotImplementedError(
            'KIND="add" (C += A @ B) is reserved but not yet implemented. '
            'Use KIND="overwrite" into a scratch buffer plus a separate '
            'T.Parallel + add (auto-fuses to plena.v_add) for now. '
            'See PIPELINE_ARCHITECTURE.md § 5.4.'
        )

    raise LowerToHLIRError(
        f"gemm kind={kind!r} is not yet supported by lower_to_hlir"
    )


def _dst_row_stride(c_buf: tir.Buffer, lane_count: int) -> int:
    """Pick the flat-memory row stride of a gemm output buffer.

    The matmul intrinsic walks the C buffer row-by-row at this stride,
    so it must reflect the **post-expansion** layout — not just the
    last-dim extent of the declared shape:

      * Rank-2 (no lane expansion):  stride = last_dim.
      * Rank-4 COL_PACK ``(1, rows, lane_count, last)``:
            stride = lane_count * last (= MLEN). Each logical row spans
            all lanes' last-dim slices in the flat memory view.
      * Rank-4 ROW_STACK ``(1, lane_count, rows, last)``:
            stride = last. Lanes are stacked separately, so a single
            head's rows are still contiguous at last-dim granularity.

    Returns last_dim as a safe default when the shape is unrecognised."""
    shape = list(c_buf.shape)
    last = int(shape[-1])
    if len(shape) == 4:
        try:
            d2 = int(shape[2])
        except (TypeError, ValueError):
            return last
        if d2 == lane_count:
            return lane_count * last     # COL_PACK: stride spans all lanes
    return last                          # ROW_STACK or rank-2 unmarked


def _lhs_rows_dim(a_buf: tir.Buffer, lane_count: int) -> int:
    """Pick the "rows" dim of a gemm LHS for matmul-vs-mv dispatch.

    Mirrors the btmm path's logic ([rows-dim == 1] → vector variant):
      * Rank-2 (pre-expansion) LHS:        shape[0] is rows.
      * Rank-4 (post-col-pack expansion):  shape[1] is rows; the
        col-pack pattern is (1, rows, lane_count, last).
      * Rank-4 row-stack expansion:        shape[2] is rows after
        ROW_STACK = (1, lane_count, rows, last).
    Returns ``-1`` when the layout is unrecognised; callers should
    treat that as "default to matmul"."""
    shape = list(a_buf.shape)
    if len(shape) == 2:
        try:
            return int(shape[0])
        except (TypeError, ValueError):
            return -1
    if len(shape) == 4:
        # Distinguish ROW_STACK vs COL_PACK by where lane_count sits.
        try:
            d1 = int(shape[1])
            d2 = int(shape[2])
        except (TypeError, ValueError):
            return -1
        if d1 == lane_count:
            return d2          # ROW_STACK: (1, lane, rows, last)
        if d2 == lane_count:
            return d1          # COL_PACK:  (1, rows, lane, last)
    return -1




# ---------------------------------------------------------------------------
# Buffer-scope rewrite of alloc_buffers + reference replacement
# ---------------------------------------------------------------------------

def _rewrite_buffer_scopes(stmt, scopes: BufferScopeMap):
    """Find every Block.alloc_buffers, rebuild buffers with the correct
    PLENA scope, and substitute every reference (data Var, BufferLoad
    buffer, region BufferLoad) with the new buffer."""
    # Collect every alloc'd buffer, build name -> new_buffer map.
    name_to_new: Dict[str, tir.Buffer] = {}
    var_to_new: Dict[tir.Var, tir.Var] = {}

    def collect(s):
        if isinstance(s, tir.Block):
            for buf in s.alloc_buffers:
                target_scope = scopes.get(buf.name)
                if target_scope in (None, "hbm"):
                    continue
                if buf.name in name_to_new:
                    continue
                new_buf = _rebuild_buffer_with_scope(buf, target_scope)
                name_to_new[buf.name] = new_buf
                var_to_new[buf.data] = new_buf.data
            collect(s.body)
            if s.init is not None:
                collect(s.init)
            return
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                collect(c)
            return
        if isinstance(s, tir.BlockRealize):
            collect(s.block)
            return
        if isinstance(s, (tir.AttrStmt, tir.For, tir.LetStmt)):
            collect(s.body)
            return
        if isinstance(s, tir.IfThenElse):
            collect(s.then_case)
            if s.else_case is not None:
                collect(s.else_case)
            return

    collect(stmt)

    def rw_expr(e):
        if isinstance(e, tir.Var):
            return var_to_new.get(e, e)
        if isinstance(e, tir.BufferLoad):
            new_buf = name_to_new.get(e.buffer.name, e.buffer)
            return tir.BufferLoad(new_buf, [rw_expr(i) for i in e.indices])
        if isinstance(e, tir.BufferStore):
            new_buf = name_to_new.get(e.buffer.name, e.buffer)
            return tir.BufferStore(new_buf, rw_expr(e.value),
                                    [rw_expr(i) for i in e.indices])
        if isinstance(e, tir.Call):
            return tir.Call(e.dtype, e.op, [rw_expr(a) for a in e.args])
        if isinstance(e, tir.Cast):
            return type(e)(e.dtype, rw_expr(e.value))
        if hasattr(e, "a") and hasattr(e, "b"):
            return type(e)(rw_expr(e.a), rw_expr(e.b))
        return e

    def rw(s):
        if isinstance(s, tir.SeqStmt):
            return tir.SeqStmt([rw(c) for c in s.seq])
        if isinstance(s, tir.BlockRealize):
            return tir.BlockRealize(
                iter_values=[rw_expr(v) for v in s.iter_values],
                predicate=rw_expr(s.predicate), block=rw(s.block),
            )
        if isinstance(s, tir.Block):
            new_allocs = [name_to_new.get(b.name, b) for b in s.alloc_buffers]
            return tir.Block(
                iter_vars=s.iter_vars, reads=s.reads, writes=s.writes,
                name_hint=s.name_hint, body=rw(s.body),
                init=rw(s.init) if s.init is not None else None,
                alloc_buffers=new_allocs, match_buffers=s.match_buffers,
                annotations=s.annotations,
            )
        if isinstance(s, tir.AttrStmt):
            return tir.AttrStmt(s.node, s.attr_key, rw_expr(s.value), rw(s.body))
        if isinstance(s, tir.For):
            return tir.For(s.loop_var, rw_expr(s.min), rw_expr(s.extent),
                           s.kind, rw(s.body), s.thread_binding, s.annotations)
        if isinstance(s, tir.LetStmt):
            return tir.LetStmt(s.var, rw_expr(s.value), rw(s.body))
        if isinstance(s, tir.IfThenElse):
            return tir.IfThenElse(
                rw_expr(s.condition), rw(s.then_case),
                rw(s.else_case) if s.else_case is not None else None,
            )
        if isinstance(s, tir.Evaluate):
            return tir.Evaluate(rw_expr(s.value))
        return s

    return rw(stmt)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = ["LowerToHLIRError",
           "_lower_copy", "_lower_gemm", "_rewrite_buffer_scopes"]
