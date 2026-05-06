"""Lower the fully-annotated tilelang IR to the plena.* extern-call form
that ``codegen.PlenaCodegen`` consumes.

Responsibilities:

  * Rewrite shared.dyn / local.fragment buffer scopes to vram / mram per
    the ``BufferScopeMap`` returned by ``scope_inference``.
  * Translate ``tl.tileop.copy`` to ``plena.dma_h2v_slice`` /
    ``plena.dma_h2m_slice`` / ``plena.dma_v2h_slice``.
  * Translate ``tl.tileop.gemm_py`` to ``plena.matmul`` (kind=overwrite) or
    ``plena.btmm`` (kind=btmm).
  * **Sync-driven multi-lane fusion**: when a ``tl.tileop.copy`` sits
    inside a ``plena.sync`` AttrStmt that itself sits inside a
    ``plena.group(extent=lane_count)``, we collapse the surrounding
    serial for-loop and emit ONE multi-lane DMA: the lane-var is
    substituted to ``0`` in the start expressions, and the extent at the
    position the lane-var indexed into is set to ``lane_count``. The
    ``plena.btmm`` gemm path collapses similarly — the for-loop wrapper
    is dropped and the gemm is emitted exactly once (the HW BTMM op is
    naturally multi-lane).
  * Pass through ``plena.v_add`` and other already-lowered plena.* calls.
  * Drop ``plena.group`` / ``plena.sync`` / ``plena.gemm_kind`` AttrStmts
    once their information has been consumed.

Pre-conditions: ``annotate_gemm_kind``, ``annotate_group``,
``annotate_sync``, ``split_lane_groups``, ``scope_inference``,
``allocate_group_memory``, ``fuse_elementwise`` have all run.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import tvm
from tvm import tir

from .annotate_group import GROUP_KEY
from .annotate_gemm_kind import KIND_KEY
from .annotate_sync import SYNC_KEY
from .scope_inference import BufferScopeMap


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"


class LowerToHLIRError(RuntimeError):
    pass


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


def _stmt_uses_var(stmt, var_name: str) -> bool:
    """Walk a Stmt + Exprs for any reference to a Var named `var_name`."""
    if isinstance(stmt, tir.SeqStmt):
        return any(_stmt_uses_var(c, var_name) for c in stmt.seq)
    if isinstance(stmt, tir.BlockRealize):
        return _stmt_uses_var(stmt.block, var_name)
    if isinstance(stmt, tir.Block):
        if _stmt_uses_var(stmt.body, var_name):
            return True
        return stmt.init is not None and _stmt_uses_var(stmt.init, var_name)
    if isinstance(stmt, tir.AttrStmt):
        return _expr_uses_var(stmt.value, var_name) or _stmt_uses_var(stmt.body, var_name)
    if isinstance(stmt, tir.For):
        return (_expr_uses_var(stmt.min, var_name)
                or _expr_uses_var(stmt.extent, var_name)
                or _stmt_uses_var(stmt.body, var_name))
    if isinstance(stmt, tir.LetStmt):
        return _expr_uses_var(stmt.value, var_name) or _stmt_uses_var(stmt.body, var_name)
    if isinstance(stmt, tir.IfThenElse):
        if _expr_uses_var(stmt.condition, var_name):
            return True
        if _stmt_uses_var(stmt.then_case, var_name):
            return True
        return stmt.else_case is not None and _stmt_uses_var(stmt.else_case, var_name)
    if isinstance(stmt, tir.Evaluate):
        return _expr_uses_var(stmt.value, var_name)
    return False


def _stmt_contains_extern(stmt, extern_name: str) -> bool:
    if isinstance(stmt, tir.SeqStmt):
        return any(_stmt_contains_extern(c, extern_name) for c in stmt.seq)
    if isinstance(stmt, tir.BlockRealize):
        return _stmt_contains_extern(stmt.block, extern_name)
    if isinstance(stmt, tir.Block):
        return _stmt_contains_extern(stmt.body, extern_name)
    if isinstance(stmt, tir.AttrStmt):
        return _stmt_contains_extern(stmt.body, extern_name)
    if isinstance(stmt, tir.For):
        return _stmt_contains_extern(stmt.body, extern_name)
    if isinstance(stmt, tir.LetStmt):
        return _stmt_contains_extern(stmt.body, extern_name)
    if isinstance(stmt, tir.IfThenElse):
        return (
            _stmt_contains_extern(stmt.then_case, extern_name)
            or (
                stmt.else_case is not None
                and _stmt_contains_extern(stmt.else_case, extern_name)
            )
        )
    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if not (isinstance(v, tir.Call)
                and getattr(v.op, "name", None) == "tir.call_extern"
                and v.args
                and isinstance(v.args[0], tir.StringImm)):
            return False
        return v.args[0].value == extern_name
    return False


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
                         in_sync: bool) -> tir.Stmt:
    """Lower one ``T.copy`` between VRAM and FPRAM to a row-wide MAP transfer.

    The HW op (S_MAP_V_FP / S_MAP_FP_V) moves VLEN=MLEN elements per
    invocation, naturally serving all lanes at once. Lane fusion is
    therefore implicit — when in_sync, we just substitute lane_var to 0
    in both index sides; we do NOT multiply any extent (HW op size is
    fixed).
    """
    if in_sync and lane_var is not None:
        zero = tir.IntImm("int32", 0)
        vram_starts = [_substitute_var(s, lane_var, zero) for s in vram_starts]
        fp_starts = [_substitute_var(s, lane_var, zero) for s in fp_starts]

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
                       lane_var: Optional[str], in_sync: bool) -> tir.Stmt:
    """Lower a vram→vram T.copy to one V_ADD_VF row transfer.

    Lane fusion handling mirrors _lower_row_v_fp_copy: when in_sync, the
    lane_var is substituted to 0 in both index sides (the HW V_ADD_VF
    processes one full MLEN-wide vector per call, naturally covering all
    lanes — no extent multiplication needed).
    """
    if in_sync and lane_var is not None:
        zero = tir.IntImm("int32", 0)
        src_starts = [_substitute_var(s, lane_var, zero) for s in src_starts]
        dst_starts = [_substitute_var(s, lane_var, zero) for s in dst_starts]

    src_offset_expr = _flatten_starts(src_buf, src_starts)
    dst_offset_expr = _flatten_starts(dst_buf, dst_starts)

    return _evaluate(_make_call_extern(
        "plena.copy_v_to_v",
        [src_buf.data, src_offset_expr, dst_buf.data, dst_offset_expr],
    ))


def _lower_copy(call: tir.Call,
                scopes: BufferScopeMap,
                lane_count: int,
                lane_var: Optional[str],
                in_sync: bool) -> tir.Stmt:
    """Lower a tl.tileop.copy to plena.dma_h2v_slice / dma_h2m_slice /
    dma_v2h_slice. When `in_sync` is True and `lane_var` is set, substitute
    the lane var to 0 and multiply the lane-position extent by lane_count
    to fold all per-lane iterations into one multi-lane DMA."""
    src_buf, src_starts, _src_exts = _region_components(call.args[0])
    dst_buf, dst_starts, _dst_exts = _region_components(call.args[1])
    src_scope = scopes.get(src_buf.name)
    dst_scope = scopes.get(dst_buf.name)

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
        )
    elif src_scope == "fpram" and dst_scope == "vram":
        return _lower_row_v_fp_copy(
            vram_buf=dst_buf, vram_starts=dst_starts,
            fp_buf=src_buf, fp_starts=src_starts,
            direction="fp_to_v",
            lane_var=lane_var, in_sync=in_sync,
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


def _lower_gemm(call: tir.Call,
                scopes: BufferScopeMap,
                kind: str,
                lane_count: int,
                target_mlen: int,
                target_hlen: int) -> tir.Stmt:
    """Lower tl.tileop.gemm_py based on its `kind` annotation."""
    a_buf, a_starts, _a_exts = _region_components(call.args[0])
    b_buf, b_starts, _b_exts = _region_components(call.args[1])
    c_buf, c_starts, c_exts = _region_components(call.args[2])

    a_scope = scopes.get(a_buf.name)
    b_scope = scopes.get(b_buf.name)
    c_scope = scopes.get(c_buf.name)
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

    if kind in ("overwrite", "mv"):
        # Per-buffer flat element offsets. Whole-buffer T.gemm calls
        # naturally produce zero starts (preserving the original
        # behaviour); sliced calls fold their starts into the trailing
        # offset args of plena.matmul / plena.mv. _flatten_starts handles
        # both static and PrimExpr starts (e.g. lane_var * stride from a
        # T.gemm(buf[..., by, ...], ...) slice), so the offsets are
        # materialised to gp registers at isa-emit time the same way
        # split_lane_groups already projects them.
        a_off = _flatten_starts(a_buf, a_starts)
        b_off = _flatten_starts(b_buf, b_starts)
        c_off = _flatten_starts(c_buf, c_starts)

        if kind == "mv":
            # plena.mv only takes the three offsets — no M_tiles / K_tiles /
            # row_stride. The M_MV/M_MV_WO HW path always processes one
            # MLEN-wide LHS row × blen-tile slices of the matrix per call;
            # the kernel author shapes the slice extents to match.
            return _evaluate(_make_call_extern(
                "plena.mv",
                [a_buf.data, b_buf.data, c_buf.data, a_off, b_off, c_off],
            ))

        c_inner_ext = int(c_exts[-1].value) if c_exts else int(c_buf.shape[-1])
        c_inner_buf = int(c_buf.shape[-1])
        N = c_inner_ext
        return _evaluate(_make_call_extern(
            "plena.matmul",
            [
                a_buf.data, b_buf.data, c_buf.data,
                tir.IntImm("int32", 1),    # M_tiles
                tir.IntImm("int32", 1),    # K_tiles
                tir.IntImm("int32", N),
                a_off, b_off, c_off,
                tir.IntImm("int32", c_inner_buf),  # dst_row_stride
            ],
        ))

    raise LowerToHLIRError(
        f"gemm kind={kind!r} is not yet supported by lower_to_hlir; "
        f"the additive-cache pass is needed for kind='add'"
    )


# ---------------------------------------------------------------------------
# Lane-for segmentation
# ---------------------------------------------------------------------------

def _flatten_seq(stmt) -> List[tir.Stmt]:
    """Flatten a (possibly nested) SeqStmt into a flat list of stmts."""
    if isinstance(stmt, tir.SeqStmt):
        out: List[tir.Stmt] = []
        for c in stmt.seq:
            out.extend(_flatten_seq(c))
        return out
    return [stmt]


def _segment_lane_for(for_stmt: tir.For, lowered_body) -> tir.Stmt:
    """Split a lane-fused for-loop's body into runs separated by sync
    points and re-emit so that:

      * every sync-fused op (no longer references the lane var) runs
        EXACTLY ONCE — outside any for-by — as a multi-lane HW op;
      * every contiguous run of per-lane ops (still references the lane
        var) is wrapped in its own for-by(0..lane_count) loop.

    The lane_var var is *itself* not by-dependent so we descend through
    any wrapping ``BlockRealize`` / ``Block`` (which hold cross-lane
    state like ``alloc_buffers``) and segment the *innermost* op
    sequence — the wrappers stay outside, hoisted above the segments.
    """

    def descend(stmt):
        # Walk through wrappers that aren't lane-iteration boundaries.
        # The wrappers stay around the segmented body; only the inner
        # statement sequence is split.
        if isinstance(stmt, tir.BlockRealize):
            return tir.BlockRealize(
                stmt.iter_values, stmt.predicate, descend(stmt.block),
            )
        if isinstance(stmt, tir.Block):
            return tir.Block(
                iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
                name_hint=stmt.name_hint, body=descend(stmt.body),
                init=stmt.init, alloc_buffers=stmt.alloc_buffers,
                match_buffers=stmt.match_buffers, annotations=stmt.annotations,
            )
        return _do_segment(for_stmt, stmt)

    return descend(lowered_body)


def _do_segment(for_stmt: tir.For, body) -> tir.Stmt:
    """Segment a flattened body relative to the lane var.

    The traversal is *recursive* on inner for-loops: any nested loop's
    body is itself segmented w.r.t. the lane var, which is equivalent to
    loop-interchange followed by per-segment lane wrapping. This handles
    patterns like ``for kv_block: { sync DMA, FP using by, sync v_add }``
    correctly — the sync ops hoist outside the for-by, the FP body wraps
    in an inner for-by, all sitting inside the original for-kv-block.
    """
    flat = _flatten_seq(body)
    lane_var_name = for_stmt.loop_var.name

    out: List[tir.Stmt] = []
    cur_lane_run: List[tir.Stmt] = []

    def is_pure_lane_run(stmt) -> bool:
        """True when an inner statement can stay inside the current
        per-lane run. This preserves `for by { for row { ... }; matmul }`
        for per-lane row loops, while still recursively segmenting loops
        that contain sync-fused ops."""
        parts = _flatten_seq(stmt)
        return bool(parts) and all(_stmt_uses_var(p, lane_var_name) for p in parts)

    def flush_lane_run():
        if not cur_lane_run:
            return
        run_body = (
            cur_lane_run[0] if len(cur_lane_run) == 1
            else tir.SeqStmt(list(cur_lane_run))
        )
        kind = (
            tir.ForKind.UNROLLED
            if _stmt_contains_extern(run_body, "plena.matmul")
            else for_stmt.kind
        )
        out.append(tir.For(
            for_stmt.loop_var, for_stmt.min, for_stmt.extent, kind,
            run_body, for_stmt.thread_binding, for_stmt.annotations,
        ))
        cur_lane_run.clear()

    for s in flat:
        if isinstance(s, tir.For):
            if is_pure_lane_run(s.body):
                cur_lane_run.append(s)
                continue
            # Inner for-loop: recursively segment its body. The result no
            # longer needs the outer for-by wrapper because the recursion
            # already places per-lane runs inside the inner body. So we
            # hoist the (transformed) inner for-loop out of the outer
            # for-by entirely.
            new_inner = _segment_lane_for(for_stmt, s.body)
            new_for = tir.For(
                s.loop_var, s.min, s.extent, s.kind,
                new_inner, s.thread_binding, s.annotations,
            )
            flush_lane_run()
            out.append(new_for)
        elif _stmt_uses_var(s, lane_var_name):
            cur_lane_run.append(s)
        else:
            flush_lane_run()
            out.append(s)
    flush_lane_run()

    if not out:
        return tir.Evaluate(tir.IntImm("int32", 0))
    return out[0] if len(out) == 1 else tir.SeqStmt(out)


# ---------------------------------------------------------------------------
# Body walker
# ---------------------------------------------------------------------------

def _lower_body(stmt,
                scopes: BufferScopeMap,
                lane_count: int,
                target_mlen: int,
                target_hlen: int,
                gemm_kind: Optional[str] = None,
                in_sync: bool = False,
                lane_var: Optional[str] = None,
                drop_outer_for: bool = False) -> Optional[tir.Stmt]:
    """Recurse and rewrite. Returns None when the input was an Evaluate
    that has been completely consumed by a fusion (caller should drop)."""
    if isinstance(stmt, tir.AttrStmt):
        # Strip plena.* annotations — they've served their purpose.
        if stmt.attr_key in (KIND_KEY, GROUP_KEY, SYNC_KEY):
            new_kind = gemm_kind
            new_in_sync = in_sync
            new_lane_var = lane_var
            new_drop = drop_outer_for
            if stmt.attr_key == KIND_KEY and isinstance(stmt.value, tir.StringImm):
                new_kind = stmt.value.value
            elif stmt.attr_key == SYNC_KEY:
                new_in_sync = True
                # If we're already inside a lane group, syncing means the
                # surrounding for-loop will be dropped (the op fuses across
                # all lanes into one multi-lane HW op).
                if lane_var is not None:
                    new_drop = True
            elif stmt.attr_key == GROUP_KEY:
                if (isinstance(stmt.value, tir.IntImm)
                        and int(stmt.value.value) == lane_count):
                    # Mark that the surrounding For's loop_var is the lane
                    # var. The for-loop itself has set lane_var already
                    # (see tir.For handling below); nothing to do here.
                    pass
            return _lower_body(stmt.body, scopes, lane_count, target_mlen,
                               target_hlen, new_kind, new_in_sync,
                               new_lane_var, new_drop)
        return _passthrough_attr(stmt, scopes, lane_count, target_mlen,
                                  target_hlen, gemm_kind, in_sync, lane_var,
                                  drop_outer_for)

    if isinstance(stmt, tir.For):
        # Detect "this For wraps a plena.group(extent=lane_count)" — that
        # makes its loop_var the lane var.
        is_lane_for = (
            isinstance(stmt.body, tir.AttrStmt)
            and stmt.body.attr_key == GROUP_KEY
            and isinstance(stmt.body.value, tir.IntImm)
            and int(stmt.body.value.value) == lane_count
        )
        new_lane_var = stmt.loop_var.name if is_lane_for else lane_var
        new_body = _lower_body(stmt.body, scopes, lane_count, target_mlen,
                                target_hlen, gemm_kind, in_sync,
                                new_lane_var, drop_outer_for=False)
        if new_body is None:
            return None
        if not is_lane_for:
            return tir.For(
                stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
                new_body, stmt.thread_binding, stmt.annotations,
            )
        # Lane-fused for: segment body at sync boundaries.
        # Each statement is either:
        #   * a sync-fused op (multi-lane HW op, body no longer references
        #     the lane var) — emitted ONCE outside any per-lane for-loop;
        #   * a per-lane op (still references the lane var) — wrapped in a
        #     for-by loop to run lane_count times.
        # Order is preserved.
        return _segment_lane_for(stmt, new_body)

    if isinstance(stmt, tir.SeqStmt):
        out = []
        for c in stmt.seq:
            r = _lower_body(c, scopes, lane_count, target_mlen, target_hlen,
                            gemm_kind, in_sync, lane_var, drop_outer_for)
            if r is not None:
                out.append(r)
        if not out:
            return tir.Evaluate(tir.IntImm("int32", 0))
        return tir.SeqStmt(out) if len(out) > 1 else out[0]

    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_lower_body(stmt.block, scopes, lane_count, target_mlen,
                               target_hlen, gemm_kind, in_sync, lane_var,
                               drop_outer_for),
        )
    if isinstance(stmt, tir.Block):
        return _rewrite_block(stmt, scopes, lane_count, target_mlen,
                               target_hlen, gemm_kind, in_sync, lane_var,
                               drop_outer_for)

    if isinstance(stmt, tir.Evaluate):
        v = stmt.value
        if isinstance(v, tir.Call):
            op_name = v.op.name
            if op_name == _TILEOP_COPY:
                return _lower_copy(v, scopes, lane_count, lane_var, in_sync)
            if op_name == _TILEOP_GEMM:
                kind = gemm_kind or "overwrite"
                return _lower_gemm(v, scopes, kind, lane_count, target_mlen,
                                   target_hlen)
            # Already-lowered plena.* extern calls — pass through.
            if op_name == "tir.call_extern":
                return _project_matmul_offsets_to_lane(stmt, lane_var)
        return stmt

    return stmt


def _passthrough_attr(stmt, scopes, lane_count, target_mlen, target_hlen,
                      gemm_kind, in_sync, lane_var, drop_outer_for):
    new_body = _lower_body(stmt.body, scopes, lane_count, target_mlen,
                            target_hlen, gemm_kind, in_sync, lane_var,
                            drop_outer_for)
    if new_body is None:
        return None
    return tir.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)


def _rewrite_block(block, scopes, lane_count, target_mlen, target_hlen,
                   gemm_kind, in_sync, lane_var, drop_outer_for):
    new_body = _lower_body(block.body, scopes, lane_count, target_mlen,
                            target_hlen, gemm_kind, in_sync, lane_var,
                            drop_outer_for)
    return tir.Block(
        iter_vars=block.iter_vars, reads=block.reads, writes=block.writes,
        name_hint=block.name_hint, body=new_body, init=block.init,
        alloc_buffers=block.alloc_buffers, match_buffers=block.match_buffers,
        annotations=block.annotations,
    )


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
# Public entry
# ---------------------------------------------------------------------------

def run(func: tir.PrimFunc,
        scopes: BufferScopeMap,
        lane_count: int = 4,
        target_mlen: int = 64,
        target_hlen: int = 16) -> tir.PrimFunc:
    rewritten = _rewrite_buffer_scopes(func.body, scopes)
    lowered = _lower_body(rewritten, scopes, lane_count, target_mlen, target_hlen)
    if lowered is None:
        lowered = tir.Evaluate(tir.IntImm("int32", 0))
    return tir.PrimFunc(
        params=func.params,
        body=lowered,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "LowerToHLIRError"]
