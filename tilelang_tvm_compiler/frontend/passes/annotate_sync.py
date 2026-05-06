"""Insert implicit `plena.sync` markers around ops that need cross-lane
fusion in the surrounding group.

A *sync* marker is the boundary at which per-iteration work of the
enclosing ``plena.group`` collapses into a single multi-lane hardware
op. Today the only ops that need it are:

  * **DMAs** — ``tl.tileop.copy`` calls where exactly one side is an HBM
    buffer (the other being a `shared.dyn` / `local.fragment`). The HW
    DMA reads/writes a packed multi-lane stripe in one shot.
  * **BTMM gemms** — ``tl.tileop.gemm_py`` calls running under a
    surrounding ``T.attr(0, "plena.gemm_kind", "btmm")``. The HW BTMM
    instruction processes ``lane_count`` heads in one shot.

Other ops (regular matmul, FP scalar / vector ops, vram→vram copies)
execute per-lane inside the group's serial loop and do not need sync.

Output: each marked Evaluate is wrapped in a structured sync marker,
``AttrStmt(plena.sync, "kind=...,domain=head,width=...")``.
The downstream ``split_lane_groups`` pass walks these markers and uses
the sync width to decide where to split a logical head group into
``outer_for × hardware_width_inner``. Different sync kinds that share the
same domain and width (for example h2v DMA, h2m DMA, and BTMM) are
intentionally compatible and can live in the same sync domain.

Invariants on output:
  * Every DMA copy has exactly one ``plena.sync`` AttrStmt around it.
  * Every BTMM gemm has exactly one ``plena.sync`` AttrStmt around it.
  * No other op carries a ``plena.sync`` annotation.
"""

from __future__ import annotations

from typing import Optional, Set

from tvm import tir

from .annotate_gemm_kind import KIND_KEY


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"

SYNC_KEY = "plena.sync"
SYNC_DOMAIN_HEAD = "head"


def make_sync_value(kind: str, width: int, domain: str = SYNC_DOMAIN_HEAD) -> tir.StringImm:
    if width <= 0:
        raise ValueError(f"sync width must be positive; got {width}")
    return tir.StringImm(f"kind={kind};domain={domain};width={int(width)}")


def parse_sync_value(value) -> dict[str, str]:
    """Parse the structured plena.sync value.

    Older tests / intermediate IR may still use the legacy integer marker;
    treat that as an untyped sync so callers can fall back to their default
    hardware width.
    """
    if isinstance(value, tir.StringImm):
        out: dict[str, str] = {}
        for part in value.value.split(";"):
            if not part:
                continue
            k, _, v = part.partition("=")
            if k:
                out[k] = v
        return out
    return {}


def sync_width(value, default: int) -> int:
    meta = parse_sync_value(value)
    raw = meta.get("width")
    return int(raw) if raw is not None else int(default)


def _wrap_sync(stmt: tir.Stmt, kind: str, width: int) -> tir.Stmt:
    return tir.AttrStmt(
        node=tir.IntImm("int32", 0),
        attr_key=SYNC_KEY,
        value=make_sync_value(kind, width),
        body=stmt,
    )


def _region_buffer(call: tir.Call) -> Optional[tir.Buffer]:
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


def _is_hbm_buffer(buf: Optional[tir.Buffer], hbm_names: Set[str]) -> bool:
    return buf is not None and buf.name in hbm_names


def _is_fpram_fragment(buf: Optional[tir.Buffer]) -> bool:
    """A rank-1 ``local.fragment`` buffer maps to FPRAM (per the convention
    used by ``scope_inference``). This is the lane-stacked FP scratch
    layout the row_load_v_to_fp / row_store_fp_to_v intrinsics target."""
    if buf is None:
        return False
    declared = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    if declared != "local.fragment":
        return False
    if len(buf.shape) != 1:
        return False
    return True


def _walk(stmt, hbm_names: Set[str], gemm_kind: Optional[str],
          sync_width: int,
          in_sync: bool = False):
    if isinstance(stmt, tir.SeqStmt):
        return tir.SeqStmt([
            _walk(c, hbm_names, gemm_kind, sync_width, in_sync)
            for c in stmt.seq
        ])
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_walk(stmt.block, hbm_names, gemm_kind, sync_width, in_sync),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, hbm_names, gemm_kind, sync_width, in_sync),
            init=stmt.init, alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        if stmt.attr_key == SYNC_KEY:
            # Already wrapped — preserve and mark in_sync so the inner
            # Evaluate doesn't get a second wrapper on repeat runs.
            return tir.AttrStmt(
                stmt.node, stmt.attr_key, stmt.value,
                _walk(stmt.body, hbm_names, gemm_kind, sync_width, in_sync=True),
            )
        if stmt.attr_key == KIND_KEY:
            new_kind = (
                stmt.value.value
                if isinstance(stmt.value, tir.StringImm)
                else None
            )
            return tir.AttrStmt(
                stmt.node, stmt.attr_key, stmt.value,
                _walk(stmt.body, hbm_names, new_kind, sync_width, in_sync),
            )
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _walk(stmt.body, hbm_names, gemm_kind, sync_width, in_sync),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, hbm_names, gemm_kind, sync_width, in_sync),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.Evaluate):
        if in_sync:
            return stmt
        v = stmt.value
        if isinstance(v, tir.Call):
            op_name = v.op.name
            if op_name == _TILEOP_COPY:
                src_buf = _region_buffer(v.args[0])
                dst_buf = _region_buffer(v.args[1])
                src_is_hbm = _is_hbm_buffer(src_buf, hbm_names)
                dst_is_hbm = _is_hbm_buffer(dst_buf, hbm_names)
                # Exactly one side HBM = a real DMA; both-HBM (HBM→HBM) or
                # both-local (vram↔vram) is not a sync site.
                if src_is_hbm ^ dst_is_hbm:
                    kind = "dma_h2local" if src_is_hbm else "dma_local2h"
                    return _wrap_sync(stmt, kind, sync_width)
                # vram <-> fpram (rank-1 fragment). The HW S_MAP_*_*
                # instructions are lane-fused: one op moves VLEN==MLEN
                # elements covering all lanes. Treat as a sync site so
                # split_lane_groups / lower_to_hlir collapse the surrounding
                # per-lane for-loop and emit the op exactly once per row.
                src_is_fp = _is_fpram_fragment(src_buf)
                dst_is_fp = _is_fpram_fragment(dst_buf)
                if src_is_fp ^ dst_is_fp:
                    kind = "row_v_to_fp" if dst_is_fp else "row_fp_to_v"
                    return _wrap_sync(stmt, kind, sync_width)
                # vram <-> vram ("tensor cache" path). One V_ADD_VF row
                # covers MLEN = lane_count * hlen elements, so it's also
                # a sync site — collapse the per-lane for-loop into a
                # single multi-lane copy.
                if (src_buf is not None and dst_buf is not None
                        and not src_is_hbm and not dst_is_hbm
                        and not src_is_fp and not dst_is_fp):
                    return _wrap_sync(stmt, "copy_v_to_v", sync_width)
            elif op_name == _TILEOP_GEMM and gemm_kind == "btmm":
                return _wrap_sync(stmt, "btmm", sync_width)
            elif op_name == "tir.call_extern" and v.args:
                # Already-lowered plena.* extern calls. Vector-style ops
                # that act on a whole packed multi-lane VRAM tile in one
                # hardware instruction are sync sites: a single op covers
                # all lanes, so it should fire exactly once per group
                # rather than once-per-lane.
                head = v.args[0]
                if isinstance(head, tir.StringImm):
                    name = head.value
                    if (name == "plena.zero_v"
                            or name.startswith("plena.v_")):
                        return _wrap_sync(stmt, name, sync_width)
        return stmt
    return stmt


def run(func: tir.PrimFunc, sync_width: int = 4) -> tir.PrimFunc:
    hbm_names = {buf.name for buf in func.buffer_map.values()}
    new_body = _walk(func.body, hbm_names, gemm_kind=None,
                     sync_width=sync_width)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "SYNC_KEY", "make_sync_value", "parse_sync_value", "sync_width"]
