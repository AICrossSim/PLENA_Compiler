"""Expand the storage of buffers that participate in lane-fused ops.

Expansion is **role-based** with two distinct modes:

  * **Column-packed (BSHD)** — applied to BTMM inputs and DMA local-side
    buffers inside a lane group. The last-dim of the buffer holds
    ``lane_count`` lanes worth of data contiguously, matching how the
    hardware DMA / BTMM consume packed BSHD::

        shape = (..., orig_last)   -->  (..., orig_last * lane_count)
        Q_sh[..., j]               -->  Q_sh[..., lane_var * orig_last + j]

  * **Row-stacked (BHSD)** — applied to BTMM outputs. The hardware
    M_BMM_WO drains all lanes into one buffer with heads stacked along
    the row direction, not packed in columns. So the *first* dim
    expands and the *first* index gets the lane offset::

        shape = (orig_first, ...)  -->  (orig_first * lane_count, ...)
        S_loc[i, ...]              -->  S_loc[lane_var * orig_first + i, ...]

  * **Lane-stacked FPRAM** — applied to per-lane FP scratch buffers
    used as scalar operands of ``plena.fp_*_at`` / ``plena.row_*_at``.
    Users declare a 1D per-lane fragment and the compiler exposes the
    lane dimension automatically::

        shape = (rows,)            -->  (lane_count, rows)
        M_old[row]                 -->  M_old[lane_var, row]

Role detection:

  * Operand 0 / 1 of a ``tl.tileop.gemm_py`` under
    ``plena.gemm_kind = "btmm"``  → column-packed.
  * Operand 2 of a btmm gemm                                 → row-stacked.
  * ``tl.tileop.copy`` local side inside a ``plena.group(lane_count)``
    AttrStmt                                                  → column-packed.
  * Matmul (``kind != "btmm"``) operands are **neutral** — they neither
    trigger nor prevent expansion. If the same buffer is also touched
    by an expanding role, that role wins.

A buffer flagged for *both* modes is rejected (an obvious
miscompilation). Buffers that match neither role are unchanged.

``lane_var`` is the loop_var of the for-loop wrapping the inner
``plena.group(extent=lane_count)`` in which the eligible op lives.

Pre-conditions:
  * ``annotate_gemm_kind`` ran (kind annotations are present).
  * ``annotate_group``, ``annotate_sync`` ran (group / sync attrs are present).
  * ``split_lane_groups`` ran with the same ``lane_count`` (lane-fusion
    groups have extent == ``lane_count``).
  * ``scope_inference`` produced a ``BufferScopeMap``.

Post-condition: every "eligible" buffer has its lane dimension made
explicit and all references to it carry the lane offset in the
appropriate index position.
"""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

import tvm
from tvm import tir

from .annotate_group import GROUP_KEY
from .annotate_gemm_kind import KIND_KEY
from .scope_inference import BufferScopeMap


_TILEOP_COPY = "tl.tileop.copy"
_TILEOP_GEMM = "tl.tileop.gemm_py"
_TILEOP_REGION = "tl.tileop.region"


class AllocateGroupMemoryError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _region_buffer(call) -> Optional[tir.Buffer]:
    if not isinstance(call, tir.Call) or call.op.name != _TILEOP_REGION:
        return None
    load = call.args[0]
    if not isinstance(load, tir.BufferLoad):
        return None
    return load.buffer


COL_PACK = "col_pack"
ROW_STACK = "row_stack"
FP_LANE = "fp_lane"


_FP_EXTERN_POSITIONS = {
    "plena.fp_copy_at": (0, 1),
    "plena.fp_add_at": (0, 1, 2),
    "plena.fp_sub_at": (0, 1, 2),
    "plena.fp_mul_at": (0, 1, 2),
    "plena.fp_max_at": (0, 1, 2),
    "plena.fp_exp_at": (0, 1),
    "plena.fp_reci_at": (0, 1),
    "plena.fp_sqrt_at": (0, 1),
    "plena.row_reduce_max_at": (1,),
    "plena.row_reduce_sum_at": (1,),
    "plena.row_sub_fp_at": (1,),
    "plena.row_mul_fp_at": (1,),
    "plena.row_add_fp_at": (1,),
}


def _collect_alloc_buffers(stmt) -> Dict[tir.Var, tir.Buffer]:
    """Walk the IR collecting every Block.alloc_buffers, keyed by the
    buffer's data Var. Used so call_extern args (which reference data
    Vars directly) can resolve back to the underlying Buffer object."""
    out: Dict[tir.Var, tir.Buffer] = {}

    def visit(s):
        if isinstance(s, tir.Block):
            for buf in s.alloc_buffers:
                out[buf.data] = buf
            visit(s.body)
            if s.init is not None:
                visit(s.init)
            return
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
            return
        if isinstance(s, tir.BlockRealize):
            visit(s.block)
            return
        if isinstance(s, (tir.AttrStmt, tir.For, tir.LetStmt)):
            visit(s.body)
            return
        if isinstance(s, tir.IfThenElse):
            visit(s.then_case)
            if s.else_case is not None:
                visit(s.else_case)

    visit(stmt)
    return out


def _expr_fpram_buffers(expr, scopes: BufferScopeMap, out: Set[tir.Buffer]) -> None:
    if isinstance(expr, tir.BufferLoad):
        if scopes.get(expr.buffer.name) == "fpram":
            out.add(expr.buffer)
        for i in expr.indices:
            _expr_fpram_buffers(i, scopes, out)
        return
    if isinstance(expr, tir.Call):
        for a in expr.args:
            _expr_fpram_buffers(a, scopes, out)
        return
    if hasattr(expr, "a") and hasattr(expr, "b"):
        _expr_fpram_buffers(expr.a, scopes, out)
        _expr_fpram_buffers(expr.b, scopes, out)
        return
    if hasattr(expr, "value"):
        _expr_fpram_buffers(expr.value, scopes, out)


def _analyze(func: tir.PrimFunc, lane_count: int,
             hbm_names: Set[str],
             scopes: BufferScopeMap) -> Dict[str, Tuple[tir.PrimExpr, int, str]]:
    """Return ``buffer_name -> (lane_expr, factor, mode)`` for every
    buffer that should be expanded.

    ``mode`` is one of ``COL_PACK`` (last-dim expansion) or ``ROW_STACK``
    (first-dim expansion). ``factor`` is the active hardware lane-domain
    width. FPRAM has no sync demand of its own; it follows the nearest
    already-established lane group instead of the logical head count.
    """
    info: Dict[str, Tuple[tir.PrimExpr, int, str]] = {}
    data_var_to_buffer = _collect_alloc_buffers(func.body)

    def record(buf: tir.Buffer, lane_expr: tir.PrimExpr, factor: int, mode: str):
        if not buf.shape:
            return
        prev = info.get(buf.name)
        if prev is not None:
            if str(prev[0]) != str(lane_expr):
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} touched by multiple lane expressions "
                    f"({prev[0]!r} and {lane_expr!r}); not yet supported"
                )
            if prev[1] != factor:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} touched with multiple lane factors "
                    f"({prev[1]} and {factor}); not yet supported"
                )
            # Mode conflict: ROW_STACK (BTMM output's BHSD layout) wins
            # because it reflects the actual hardware-produced layout.
            # A DMA touching the same buffer must work per-head against
            # that layout — handled later in lowering.
            if prev[2] == ROW_STACK:
                return  # keep existing row_stack assignment
            if mode == ROW_STACK:
                pass  # fall through, overwrite previous col_pack
            elif prev[2] != mode:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} flagged for both {prev[2]!r} and "
                    f"{mode!r} expansion — that's a miscompilation"
                )
        info[buf.name] = (lane_expr, factor, mode)

    def visit(stmt, lane_var: Optional[tir.Var], gemm_kind: Optional[str]):
        if isinstance(stmt, tir.AttrStmt):
            new_kind = gemm_kind
            if stmt.attr_key == KIND_KEY and isinstance(stmt.value, tir.StringImm):
                new_kind = stmt.value.value
            visit(stmt.body, lane_var, new_kind)
            return
        if isinstance(stmt, tir.For):
            inner_lane = lane_var
            if (isinstance(stmt.body, tir.AttrStmt)
                    and stmt.body.attr_key == GROUP_KEY
                    and isinstance(stmt.body.value, tir.IntImm)
                    and int(stmt.body.value.value) == lane_count):
                inner_lane = stmt.loop_var
            visit(stmt.body, inner_lane, gemm_kind)
            return
        if isinstance(stmt, tir.SeqStmt):
            for c in stmt.seq:
                visit(c, lane_var, gemm_kind)
            return
        if isinstance(stmt, tir.BlockRealize):
            visit(stmt.block, lane_var, gemm_kind)
            return
        if isinstance(stmt, tir.Block):
            visit(stmt.body, lane_var, gemm_kind)
            if stmt.init is not None:
                visit(stmt.init, lane_var, gemm_kind)
            return
        if isinstance(stmt, tir.LetStmt):
            visit(stmt.body, lane_var, gemm_kind)
            return
        if isinstance(stmt, tir.IfThenElse):
            visit(stmt.then_case, lane_var, gemm_kind)
            if stmt.else_case is not None:
                visit(stmt.else_case, lane_var, gemm_kind)
            return
        if isinstance(stmt, tir.Evaluate):
            v = stmt.value
            if not isinstance(v, tir.Call):
                return
            op_name = v.op.name
            if op_name == _TILEOP_GEMM and gemm_kind == "btmm" and lane_var is not None:
                lhs = _region_buffer(v.args[0])
                rhs = _region_buffer(v.args[1])
                dst = _region_buffer(v.args[2])
                if lhs is not None:
                    record(lhs, lane_var, lane_count, COL_PACK)
                if rhs is not None:
                    record(rhs, lane_var, lane_count, COL_PACK)
                if dst is not None:
                    record(dst, lane_var, lane_count, ROW_STACK)
            elif op_name == _TILEOP_COPY and lane_var is not None:
                src = _region_buffer(v.args[0])
                dst = _region_buffer(v.args[1])
                src_is_hbm = src is not None and src.name in hbm_names
                dst_is_hbm = dst is not None and dst.name in hbm_names
                if src_is_hbm and dst is not None and not dst_is_hbm:
                    record(dst, lane_var, lane_count, COL_PACK)
                elif dst_is_hbm and src is not None and not src_is_hbm:
                    record(src, lane_var, lane_count, COL_PACK)
                else:
                    # vram <-> fpram. The S_MAP_*_* HW op moves MLEN
                    # elements per call regardless of fragment shape, so
                    # the rank-1 fpram side MUST be lane-stacked to
                    # (lane_count, hlen) = MLEN; otherwise the HW
                    # transfer corrupts neighbouring FPRAM slots.
                    for buf in (src, dst):
                        if (buf is not None
                                and scopes.get(buf.name) == "fpram"
                                and len(buf.shape) == 1):
                            record(buf, lane_var, lane_count, FP_LANE)
            elif op_name == "tir.call_extern" and lane_var is not None and v.args:
                # Already-lowered plena.* extern calls. Their buffer-Var
                # args refer to lane-shared VRAM tiles; mark them
                # COL_PACK so the per-lane shape gets expanded into the
                # 4D BSHD-packed layout the existing intrinsics (and the
                # matmul / row_*_at backends) expect.
                head = v.args[0]
                if not isinstance(head, tir.StringImm):
                    return
                name = head.value
                raw_args = list(v.args[1:])
                for pos in _FP_EXTERN_POSITIONS.get(name, ()):
                    if pos >= len(raw_args):
                        continue
                    arg = raw_args[pos]
                    if isinstance(arg, tir.BufferLoad):
                        record(arg.buffer, lane_var, lane_count, FP_LANE)
                if not (name == "plena.zero_v"
                        or name == "plena.matmul"
                        or name.startswith("plena.v_")
                        or name.startswith("plena.row_")):
                    return
                # Walk trailing args; for each Var that resolves to an
                # alloc'd VRAM buffer, mark COL_PACK.
                for arg in raw_args:
                    if not isinstance(arg, tir.Var):
                        continue
                    buf = data_var_to_buffer.get(arg)
                    if buf is not None:
                        record(buf, lane_var, lane_count, COL_PACK)
            # Matmul / FP-scalar ops without buffer-Vars (e.g. fp_*_at
            # on raw FPRAM addresses) are neutral.
            return
        if isinstance(stmt, tir.BufferStore) and lane_var is not None:
            if scopes.get(stmt.buffer.name) == "fpram":
                record(stmt.buffer, lane_var, lane_count, FP_LANE)
            bufs: Set[tir.Buffer] = set()
            _expr_fpram_buffers(stmt.value, scopes, bufs)
            for buf in bufs:
                record(buf, lane_var, lane_count, FP_LANE)

    visit(func.body, lane_var=None, gemm_kind=None)
    return info


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

def _expand_buffer(buf: tir.Buffer, factor: int, mode: str) -> tir.Buffer:
    """Expand a per-lane buffer to a multi-lane buffer.

    The 4D output matches the layouts the row_*_at / matmul intrinsics
    in `isa_pass` expect:

      * COL_PACK: ``(rows, last) → (1, rows, lane_count, last)``
        BSHD-packed-narrow; head h's data occupies cols
        [h*last, (h+1)*last) within an mlen-wide row.
      * ROW_STACK: ``(rows, mlen) → (1, lane_count, rows, mlen)``
        BHSD-stacked; head h's tile starts at row h*rows in the flat
        memory view.

    The 4D VRAM form keeps logical 2D arithmetic correct (matmul / DMA see
    the same flat layout) and lets `_resolve_row_at_coords` apply its
    existing packed-vs-full-width detection rules unchanged.
    """
    shape = list(buf.shape)
    one = tir.IntImm("int32", 1)
    lane_imm = tir.IntImm("int32", int(factor))
    if mode == FP_LANE:
        if len(shape) != 1:
            raise AllocateGroupMemoryError(
                f"buffer {buf.name!r}: FPRAM lane expansion expects rank-1 pre-shape; "
                f"got rank {len(shape)} ({shape})"
            )
        new_shape = [lane_imm, shape[0]]
    elif len(shape) != 2:
        raise AllocateGroupMemoryError(
            f"buffer {buf.name!r}: expansion only supports 2D pre-shapes for VRAM/MRAM roles; "
            f"got rank {len(shape)} ({shape})"
        )
    else:
        rows, last = shape
        if mode == COL_PACK:
            new_shape = [one, rows, lane_imm, last]
        elif mode == ROW_STACK:
            new_shape = [one, lane_imm, rows, last]
        else:
            raise AllocateGroupMemoryError(f"unknown mode {mode!r}")
    declared_scope = buf.scope() if callable(getattr(buf, "scope", None)) else "global"
    new_data = tir.Var(buf.data.name, tvm.ir.PointerType(
        tvm.ir.PrimType(buf.dtype), declared_scope,
    ))
    return tir.decl_buffer(
        shape=new_shape,
        dtype=buf.dtype,
        name=buf.name,
        data=new_data,
        scope=declared_scope,
    )


class _Rewriter:
    def __init__(self, info: Dict[str, Tuple[tir.PrimExpr, int, str]], lane_count: int):
        self.info = info
        self.lane_count = lane_count
        self.name_to_new: Dict[str, tir.Buffer] = {}
        self.var_to_new: Dict[tir.Var, tir.Var] = {}

    def _expand(self, buf: tir.Buffer) -> tir.Buffer:
        if buf.name not in self.info:
            return buf
        if buf.name in self.name_to_new:
            return self.name_to_new[buf.name]
        _lane_expr, factor, mode = self.info[buf.name]
        # Idempotent on repeat runs.
        if mode == FP_LANE:
            if len(buf.shape) == 2:
                new_buf = buf
            elif len(buf.shape) == 1:
                new_buf = _expand_buffer(buf, factor, mode)
            else:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} has unexpected rank {len(buf.shape)}; "
                    f"expected 1 (per-lane) or 2 (already expanded) for fpram"
                )
        else:
            if len(buf.shape) == 4:
                new_buf = buf
            elif len(buf.shape) == 2:
                new_buf = _expand_buffer(buf, factor, mode)
            else:
                raise AllocateGroupMemoryError(
                    f"buffer {buf.name!r} has unexpected rank {len(buf.shape)}; "
                    f"expected 2 (per-lane) or 4 (already expanded)"
                )
        self.name_to_new[buf.name] = new_buf
        self.var_to_new[buf.data] = new_buf.data
        return new_buf

    def visit(self, n):
        if isinstance(n, tir.SeqStmt):
            return tir.SeqStmt([self.visit(c) for c in n.seq])
        if isinstance(n, tir.BlockRealize):
            return tir.BlockRealize(
                iter_values=[self.visit_expr(v) for v in n.iter_values],
                predicate=self.visit_expr(n.predicate),
                block=self.visit(n.block),
            )
        if isinstance(n, tir.Block):
            new_allocs = [self._expand(b) for b in n.alloc_buffers]
            return tir.Block(
                iter_vars=n.iter_vars, reads=n.reads, writes=n.writes,
                name_hint=n.name_hint, body=self.visit(n.body),
                init=self.visit(n.init) if n.init is not None else None,
                alloc_buffers=new_allocs,
                match_buffers=n.match_buffers, annotations=n.annotations,
            )
        if isinstance(n, tir.AttrStmt):
            return tir.AttrStmt(
                n.node, n.attr_key,
                self.visit_expr(n.value), self.visit(n.body),
            )
        if isinstance(n, tir.For):
            return tir.For(
                n.loop_var, self.visit_expr(n.min), self.visit_expr(n.extent),
                n.kind, self.visit(n.body), n.thread_binding, n.annotations,
            )
        if isinstance(n, tir.LetStmt):
            return tir.LetStmt(n.var, self.visit_expr(n.value), self.visit(n.body))
        if isinstance(n, tir.IfThenElse):
            return tir.IfThenElse(
                self.visit_expr(n.condition),
                self.visit(n.then_case),
                self.visit(n.else_case) if n.else_case is not None else None,
            )
        if isinstance(n, tir.Evaluate):
            return tir.Evaluate(self.visit_expr(n.value))
        if isinstance(n, tir.BufferStore):
            return self.visit_expr(n)
        return n

    def _fold_lane(self, indices, buf_name):
        """Lift 2D per-lane indices to the 4D layout produced by
        `_expand_buffer`. The lane var is inserted at the new lane slot;
        the original (row, col) keep their slots in the new shape:

          COL_PACK  2D [r, c] → 4D [0, r, by, c]
          ROW_STACK 2D [r, c] → 4D [0, by, r, c]

        Already-4D indices (idempotent re-walk) are left untouched.
        """
        if buf_name not in self.info or not indices:
            return indices
        lane_expr, _factor, mode = self.info[buf_name]
        if mode == FP_LANE:
            if len(indices) == 2:
                return list(indices)
            if len(indices) != 1:
                raise AllocateGroupMemoryError(
                    f"buffer {buf_name!r} access has rank {len(indices)}; "
                    f"_fold_lane expects pre-expansion rank 1 for fpram"
                )
            return [lane_expr, indices[0]]
        if len(indices) == 4:
            return list(indices)
        if len(indices) != 2:
            raise AllocateGroupMemoryError(
                f"buffer {buf_name!r} access has rank {len(indices)}; "
                f"_fold_lane expects pre-expansion rank 2"
            )
        zero_dtype = getattr(lane_expr, "dtype", "int32")
        zero = tir.IntImm(zero_dtype, 0)
        r, c = indices
        if mode == COL_PACK:
            return [zero, r, lane_expr, c]
        return [zero, lane_expr, r, c]

    def visit_expr(self, e):
        if isinstance(e, tir.Var):
            return self.var_to_new.get(e, e)
        if isinstance(e, tir.BufferLoad):
            new_buf = self.name_to_new.get(e.buffer.name, e.buffer)
            indices = [self.visit_expr(i) for i in e.indices]
            indices = self._fold_lane(indices, e.buffer.name)
            return tir.BufferLoad(new_buf, indices)
        if isinstance(e, tir.BufferStore):
            new_buf = self.name_to_new.get(e.buffer.name, e.buffer)
            indices = [self.visit_expr(i) for i in e.indices]
            indices = self._fold_lane(indices, e.buffer.name)
            return tir.BufferStore(new_buf, self.visit_expr(e.value), indices)
        if isinstance(e, tir.Call):
            return tir.Call(e.dtype, e.op, [self.visit_expr(a) for a in e.args])
        if isinstance(e, tir.Cast):
            return type(e)(e.dtype, self.visit_expr(e.value))
        if hasattr(e, "a") and hasattr(e, "b"):
            return type(e)(self.visit_expr(e.a), self.visit_expr(e.b))
        return e


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def run(func: tir.PrimFunc, scopes: BufferScopeMap, lane_count: int = 4) -> tir.PrimFunc:
    if lane_count <= 0:
        raise AllocateGroupMemoryError(f"lane_count must be positive; got {lane_count}")

    hbm_names = {n for n, sc in scopes.items() if sc == "hbm"}
    info = _analyze(func, lane_count, hbm_names, scopes)
    if not info:
        return func

    rw = _Rewriter(info, lane_count)
    new_body = rw.visit(func.body)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "AllocateGroupMemoryError"]
