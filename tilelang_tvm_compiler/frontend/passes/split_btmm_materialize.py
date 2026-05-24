"""Split a ``btmm_mm`` GEMM into compute (M_BTMM) + a deferred drain
(M_BMM_WO / "materialize").

Why this pass exists
--------------------

The ``linear_min`` kernel computes its MLEN×MLEN×MLEN GEMM via the BTMM
systolic path (KIND ``"btmm_mm"``). The hardware accumulates each k_block's
M_BTMM into ``hm_accum`` (``hm_accum +=``), so the K-loop accumulation is
done in hardware — no software K-add. The accumulator is drained to VRAM
exactly once, by a single M_BMM_WO, *after* the K-loop.

The kernel only writes the compute side::

    for k_block in T.serial(k_blocks):
        T.copy(A_hbm[...], A_sh)
        T.copy(B_hbm[...], B_sh)
        with T.attr(0, KIND, "btmm_mm"):
            T.gemm(A_sh, B_sh, C_loc, transpose_B=True)
    T.copy(C_loc, C_sh)          # first read of C_loc as a src

This pass finds the serial K-loop wrapping a ``btmm_mm`` gemm, extracts the
gemm's dst buffer (``C_loc``), and inserts a drain right after the loop —
before ``C_loc`` is first read as a src::

    for k_block ...:  (compute only)
    T.evaluate(T.call_extern("", "plena.bmm_wo_at", C_loc.data, tile_count))
    T.copy(C_loc, C_sh)

``fold`` turns the ``plena.bmm_wo_at`` extern into a mid_ir drain node;
``to_plena`` lowers it to an HLIR ``bmm_wo`` op (M_BMM_WO). The ``btmm_mm``
gemm itself lowers to an HLIR ``btmm_mm`` compute-only op (M_BTMM, no
writeback).

First cut: "drain right after the K-loop" — the loop body containing the
btmm_mm gemm is the natural last-use boundary. A fully general use-analysis
(drain right before the dst's first read anywhere) can extend this later.

No-op for kernels without a ``btmm_mm`` gemm.
"""

from __future__ import annotations

from typing import List, Optional

import tvm
from tvm import tir

from ..gemm_macros import KIND, BTMM_MM


_TILEOP_GEMM = "tl.tileop.gemm_py"
BMM_WO_INTRIN = "plena.bmm_wo_at"


class SplitBtmmMaterializeError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# detection helpers
# --------------------------------------------------------------------------
def _call_kind(call: tir.Call) -> Optional[str]:
    if not isinstance(call, tir.Call):
        return None
    op_name = getattr(call.op, "name", "")
    if op_name and not op_name.startswith("tir."):
        return op_name
    if op_name == "tir.call_extern" and call.args:
        head = call.args[0]
        if isinstance(head, tir.StringImm):
            return str(head.value)
    return None


def _gemm_call_in(stmt: tir.Stmt) -> Optional[tir.Call]:
    """If ``stmt`` is (or wraps) an Evaluate of a tl.tileop.gemm_py call,
    return that Call. Handles a bare Evaluate (the gemm under the KIND
    AttrStmt body)."""
    if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call):
        if _call_kind(stmt.value) == _TILEOP_GEMM:
            return stmt.value
    return None


def _region_buffer(region_arg) -> Optional[tir.Buffer]:
    """Extract the underlying tir.Buffer from a tl.tileop.region(...) call
    arg or a bare BufferLoad. region's first arg is a BufferLoad of the
    operand buffer."""
    node = region_arg
    if isinstance(node, tir.Call):
        # tl.tileop.region(BufferLoad(buf, [starts]), mode, *extents)
        if node.args and isinstance(node.args[0], tir.BufferLoad):
            return node.args[0].buffer
        return None
    if isinstance(node, tir.BufferLoad):
        return node.buffer
    return None


def _btmm_mm_gemm_dst(stmt: tir.Stmt) -> Optional[tir.Buffer]:
    """If ``stmt`` is an AttrStmt(KIND, "btmm_mm") wrapping a gemm, return
    the gemm's dst (C) buffer; else None."""
    if not (isinstance(stmt, tir.AttrStmt) and stmt.attr_key == KIND):
        return None
    val = stmt.value
    kind = val.value if isinstance(val, tir.StringImm) else str(val)
    if kind != BTMM_MM:
        return None
    call = _gemm_call_in(stmt.body)
    if call is None:
        return None
    # gemm ABI: region(A), region(B), region(C), [flags...]
    args = (list(call.args[1:]) if getattr(call.op, "name", "")
            == "tir.call_extern" else list(call.args))
    if len(args) < 3:
        return None
    return _region_buffer(args[2])


def _contains_btmm_mm(stmt: tir.Stmt) -> Optional[tir.Buffer]:
    """Search ``stmt`` (a SeqStmt, a For K-loop, or a bare AttrStmt) for a
    btmm_mm gemm; return its dst buffer if found. Descends into For bodies
    (the `for k_block` K-loop case) and SeqStmts; the leaf is the
    AttrStmt(KIND, btmm_mm) wrapping the gemm."""
    if isinstance(stmt, tir.SeqStmt):
        for c in stmt.seq:
            dst = _contains_btmm_mm(c)
            if dst is not None:
                return dst
        return None
    if isinstance(stmt, tir.For):
        return _contains_btmm_mm(stmt.body)
    return _btmm_mm_gemm_dst(stmt)


# --------------------------------------------------------------------------
# "reads buffer X" detection (loop-agnostic use analysis)
# --------------------------------------------------------------------------
def _expr_reads(expr, name: str) -> bool:
    if expr is None:
        return False
    if isinstance(expr, tir.BufferLoad):
        if expr.buffer.name == name:
            return True
        return any(_expr_reads(i, name) for i in expr.indices)
    if isinstance(expr, tir.Call):
        # tl.tileop.region(BufferLoad(buf,...), ...) / gemm / copy args.
        return any(_expr_reads(a, name) for a in expr.args)
    if isinstance(expr, tir.Cast):
        return _expr_reads(expr.value, name)
    if hasattr(expr, "a") and hasattr(expr, "b"):
        return _expr_reads(expr.a, name) or _expr_reads(expr.b, name)
    if hasattr(expr, "value"):
        return _expr_reads(expr.value, name)
    return False


def _stmt_reads(stmt, name: str) -> bool:
    """True if ``stmt`` (recursively) reads buffer ``name`` as a SOURCE.
    A btmm_mm gemm WRITING ``name`` (accumulate) is not a triggering read —
    callers exclude that case by registering the dst after walking."""
    if stmt is None:
        return False
    if isinstance(stmt, tir.SeqStmt):
        return any(_stmt_reads(c, name) for c in stmt.seq)
    if isinstance(stmt, tir.For):
        return _stmt_reads(stmt.body, name)
    if isinstance(stmt, (tir.BlockRealize,)):
        return _stmt_reads(stmt.block, name)
    if isinstance(stmt, tir.Block):
        return _stmt_reads(stmt.body, name) or _stmt_reads(stmt.init, name)
    if isinstance(stmt, tir.AttrStmt):
        return _stmt_reads(stmt.body, name)
    if isinstance(stmt, tir.IfThenElse):
        return (_expr_reads(stmt.condition, name)
                or _stmt_reads(stmt.then_case, name)
                or _stmt_reads(stmt.else_case, name))
    if isinstance(stmt, tir.LetStmt):
        return _expr_reads(stmt.value, name) or _stmt_reads(stmt.body, name)
    if isinstance(stmt, tir.Allocate):
        return _stmt_reads(stmt.body, name)
    if isinstance(stmt, tir.BufferStore):
        return _expr_reads(stmt.value, name) or any(
            _expr_reads(i, name) for i in stmt.indices)
    if isinstance(stmt, tir.Evaluate):
        return _expr_reads(stmt.value, name)
    return False


# --------------------------------------------------------------------------
# scratch buffer + drain + accumulate construction
# --------------------------------------------------------------------------
class _Ctx:
    """Holds the mlen/lane_count config and accumulates the scratch
    buffers this pass auto-allocates (injected into the kernel block)."""

    def __init__(self, mlen: int, lane_count: int) -> None:
        self.mlen = mlen
        self.lane_count = lane_count
        self.next_id = 0
        self.new_buffers: List[tir.Buffer] = []

    def fresh_scratch(self, dtype: str) -> tir.Buffer:
        """A (lane_count*mlen, mlen) shared buffer to receive the BMM_WO
        drain — one (mlen,mlen) tile per lane, stacked along rows."""
        name = f"__btmm_scratch_{self.next_id}"
        self.next_id += 1
        rows = self.lane_count * self.mlen
        data = tir.Var(
            name, tvm.ir.PointerType(tvm.ir.PrimType(dtype), "shared.dyn"))
        buf = tir.decl_buffer(
            shape=[rows, self.mlen], dtype=dtype, name=name, data=data,
            scope="shared.dyn",
        )
        self.new_buffers.append(buf)
        return buf


def _make_drain(scratch: tir.Buffer, dst: tir.Buffer, ctx: _Ctx) -> tir.Stmt:
    """``call_extern("", "plena.bmm_wo_at", scratch.data, dst.data,
    lane_count)``.

    Drain hm_accum's ``lane_count`` (mlen,mlen) tiles into ``scratch``
    (which is (lane_count*mlen, mlen)), then accumulate the tiles into the
    real ``dst`` (mlen,mlen). The accumulate is emitted as a small V_ADD
    loop directly in the backend (pre_isa_pass_v2._emit_bmm_wo) — no fold
    pattern-matching, so the per-tile row offset isn't a problem."""
    call = tir.call_extern(
        "", BMM_WO_INTRIN, scratch.data, dst.data,
        tir.IntImm("int32", ctx.lane_count),
    )
    return tir.Evaluate(call)


# --------------------------------------------------------------------------
# walk: after a K-loop with a btmm_mm gemm, drain to scratch (+ backend accum)
# --------------------------------------------------------------------------
def _walk(stmt, ctx: _Ctx):
    if stmt is None:
        return None
    if isinstance(stmt, tir.SeqStmt):
        out: List[tir.Stmt] = []
        # Use-driven drain: each btmm_mm dst is "pending" until something
        # READS it as a src; the drain (M_BMM_WO + accumulate) goes right
        # before that first read. Loop-agnostic — works whether the
        # btmm_mm is in a K-loop (multiple accumulate, one drain after) or
        # a single bare matmul.  pending: dst_name -> (scratch, dst_buf)
        pending: dict = {}   # dst_name -> dst_buf (scratch made at drain)
        for c in stmt.seq:
            # 1. Does this child read any pending dst? If so, drain first.
            for name in list(pending.keys()):
                if _stmt_reads(c, name):
                    dst_buf = pending.pop(name)
                    scratch = ctx.fresh_scratch(str(dst_buf.dtype))
                    out.append(_make_drain(scratch, dst_buf, ctx))
            # 2. Walk the child.
            new_c = _walk(c, ctx)
            out.append(new_c)
            # 3. Did this child WRITE a btmm_mm dst? Register it pending.
            #    (A K-loop child accumulates into the dst; the dst is only
            #    READ later — at THIS scope — by the consumer, so the drain
            #    fires there via step 1. No end-of-scope drain: the dst is
            #    always consumed at the scope where it was produced, and a
            #    premature end-of-scope drain inside a nested K-loop body
            #    would split the accumulation. If a btmm dst is genuinely
            #    never read, that's a dead matmul — leave it undrained.)
            dst = _contains_btmm_mm(c)
            if dst is not None and dst.name not in pending:
                pending[dst.name] = dst
        return tir.SeqStmt(out)
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _walk(stmt.body, ctx), stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_walk(stmt.block, ctx),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint,
            body=_walk(stmt.body, ctx),
            init=_walk(stmt.init, ctx) if stmt.init is not None else None,
            alloc_buffers=stmt.alloc_buffers,
            match_buffers=stmt.match_buffers,
            annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value, _walk(stmt.body, ctx),
        )
    if isinstance(stmt, tir.IfThenElse):
        return tir.IfThenElse(
            stmt.condition, _walk(stmt.then_case, ctx),
            _walk(stmt.else_case, ctx) if stmt.else_case is not None else None,
        )
    if isinstance(stmt, tir.LetStmt):
        return tir.LetStmt(stmt.var, stmt.value, _walk(stmt.body, ctx))
    if isinstance(stmt, tir.Allocate):
        return tir.Allocate(
            stmt.buffer_var, stmt.dtype, list(stmt.extents),
            stmt.condition, _walk(stmt.body, ctx), stmt.annotations,
        )
    return stmt


def _inject_alloc_buffers(stmt, new_buffers: List[tir.Buffer]):
    """Append ``new_buffers`` to the first tir.Block's alloc_buffers (the
    kernel root block). Mirrors fission_vector_chains / lower_compound."""
    if not new_buffers:
        return stmt
    if isinstance(stmt, tir.SeqStmt):
        out = []
        injected = False
        for c in stmt.seq:
            if injected:
                out.append(c)
            else:
                new_c = _inject_alloc_buffers(c, new_buffers)
                injected = new_c is not c
                out.append(new_c)
        return tir.SeqStmt(out)
    if isinstance(stmt, tir.BlockRealize):
        return tir.BlockRealize(
            iter_values=stmt.iter_values, predicate=stmt.predicate,
            block=_inject_alloc_buffers(stmt.block, new_buffers),
        )
    if isinstance(stmt, tir.Block):
        return tir.Block(
            iter_vars=stmt.iter_vars, reads=stmt.reads, writes=stmt.writes,
            name_hint=stmt.name_hint, body=stmt.body, init=stmt.init,
            alloc_buffers=list(stmt.alloc_buffers) + list(new_buffers),
            match_buffers=stmt.match_buffers, annotations=stmt.annotations,
        )
    if isinstance(stmt, tir.AttrStmt):
        return tir.AttrStmt(
            stmt.node, stmt.attr_key, stmt.value,
            _inject_alloc_buffers(stmt.body, new_buffers),
        )
    if isinstance(stmt, tir.For):
        return tir.For(
            stmt.loop_var, stmt.min, stmt.extent, stmt.kind,
            _inject_alloc_buffers(stmt.body, new_buffers),
            stmt.thread_binding, stmt.annotations,
        )
    if isinstance(stmt, tir.LetStmt):
        return tir.LetStmt(stmt.var, stmt.value,
                           _inject_alloc_buffers(stmt.body, new_buffers))
    return stmt


def run(func: tir.PrimFunc, *, mlen: int, lane_count: int) -> tir.PrimFunc:
    ctx = _Ctx(mlen, lane_count)
    new_body = _walk(func.body, ctx)
    new_body = _inject_alloc_buffers(new_body, ctx.new_buffers)
    return tir.PrimFunc(
        params=func.params,
        body=new_body,
        ret_type=func.ret_type,
        buffer_map=func.buffer_map,
        attrs=func.attrs,
    )


__all__ = ["run", "SplitBtmmMaterializeError", "BMM_WO_INTRIN"]
