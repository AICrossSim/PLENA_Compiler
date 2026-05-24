"""PreIsaPass v2 — the clean producer.

Each HLIR op handler emits a sequence of PreIsaOp + LoopRegion items
into a PreIsaModule. The producer does NOT think about registers,
materialisation, GP cache, addr-reg cache, or scope nesting. Its
single job is "what PLENA ISA instructions does this HLIR op
correspond to, with what symbolic operand expressions".

PrimExpr expansion (turning ``a + b * c`` into S_ADD_INT / S_MUL_INT
chains), simplification (folding hw consts), SSA naming, register
allocation, and ISA text emission all live in later passes
(:mod:`pre_isa_to_mir`, MIR optimise passes, :mod:`mir_to_isa`).

Currently migrated handlers (everything else raises):
  * fp_zero_at

The pre_isa_pass_v2 dispatch table only contains handlers we've
already migrated. Callers fall back to the legacy ``isa_pass`` for
non-migrated ops.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from tvm import tir

from . import hlir as _hlir
from . import pre_isa_ir_v2 as pi
from . import scope as _scope
from .program_shim import ProgramShim


# Helper to combine a base address + an offset PrimExpr into a single
# operand expression. Returns a PrimExpr; the PreIsa→MIR conversion
# pass simplifies/folds.
def _addr(base: int, offset) -> tir.PrimExpr:
    base_imm = tir.IntImm("int32", int(base))
    if isinstance(offset, int) and offset == 0:
        return base_imm
    if isinstance(offset, tir.IntImm) and int(offset.value) == 0:
        return base_imm
    return tir.Add(base_imm, offset)


def _try_const(expr) -> "int | None":
    """Return ``int(expr)`` if it simplifies to a compile-time constant,
    else None. Uses TVM's arithmetic so symbolic-but-cancelling diffs
    (e.g. ``(b + c*MLEN) - (b + (c-1)*MLEN)``) fold to the literal step."""
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tir.IntImm):
        return int(expr.value)
    import tvm
    simp = tvm.arith.Analyzer().simplify(expr)
    if isinstance(simp, tir.IntImm):
        return int(simp.value)
    return None


def _const_stride_pair(src_chunks, dst_chunks):
    """If both chunk-offset lists are constant arithmetic progressions,
    return ``(src_base, src_stride, dst_base, dst_stride)`` (base = chunk
    0's offset expr, stride = the constant int step). Else None — caller
    must fall back to static expansion. A 1-element list has no stride to
    infer, so it is treated as non-arithmetic (caller handles len==1
    separately anyway)."""
    def _stride_of(chunks):
        if len(chunks) < 2:
            return None
        step = _try_const(tir.Sub(_as_expr(chunks[1]), _as_expr(chunks[0])))
        if step is None:
            return None
        # Verify EVERY consecutive gap equals that same step.
        for a, b in zip(chunks, chunks[1:]):
            d = _try_const(tir.Sub(_as_expr(b), _as_expr(a)))
            if d != step:
                return None
        return step

    s_stride = _stride_of(src_chunks)
    d_stride = _stride_of(dst_chunks)
    if s_stride is None or d_stride is None:
        return None
    return src_chunks[0], s_stride, dst_chunks[0], d_stride


def _stride_off(base, stride: int, iv: tir.Var) -> tir.PrimExpr:
    """``base + iv*stride`` as a PrimExpr (the chunk offset for iteration
    ``iv``). ``base`` is chunk 0's offset expr; for iv=0 this is exactly
    ``base``, matching the static-expansion chunk-0 offset."""
    term = tir.Mul(iv, tir.IntImm("int32", int(stride)))
    return tir.Add(_as_expr(base), term)


def _as_expr(v) -> tir.PrimExpr:
    """Coerce ``v`` (int / IntImm / PrimExpr) to a PrimExpr."""
    if isinstance(v, int):
        return tir.IntImm("int32", v)
    if isinstance(v, tir.IntImm):
        return v
    if isinstance(v, tir.PrimExpr):
        return v
    raise TypeError(
        f"_as_expr: expected int/IntImm/PrimExpr; got "
        f"{type(v).__name__}: {v!r}"
    )


class PreIsaPassV2Error(RuntimeError):
    pass


class PreIsaPassV2:
    """Lower an HLIRModule to a clean PreIsaModule.

    Construction takes a ProgramShim — handlers read hardware-shape
    constants (mlen, blen, btmm_hlen) via the shim and may also
    delegate to legacy helpers (notably ``_resolve_fp_scalar_addr_arg``
    on the legacy IsaEmitterPass) for buffer-address resolution.
    """

    def __init__(self, shim: ProgramShim) -> None:
        from .isa_pass import IsaEmitterPass
        self.shim = shim
        self._legacy = IsaEmitterPass(shim)
        self.pre_isa = pi.PreIsaModule(name="<unset>")
        # Stack of "current append targets" — handlers write into the
        # top of this stack. The bottom is the module's top-level body;
        # entering a LoopRegion pushes its body list on top so nested
        # ops land inside the loop region naturally. Mirrors a scope
        # stack — loops ARE scopes.
        self._cursor_stack: List[List] = []
        self._dispatch: Dict[
            str, Callable[[_hlir.HLIRModule, _hlir.Op], None],
        ] = {
            "fp_zero_at": self._emit_fp_zero_at,
            "fp_copy_at": lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="copy"),
            "fp_exp_at":  lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="exp"),
            "fp_reci_at": lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="reci"),
            "fp_sqrt_at": lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="sqrt"),
            "fp_add_at":  lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="add"),
            "fp_sub_at":  lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="sub"),
            "fp_mul_at":  lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="mul"),
            "fp_max_at":  lambda m, o: self._emit_fp_scalar_op(m, o, kernel_op="max"),
            "v_zero":     self._emit_v_zero,
            "v_add":      lambda m, o: self._emit_v_binary(m, o, opcode="V_ADD_VV"),
            "v_sub":      lambda m, o: self._emit_v_binary(m, o, opcode="V_SUB_VV"),
            "v_mul":      lambda m, o: self._emit_v_binary(m, o, opcode="V_MUL_VV"),
            "v_exp":      lambda m, o: self._emit_v_unary(m, o, opcode="V_EXP_V"),
            "v_reci":     lambda m, o: self._emit_v_unary(m, o, opcode="V_RECI_V"),
            "v_sqrt":     lambda m, o: self._emit_v_unary(m, o, opcode="V_SQRT_V"),
            "copy_v_to_v": self._emit_copy_v_to_v,
            "v_fp_transfer_slice_v_to_fp": lambda m, o:
                self._emit_v_fp_transfer_slice(m, o, direction="v_to_fp"),
            "v_fp_transfer_slice_fp_to_v": lambda m, o:
                self._emit_v_fp_transfer_slice(m, o, direction="fp_to_v"),
            "for": self._emit_for,
            "row_reduce_max_at": lambda m, o: self._emit_row_scalar(
                m, o, row_op="reduce_max", reduce=True, masked=True,
            ),
            "row_reduce_sum_at": lambda m, o: self._emit_row_scalar(
                m, o, row_op="reduce_sum", reduce=True, masked=True,
            ),
            "row_exp": lambda m, o: self._emit_row_scalar(
                m, o, row_op="exp", masked=True,
            ),
            "row_reci": lambda m, o: self._emit_row_scalar(
                m, o, row_op="reci", masked=True,
            ),
            "row_sub_fp": lambda m, o: self._emit_row_scalar(
                m, o, row_op="sub", masked=True, has_fp=True,
            ),
            "row_mul_fp": lambda m, o: self._emit_row_scalar(
                m, o, row_op="mul", masked=True, has_fp=True,
            ),
            "row_add_fp": lambda m, o: self._emit_row_scalar(
                m, o, row_op="add", masked=True, has_fp=True,
            ),
            "btmm": self._emit_btmm,
            "btmm_mm": self._emit_btmm_mm,
            "bmm_wo": self._emit_bmm_wo,
            "btmv": self._emit_btmv,
            "mv": self._emit_mv,
            "mm": self._emit_mm,
            "mm_slot": self._emit_mm_slot,
            "matmul": self._emit_matmul,
            "dma_h2v": self._emit_dma_h2v,
            "dma_h2m": self._emit_dma_h2m,
            "dma_v2h": self._emit_dma_v2h,
            "dma_h2v_slice": self._emit_dma_h2v_slice,
            "dma_h2m_slice": self._emit_dma_h2m_slice,
            "dma_v2h_slice": self._emit_dma_v2h_slice,
        }

    # ---- "cursor" scope helpers — handlers append via _append ----
    def _append(self, item) -> None:
        """Add a PreIsaOp / LoopRegion to the current scope (top of
        ``_cursor_stack``, or the module's top-level body if stack is
        empty)."""
        if self._cursor_stack:
            self._cursor_stack[-1].append(item)
        else:
            self.pre_isa.append(item)

    def _comment(self, text: str) -> None:
        self._append(pi.PreIsaOp(opcode="_COMMENT", operands=[text]))

    def _push_scope(self, body_list: List) -> None:
        """Enter a nested scope (LoopRegion body). Subsequent appends
        land in ``body_list``."""
        self._cursor_stack.append(body_list)

    def _pop_scope(self) -> None:
        self._cursor_stack.pop()

    def run(self, mod: _hlir.HLIRModule) -> pi.PreIsaModule:
        _hlir.assert_addresses_resolved(mod)
        self.pre_isa = pi.PreIsaModule(
            name=mod.name, buffers=dict(mod.buffers),
        )
        self._cursor_stack = []
        for hlir_op in mod.ops:
            self._dispatch_op(mod, hlir_op)
        return self.pre_isa

    def _dispatch_op(
        self, mod: _hlir.HLIRModule, hlir_op: _hlir.Op,
    ) -> None:
        """Find + run the v2 handler for one HLIR op. Used by the
        top-level run loop AND recursively by the ``for`` handler
        for body sub-ops."""
        handler = self._dispatch.get(hlir_op.kind)
        if handler is None:
            raise PreIsaPassV2Error(
                f"PreIsaPassV2: no handler migrated for HLIR op kind "
                f"{hlir_op.kind!r}. Migrate it or dispatch to legacy "
                f"isa_pass for this op."
            )
        handler(mod, hlir_op)

    # ==================================================================
    # migrated handlers
    # ==================================================================
    def _emit_fp_zero_at(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        """HLIR ``fp_zero_at`` — store FP zero to one FPRAM slot.

        Legacy ISA:
            ; fp scalar task <intrinsic> op=zero
            S_ADDI_INT gp{r}, gp0, <dst.address + slot offset>
            S_ST_FP f0, gp{r}, 0

        PreIsaIR v2 form — no GPs, no materialisation:
            _COMMENT "fp scalar task ..."
            S_ST_FP   ["f0", <dst_addr_expr>, 0]
        The conversion pass turns dst_addr_expr into a SSA value chain
        producing the right physical GP at MIR-emit time.
        """
        if len(op.scalar_args) != 1:
            raise PreIsaPassV2Error(
                f"fp_zero_at expects 1 scalar address arg; got "
                f"{len(op.scalar_args)}"
            )
        # Reuse legacy resolver — returns a tir.PrimExpr already
        # carrying buf.address + buffer-element offset.
        dst_addr_expr = self._legacy._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "dst",
        )
        intrinsic = op.annotations.get("intrinsic", op.kind)
        self._comment(f"fp scalar task {intrinsic} op=zero")
        self._append(pi.PreIsaOp(
            opcode="S_ST_FP",
            operands=["f0", dst_addr_expr, 0],
        ))

    def _emit_fp_scalar_op(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, kernel_op: str,
    ) -> None:
        """``fp_<copy/exp/reci/sqrt>_at`` (2 scalar args: src, dst) or
        ``fp_<add/sub/mul/max>_at`` (3 scalar args: lhs, rhs, dst).

        Legacy ISA(unary, e.g. exp):
            S_ADDI_INT gp{src}, gp0, <src>
            S_ADDI_INT gp{dst}, gp0, <dst>
            ; fp scalar task ... op=exp
            S_LD_FP f1, gp{src}, 0
            S_EXP_FP f1, f1, 0
            S_ST_FP f1, gp{dst}, 0

        Legacy ISA(binary, e.g. mul):
            S_ADDI gp{lhs}, ...; S_ADDI gp{rhs}, ...; S_ADDI gp{dst}, ...
            ; fp scalar task ... op=mul
            S_LD_FP f1, gp{lhs}, 0
            S_LD_FP f2, gp{rhs}, 0
            S_MUL_FP f1, f1, f2
            S_ST_FP f1, gp{dst}, 0

        PreIsaIR v2 form — no GPs, no materialisation:
            _COMMENT "fp scalar task ..."
            S_LD_FP   ["f1", <src_or_lhs_expr>, 0]
            [S_LD_FP  ["f2", <rhs_expr>, 0]]      ; binary only
            <opcode>  [...fpreg arguments...]
            S_ST_FP   ["f1", <dst_expr>, 0]
        """
        if kernel_op in ("copy", "exp", "reci", "sqrt"):
            expected = 2
        else:
            expected = 3
        if len(op.scalar_args) != expected:
            raise PreIsaPassV2Error(
                f"{op.kind} expects {expected} scalar address args; "
                f"got {len(op.scalar_args)}"
            )
        addr_exprs = [
            self._legacy._resolve_fp_scalar_addr_arg(
                mod, a, op.kind, f"arg{i}",
            )
            for i, a in enumerate(op.scalar_args)
        ]
        intrinsic = op.annotations.get("intrinsic", op.kind)
        self._comment(
            f"fp scalar task {intrinsic} op={kernel_op}"
        )
        if kernel_op in ("copy", "exp", "reci", "sqrt"):
            src_expr, dst_expr = addr_exprs
            # Load src into f1.
            self._append(pi.PreIsaOp(
                opcode="S_LD_FP", operands=["f1", src_expr, 0],
            ))
            # Per-op compute (copy is the no-op variant).
            if kernel_op == "exp":
                self._append(pi.PreIsaOp(
                    opcode="S_EXP_FP", operands=["f1", "f1", 0],
                ))
            elif kernel_op == "reci":
                self._append(pi.PreIsaOp(
                    opcode="S_RECI_FP", operands=["f1", "f1"],
                ))
            elif kernel_op == "sqrt":
                self._append(pi.PreIsaOp(
                    opcode="S_SQRT_FP", operands=["f1", "f1"],
                ))
            # ``copy`` has no compute — f1 is already src; just store.
            # Store f1 into dst.
            self._append(pi.PreIsaOp(
                opcode="S_ST_FP", operands=["f1", dst_expr, 0],
            ))
        else:
            lhs_expr, rhs_expr, dst_expr = addr_exprs
            self._append(pi.PreIsaOp(
                opcode="S_LD_FP", operands=["f1", lhs_expr, 0],
            ))
            self._append(pi.PreIsaOp(
                opcode="S_LD_FP", operands=["f2", rhs_expr, 0],
            ))
            opcode_map = {
                "add": "S_ADD_FP",
                "sub": "S_SUB_FP",
                "mul": "S_MUL_FP",
                "max": "S_MAX_FP",
            }
            self._append(pi.PreIsaOp(
                opcode=opcode_map[kernel_op],
                operands=["f1", "f1", "f2"],
            ))
            self._append(pi.PreIsaOp(
                opcode="S_ST_FP", operands=["f1", dst_expr, 0],
            ))

    # ------------------------------------------------------------------
    # vector ops — VRAM region elementwise
    # ------------------------------------------------------------------
    def _emit_v_zero(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """``v_zero`` — dst[region] = 0, lowered to one
        ``V_MUL_VF dst, dst, f0, 0`` per MLEN-wide chunk.

        Iterates the legacy ``_vram_region_iter_chunks`` to walk the
        region; each chunk's VRAM offset becomes a PrimExpr operand.
        """
        if len(op.buffer_args) != 1 or not isinstance(
            op.buffer_args[0], _hlir.VramRegion,
        ):
            raise PreIsaPassV2Error(
                f"v_zero expects 1 VramRegion buffer_arg; got "
                f"{op.buffer_args!r}"
            )
        dst_region: _hlir.VramRegion = op.buffer_args[0]
        dst = mod.get_buffer(dst_region.parent)
        self._comment(
            f"v_zero dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}"
        )
        for d_off, _fp_step in self._legacy._vram_region_iter_chunks(
            dst, dst_region,
        ):
            dst_addr = _addr(int(dst.address), d_off)
            # V_MUL_VF dst, dst, f0, 0  (the conversion pass will
            # CSE the two ``dst_addr`` operands into a single SSA
            # value since they're the same Python object.)
            self._append(pi.PreIsaOp(
                opcode="V_MUL_VF",
                operands=[dst_addr, dst_addr, "f0", 0],
            ))

    def _emit_v_binary(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, opcode: str,
    ) -> None:
        """``v_add`` / ``v_sub`` / ``v_mul`` — elementwise VV.
        Per chunk:  <opcode> dst_addr, lhs_addr, rhs_addr, 0
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"{op.kind} expects 3 buffer_args (lhs, rhs, dst regions); "
                f"got {len(op.buffer_args)}"
            )
        lhs_region, rhs_region, dst_region = op.buffer_args
        for slot, name in enumerate(("lhs", "rhs", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassV2Error(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        lhs = mod.get_buffer(lhs_region.parent)
        rhs = mod.get_buffer(rhs_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self._comment(
            f"v binary {op.kind} {opcode} "
            f"dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}"
        )
        lhs_iter = self._legacy._vram_region_iter_chunks(lhs, lhs_region)
        rhs_iter = self._legacy._vram_region_iter_chunks(rhs, rhs_region)
        dst_iter = self._legacy._vram_region_iter_chunks(dst, dst_region)
        for (l_off, _), (r_off, _), (d_off, _) in zip(
            lhs_iter, rhs_iter, dst_iter,
        ):
            lhs_addr = _addr(int(lhs.address), l_off)
            rhs_addr = _addr(int(rhs.address), r_off)
            dst_addr = _addr(int(dst.address), d_off)
            self._append(pi.PreIsaOp(
                opcode=opcode,
                operands=[dst_addr, lhs_addr, rhs_addr, 0],
            ))

    def _emit_v_unary(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, opcode: str,
    ) -> None:
        """``v_exp`` / ``v_reci`` / ``v_sqrt`` — elementwise unary.
        Per chunk:  <opcode> dst_addr, src_addr, 0
        """
        if len(op.buffer_args) != 2:
            raise PreIsaPassV2Error(
                f"{op.kind} expects 2 buffer_args (src, dst regions); "
                f"got {len(op.buffer_args)}"
            )
        src_region, dst_region = op.buffer_args
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassV2Error(
                    f"{op.kind} {name}: expected VramRegion"
                )
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self._comment(
            f"v unary {op.kind} {opcode} "
            f"dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}"
        )
        src_iter = self._legacy._vram_region_iter_chunks(src, src_region)
        dst_iter = self._legacy._vram_region_iter_chunks(dst, dst_region)
        for (s_off, _), (d_off, _) in zip(src_iter, dst_iter):
            src_addr = _addr(int(src.address), s_off)
            dst_addr = _addr(int(dst.address), d_off)
            self._append(pi.PreIsaOp(
                opcode=opcode,
                operands=[dst_addr, src_addr, 0],
            ))

    # ------------------------------------------------------------------
    # VRAM-to-VRAM copy and VRAM ↔ FPRAM slice transfers
    # ------------------------------------------------------------------
    def _emit_copy_v_to_v(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        """``copy_v_to_v`` — dst[region] = src[region]. Each chunk
        emits ``V_ADD_VF dst, src, f0, 0`` (f0 == 0 so dst = src + 0)."""
        if len(op.buffer_args) != 2:
            raise PreIsaPassV2Error(
                f"copy_v_to_v expects 2 buffer_args (src, dst); "
                f"got {len(op.buffer_args)}"
            )
        src_region, dst_region = op.buffer_args
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassV2Error(
                    f"copy_v_to_v {name}: expected VramRegion"
                )
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self._comment(
            f"copy_v_to_v src.parent={src_region.parent} -> "
            f"dst.parent={dst_region.parent} "
            f"extents={list(dst_region.extents)!r}"
        )
        src_chunks = [s_off for (s_off, _) in
                      self._legacy._vram_region_iter_chunks(src, src_region)]
        dst_chunks = [d_off for (d_off, _) in
                      self._legacy._vram_region_iter_chunks(dst, dst_region)]
        if len(src_chunks) != len(dst_chunks):
            raise PreIsaPassV2Error(
                f"copy_v_to_v: src has {len(src_chunks)} chunks but dst "
                f"has {len(dst_chunks)}"
            )

        def _emit_one(src_off, dst_off):
            self._append(pi.PreIsaOp(
                opcode="V_ADD_VF",
                operands=[
                    _addr(int(dst.address), dst_off),
                    _addr(int(src.address), src_off),
                    "f0", 0,
                ],
            ))

        # A whole-buffer T.copy (e.g. C_loc -> C_sh, 1024 rows) has no
        # source for-loop wrapping it, so the naive lowering emits ONE
        # V_ADD_VF per chunk — at MLEN=1024 that is 1024 instructions
        # with 1024 literal addresses, which LICM then hoists into a
        # ~2000-line constant prologue. When the chunk offsets are a
        # constant arithmetic stride (the common case — row-major-flat
        # chunks step by exactly MLEN) we instead emit ONE serial loop
        # whose body computes addr = base + i*stride from the loop_var,
        # collapsing the 1024 instrs to a 1-instr C_LOOP body. Anything
        # not detectably arithmetic falls back to static expansion so we
        # never emit a wrong address.
        strides = _const_stride_pair(src_chunks, dst_chunks)
        if len(src_chunks) > 1 and strides is not None:
            src_base, src_stride, dst_base, dst_stride = strides
            iv = tir.Var(f"copy_row_{id(op) & 0xffff:x}", "int32")
            loop = pi.LoopRegion(
                loop_var=iv, init_imm=0, extent_imm=len(src_chunks),
                loop_kind="serial", body=[],
            )
            self._append(loop)
            self._push_scope(loop.body)
            try:
                _emit_one(
                    _stride_off(src_base, src_stride, iv),
                    _stride_off(dst_base, dst_stride, iv),
                )
            finally:
                self._pop_scope()
            return

        for s_off, d_off in zip(src_chunks, dst_chunks):
            _emit_one(s_off, d_off)

    def _emit_v_fp_transfer_slice(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, direction: str,
    ) -> None:
        """``v_fp_transfer_slice_<v_to_fp | fp_to_v>``. Each chunk
        emits one ``S_MAP_FP_V`` (vram→fpram) or ``S_MAP_V_FP``
        (fpram→vram). The FPRAM address steps by the cumulative
        ``fp_step_elems`` returned by ``_vram_region_iter_chunks``.
        """
        if len(op.buffer_args) != 1 or not isinstance(
            op.buffer_args[0], _hlir.VramRegion,
        ):
            raise PreIsaPassV2Error(
                f"{op.kind}: buffer_args[0] must be VramRegion"
            )
        if len(op.scalar_args) != 1:
            raise PreIsaPassV2Error(
                f"{op.kind} expects 1 scalar arg (fp address); got "
                f"{len(op.scalar_args)}"
            )
        region = op.buffer_args[0]
        vram = mod.get_buffer(region.parent)
        fp_base_expr = self._legacy._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "fp",
        )
        opcode = "S_MAP_FP_V" if direction == "v_to_fp" else "S_MAP_V_FP"
        self._comment(
            f"v↔fp transfer slice {op.kind} parent={region.parent} "
            f"starts={list(region.starts)!r} "
            f"extents={list(region.extents)!r}"
        )
        for vram_off, fp_step in self._legacy._vram_region_iter_chunks(
            vram, region,
        ):
            vram_addr = _addr(int(vram.address), vram_off)
            if fp_step == 0:
                fp_addr = fp_base_expr
            else:
                fp_addr = tir.Add(
                    fp_base_expr, tir.IntImm("int32", int(fp_step)),
                )
            if direction == "v_to_fp":
                # S_MAP_FP_V fp_dst, vram_src, 0
                self._append(pi.PreIsaOp(
                    opcode="S_MAP_FP_V",
                    operands=[fp_addr, vram_addr, 0],
                ))
            else:
                # S_MAP_V_FP vram_dst, fp_src, 0
                self._append(pi.PreIsaOp(
                    opcode="S_MAP_V_FP",
                    operands=[vram_addr, fp_addr, 0],
                ))

    # ------------------------------------------------------------------
    # HLIR for-loop — emits a single LoopRegion holding the body
    # sub-ops. THIS IS THE WHOLE HANDLER: no GP pinning, no
    # symbol_table push/pop, no idx_addr management. The MIR loop's
    # body IS a scope (MirBlock), and the pre_isa_to_mir conversion
    # binds the loop_var to a fresh ``_LOOP_VAR_DEF`` MirValue at
    # the top of that block — body ops referencing the loop_var via
    # PrimExpr operands resolve through the converter's symbol table
    # automatically.
    # ------------------------------------------------------------------
    def _emit_for(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        loop_var = op.annotations.get("loop_var")
        extent = op.annotations.get("extent")
        init = op.annotations.get("init", 0)
        loop_kind = op.annotations.get("loop_kind", "serial")
        if loop_var is None or extent is None:
            raise PreIsaPassV2Error(
                f"for-op missing loop_var or extent annotation: {op!r}"
            )
        if not isinstance(extent, (int, tir.IntImm)):
            raise PreIsaPassV2Error(
                f"for-op extent must be a compile-time integer; got "
                f"{extent!r}"
            )
        if not isinstance(init, (int, tir.IntImm)):
            raise PreIsaPassV2Error(
                f"for-op init must be a compile-time integer; got "
                f"{init!r}"
            )
        if loop_kind == "unrolled":
            loop_kind = "unroll"
        if loop_kind not in ("serial", "unroll"):
            raise PreIsaPassV2Error(
                f"for-op unknown loop_kind {loop_kind!r}"
            )
        ext_imm = int(extent.value) if isinstance(extent, tir.IntImm) else int(extent)
        init_imm = int(init.value) if isinstance(init, tir.IntImm) else int(init)

        # Build the empty LoopRegion, push its body as the current
        # scope, dispatch sub-ops, pop. The append-into-current-scope
        # discipline means every sub-op handler's _append lands in
        # this loop's body without any further coordination.
        loop = pi.LoopRegion(
            loop_var=loop_var,
            init_imm=init_imm,
            extent_imm=ext_imm,
            loop_kind=loop_kind,
            body=[],
        )
        # Forward the ``order_independent`` hint if the kernel
        # author marked the source for-op as such. Backend uses
        # this to drop the IntRAM idx slot + per-iter LD/ADDI/ST
        # by running the hw counter directly as the loop_var,
        # iterating N..1 instead of 0..N-1. Safe iff body is
        # genuinely independent of iteration order.
        if "order_independent" in op.annotations:
            loop.annotations["order_independent"] = bool(
                op.annotations["order_independent"]
            )
        self._append(loop)
        self._push_scope(loop.body)
        try:
            for sub_op in op.body or []:
                self._dispatch_op(mod, sub_op)
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # row_*_at — one logical VRAM row across n_d_tiles d-tiles.
    # ------------------------------------------------------------------
    def _emit_row_scalar(
        self,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        *,
        row_op: str,
        reduce: bool = False,
        masked: bool = False,
        has_fp: bool = False,
    ) -> None:
        """Migrated form of legacy ``_emit_row_scalar_op_at``.

        Three flavours:
          * reduce_max / reduce_sum  (buffer_args=[src], scalar=[fp_addr])
              ``V_RED_MAX/SUM f1, src, mask`` accumulated across d_tiles,
              with ``S_LD_FP f1, fp_dst, 0`` before and ``S_ST_FP f1,
              fp_dst, 0`` after.
          * exp / reci  (buffer_args=[src, dst], scalar=[])
              ``V_EXP_V / V_RECI_V dst, src, mask`` per d_tile.
          * add / sub / mul  (buffer_args=[src, dst], scalar=[fp_rhs])
              ``S_LD_FP f1, fp_rhs, 0`` once, then ``V_*_VF dst, src,
              f1, mask`` per d_tile.

        In PreIsaIR v2 form, d-tile iteration is a ``LoopRegion(unroll)``
        with the body's src/dst PrimExpr operands referencing the
        d_tile loop var. MIR conversion expands them into SSA chains;
        MIR→ISA unrolls.

        Masking: when the source buffer is packed-head, we emit
        ``C_SET_V_MASK_REG <mask_expr>`` before the loop and
        ``C_SET_V_MASK_REG 0`` after (mask reset).
        """
        has_fp = has_fp or reduce
        if reduce:
            if len(op.buffer_args) != 1:
                raise PreIsaPassV2Error(
                    f"{op.kind} expects 1 buffer_arg (src region)"
                )
            expected_scalar = 1
        elif has_fp:
            if len(op.buffer_args) != 2:
                raise PreIsaPassV2Error(
                    f"{op.kind} expects 2 buffer_args (src, dst regions)"
                )
            expected_scalar = 1
        else:
            if len(op.buffer_args) != 2:
                raise PreIsaPassV2Error(
                    f"{op.kind} expects 2 buffer_args (src, dst regions)"
                )
            expected_scalar = 0
        if len(op.scalar_args) != expected_scalar:
            raise PreIsaPassV2Error(
                f"{op.kind} expects {expected_scalar} scalar args"
            )
        for slot, name in enumerate(
            ("src",) if reduce else ("src", "dst"),
        ):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassV2Error(
                    f"{op.kind} {name}: expected VramRegion"
                )

        src_region = op.buffer_args[0]
        src = mod.get_buffer(src_region.parent)

        fp_addr_expr = None
        if has_fp:
            fp_addr_expr = self._legacy._resolve_fp_scalar_addr_arg(
                mod, op.scalar_args[0], op.kind, "fp",
            )

        src_base_off, src_mask_expr, src_info = (
            self._legacy._logical_to_phys_row_offset(src, src_region)
        )
        emit_v_mask = masked and src_mask_expr is not None
        mask_flag = 1 if emit_v_mask else 0
        n_d_tiles = int(src_info["d_tiles"])
        d_tile_stride_s = int(src_info["d_tile_stride"])

        # dst region only for non-reduce.
        dst = None
        dst_base_off = None
        d_tile_stride_d = 0
        if not reduce:
            dst_region = op.buffer_args[1]
            dst = mod.get_buffer(dst_region.parent)
            dst_base_off, _dm, dst_info = (
                self._legacy._logical_to_phys_row_offset(dst, dst_region)
            )
            d_tile_stride_d = int(dst_info["d_tile_stride"])

        intrinsic = op.annotations.get("intrinsic", op.kind)
        self._comment(
            f"row scalar task {intrinsic} op={row_op} "
            f"src.parent={src_region.parent} "
            f"starts={list(src_region.starts)!r}"
        )

        # Mask arm — legacy emits ``C_SET_V_MASK_REG gp{mask_expr}``
        # ONCE before the body loop, and ``C_SET_V_MASK_REG gp0`` to
        # reset afterwards.
        if emit_v_mask:
            self._append(pi.PreIsaOp(
                opcode="C_SET_V_MASK_REG",
                operands=[src_mask_expr],
            ))

        # F1 seed for reduce/binary-fp.
        if reduce:
            self._append(pi.PreIsaOp(
                opcode="S_LD_FP",
                operands=["f1", fp_addr_expr, 0],
            ))
        elif fp_addr_expr is not None:
            self._append(pi.PreIsaOp(
                opcode="S_LD_FP",
                operands=["f1", fp_addr_expr, 0],
            ))

        # d_tile sweep — wrap in an unroll LoopRegion when n_d_tiles
        # > 1; for n_d_tiles == 1, emit the body straight into the
        # current scope. Otherwise an extent=1 loop would emit a
        # vestigial ``loop_var * stride`` SSA chain that the current
        # arith.simplify doesn't fold (loop_var stays symbolic in
        # PreIsaIR → MIR conversion).
        if n_d_tiles > 1:
            t_var = tir.Var(f"d_tile_{id(op) & 0xffff:x}", "int32")
            loop = pi.LoopRegion(
                loop_var=t_var, init_imm=0, extent_imm=n_d_tiles,
                loop_kind="unroll", body=[],
            )
            self._append(loop)
            self._push_scope(loop.body)
            t_var_term_s = (
                tir.Mul(t_var, tir.IntImm("int32", d_tile_stride_s))
                if d_tile_stride_s != 0 else None
            )
            t_var_term_d = (
                tir.Mul(t_var, tir.IntImm("int32", d_tile_stride_d))
                if d_tile_stride_d != 0 else None
            )
        else:
            t_var_term_s = None
            t_var_term_d = None

        try:
            # Per d_tile addresses.
            src_off_expr = src_base_off
            if t_var_term_s is not None:
                src_off_expr = tir.Add(src_base_off, t_var_term_s)
            src_addr = _addr(int(src.address), src_off_expr)
            if dst is not None:
                dst_off_expr = dst_base_off
                if t_var_term_d is not None:
                    dst_off_expr = tir.Add(dst_base_off, t_var_term_d)
                dst_addr = _addr(int(dst.address), dst_off_expr)

            if reduce:
                opcode = {
                    "reduce_max": "V_RED_MAX",
                    "reduce_sum": "V_RED_SUM",
                }[row_op]
                # V_RED_* f1, gp_src, mask_flag — accumulates into f1.
                self._append(pi.PreIsaOp(
                    opcode=opcode,
                    operands=["f1", src_addr, mask_flag],
                ))
            elif fp_addr_expr is None:
                # exp / reci.
                opcode = {
                    "exp": "V_EXP_V",
                    "reci": "V_RECI_V",
                }[row_op]
                self._append(pi.PreIsaOp(
                    opcode=opcode,
                    operands=[dst_addr, src_addr, mask_flag],
                ))
            else:
                # add / sub / mul with FP scalar f1. PLENA quirk:
                # V_SUB_VF takes 5 operands (extra trailing 0 flag);
                # V_ADD_VF / V_MUL_VF take 4.
                opcode = {
                    "add": "V_ADD_VF",
                    "sub": "V_SUB_VF",
                    "mul": "V_MUL_VF",
                }[row_op]
                if row_op == "sub":
                    self._append(pi.PreIsaOp(
                        opcode=opcode,
                        operands=[dst_addr, src_addr, "f1", mask_flag, 0],
                    ))
                else:
                    self._append(pi.PreIsaOp(
                        opcode=opcode,
                        operands=[dst_addr, src_addr, "f1", mask_flag],
                    ))
        finally:
            if n_d_tiles > 1:
                self._pop_scope()

        # F1 flush (reduce only).
        if reduce:
            self._append(pi.PreIsaOp(
                opcode="S_ST_FP",
                operands=["f1", fp_addr_expr, 0],
            ))

        # Mask reset.
        if emit_v_mask:
            zero = tir.IntImm("int32", 0)
            self._append(pi.PreIsaOp(
                opcode="C_SET_V_MASK_REG",
                operands=[zero],
            ))

    # ------------------------------------------------------------------
    # btmm / btmv — lane-fused matrix × matrix / matrix × vector.
    # Each op emits one ``M_BTMM`` / ``M_BTMV`` and a paired
    # write-only ``M_BMM_WO`` / ``M_BMV_WO``. No tile loop.
    # ------------------------------------------------------------------
    def _emit_btmm_like(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
        *, op_mnemonic: str, wo_mnemonic: str, task_default: str,
    ) -> None:
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"{op.kind} expects 3 buffer_args (a/b/c regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(
                f"{op.kind} a: expected VramRegion"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassV2Error(
                f"{op.kind} b: expected MramRegion"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(
                f"{op.kind} c: expected VramRegion"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        task_id = op.annotations.get("intrinsic", task_default)
        # rhs / lhs addresses are just the buffer.address ints
        # (legacy uses ``rhs.address`` literally — region.starts are
        # always zero on this op path).
        rhs_addr = tir.IntImm("int32", int(rhs.address))
        lhs_addr = tir.IntImm("int32", int(lhs.address))
        dst_addr = tir.IntImm("int32", int(dst.address))
        # Matching tile_count for the write-back.
        tile_count = max(1, dst.num_elements // self.shim.tile_elems)
        # Header + main op.
        self._comment(
            f"{task_default} task {task_id} "
            f"lhs_packed=vram[{int(lhs.address)}] "
            f"rhs_mram={int(rhs.address)} "
            f"lanes={self.shim.btmm_lane_count} "
            f"head_width={self.shim.btmm_hlen}"
        )
        self._append(pi.PreIsaOp(
            opcode=op_mnemonic,
            operands=["gp0", rhs_addr, lhs_addr],
        ))
        # Write-back header + op.
        self._comment(
            f"{task_default} write-only task {task_id}.wo "
            f"out=vram[{int(dst.address)}] "
            f"tiles={tile_count} "
            f"lanes={self.shim.btmm_lane_count} "
            f"head_width={self.shim.btmm_hlen}"
        )
        self._append(pi.PreIsaOp(
            opcode=wo_mnemonic,
            operands=[dst_addr, 0],
        ))

    def _emit_btmm(self, mod, op):
        self._emit_btmm_like(
            mod, op,
            op_mnemonic="M_BTMM", wo_mnemonic="M_BMM_WO",
            task_default="btmm",
        )

    def _emit_btmv(self, mod, op):
        self._emit_btmm_like(
            mod, op,
            op_mnemonic="M_BTMV", wo_mnemonic="M_BMV_WO",
            task_default="btmv",
        )

    def _emit_btmm_mm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Non-fused MLEN×MLEN matmul: COMPUTE ONLY (one M_BTMM into the
        hm_accum). No write-back — a separate ``bmm_wo`` op drains the
        accumulator after the K-loop, so multiple k_blocks accumulate in
        hardware before a single drain."""
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"btmm_mm expects 3 buffer_args (a/b/c regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error("btmm_mm a: expected VramRegion")
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassV2Error("btmm_mm b: expected MramRegion")
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        task_id = op.annotations.get("intrinsic", "btmm_mm")
        rhs_addr = tir.IntImm("int32", int(rhs.address))
        lhs_addr = tir.IntImm("int32", int(lhs.address))
        self._comment(
            f"btmm_mm compute {task_id} "
            f"lhs_packed=vram[{int(lhs.address)}] rhs_mram={int(rhs.address)} "
            f"(accumulate into hm_accum; drained by bmm_wo)"
        )
        self._append(pi.PreIsaOp(
            opcode="M_BTMM",
            operands=["gp0", rhs_addr, lhs_addr],
        ))

    def _emit_bmm_wo(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Drain ("materialize") + accumulate. BTMM split K into
        ``lane_count`` partial products, so hm_accum holds lane_count
        (mlen,mlen) tiles whose SUM is the full result.

        1. ``M_BMM_WO scratch`` writes the lane_count tiles into the
           scratch buffer ((lane_count*mlen, mlen): tile ``lane`` occupies
           rows [lane*mlen:(lane+1)*mlen]).
        2. A V_ADD_VV loop sums the lane_count tiles into ``dst``
           (mlen,mlen), one mlen-wide chunk (row) at a time:
               dst[i]  = scratch[0*mlen + i]              (lane 0, copy)
               dst[i] += scratch[lane*mlen + i]           (lane > 0)
        """
        if len(op.buffer_args) != 2:
            raise PreIsaPassV2Error(
                f"bmm_wo expects 2 buffer_args (scratch, dst regions); "
                f"got {len(op.buffer_args)}"
            )
        scratch_reg, dst_reg = op.buffer_args
        if not isinstance(scratch_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error("bmm_wo scratch: expected VramRegion")
        if not isinstance(dst_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error("bmm_wo dst: expected VramRegion")
        scratch = mod.get_buffer(scratch_reg.parent)
        dst = mod.get_buffer(dst_reg.parent)
        lane_count = int(op.scalar_args[0]) if op.scalar_args else 1
        mlen = self.shim.mlen

        # 1. drain hm_accum -> scratch
        self._comment(
            f"bmm_wo drain out=vram[{int(scratch.address)}] "
            f"tiles={lane_count}"
        )
        self._append(pi.PreIsaOp(
            opcode="M_BMM_WO",
            operands=[tir.IntImm("int32", int(scratch.address)), 0],
        ))

        # 2. accumulate lane_count tiles of scratch into dst, one mlen-wide
        #    row per iteration of a hardware C_LOOP (loop var = row i).
        #    Row i address offset = i*mlen; scratch tile `lane` row i =
        #    (lane*mlen + i)*mlen = lane*mlen*mlen + i*mlen. Only the row
        #    axis is a runtime loop; the few lanes stay statically unrolled
        #    inside the body (each has a distinct constant tile base).
        self._comment(
            f"bmm_wo accumulate {lane_count} tiles "
            f"scratch[{int(scratch.address)}] -> dst[{int(dst.address)}] "
            f"(C_LOOP over {mlen} rows)"
        )
        i_var = tir.Var(f"bmm_wo_row_{id(op) & 0xffff:x}", "int32")
        row_off = tir.Mul(i_var, tir.IntImm("int32", mlen))  # i*mlen
        loop = pi.LoopRegion(
            loop_var=i_var, init_imm=0, extent_imm=mlen,
            loop_kind="serial", body=[],
        )
        loop.annotations["order_independent"] = True
        self._append(loop)
        self._push_scope(loop.body)
        try:
            dst_addr = _addr(int(dst.address), row_off)
            # lane 0: dst[i] = scratch[0*tile + i*mlen] + 0  (copy)
            lane0_addr = _addr(int(scratch.address), row_off)
            self._append(pi.PreIsaOp(
                opcode="V_ADD_VF",
                operands=[dst_addr, lane0_addr, "f0", 0],
            ))
            # lanes 1..: dst[i] += scratch[lane*mlen*mlen + i*mlen]
            for lane in range(1, lane_count):
                tile_base = lane * mlen * mlen
                src_off = tir.Add(tir.IntImm("int32", tile_base), row_off)
                src_addr = _addr(int(scratch.address), src_off)
                self._append(pi.PreIsaOp(
                    opcode="V_ADD_VV",
                    operands=[dst_addr, dst_addr, src_addr, 0],
                ))
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # mm — single-tile (mlen*mlen) matmul. Walks (oc, orow) grid;
    # each iter emits one M_MM + M_MM_WO pair. K stays at 1 here
    # (single-tile). Narrow path (rhs_cols < mlen) is a TODO.
    # ------------------------------------------------------------------
    def _emit_mm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"plena.mm expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        lhs = mod.get_buffer(op.buffer_args[0])
        rhs = mod.get_buffer(op.buffer_args[1])
        dst = mod.get_buffer(op.buffer_args[2])
        mlen = int(self.shim.mlen)
        blen = int(self.shim.blen)
        lhs_rows, lhs_cols = self._legacy._logical_2d(lhs.shape)
        rhs_rows, rhs_cols = self._legacy._logical_2d(rhs.shape)
        dst_rows, dst_cols = self._legacy._logical_2d(dst.shape)
        if lhs_rows != mlen or lhs_cols != mlen:
            raise PreIsaPassV2Error(
                f"plena.mm lhs must be mlen*mlen; got ({lhs_rows},{lhs_cols})"
            )
        if rhs_rows != mlen or dst_rows != mlen:
            raise PreIsaPassV2Error(
                f"plena.mm rhs/dst must have mlen rows"
            )
        if not (rhs_cols == mlen and dst_cols == mlen):
            raise PreIsaPassV2Error(
                f"plena.mm narrow-tile path not yet migrated to v2"
            )

        tiles_per_mlen = mlen // blen
        output_row_stride = blen * mlen
        task_id = op.annotations.get("intrinsic", "mm")
        self._comment(
            f"matmul (single-tile, symbolic unroll) task {task_id} "
            f"lhs=vram[{int(lhs.address)}] rhs=mram[{int(rhs.address)}] "
            f"dst=vram[{int(dst.address)}]"
        )

        oc_var = tir.Var(f"mm_oc_{id(op) & 0xffff:x}", "int32")
        orow_var = tir.Var(f"mm_orow_{id(op) & 0xffff:x}", "int32")

        # Outer oc loop.
        oc_loop = pi.LoopRegion(
            loop_var=oc_var, init_imm=0, extent_imm=tiles_per_mlen,
            loop_kind="unroll", body=[],
        )
        self._append(oc_loop)
        self._push_scope(oc_loop.body)
        try:
            orow_loop = pi.LoopRegion(
                loop_var=orow_var, init_imm=0, extent_imm=tiles_per_mlen,
                loop_kind="unroll", body=[],
            )
            self._append(orow_loop)
            self._push_scope(orow_loop.body)
            try:
                # Address PrimExprs.
                mat_col = tir.Add(
                    tir.IntImm("int32", int(rhs.address)),
                    tir.Mul(oc_var, tir.IntImm("int32", blen)),
                )
                act_row = tir.Add(
                    tir.IntImm("int32", int(lhs.address)),
                    tir.Mul(orow_var,
                            tir.IntImm("int32", output_row_stride)),
                )
                result_addr = tir.Add(
                    tir.Add(
                        tir.IntImm("int32", int(dst.address)),
                        tir.Mul(oc_var, tir.IntImm("int32", blen)),
                    ),
                    tir.Mul(orow_var,
                            tir.IntImm("int32", output_row_stride)),
                )
                # M_MM 0, gp{mat_col}, gp{act_row}
                self._append(pi.PreIsaOp(
                    opcode="M_MM",
                    operands=[0, mat_col, act_row],
                ))
                # M_MM_WO gp{result_addr}, gp0, 0
                self._append(pi.PreIsaOp(
                    opcode="M_MM_WO",
                    operands=[result_addr, "gp0", 0],
                ))
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # mm_slot — apply M_MM/M_MM_WO over a col-slot of rhs/dst, with
    # optional dynamic LHS row, RHS col, and DST col offsets carried as
    # PrimExprs through scalar_args. Mirrors legacy
    # ``ISAEmitter.emit_slot_matmul``: nested (oc, t) loops, oc walking
    # by ``blen`` columns, t walking by ``blen*mlen`` rows.
    # ------------------------------------------------------------------
    def _emit_mm_slot(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"plena.mm_slot expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        if len(op.scalar_args) != 4:
            raise PreIsaPassV2Error(
                f"plena.mm_slot expects 4 scalar args "
                f"(lhs_row_off, rhs_col_off, dst_col_off, col_count)"
            )
        lhs = mod.get_buffer(op.buffer_args[0])
        rhs = mod.get_buffer(op.buffer_args[1])
        dst = mod.get_buffer(op.buffer_args[2])
        lhs_row_raw, rhs_col_raw, dst_col_raw, col_count_raw = op.scalar_args
        try:
            col_count = int(col_count_raw)
        except TypeError as exc:
            raise PreIsaPassV2Error(
                f"plena.mm_slot col_count must be compile-time int; "
                f"got {col_count_raw!r}"
            ) from exc

        mlen = int(self.shim.mlen)
        blen = int(self.shim.blen)
        if col_count <= 0 or col_count % blen != 0:
            raise PreIsaPassV2Error(
                f"plena.mm_slot col_count must be positive multiple of "
                f"blen={blen}; got {col_count}"
            )
        tiles_per_slot = col_count // blen
        tiles_per_mlen = mlen // blen
        row_stride = blen * mlen

        # Coerce raw scalars to PrimExpr-or-int. Both kinds end up
        # combined into a tir.Add chain; arith.simplify in
        # pre_isa_to_mir folds the static parts.
        def _expr_or_int(x) -> tir.PrimExpr:
            if isinstance(x, int):
                return tir.IntImm("int32", x)
            if isinstance(x, tir.IntImm):
                return x
            if isinstance(x, tir.PrimExpr):
                return x
            raise PreIsaPassV2Error(
                f"plena.mm_slot scalar arg must be int or PrimExpr; "
                f"got {type(x).__name__}: {x!r}"
            )

        lhs_off_e = _expr_or_int(lhs_row_raw)
        rhs_off_e = _expr_or_int(rhs_col_raw)
        dst_off_e = _expr_or_int(dst_col_raw)

        task_id = op.annotations.get("intrinsic", "mm_slot")
        self._comment(
            f"slot matmul task {task_id} "
            f"lhs=vram[{int(lhs.address)}] rhs=mram[{int(rhs.address)}] "
            f"dst=vram[{int(dst.address)}] col_count={col_count}"
        )

        oc_var = tir.Var(f"slot_oc_{id(op) & 0xffff:x}", "int32")
        t_var = tir.Var(f"slot_t_{id(op) & 0xffff:x}", "int32")

        oc_loop = pi.LoopRegion(
            loop_var=oc_var, init_imm=0, extent_imm=tiles_per_slot,
            loop_kind="unroll", body=[],
        )
        self._append(oc_loop)
        self._push_scope(oc_loop.body)
        try:
            t_loop = pi.LoopRegion(
                loop_var=t_var, init_imm=0, extent_imm=tiles_per_mlen,
                loop_kind="unroll", body=[],
            )
            self._append(t_loop)
            self._push_scope(t_loop.body)
            try:
                # act = lhs.base + lhs_row_off + t * row_stride
                act_addr = tir.Add(
                    tir.Add(tir.IntImm("int32", int(lhs.address)),
                            lhs_off_e),
                    tir.Mul(t_var, tir.IntImm("int32", row_stride)),
                )
                # mat = rhs.base + rhs_col_off + oc * blen
                mat_addr = tir.Add(
                    tir.Add(tir.IntImm("int32", int(rhs.address)),
                            rhs_off_e),
                    tir.Mul(oc_var, tir.IntImm("int32", blen)),
                )
                # out = dst.base + dst_col_off + oc * blen + t * row_stride
                out_addr = tir.Add(
                    tir.Add(
                        tir.Add(tir.IntImm("int32", int(dst.address)),
                                dst_off_e),
                        tir.Mul(oc_var, tir.IntImm("int32", blen)),
                    ),
                    tir.Mul(t_var, tir.IntImm("int32", row_stride)),
                )
                self._append(pi.PreIsaOp(
                    opcode="M_MM",
                    operands=[0, mat_addr, act_addr],
                ))
                self._append(pi.PreIsaOp(
                    opcode="M_MM_WO",
                    operands=[out_addr, "gp0", 0],
                ))
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # matmul — unified (M, K) @ (K, N) -> (M, N).
    # Region-aware: each operand is a Vram/MramRegion with dim_roles
    # ("M"/"K"/"N"/"_") scalar arg. transpose_b inferred from B-region
    # axis order. K is folded into the systolic-array accumulator:
    # K_tiles M_MM/M_TMM issuances feed one M_MM_WO. 5-level unroll
    # over (m, n_mlen, oc, orow, k).
    # ------------------------------------------------------------------
    def _emit_matmul(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"plena.matmul expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(
                f"plena.matmul a: expected VramRegion, got {type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassV2Error(
                f"plena.matmul b: expected MramRegion, got {type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(
                f"plena.matmul c: expected VramRegion, got {type(c_reg).__name__}"
            )
        if len(op.scalar_args) != 3:
            raise PreIsaPassV2Error(
                f"plena.matmul expects 3 scalar_args (a/b/c dim_roles)"
            )
        a_roles, b_roles, c_roles = op.scalar_args
        if len(a_roles) != 4 or len(b_roles) != 4 or len(c_roles) != 4:
            raise PreIsaPassV2Error(
                f"plena.matmul dim_roles must be 4-tuples"
            )

        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)

        def _find_role(roles, role, operand):
            hits = [i for i, r in enumerate(roles) if r == role]
            if not hits:
                raise PreIsaPassV2Error(
                    f"plena.matmul {operand}: missing role {role!r}"
                )
            if len(hits) > 1:
                raise PreIsaPassV2Error(
                    f"plena.matmul {operand}: role {role!r} at multiple axes"
                )
            return hits[0]

        c_M_axis = _find_role(c_roles, "M", "c")
        a_M_axis = _find_role(a_roles, "M", "a")
        a_K_axis = _find_role(a_roles, "K", "a")
        b_K_axis = _find_role(b_roles, "K", "b")
        b_N_axis = _find_role(b_roles, "N", "b")

        mlen = int(self.shim.mlen)
        blen = int(self.shim.blen)
        hlen = int(self.shim.btmm_hlen)

        M = int(a_reg.extents[a_M_axis])
        K = int(a_reg.extents[a_K_axis])
        N = int(b_reg.extents[b_N_axis])
        if M % mlen != 0 or K % mlen != 0:
            raise PreIsaPassV2Error(
                f"plena.matmul: M ({M}) and K ({K}) must be multiples of MLEN ({mlen})"
            )
        if N % hlen != 0:
            raise PreIsaPassV2Error(
                f"plena.matmul: N ({N}) must be multiple of hlen ({hlen})"
            )
        M_tiles = M // mlen
        K_tiles = K // mlen
        N_mlen_tiles = (N + mlen - 1) // mlen
        transpose_b = b_N_axis < b_K_axis

        # dst_row_stride — packed-head (cluster_dim==2, lane_count>1, M
        # on S axis) uses physical s_inner_stride; otherwise extents
        # product after M.
        dst_cluster_dim = getattr(dst, "cluster_dim", None)
        tl_info = self._legacy._tile_layout_strides(dst)
        packed_head_dst = (
            tl_info is not None
            and dst_cluster_dim == 2
            and int(tl_info["lane_count"]) > 1
            and c_M_axis == 1
        )
        if packed_head_dst:
            dst_row_stride = int(tl_info["s_inner_stride"])
        else:
            dst_row_stride = 1
            for ax in range(c_M_axis + 1, len(c_reg.extents)):
                dst_row_stride *= int(c_reg.extents[ax])
        if dst_row_stride <= 0:
            dst_row_stride = 1

        # Per-side raw origin offsets — int or PrimExpr. arith.simplify
        # in pre_isa_to_mir folds the static parts.
        lhs_raw_off = self._legacy._region_origin_offset(lhs, a_reg)
        rhs_raw_off = self._legacy._region_origin_offset(rhs, b_reg)
        dst_raw_off = self._legacy._region_origin_offset(dst, c_reg)

        def _as_expr(x) -> tir.PrimExpr:
            if isinstance(x, int):
                return tir.IntImm("int32", x)
            return x

        lhs_off_e = _as_expr(lhs_raw_off)
        rhs_off_e = _as_expr(rhs_raw_off)
        dst_off_e = _as_expr(dst_raw_off)

        # Strides matching emit_matmul_general defaults.
        lhs_k_tile_stride = mlen * mlen
        lhs_m_tile_stride = K_tiles * mlen * mlen
        if transpose_b:
            rhs_n_mlen_tile_stride = K_tiles * mlen * mlen
            rhs_k_tile_stride = mlen * mlen
            oc_b_step = blen * mlen
            mm_opcode = "M_TMM"
        else:
            rhs_n_mlen_tile_stride = mlen * mlen
            rhs_k_tile_stride = N_mlen_tiles * mlen * mlen
            oc_b_step = blen
            mm_opcode = "M_MM"
        dst_m_tile_stride = mlen * int(dst_row_stride)

        tiles_per_mlen = mlen // blen
        a_orow_step = blen * mlen
        c_orow_step = blen * mlen

        task_id = op.annotations.get("intrinsic", "matmul")
        self._comment(
            f"matmul (general) task {task_id} M={M_tiles*mlen} K={K_tiles*mlen} N={N} "
            f"(M_tiles={M_tiles} K_tiles={K_tiles} N_mlen_tiles={N_mlen_tiles}"
            f"{', transpose_b' if transpose_b else ''})"
        )

        # Loop vars for the 5-level nest.
        op_id = f"{id(op) & 0xffff:x}"
        m_var = tir.Var(f"mm_m_{op_id}", "int32")
        nm_var = tir.Var(f"mm_nm_{op_id}", "int32")
        oc_var = tir.Var(f"mm_oc_{op_id}", "int32")
        orow_var = tir.Var(f"mm_orow_{op_id}", "int32")
        k_var = tir.Var(f"mm_k_{op_id}", "int32")

        # Build nested LoopRegions. We enter scopes from outer to inner;
        # innermost emits two PreIsaOps: a K-loop with M_MM/M_TMM in its
        # body, then one M_MM_WO after the K loop in the orow scope.
        m_loop = pi.LoopRegion(
            loop_var=m_var, init_imm=0, extent_imm=M_tiles,
            loop_kind="unroll", body=[],
        )
        self._append(m_loop)
        self._push_scope(m_loop.body)
        try:
            for n_mlen_static in range(N_mlen_tiles):
                # N_mlen_tiles may have a partial trailing block (cols < mlen);
                # tiles_per_n_mlen varies per iter, so we materialise the
                # n_mlen loop as a Python-side static unroll over n_mlen.
                # The remaining 3 inner levels stay as LoopRegions.
                cols_here = min(mlen, N - n_mlen_static * mlen)
                tiles_per_n_mlen = cols_here // blen
                if tiles_per_n_mlen <= 0:
                    continue
                oc_loop = pi.LoopRegion(
                    loop_var=oc_var, init_imm=0,
                    extent_imm=tiles_per_n_mlen,
                    loop_kind="unroll", body=[],
                )
                self._append(oc_loop)
                self._push_scope(oc_loop.body)
                try:
                    orow_loop = pi.LoopRegion(
                        loop_var=orow_var, init_imm=0,
                        extent_imm=tiles_per_mlen,
                        loop_kind="unroll", body=[],
                    )
                    self._append(orow_loop)
                    self._push_scope(orow_loop.body)
                    try:
                        # Inner K loop — emits one M_MM/M_TMM per iter,
                        # accumulating into the systolic-array
                        # accumulator. M_MM_WO sits AFTER the K loop,
                        # at the orow-scope level.
                        k_loop = pi.LoopRegion(
                            loop_var=k_var, init_imm=0, extent_imm=K_tiles,
                            loop_kind="unroll", body=[],
                        )
                        self._append(k_loop)
                        self._push_scope(k_loop.body)
                        try:
                            # act = lhs.base + lhs_off + m*lhs_m_tile_stride
                            #     + orow*a_orow_step + k*lhs_k_tile_stride
                            act_addr = tir.Add(
                                tir.Add(
                                    tir.Add(
                                        tir.Add(
                                            tir.IntImm("int32", int(lhs.address)),
                                            lhs_off_e,
                                        ),
                                        tir.Mul(m_var, tir.IntImm("int32", lhs_m_tile_stride)),
                                    ),
                                    tir.Mul(orow_var, tir.IntImm("int32", a_orow_step)),
                                ),
                                tir.Mul(k_var, tir.IntImm("int32", lhs_k_tile_stride)),
                            )
                            # mat = rhs.base + rhs_off + n_mlen_static*rhs_n_mlen_tile_stride
                            #     + oc*oc_b_step + k*rhs_k_tile_stride
                            mat_addr = tir.Add(
                                tir.Add(
                                    tir.Add(
                                        tir.Add(
                                            tir.IntImm("int32", int(rhs.address)),
                                            rhs_off_e,
                                        ),
                                        tir.IntImm("int32",
                                                   n_mlen_static * rhs_n_mlen_tile_stride),
                                    ),
                                    tir.Mul(oc_var, tir.IntImm("int32", oc_b_step)),
                                ),
                                tir.Mul(k_var, tir.IntImm("int32", rhs_k_tile_stride)),
                            )
                            if transpose_b:
                                # M_TMM 0, act, mat
                                self._append(pi.PreIsaOp(
                                    opcode="M_TMM",
                                    operands=[0, act_addr, mat_addr],
                                ))
                            else:
                                # M_MM 0, mat, act
                                self._append(pi.PreIsaOp(
                                    opcode="M_MM",
                                    operands=[0, mat_addr, act_addr],
                                ))
                        finally:
                            self._pop_scope()
                        # M_MM_WO: at orow scope. dst_col within the
                        # output tile = n_mlen*mlen + oc*blen.
                        dst_col_static = n_mlen_static * mlen
                        out_addr = tir.Add(
                            tir.Add(
                                tir.Add(
                                    tir.Add(
                                        tir.IntImm("int32", int(dst.address)),
                                        dst_off_e,
                                    ),
                                    tir.Mul(m_var, tir.IntImm("int32", dst_m_tile_stride)),
                                ),
                                tir.Mul(orow_var, tir.IntImm("int32", c_orow_step)),
                            ),
                            tir.Add(
                                tir.IntImm("int32", dst_col_static),
                                tir.Mul(oc_var, tir.IntImm("int32", blen)),
                            ),
                        )
                        self._append(pi.PreIsaOp(
                            opcode="M_MM_WO",
                            operands=[out_addr, "gp0", 0],
                        ))
                    finally:
                        self._pop_scope()
                finally:
                    self._pop_scope()
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # DMA — H↔V / H↔M / V↔H tile-wise transfers. Each HBM buffer carries
    # a (row_blocks × col_blocks) tile grid annotation; we walk it via
    # legacy's ``_iter_tile_offsets`` and emit one preload/store body
    # per tile. Per-tile body is the canonical
    # ``C_SET_ADDR_REG`` + ``C_SET_SCALE_REG`` + ``C_SET_STRIDE_REG`` +
    # (vlen/preload-stripe of) ``H_PREFETCH_V`` / ``H_PREFETCH_M`` /
    # ``H_STORE_V`` sequence.
    # ------------------------------------------------------------------
    def _emit_dma_h2v(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        if src.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_h2v src must be HBM; got {src.scope}"
            )
        if dst.scope != _scope.VRAM:
            raise PreIsaPassV2Error(
                f"dma_h2v dst must be VRAM; got {dst.scope}"
            )
        for vram_off, hbm_off in self._legacy._iter_tile_offsets(src):
            self._comment(
                f"dma_h2v tile  {src.name}[hbm+{hbm_off}] -> "
                f"{dst.name}[vram+{vram_off}]"
            )
            self._emit_h2v_tile_body(
                hbm_addr=int(src.address),
                vram_addr=int(dst.address) + int(vram_off),
                hbm_stride=src.hbm_stride,
                hbm_scale_size=src.hbm_scale_size,
                hbm_start_offset=int(src.hbm_offset) + int(hbm_off),
            )

    def _emit_dma_h2m(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        if src.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_h2m src must be HBM; got {src.scope}"
            )
        if dst.scope != _scope.MRAM:
            raise PreIsaPassV2Error(
                f"dma_h2m dst must be MRAM; got {dst.scope}"
            )
        for mram_off, hbm_off in self._legacy._iter_tile_offsets(src):
            self._comment(
                f"dma_h2m tile  {src.name}[hbm+{hbm_off}] -> "
                f"{dst.name}[mram+{mram_off}]"
            )
            self._emit_h2m_tile_body(
                hbm_addr=int(src.address),
                mram_addr=int(dst.address) + int(mram_off),
                hbm_offset=int(src.hbm_offset) + int(hbm_off),
                hbm_scale=src.hbm_scale_size,
                hbm_stride=src.hbm_stride,
            )

    def _emit_dma_v2h(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        if src.scope != _scope.VRAM:
            raise PreIsaPassV2Error(
                f"dma_v2h src must be VRAM; got {src.scope}"
            )
        if dst.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_v2h dst must be HBM; got {dst.scope}"
            )
        if src.num_elements != dst.num_elements:
            raise PreIsaPassV2Error(
                f"dma_v2h: src ({src.name}, {src.num_elements} elems) "
                f"and dst ({dst.name}, {dst.num_elements} elems) must "
                f"have the same total size"
            )
        # Walk the HBM (dst) grid; vram_off = idx * tile_elems lines
        # up with BMM_WO's BHSD VRAM packing — same convention as
        # legacy ``_emit_dma_v2h``.
        for vram_off, hbm_off in self._legacy._iter_tile_offsets(dst):
            self._comment(
                f"dma_v2h tile  {src.name}[vram+{vram_off}] -> "
                f"{dst.name}[hbm+{hbm_off}]"
            )
            self._emit_v2h_tile_body(
                vram_addr=int(src.address) + int(vram_off),
                hbm_addr=int(dst.address),
                hbm_stride=dst.hbm_stride,
                hbm_scale_size=dst.hbm_scale_size,
                hbm_start_offset=int(dst.hbm_offset) + int(hbm_off),
            )

    # ------------------------------------------------------------------
    # DMA slice variants — same as dma_h2v / _h2m / _v2h but with a
    # BufferSlice describing a sub-region of the HBM buffer. Walks a
    # 4-level (d_tile, s_tile, h_grp, b) tile grid as nested
    # LoopRegions; per-tile body is the same preload/store helper as
    # the whole-buffer DMA path, with hbm_off / vram_off addresses
    # expressed as PrimExprs referencing the four loop_vars. Slice
    # starts that are themselves PrimExprs (e.g. derived from an
    # outer kernel-loop var) flow through arith.simplify and only
    # crystallise into S_ADDI_INT/S_MUL_INT chains at MIR lowering.
    # ------------------------------------------------------------------
    def _slice_offset_expr(
        self, parent: _hlir.Buffer, sl: _hlir.BufferSlice,
    ) -> tir.PrimExpr:
        """Build a unified PrimExpr for ``slice.starts``'s element
        offset in ``parent``. Mixes static and dynamic starts;
        arith.simplify folds the static sub-trees."""
        offset = tir.IntImm("int32", 0)
        shape = parent.shape
        for i, s in enumerate(sl.starts):
            stride_below = 1
            for d in shape[i + 1:]:
                stride_below *= int(d)
            if isinstance(s, int):
                term = tir.IntImm("int32", s * stride_below)
            elif isinstance(s, tir.IntImm):
                term = tir.IntImm("int32", int(s.value) * stride_below)
            else:
                term = tir.Mul(s, tir.IntImm("int32", stride_below))
            offset = tir.Add(offset, term)
        if int(parent.hbm_offset):
            offset = tir.Add(
                offset, tir.IntImm("int32", int(parent.hbm_offset))
            )
        return offset

    def _emit_dma_h2v_slice(self, mod, op) -> None:
        sl = op.buffer_args[0]
        arg1 = op.buffer_args[1]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassV2Error(
                f"dma_h2v_slice: buffer_args[0] must be BufferSlice"
            )
        if isinstance(arg1, _hlir.BufferSlice):
            raise PreIsaPassV2Error(
                f"dma_h2v_slice: dst must be a whole-buffer name"
            )
        dst = mod.get_buffer(arg1)
        parent = mod.get_buffer(sl.parent)
        if parent.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_h2v_slice: src.parent must be HBM"
            )
        if dst.scope != _scope.VRAM:
            raise PreIsaPassV2Error(
                f"dma_h2v_slice: dst must be VRAM"
            )
        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._legacy._slice_tile_grid(parent, sl, dst)
        )
        base_off_expr = self._slice_offset_expr(parent, sl)

        self._comment(
            f"dma_h2v_slice {parent.name} -> {dst.name}  "
            f"(grid d={d_tiles} s={s_tiles} h={h_groups} b={logical_b})"
        )
        self._emit_slice_grid_h2v(
            parent=parent, dst=dst, base_off_expr=base_off_expr,
            d_tiles=d_tiles, s_tiles=s_tiles, h_groups=h_groups,
            logical_b=logical_b, inner_mlen=inner_mlen,
            lane_count=lane_count,
            hbm_stride_b=hbm_stride_b, hbm_stride_s=hbm_stride_s,
            hbm_stride_h=hbm_stride_h,
            d_tile_stride=d_tile_stride, s_tile_stride=s_tile_stride,
            h_grp_stride=h_grp_stride, b_stride=b_stride,
        )

    def _emit_dma_h2m_slice(self, mod, op) -> None:
        sl = op.buffer_args[0]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassV2Error(
                f"dma_h2m_slice: buffer_args[0] must be BufferSlice"
            )
        dst = mod.get_buffer(op.buffer_args[1])
        parent = mod.get_buffer(sl.parent)
        if parent.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_h2m_slice: src.parent must be HBM"
            )
        if dst.scope != _scope.MRAM:
            raise PreIsaPassV2Error(
                f"dma_h2m_slice: dst must be MRAM"
            )
        # h2m_slice is single-tile (legacy enforces via
        # ``_check_slice_single_tile``); we just emit one
        # H_PREFETCH_M with the slice offset folded in.
        self._legacy._check_slice_single_tile(parent, sl)
        base_off_expr = self._slice_offset_expr(parent, sl)

        self._comment(
            f"dma_h2m_slice {parent.name} -> {dst.name}"
        )
        self._emit_h2m_tile_body(
            hbm_addr=int(parent.address),
            mram_addr=int(dst.address),
            hbm_offset=base_off_expr,
            hbm_scale=parent.hbm_scale_size,
            hbm_stride=parent.hbm_stride,
        )

    def _emit_dma_v2h_slice(self, mod, op) -> None:
        src = mod.get_buffer(op.buffer_args[0])
        sl = op.buffer_args[1]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassV2Error(
                f"dma_v2h_slice: buffer_args[1] must be BufferSlice"
            )
        parent = mod.get_buffer(sl.parent)
        if src.scope != _scope.VRAM:
            raise PreIsaPassV2Error(
                f"dma_v2h_slice: src must be VRAM"
            )
        if parent.scope != _scope.HBM:
            raise PreIsaPassV2Error(
                f"dma_v2h_slice: dst.parent must be HBM"
            )
        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._legacy._slice_tile_grid(parent, sl, src)
        )
        base_off_expr = self._slice_offset_expr(parent, sl)

        self._comment(
            f"dma_v2h_slice {src.name} -> {parent.name}  "
            f"(grid d={d_tiles} s={s_tiles} h={h_groups} b={logical_b})"
        )
        self._emit_slice_grid_v2h(
            src=src, parent=parent, base_off_expr=base_off_expr,
            d_tiles=d_tiles, s_tiles=s_tiles, h_groups=h_groups,
            logical_b=logical_b, inner_mlen=inner_mlen,
            lane_count=lane_count,
            hbm_stride_b=hbm_stride_b, hbm_stride_s=hbm_stride_s,
            hbm_stride_h=hbm_stride_h,
            d_tile_stride=d_tile_stride, s_tile_stride=s_tile_stride,
            h_grp_stride=h_grp_stride, b_stride=b_stride,
        )

    # ----- Slice tile-grid walkers (shared between h2v / v2h slice) -----

    def _slice_per_tile_addresses(
        self, *, base_off_expr, inner_mlen, lane_count,
        hbm_stride_b, hbm_stride_s, hbm_stride_h,
        d_tile_stride, s_tile_stride, h_grp_stride, b_stride,
        d_var, s_var, h_var, b_var,
    ):
        """Per-tile (hbm_off, vram_off) PrimExprs given the four
        loop_vars and the layout strides. Mirrors legacy's per-tile
        offset math in ``_emit_dma_h2v_slice`` /
        ``_emit_dma_v2h_slice``."""
        hbm_off = tir.Add(
            tir.Add(
                tir.Add(
                    tir.Add(
                        base_off_expr,
                        tir.Mul(b_var, tir.IntImm("int32", hbm_stride_b)),
                    ),
                    tir.Mul(s_var,
                            tir.IntImm("int32", inner_mlen * hbm_stride_s)),
                ),
                tir.Mul(h_var,
                        tir.IntImm("int32", lane_count * hbm_stride_h)),
            ),
            tir.Mul(d_var, tir.IntImm("int32", inner_mlen)),
        )
        vram_off = tir.Add(
            tir.Add(
                tir.Add(
                    tir.Mul(d_var, tir.IntImm("int32", d_tile_stride)),
                    tir.Mul(s_var, tir.IntImm("int32", s_tile_stride)),
                ),
                tir.Mul(h_var, tir.IntImm("int32", h_grp_stride)),
            ),
            tir.Mul(b_var, tir.IntImm("int32", b_stride)),
        )
        return hbm_off, vram_off

    def _emit_slice_grid_h2v(
        self, *, parent, dst, base_off_expr,
        d_tiles, s_tiles, h_groups, logical_b,
        inner_mlen, lane_count,
        hbm_stride_b, hbm_stride_s, hbm_stride_h,
        d_tile_stride, s_tile_stride, h_grp_stride, b_stride,
    ) -> None:
        op_id = f"{id(parent) & 0xffff:x}_{id(dst) & 0xffff:x}"
        d_var = tir.Var(f"sl_d_{op_id}", "int32")
        s_var = tir.Var(f"sl_s_{op_id}", "int32")
        h_var = tir.Var(f"sl_h_{op_id}", "int32")
        b_var = tir.Var(f"sl_b_{op_id}", "int32")

        d_loop = pi.LoopRegion(
            loop_var=d_var, init_imm=0, extent_imm=d_tiles,
            loop_kind="unroll", body=[],
        )
        self._append(d_loop)
        self._push_scope(d_loop.body)
        try:
            s_loop = pi.LoopRegion(
                loop_var=s_var, init_imm=0, extent_imm=s_tiles,
                loop_kind="unroll", body=[],
            )
            self._append(s_loop)
            self._push_scope(s_loop.body)
            try:
                h_loop = pi.LoopRegion(
                    loop_var=h_var, init_imm=0, extent_imm=h_groups,
                    loop_kind="unroll", body=[],
                )
                self._append(h_loop)
                self._push_scope(h_loop.body)
                try:
                    b_loop = pi.LoopRegion(
                        loop_var=b_var, init_imm=0, extent_imm=logical_b,
                        loop_kind="unroll", body=[],
                    )
                    self._append(b_loop)
                    self._push_scope(b_loop.body)
                    try:
                        hbm_off, vram_off = self._slice_per_tile_addresses(
                            base_off_expr=base_off_expr,
                            inner_mlen=inner_mlen, lane_count=lane_count,
                            hbm_stride_b=hbm_stride_b,
                            hbm_stride_s=hbm_stride_s,
                            hbm_stride_h=hbm_stride_h,
                            d_tile_stride=d_tile_stride,
                            s_tile_stride=s_tile_stride,
                            h_grp_stride=h_grp_stride,
                            b_stride=b_stride,
                            d_var=d_var, s_var=s_var,
                            h_var=h_var, b_var=b_var,
                        )
                        vram_addr_expr = tir.Add(
                            tir.IntImm("int32", int(dst.address)),
                            vram_off,
                        )
                        self._emit_h2v_tile_body(
                            hbm_addr=int(parent.address),
                            vram_addr=vram_addr_expr,
                            hbm_stride=parent.hbm_stride,
                            hbm_scale_size=parent.hbm_scale_size,
                            hbm_start_offset=hbm_off,
                        )
                    finally:
                        self._pop_scope()
                finally:
                    self._pop_scope()
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    def _emit_slice_grid_v2h(
        self, *, src, parent, base_off_expr,
        d_tiles, s_tiles, h_groups, logical_b,
        inner_mlen, lane_count,
        hbm_stride_b, hbm_stride_s, hbm_stride_h,
        d_tile_stride, s_tile_stride, h_grp_stride, b_stride,
    ) -> None:
        op_id = f"{id(parent) & 0xffff:x}_{id(src) & 0xffff:x}"
        d_var = tir.Var(f"sl_d_{op_id}", "int32")
        s_var = tir.Var(f"sl_s_{op_id}", "int32")
        h_var = tir.Var(f"sl_h_{op_id}", "int32")
        b_var = tir.Var(f"sl_b_{op_id}", "int32")

        d_loop = pi.LoopRegion(
            loop_var=d_var, init_imm=0, extent_imm=d_tiles,
            loop_kind="unroll", body=[],
        )
        self._append(d_loop)
        self._push_scope(d_loop.body)
        try:
            s_loop = pi.LoopRegion(
                loop_var=s_var, init_imm=0, extent_imm=s_tiles,
                loop_kind="unroll", body=[],
            )
            self._append(s_loop)
            self._push_scope(s_loop.body)
            try:
                h_loop = pi.LoopRegion(
                    loop_var=h_var, init_imm=0, extent_imm=h_groups,
                    loop_kind="unroll", body=[],
                )
                self._append(h_loop)
                self._push_scope(h_loop.body)
                try:
                    b_loop = pi.LoopRegion(
                        loop_var=b_var, init_imm=0, extent_imm=logical_b,
                        loop_kind="unroll", body=[],
                    )
                    self._append(b_loop)
                    self._push_scope(b_loop.body)
                    try:
                        hbm_off, vram_off = self._slice_per_tile_addresses(
                            base_off_expr=base_off_expr,
                            inner_mlen=inner_mlen, lane_count=lane_count,
                            hbm_stride_b=hbm_stride_b,
                            hbm_stride_s=hbm_stride_s,
                            hbm_stride_h=hbm_stride_h,
                            d_tile_stride=d_tile_stride,
                            s_tile_stride=s_tile_stride,
                            h_grp_stride=h_grp_stride,
                            b_stride=b_stride,
                            d_var=d_var, s_var=s_var,
                            h_var=h_var, b_var=b_var,
                        )
                        vram_addr_expr = tir.Add(
                            tir.IntImm("int32", int(src.address)),
                            vram_off,
                        )
                        self._emit_v2h_tile_body(
                            vram_addr=vram_addr_expr,
                            hbm_addr=int(parent.address),
                            hbm_stride=parent.hbm_stride,
                            hbm_scale_size=parent.hbm_scale_size,
                            hbm_start_offset=hbm_off,
                        )
                    finally:
                        self._pop_scope()
                finally:
                    self._pop_scope()
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    # ----- DMA tile bodies (one HBM tile worth of preload/store) -----
    #
    # These mirror ``ISAEmitter._emit_preload_tile_isa`` /
    # ``_emit_store_tile_isa`` for the ``batch = mlen`` (multi-row)
    # case, which is the only one v2 supports today. The ``batch = 1``
    # narrow-row path exists in legacy but is unreachable from our
    # shim's DMA dispatcher (it always sets ``batch = mlen``).

    def _emit_h2v_tile_body(
        self, *, hbm_addr: int, vram_addr,
        hbm_stride, hbm_scale_size, hbm_start_offset,
    ) -> None:
        """One HBM-tile's worth of preload, as PreIsaIR.

        ``vram_addr`` and ``hbm_start_offset`` may be int OR
        ``tir.PrimExpr`` (slice-aware DMA passes in PrimExprs that
        reference outer LoopRegion loop_vars).

        Structure: ``C_SET_ADDR_REG`` + scale/stride + a nested
        (outer, inner) LoopRegion whose body issues one
        ``H_PREFETCH_V`` with PrimExpr addresses referencing both
        loop vars.
        """
        mlen = int(self.shim.mlen)
        v_prefetch = int(self.shim.v_prefetch_amount)
        tile_elems = mlen * mlen
        stride_len = mlen if hbm_stride is None else int(hbm_stride)
        scale_len = tile_elems if hbm_scale_size is None else int(hbm_scale_size)
        vram_addr_e = _as_expr(vram_addr)
        hbm_start_e = _as_expr(hbm_start_offset)

        # batch = hidden = mlen for our shim's DMA path, so
        # load_amount_per_hidden = ceil(mlen / mlen) = 1 and
        # inner_count = ceil(mlen / v_prefetch). When batch <=
        # preload, the inner stride term collapses to 0 (no per-inner
        # advance) — matches legacy's ``if batch > preload_len``
        # branch.
        load_amount_per_hidden = (mlen + mlen - 1) // mlen
        if mlen > v_prefetch:
            inner_count = (mlen + v_prefetch - 1) // v_prefetch
            inner_stride_per_iter = stride_len * v_prefetch
        else:
            inner_count = 1
            inner_stride_per_iter = 0

        # 1) Bind a fresh addr_reg to ``hbm_addr``.
        addr = pi.PreIsaOp(
            opcode="C_SET_ADDR_REG",
            # high word = constant-zero gp0; low word = address.
            operands=["gp0", int(hbm_addr)],
        )
        self._append(addr)
        # 2) Scale + stride for this DMA.
        self._append(pi.PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[int(scale_len)],
        ))
        self._append(pi.PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[int(stride_len)],
        ))

        # 3) Nested LoopRegions over (outer, inner). Addresses become
        #    PrimExprs in loop_var; arith.simplify in pre_isa_to_mir
        #    folds the static constants.
        op_id = f"{id(addr) & 0xffff:x}"
        outer_var = tir.Var(f"h2v_outer_{op_id}", "int32")
        inner_var = tir.Var(f"h2v_inner_{op_id}", "int32")

        outer_loop = pi.LoopRegion(
            loop_var=outer_var, init_imm=0,
            extent_imm=load_amount_per_hidden,
            loop_kind="unroll", body=[],
        )
        self._append(outer_loop)
        self._push_scope(outer_loop.body)
        try:
            inner_loop = pi.LoopRegion(
                loop_var=inner_var, init_imm=0, extent_imm=inner_count,
                loop_kind="unroll", body=[],
            )
            self._append(inner_loop)
            self._push_scope(inner_loop.body)
            try:
                # result_addr = vram_addr
                #             + (outer * inner_count + inner)
                #             * (mlen * v_prefetch)
                row_idx = tir.Add(
                    tir.Mul(outer_var,
                            tir.IntImm("int32", inner_count)),
                    inner_var,
                )
                result_addr = tir.Add(
                    vram_addr_e,
                    tir.Mul(row_idx,
                            tir.IntImm("int32", mlen * v_prefetch)),
                )
                # hbm_off = hbm_start_offset
                #         + outer * mlen
                #         + inner * inner_stride_per_iter
                hbm_off = tir.Add(
                    tir.Add(
                        hbm_start_e,
                        tir.Mul(outer_var,
                                tir.IntImm("int32", mlen)),
                    ),
                    tir.Mul(inner_var,
                            tir.IntImm("int32",
                                       int(inner_stride_per_iter))),
                )
                self._append(pi.PreIsaOp(
                    opcode="H_PREFETCH_V",
                    operands=[result_addr, hbm_off, addr, 1, 0],
                ))
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    def _emit_h2m_tile_body(
        self, *, hbm_addr: int, mram_addr,
        hbm_offset, hbm_scale, hbm_stride,
    ) -> None:
        """One HBM tile → MRAM. ``mram_addr`` and ``hbm_offset`` may
        be int OR PrimExpr (slice path with dynamic starts)."""
        mlen = int(self.shim.mlen)
        tile_elems = mlen * mlen
        scale_val = tile_elems if hbm_scale is None else int(hbm_scale)
        stride_val = mlen if hbm_stride is None else int(hbm_stride)

        addr = pi.PreIsaOp(
            opcode="C_SET_ADDR_REG",
            # high word = constant-zero gp0; low word = address.
            operands=["gp0", int(hbm_addr)],

        )
        self._append(addr)
        self._append(pi.PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[int(scale_val)],
        ))
        self._append(pi.PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[int(stride_val)],
        ))
        # One H_PREFETCH_M per tile (mram_addr, hbm_offset both as
        # PrimExpr — converter folds static cases via arith.simplify).
        self._append(pi.PreIsaOp(
            opcode="H_PREFETCH_M",
            operands=[
                _as_expr(mram_addr),
                _as_expr(hbm_offset),
                addr,
                1, 0,
            ],
        ))
        # Reset scale/stride to canonical (tile_elems / mlen) so subsequent
        # non-DMA ops see the defaults the rest of the kernel expects —
        # matches legacy reset after H_PREFETCH_M.
        self._append(pi.PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[int(tile_elems)],
        ))
        self._append(pi.PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[int(mlen)],
        ))

    def _emit_v2h_tile_body(
        self, *, vram_addr, hbm_addr: int,
        hbm_stride, hbm_scale_size, hbm_start_offset,
    ) -> None:
        """One HBM tile's worth of writeback. Mirror of
        ``_emit_h2v_tile_body``: nested (outer, inner) LoopRegions
        with PrimExpr addresses; emits ``H_STORE_V`` instead of
        ``H_PREFETCH_V``. ``vram_addr`` / ``hbm_start_offset`` may
        be int or PrimExpr.
        """
        mlen = int(self.shim.mlen)
        v_writeback = int(self.shim.v_writeback_amount)
        tile_elems = mlen * mlen
        stride_len = mlen if hbm_stride is None else int(hbm_stride)
        scale_len = tile_elems if hbm_scale_size is None else int(hbm_scale_size)
        vram_addr_e = _as_expr(vram_addr)
        hbm_start_e = _as_expr(hbm_start_offset)

        store_amount_per_hidden = (mlen + mlen - 1) // mlen
        if mlen > v_writeback:
            inner_count = (mlen + v_writeback - 1) // v_writeback
            inner_stride_per_iter = stride_len * v_writeback
        else:
            inner_count = 1
            inner_stride_per_iter = 0

        addr = pi.PreIsaOp(
            opcode="C_SET_ADDR_REG",
            # high word = constant-zero gp0; low word = address.
            operands=["gp0", int(hbm_addr)],
        )
        self._append(addr)
        self._append(pi.PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[int(scale_len)],
        ))
        self._append(pi.PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[int(stride_len)],
        ))

        op_id = f"{id(addr) & 0xffff:x}"
        outer_var = tir.Var(f"v2h_outer_{op_id}", "int32")
        inner_var = tir.Var(f"v2h_inner_{op_id}", "int32")

        outer_loop = pi.LoopRegion(
            loop_var=outer_var, init_imm=0,
            extent_imm=store_amount_per_hidden,
            loop_kind="unroll", body=[],
        )
        self._append(outer_loop)
        self._push_scope(outer_loop.body)
        try:
            inner_loop = pi.LoopRegion(
                loop_var=inner_var, init_imm=0, extent_imm=inner_count,
                loop_kind="unroll", body=[],
            )
            self._append(inner_loop)
            self._push_scope(inner_loop.body)
            try:
                row_idx = tir.Add(
                    tir.Mul(outer_var,
                            tir.IntImm("int32", inner_count)),
                    inner_var,
                )
                vram_chunk = tir.Add(
                    vram_addr_e,
                    tir.Mul(row_idx,
                            tir.IntImm("int32", mlen * v_writeback)),
                )
                hbm_off = tir.Add(
                    tir.Add(
                        hbm_start_e,
                        tir.Mul(outer_var,
                                tir.IntImm("int32", mlen)),
                    ),
                    tir.Mul(inner_var,
                            tir.IntImm("int32",
                                       int(inner_stride_per_iter))),
                )
                self._append(pi.PreIsaOp(
                    opcode="H_STORE_V",
                    operands=[vram_chunk, hbm_off, addr, 1, 0],
                ))
            finally:
                self._pop_scope()
        finally:
            self._pop_scope()

    # ------------------------------------------------------------------
    # mv — per-head matrix × vector with tile loop over n/blen.
    # ------------------------------------------------------------------
    def _emit_mv(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Legacy emits ``tiles = n // blen`` (M_MV, M_MV_WO) pairs,
        each iter advancing the mat/dst pointers by ``blen``. In v2
        we model the tile loop as an unroll LoopRegion with the
        per-tile addresses as ``base + t * blen``.

        Static-offset path only — dynamic region offsets are a TODO.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassV2Error(
                f"plena.mv expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(f"plena.mv a: expected VramRegion")
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassV2Error(f"plena.mv b: expected MramRegion")
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassV2Error(f"plena.mv c: expected VramRegion")
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        # Region origin offsets — static-only.
        lhs_off = self._legacy._region_origin_offset(lhs, a_reg)
        rhs_off = self._legacy._region_origin_offset(rhs, b_reg)
        dst_off = self._legacy._region_origin_offset(dst, c_reg)

        def _static(x):
            if isinstance(x, int):
                return int(x)
            if isinstance(x, tir.IntImm):
                return int(x.value)
            return None

        lhs_off_s = _static(lhs_off)
        rhs_off_s = _static(rhs_off)
        dst_off_s = _static(dst_off)
        if (lhs_off_s is None or rhs_off_s is None or dst_off_s is None):
            raise PreIsaPassV2Error(
                f"plena.mv: dynamic region offsets not yet supported on v2"
            )
        task_id = op.annotations.get("intrinsic", "mv")
        n = int(self.shim.btmm_hlen)
        blen = int(self.shim.blen)
        if n % blen != 0:
            raise PreIsaPassV2Error(
                f"plena.mv: n={n} must be a multiple of blen={blen}"
            )
        tiles = n // blen
        lhs_vram_addr = int(lhs.address) + lhs_off_s
        rhs_mram_addr = int(rhs.address) + rhs_off_s
        dst_vram_addr = int(dst.address) + dst_off_s
        self._comment(
            f"mv task {task_id} "
            f"v=vram[{lhs_vram_addr}] "
            f"m=mram[{rhs_mram_addr}] "
            f"dst=vram[{dst_vram_addr}] "
            f"tiles={tiles} blen={blen}"
        )
        # Per-tile address PrimExprs. lhs is fixed across tiles (the
        # vector base); rhs and dst advance by ``blen`` per tile.
        lhs_addr = tir.IntImm("int32", lhs_vram_addr)
        if tiles > 1:
            t_var = tir.Var(f"mv_t_{id(op) & 0xffff:x}", "int32")
            loop = pi.LoopRegion(
                loop_var=t_var, init_imm=0, extent_imm=tiles,
                loop_kind="unroll", body=[],
            )
            self._append(loop)
            self._push_scope(loop.body)
            try:
                rhs_addr = tir.Add(
                    tir.IntImm("int32", rhs_mram_addr),
                    tir.Mul(t_var, tir.IntImm("int32", blen)),
                )
                dst_addr = tir.Add(
                    tir.IntImm("int32", dst_vram_addr),
                    tir.Mul(t_var, tir.IntImm("int32", blen)),
                )
                # M_MV gp0, gp{rhs}, gp{lhs}
                self._append(pi.PreIsaOp(
                    opcode="M_MV",
                    operands=["gp0", rhs_addr, lhs_addr],
                ))
                # M_MV_WO gp{dst}, 0
                self._append(pi.PreIsaOp(
                    opcode="M_MV_WO",
                    operands=[dst_addr, 0],
                ))
            finally:
                self._pop_scope()
        else:
            rhs_addr = tir.IntImm("int32", rhs_mram_addr)
            dst_addr = tir.IntImm("int32", dst_vram_addr)
            self._append(pi.PreIsaOp(
                opcode="M_MV",
                operands=["gp0", rhs_addr, lhs_addr],
            ))
            self._append(pi.PreIsaOp(
                opcode="M_MV_WO",
                operands=[dst_addr, 0],
            ))


__all__ = ["PreIsaPassV2", "PreIsaPassV2Error"]
