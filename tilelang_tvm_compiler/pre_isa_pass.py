"""PreIsaPass — lower HLIR to PreIsaIR.

Replaces the emit half of the legacy ``IsaEmitterPass``. For each HLIR
op:
  * Same address-math code as before (reuses helpers on ``IsaEmitterPass``
    by composition; this pass holds an instance to delegate to for now,
    while the migration is in progress).
  * Instead of calling ``materializer.materialize`` + ``ISAEmitter.emit_*``
    + ``generated_code +=``, the handler appends one or more PreIsaOps
    to ``self.pre_isa`` (a ``PreIsaModule``). Operands are kept as
    ``tir.PrimExpr`` (var-ref form); register allocation happens later
    in :class:`backend_emit.BackendEmit`.

Iron rule: one PreIsaOp == one HW ISA instruction. ``C_LOOP_START`` /
``C_LOOP_END`` are themselves PreIsaOps in the same flat stream.

Migration status:
  Migrated handlers (produce PreIsaIR):
    * fp_zero_at
  All other op kinds delegate to the legacy ``IsaEmitterPass`` and run
  through the byte-equal old path. As each handler migrates, the
  delegate fallback is removed.
"""

from __future__ import annotations

import warnings
from typing import Dict, Callable

from tvm import tir

from . import hlir as _hlir
from . import scope as _scope
from .hw_consts import (
    BLEN_VAR, BTMM_HLEN_VAR, MLEN_VAR,
    V_PREFETCH_AMOUNT_VAR, V_WRITEBACK_AMOUNT_VAR,
)
from .pre_isa_ir import PreIsaModule, PreIsaOp


class PreIsaPassError(RuntimeError):
    pass


class PreIsaPass:
    """Lower an HLIRModule into a PreIsaModule.

    Construction takes the same shim as the legacy IsaEmitterPass —
    we reuse its address-resolution helpers (``_resolve_fp_scalar_addr_arg``
    in particular) by composing the legacy pass instance.
    """

    def __init__(self, shim) -> None:
        # Lazy import to avoid the dependency cycle isa_pass ↔ pre_isa_pass
        # at module-import time.
        from .isa_pass import IsaEmitterPass
        self.shim = shim
        self._legacy = IsaEmitterPass(shim)
        self.pre_isa = PreIsaModule(name="<unset>")

        # Counter for ``annotations["group_id"]`` — BackendEmit groups
        # consecutive PreIsaOps sharing this id into one materialisation
        # scope (so e.g. an FP-binary's 3 addresses get materialised
        # once across its 5 ISA lines, mirroring the legacy
        # ra.pin_gp pattern in IsaEmitterPass).
        self._next_group_id: int = 0

        self._dispatch: Dict[str, Callable[[_hlir.HLIRModule, _hlir.Op], None]] = {
            "fp_zero_at":  self._emit_fp_zero_at,
            "fp_copy_at":  lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="copy"),
            "fp_exp_at":   lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="exp"),
            "fp_reci_at":  lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="reci"),
            "fp_sqrt_at":  lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="sqrt"),
            "fp_add_at":   lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="add"),
            "fp_sub_at":   lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="sub"),
            "fp_mul_at":   lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="mul"),
            "fp_max_at":   lambda m, o: self._emit_fp_scalar_op_at(m, o, kernel_op="max"),
            "v_zero":      self._emit_v_zero,
            "v_add":       lambda m, o: self._emit_v_binary(m, o, binary_op="add"),
            "v_sub":       lambda m, o: self._emit_v_binary(m, o, binary_op="sub"),
            "v_mul":       lambda m, o: self._emit_v_binary(m, o, binary_op="mul"),
            "v_exp":       lambda m, o: self._emit_v_unary(m, o, opcode="V_EXP_V"),
            "v_reci":      lambda m, o: self._emit_v_unary(m, o, opcode="V_RECI_V"),
            "v_sqrt":      lambda m, o: self._emit_v_unary(m, o, opcode="V_SQRT_V"),
            "copy_v_to_v": self._emit_copy_v_to_v,
            "v_fp_transfer_slice_v_to_fp":
                lambda m, o: self._emit_v_fp_transfer_slice(m, o, direction="v_to_fp"),
            "v_fp_transfer_slice_fp_to_v":
                lambda m, o: self._emit_v_fp_transfer_slice(m, o, direction="fp_to_v"),
            "for":         self._emit_for,
            "row_reduce_max_at":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="reduce_max", reduce=True, masked=True,
                ),
            "row_reduce_sum_at":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="reduce_sum", reduce=True, masked=True,
                ),
            "row_exp":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="exp", masked=True,
                ),
            "row_sub_fp":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="sub", masked=True, has_fp=True,
                ),
            "row_mul_fp":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="mul", masked=True, has_fp=True,
                ),
            "row_add_fp":
                lambda m, o: self._emit_row_scalar_op_at(
                    m, o, row_op="add", masked=True, has_fp=True,
                ),
            "btmm":     self._emit_btmm,
            "btmv":     self._emit_btmv,
            "mv":       self._emit_mv,
            "mm":       self._emit_mm,
            "mm_slot":  self._emit_mm_slot,
            "matmul":   self._emit_matmul,
            "dma_h2v":        self._emit_dma_h2v,
            "dma_h2m":        self._emit_dma_h2m,
            "dma_v2h":        self._emit_dma_v2h,
            "dma_h2v_slice":  self._emit_dma_h2v_slice,
            "dma_h2m_slice":  self._emit_dma_h2m_slice,
            "dma_v2h_slice":  self._emit_dma_v2h_slice,
        }

    def _new_group(self) -> int:
        gid = self._next_group_id
        self._next_group_id += 1
        return gid

    def run(self, mod: _hlir.HLIRModule) -> PreIsaModule:
        """Produce a PreIsaModule for ``mod``. Buffers / addresses are
        forwarded verbatim onto the PreIsaModule for BackendEmit /
        the dump."""
        _hlir.assert_addresses_resolved(mod)
        self.pre_isa = PreIsaModule(name=mod.name, buffers=dict(mod.buffers))
        self.pre_isa.comment(f"PLENA ISA  --  kernel: {mod.name}")
        self.pre_isa.comment("generated by tilelang_tvm_compiler (PreIsaIR path)")
        self.pre_isa.comment("=" * 60)
        self.pre_isa.comment("buffer layout:")
        for buf in mod.buffers.values():
            shape_s = "x".join(str(s) for s in buf.shape)
            self.pre_isa.comment(
                f"  {buf.name:<10s} scope={buf.scope:<5s} addr={buf.address}  "
                f"shape={shape_s}"
            )
        self.pre_isa.comment("=" * 60)
        self.pre_isa.comment("")
        for op in mod.ops:
            handler = self._dispatch.get(op.kind)
            if handler is None:
                raise PreIsaPassError(
                    f"PreIsaPass: no handler migrated yet for HLIR op "
                    f"kind {op.kind!r}. While migration is in progress "
                    f"callers must dispatch to the legacy IsaEmitterPass "
                    f"for non-migrated kinds."
                )
            handler(mod, op)
        return self.pre_isa

    # ==================================================================
    # migrated handlers
    # ==================================================================
    def _emit_fp_zero_at(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Store FP zero to one FPRAM slot.

        Legacy emission (isa_pass._emit_fp_zero_at):
            ; fp scalar task <intrinsic> op=zero
            S_ST_FP f0, gp{dst}, 0

        PreIsaIR emission (var-ref; one PreIsaOp per ISA line):
            _COMMENT  "fp scalar task <intrinsic> op=zero"
            S_ST_FP   ["f0", <dst PrimExpr>, 0]
        """
        if len(op.scalar_args) != 1:
            raise PreIsaPassError(
                f"{op.kind} expects 1 scalar address arg, got {len(op.scalar_args)}"
            )
        dst_addr_expr = self._legacy._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "dst",
        )
        intrinsic = op.annotations.get("intrinsic", op.kind)
        gid = self._new_group()
        # Order matches legacy isa_pass._emit_fp_zero_at: there is no
        # upfront materialisation loop in legacy fp_zero_at (only one
        # address) — the S_ADDI_INT comes from materialise() right
        # before the S_ST_FP. So we DON'T emit a _PRELOAD_ADDR here.
        self.pre_isa.append(
            PreIsaOp(
                opcode="_COMMENT",
                operands=[f"fp scalar task {intrinsic} op=zero"],
                annotations={"group_id": gid},
            )
        )
        self.pre_isa.append(
            PreIsaOp(
                opcode="S_ST_FP",
                operands=["f0", dst_addr_expr, 0],
                annotations={"group_id": gid},
            )
        )

    def _emit_fp_scalar_op_at(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, kernel_op: str,
    ) -> None:
        """Mirror of legacy isa_pass._emit_fp_scalar_op_at — FP scalar
        load/op/store sequence on FPRAM operands.

        For copy/exp/reci/sqrt (2 scalar_args = src, dst):
            ; fp scalar task <intrinsic> op=<kernel_op>
            S_LD_FP f1, gp{src}, 0
            [S_EXP_FP f1, f1, 0  |  S_RECI_FP f1, f1  |  S_SQRT_FP f1, f1]   (only for exp/reci/sqrt)
            S_ST_FP f1, gp{dst}, 0

        For add/sub/mul/max (3 scalar_args = lhs, rhs, dst):
            ; fp scalar task <intrinsic> op=<kernel_op>
            S_LD_FP f1, gp{lhs}, 0
            S_LD_FP f2, gp{rhs}, 0
            <opcode> f1, f1, f2
            S_ST_FP f1, gp{dst}, 0

        All addresses go through ``_resolve_fp_scalar_addr_arg`` so a
        ``BufferElement`` resolves to ``buf.address + flat-offset`` —
        identical to legacy. The address PrimExpr objects are CREATED
        ONCE and reused across the multiple PreIsaOps in this op so
        BackendEmit's group cache materialises each address one time.
        """
        if kernel_op in {"copy", "exp", "reci", "sqrt"}:
            expected = 2
        else:
            expected = 3
        if len(op.scalar_args) != expected:
            raise PreIsaPassError(
                f"{op.kind} expects {expected} scalar address args, got {len(op.scalar_args)}"
            )

        addr_exprs = [
            self._legacy._resolve_fp_scalar_addr_arg(
                mod, a, op.kind, f"arg{i}",
            )
            for i, a in enumerate(op.scalar_args)
        ]
        intrinsic = op.annotations.get("intrinsic", op.kind)
        gid = self._new_group()

        def _stamp(op_):
            op_.annotations["group_id"] = gid
            return op_

        # Order matches legacy isa_pass._emit_fp_scalar_op_at:
        #   1. for each address, materialise (S_ADDI_INT / S_LUI_INT)
        #   2. emit the ``; fp scalar task ...`` comment
        #   3. emit the S_LD_FP / OP / S_ST_FP burst
        # PreIsaIR encodes step 1 as _PRELOAD_ADDR meta-ops so the
        # group cache pre-populates before the FP ops materialise
        # operands lazily.
        for addr in addr_exprs:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[addr],
            )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_COMMENT",
            operands=[f"fp scalar task {intrinsic} op={kernel_op}"],
        )))
        if kernel_op in {"copy", "exp", "reci", "sqrt"}:
            src, dst = addr_exprs
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="S_LD_FP", operands=["f1", src, 0],
            )))
            if kernel_op == "exp":
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="S_EXP_FP", operands=["f1", "f1", 0],
                )))
            elif kernel_op == "reci":
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="S_RECI_FP", operands=["f1", "f1"],
                )))
            elif kernel_op == "sqrt":
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="S_SQRT_FP", operands=["f1", "f1"],
                )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="S_ST_FP", operands=["f1", dst, 0],
            )))
        else:
            lhs, rhs, dst = addr_exprs
            opcode_map = {
                "add": "S_ADD_FP",
                "sub": "S_SUB_FP",
                "mul": "S_MUL_FP",
                "max": "S_MAX_FP",
            }
            opcode = opcode_map[kernel_op]
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="S_LD_FP", operands=["f1", lhs, 0],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="S_LD_FP", operands=["f2", rhs, 0],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode=opcode, operands=["f1", "f1", "f2"],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="S_ST_FP", operands=["f1", dst, 0],
            )))

    # ------------------------------------------------------------------
    # vector ops — VRAM region-based, walks chunks via the legacy
    # ``_vram_region_iter_chunks`` helper. One HLIR op may emit N
    # PreIsaOps (N = total chunk count across non-cluster outer dims).
    # ------------------------------------------------------------------
    def _emit_v_zero(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_v_zero``:
            ; v_zero dst.parent=... starts=... extents=...
            for each chunk:
                S_ADDI_INT gp{r}, gp0, dst.address + d_off
                V_MUL_VF gp{r}, gp{r}, f0, 0
        """
        if len(op.buffer_args) != 1:
            raise PreIsaPassError(
                f"v_zero expects 1 buffer_arg; got {len(op.buffer_args)}"
            )
        if not isinstance(op.buffer_args[0], _hlir.VramRegion):
            raise PreIsaPassError(
                f"v_zero dst: expected VramRegion, got "
                f"{type(op.buffer_args[0]).__name__}"
            )
        if op.scalar_args:
            raise PreIsaPassError(
                f"v_zero expects 0 scalar_args; got {len(op.scalar_args)}"
            )
        dst_region: _hlir.VramRegion = op.buffer_args[0]
        dst = mod.get_buffer(dst_region.parent)
        self.pre_isa.comment(
            f"v_zero dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}"
        )
        for d_off, _ in self._legacy._vram_region_iter_chunks(dst, dst_region):
            dst_addr = tir.Add(
                tir.IntImm("int32", int(dst.address)), d_off,
            )
            gid = self._new_group()
            # V_MUL_VF dst, dst, f0, 0 — both vector operand slots
            # point at the SAME ``dst_addr`` PrimExpr object so
            # BackendEmit's id()-keyed group cache materialises it
            # once and reuses the GP for both gpN positions.
            self.pre_isa.append(PreIsaOp(
                opcode="V_MUL_VF",
                operands=[dst_addr, dst_addr, "f0", 0],
                annotations={"group_id": gid},
            ))

    def _emit_v_binary(self, mod: _hlir.HLIRModule, op: _hlir.Op,
                       *, binary_op: str) -> None:
        """Mirror of legacy ``isa_pass._emit_v_binary``.
            ; v binary <kind> <opcode> dst.parent=... starts=... extents=...
            for each (l_off, r_off, d_off):
                S_ADDI_INT gp{lhs}, ...
                S_ADDI_INT gp{rhs}, ...
                S_ADDI_INT gp{dst}, ...
                <opcode> gp{dst}, gp{lhs}, gp{rhs}, 0
        """
        op_to_insn = {"add": "V_ADD_VV", "sub": "V_SUB_VV", "mul": "V_MUL_VV"}
        opcode = op_to_insn[binary_op]
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"{op.kind} expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("lhs", "rhs", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise PreIsaPassError(
                f"{op.kind} expects 0 scalar_args; got {len(op.scalar_args)}"
            )
        lhs_region: _hlir.VramRegion = op.buffer_args[0]
        rhs_region: _hlir.VramRegion = op.buffer_args[1]
        dst_region: _hlir.VramRegion = op.buffer_args[2]
        lhs = mod.get_buffer(lhs_region.parent)
        rhs = mod.get_buffer(rhs_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self.pre_isa.comment(
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
            lhs_addr = tir.Add(tir.IntImm("int32", int(lhs.address)), l_off)
            rhs_addr = tir.Add(tir.IntImm("int32", int(rhs.address)), r_off)
            dst_addr = tir.Add(tir.IntImm("int32", int(dst.address)), d_off)
            gid = self._new_group()

            def _stamp(o):
                o.annotations["group_id"] = gid
                return o

            # Preload order = legacy materialise order = lhs, rhs, dst.
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[lhs_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[rhs_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[dst_addr],
            )))
            # ISA op operand order in template = dst, lhs, rhs, 0
            # (matches legacy
            # ``f"{opcode} gp{m_dst}, gp{m_lhs}, gp{m_rhs}, 0\n"``).
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode=opcode,
                operands=[dst_addr, lhs_addr, rhs_addr, 0],
            )))

    def _emit_v_unary(self, mod: _hlir.HLIRModule, op: _hlir.Op,
                      *, opcode: str) -> None:
        """Mirror of legacy ``isa_pass._emit_v_unary``:
            ; v unary <kind> <opcode> dst.parent=... starts=... extents=...
            for each (s_off, d_off):
                S_ADDI_INT gp{src}, ...
                S_ADDI_INT gp{dst}, ...
                <opcode> gp{dst}, gp{src}, 0
        """
        if len(op.buffer_args) != 2:
            raise PreIsaPassError(
                f"{op.kind} expects 2 buffer_args; got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise PreIsaPassError(
                f"{op.kind} expects 0 scalar_args; got {len(op.scalar_args)}"
            )
        src_region: _hlir.VramRegion = op.buffer_args[0]
        dst_region: _hlir.VramRegion = op.buffer_args[1]
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self.pre_isa.comment(
            f"v unary {op.kind} {opcode} "
            f"dst.parent={dst_region.parent} "
            f"starts={list(dst_region.starts)!r} "
            f"extents={list(dst_region.extents)!r}"
        )
        src_iter = self._legacy._vram_region_iter_chunks(src, src_region)
        dst_iter = self._legacy._vram_region_iter_chunks(dst, dst_region)
        for (s_off, _), (d_off, _) in zip(src_iter, dst_iter):
            src_addr = tir.Add(tir.IntImm("int32", int(src.address)), s_off)
            dst_addr = tir.Add(tir.IntImm("int32", int(dst.address)), d_off)
            gid = self._new_group()

            def _stamp(o):
                o.annotations["group_id"] = gid
                return o

            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[src_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[dst_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode=opcode,
                operands=[dst_addr, src_addr, 0],
            )))

    # ------------------------------------------------------------------
    # VRAM <-> VRAM copy and VRAM <-> FPRAM transfer
    # ------------------------------------------------------------------
    def _emit_copy_v_to_v(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_copy_v_to_v``:
            ; copy_v_to_v ...
            for each chunk:
                S_ADDI_INT gp{src}, ...
                S_ADDI_INT gp{dst}, ...
                V_ADD_VF gp{dst}, gp{src}, f0, 0
        """
        if len(op.buffer_args) != 2:
            raise PreIsaPassError(
                f"copy_v_to_v expects 2 buffer_args; got {len(op.buffer_args)}"
            )
        for slot, name in enumerate(("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassError(
                    f"copy_v_to_v {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )
        if op.scalar_args:
            raise PreIsaPassError(
                f"copy_v_to_v expects 0 scalar_args; got {len(op.scalar_args)}"
            )
        src_region: _hlir.VramRegion = op.buffer_args[0]
        dst_region: _hlir.VramRegion = op.buffer_args[1]
        src = mod.get_buffer(src_region.parent)
        dst = mod.get_buffer(dst_region.parent)
        self.pre_isa.comment(
            f"copy_v_to_v src.parent={src_region.parent} -> "
            f"dst.parent={dst_region.parent} "
            f"extents={list(dst_region.extents)!r}"
        )
        src_iter = self._legacy._vram_region_iter_chunks(src, src_region)
        dst_iter = self._legacy._vram_region_iter_chunks(dst, dst_region)
        for (s_off, _), (d_off, _) in zip(src_iter, dst_iter):
            src_addr = tir.Add(tir.IntImm("int32", int(src.address)), s_off)
            dst_addr = tir.Add(tir.IntImm("int32", int(dst.address)), d_off)
            gid = self._new_group()

            def _stamp(o):
                o.annotations["group_id"] = gid
                return o

            # Legacy materialise order = src, dst.
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[src_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[dst_addr],
            )))
            # V_ADD_VF gp{dst}, gp{src}, f0, 0
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="V_ADD_VF",
                operands=[dst_addr, src_addr, "f0", 0],
            )))

    def _emit_v_fp_transfer_slice(
        self, mod: _hlir.HLIRModule, op: _hlir.Op, *, direction: str,
    ) -> None:
        """Mirror of legacy ``isa_pass._emit_v_fp_transfer_slice``.

        direction == "v_to_fp":
            ; v↔fp transfer slice ...
            for each chunk:
                S_ADDI_INT gp{vram}, ...
                S_ADDI_INT gp{fp},   ...
                S_MAP_FP_V gp{fp}, gp{vram}, 0
        direction == "fp_to_v":
            same prelude, then  S_MAP_V_FP gp{vram}, gp{fp}, 0
        """
        if len(op.buffer_args) != 1 or not isinstance(op.buffer_args[0], _hlir.VramRegion):
            raise PreIsaPassError(
                f"{op.kind}: buffer_args[0] must be VramRegion"
            )
        if len(op.scalar_args) != 1:
            raise PreIsaPassError(
                f"{op.kind}: expected 1 scalar arg (fp_addr); got {len(op.scalar_args)}"
            )
        region: _hlir.VramRegion = op.buffer_args[0]
        vram = mod.get_buffer(region.parent)
        fp_addr_base = self._legacy._resolve_fp_scalar_addr_arg(
            mod, op.scalar_args[0], op.kind, "fp",
        )
        opcode = "S_MAP_FP_V" if direction == "v_to_fp" else "S_MAP_V_FP"
        self.pre_isa.comment(
            f"v↔fp transfer slice {op.kind} parent={region.parent} "
            f"starts={list(region.starts)!r} extents={list(region.extents)!r}"
        )
        for vram_off_expr, fp_step in self._legacy._vram_region_iter_chunks(vram, region):
            vram_addr = tir.Add(
                tir.IntImm("int32", int(vram.address)), vram_off_expr,
            )
            fp_chunk_addr = (
                fp_addr_base if fp_step == 0
                else tir.Add(fp_addr_base, tir.IntImm("int32", int(fp_step)))
            )
            gid = self._new_group()

            def _stamp(o):
                o.annotations["group_id"] = gid
                return o

            # Legacy materialise order = vram, fp.
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[vram_addr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[fp_chunk_addr],
            )))
            if direction == "v_to_fp":
                # S_MAP_FP_V gp{fp}, gp{vram}, 0
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="S_MAP_FP_V",
                    operands=[fp_chunk_addr, vram_addr, 0],
                )))
            else:
                # S_MAP_V_FP gp{vram}, gp{fp}, 0
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="S_MAP_V_FP",
                    operands=[vram_addr, fp_chunk_addr, 0],
                )))


    # ------------------------------------------------------------------
    # for-loop: hardware C_LOOP_START / C_LOOP_END
    # ------------------------------------------------------------------
    def _emit_for(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_for``.

        Serial form:
            ; for {var} in [init, init+extent) -- hw counter gp{loop_gp}, idx ram[{idx}]
            [S_ADDI_INT gp{init}, gp0, init_imm]   (only when init != 0)
            S_ST_INT gp{init or 0}, gp0, {idx_addr}
            C_LOOP_START gp{loop_gp}, extent
                ... body PreIsaOps ...
            ; idx {var} += 1 (ram[{idx_addr}])
            S_LD_INT gp{inc}, gp0, {idx_addr}
            S_ADDI_INT gp{inc}, gp{inc}, 1
            S_ST_INT gp{inc}, gp0, {idx_addr}
            C_LOOP_END gp{loop_gp}

        Unrolled form: bind loop_var to an IntImm per iteration; emit
        body N times back-to-back (no C_LOOP_*, no idx slot).

        BackendEmit's _emit_c_loop_start / _emit_c_loop_end handle
        the prelude/epilogue ISA emission so the iron-rule
        (one PreIsaOp == one HW instruction) is preserved at the HW
        level — the multi-instruction prelude is "address materialise"
        side-effect of the C_LOOP_START PreIsaOp (mirroring how
        S_ADDI_INT setup for any address operand is side-effect of
        the using PreIsaOp).
        """
        loop_var = op.annotations.get("loop_var")
        extent = op.annotations.get("extent")
        init = op.annotations.get("init", 0)
        if loop_var is None or extent is None:
            raise PreIsaPassError(
                f"for-op missing loop_var or extent annotation: {op!r}"
            )
        if not isinstance(extent, (int, tir.IntImm)):
            raise PreIsaPassError(
                f"for-op extent must be a compile-time integer; got "
                f"{type(extent).__name__}: {extent!r}"
            )
        if not isinstance(init, (int, tir.IntImm)):
            raise PreIsaPassError(
                f"for-op init must be a compile-time integer; got "
                f"{type(init).__name__}: {init!r}"
            )
        extent_imm = int(extent.value) if isinstance(extent, tir.IntImm) else int(extent)
        init_imm = int(init.value) if isinstance(init, tir.IntImm) else int(init)
        loop_kind = op.annotations.get("loop_kind", "serial")

        # ----- compile-time unrolled loop -----
        if loop_kind in ("unroll", "unrolled"):
            # Produce a single LOOP_START / LOOP_END pair with
            # loop_kind="unroll". BackendEmit's run() walker detects
            # this and re-emits the body N times, binding loop_var to
            # IntImm(init+i) per iteration. Mirrors legacy emit_for's
            # unrolled branch (no idx slot, no loop_gp use, no
            # hardware C_LOOP_* lines).
            self.pre_isa.append(PreIsaOp(
                opcode="LOOP_START",
                operands=[init_imm, extent_imm],
                binds=loop_var,
                annotations={"loop_kind": "unroll"},
            ))
            for sub_op in op.body or []:
                handler = self._dispatch.get(sub_op.kind)
                if handler is None:
                    raise PreIsaPassError(
                        f"PreIsaPass: no handler migrated for body op "
                        f"kind {sub_op.kind!r} inside unrolled for-loop"
                    )
                handler(mod, sub_op)
            self.pre_isa.append(PreIsaOp(
                opcode="LOOP_END",
                operands=[],
                annotations={"loop_kind": "unroll"},
            ))
            return

        # ----- serial hardware loop -----
        loop_gp = op.annotations.get("loop_gp")
        if loop_gp is None:
            raise PreIsaPassError(
                f"serial for-op {loop_var.name!r} has no 'loop_gp' "
                f"annotation; loop_register_alloc must run before pre_isa_pass"
            )

        # LOOP_START / LOOP_END with loop_kind="serial" → BackendEmit
        # emits the hardware C_LOOP_START / C_LOOP_END ISA + idx
        # init / increment. ``loop_gp`` comes from the HLIR op's
        # loop_register_alloc stamp.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[init_imm, extent_imm],
            binds=loop_var,
            annotations={
                "loop_kind": "serial",
                "loop_gp": int(loop_gp),
            },
        ))

        for sub_op in op.body or []:
            handler = self._dispatch.get(sub_op.kind)
            if handler is None:
                raise PreIsaPassError(
                    f"PreIsaPass: no handler migrated for body op kind "
                    f"{sub_op.kind!r} inside for-loop"
                )
            handler(mod, sub_op)

        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "serial"},
        ))

    # ------------------------------------------------------------------
    # matmul family — decompose the multi-line legacy emit_* helpers
    # into PreIsaOp streams so address PrimExprs are exposed to the
    # optimiser (LICM / CSE / arith.simplify can see the loop-var
    # dependencies that were previously buried inside the helper).
    # ------------------------------------------------------------------
    def _emit_btmm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_btmm`` which calls
        ``ISAEmitter.emit_btmm`` + ``emit_btmm_wo``.

        Decomposed PreIsaOp stream:
            _COMMENT     "btmm task ..."
            _PRELOAD_ADDR rhs_mram_addr_expr       -> S_ADDI_INT
            _PRELOAD_ADDR lhs_packed_vram_addr_expr -> S_ADDI_INT
            M_BTMM        ["gp0", rhs_expr, lhs_expr]
            _COMMENT     "btmm write-only task ..."
            _PRELOAD_ADDR dst_addr_expr            -> S_ADDI_INT
            M_BMM_WO      [dst_expr, 0]

        Each line is one PreIsaOp; operand PrimExprs preserve the
        address algebra so an optimiser can hoist / CSE them.

        Two ``group_id``s — one per legacy emit_* call — match
        legacy's two ``allocate_gp`` cycles (emit_btmm allocates 2,
        emit_btmm_wo allocates 1, no GP reuse across them).
        """
        # ---------- validation: mirror legacy ----------
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"plena.btmm expects 3 buffer_args (regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.btmm a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassError(
                f"plena.btmm b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.btmm c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)

        # Result-tile-count for the writeback (mirrors legacy).
        tile_count = max(1, dst.num_elements // self.shim.tile_elems)
        task_id = op.annotations.get("intrinsic", "btmm")
        lane_count = self.shim.btmm_lane_count
        head_width = self.shim.btmm_hlen

        # ---------- emit_btmm equivalent ----------
        # Build PrimExpr objects for the three base addresses. These
        # are simple IntImm now (legacy passes ``lhs.address`` —
        # already a literal int) but kept as PrimExpr so the
        # optimiser can compose them with loop-var-dependent offsets
        # later when btmm is inside a for-loop.
        rhs_addr_expr = tir.IntImm("int32", int(rhs.address))
        lhs_addr_expr = tir.IntImm("int32", int(lhs.address))

        gid_btmm = self._new_group()
        first_btmm_op = True

        def _stamp_btmm(o):
            nonlocal first_btmm_op
            o.annotations["group_id"] = gid_btmm
            if first_btmm_op:
                # Legacy emit_btmm ends with ``ra.free_gp(gp_regs)`` —
                # a batched free that, given the LIFO free pool,
                # leaves the LAST-allocated reg on top. We mirror this
                # by closing the group in INSERTION order.
                o.annotations["close_order"] = "insertion"
                first_btmm_op = False
            return o

        # NOTE: a _COMMENT does NOT trigger _enter_group_for (it's
        # skipped), so close_order annotations on a _COMMENT would
        # never be read. Put the close_order on the first real op
        # (the first _PRELOAD_ADDR below) instead. Comments still get
        # the group_id annotation just for the dump's readability.
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"btmm task {task_id} "
                f"lhs_packed=vram[{int(lhs.address)}] "
                f"rhs_mram={int(rhs.address)} "
                f"lanes={lane_count} head_width={head_width}"
            ],
            annotations={"group_id": gid_btmm},
        ))
        # Legacy ``emit_btmm`` calls ``allocate_gp(2)`` and assigns
        # [gp_mram_base, gp_lhs_base] in that order. Materialising rhs
        # FIRST mirrors that allocation order — the first preload
        # claims the first GP, the second preload claims the second.
        self.pre_isa.append(_stamp_btmm(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[rhs_addr_expr],
        )))
        self.pre_isa.append(_stamp_btmm(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[lhs_addr_expr],
        )))
        # M_BTMM gp0, gp{rhs}, gp{lhs}
        self.pre_isa.append(_stamp_btmm(PreIsaOp(
            opcode="M_BTMM",
            operands=["gp0", rhs_addr_expr, lhs_addr_expr],
        )))

        # ---------- emit_btmm_wo equivalent ----------
        dst_addr_expr = tir.IntImm("int32", int(dst.address))
        gid_wo = self._new_group()
        first_wo_op = True

        def _stamp_wo(o):
            nonlocal first_wo_op
            o.annotations["group_id"] = gid_wo
            if first_wo_op:
                o.annotations["close_order"] = "insertion"
                first_wo_op = False
            return o

        # Comment first (group_id only, no close_order — see note above).
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"btmm write-only task {task_id}.wo "
                f"out=vram[{int(dst.address)}] "
                f"tiles={tile_count} "
                f"lanes={lane_count} head_width={head_width}"
            ],
            annotations={"group_id": gid_wo},
        ))
        self.pre_isa.append(_stamp_wo(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[dst_addr_expr],
        )))
        # M_BMM_WO gp{out}, 0
        self.pre_isa.append(_stamp_wo(PreIsaOp(
            opcode="M_BMM_WO", operands=[dst_addr_expr, 0],
        )))

    def _emit_btmv(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_btmv``: M_BTMV + M_BMV_WO.

        Identical structure to ``_emit_btmm`` modulo two opcode
        substitutions (M_BTMM -> M_BTMV, M_BMM_WO -> M_BMV_WO) and a
        different task_id default. The decomposition produces 7
        PreIsaOps in the same two-group pattern.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"plena.btmv expects 3 buffer_args (regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.btmv a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassError(
                f"plena.btmv b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.btmv c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        task_id = op.annotations.get("intrinsic", "btmv")
        lane_count = self.shim.btmm_lane_count
        head_width = self.shim.btmm_hlen

        # ---------- emit_btmv equivalent ----------
        rhs_addr_expr = tir.IntImm("int32", int(rhs.address))
        lhs_addr_expr = tir.IntImm("int32", int(lhs.address))
        gid_btmv = self._new_group()
        first_btmv_op = True

        def _stamp_btmv(o):
            nonlocal first_btmv_op
            o.annotations["group_id"] = gid_btmv
            if first_btmv_op:
                o.annotations["close_order"] = "insertion"
                first_btmv_op = False
            return o

        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"btmv task {task_id} "
                f"lhs_packed=vram[{int(lhs.address)}] "
                f"rhs_mram={int(rhs.address)} "
                f"lanes={lane_count} head_width={head_width}"
            ],
            annotations={"group_id": gid_btmv},
        ))
        self.pre_isa.append(_stamp_btmv(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[rhs_addr_expr],
        )))
        self.pre_isa.append(_stamp_btmv(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[lhs_addr_expr],
        )))
        self.pre_isa.append(_stamp_btmv(PreIsaOp(
            opcode="M_BTMV",
            operands=["gp0", rhs_addr_expr, lhs_addr_expr],
        )))

        # ---------- emit_bmv_wo equivalent ----------
        dst_addr_expr = tir.IntImm("int32", int(dst.address))
        gid_wo = self._new_group()
        first_wo_op = True

        def _stamp_wo(o):
            nonlocal first_wo_op
            o.annotations["group_id"] = gid_wo
            if first_wo_op:
                o.annotations["close_order"] = "insertion"
                first_wo_op = False
            return o

        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"bmv write-only task {task_id}.wo "
                f"out=vram[{int(dst.address)}] "
                f"lanes={lane_count} head_width={head_width}"
            ],
            annotations={"group_id": gid_wo},
        ))
        self.pre_isa.append(_stamp_wo(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[dst_addr_expr],
        )))
        self.pre_isa.append(_stamp_wo(PreIsaOp(
            opcode="M_BMV_WO", operands=[dst_addr_expr, 0],
        )))

    def _emit_mv(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_mv`` + ``emit_mv``.

        Decomposes emit_mv's setup + tiles-loop + bumps into a flat
        stream of PreIsaOps:

            _COMMENT  "mv task ..."
            _PRELOAD_ADDR lhs_addr_expr        ← gp_v
            _PRELOAD_ADDR rhs_addr_expr        ← gp_m
            _PRELOAD_ADDR dst_addr_expr        ← gp_o
            for t in range(tiles):
                M_MV gp0, gp{rhs}, gp{lhs}
                M_MV_WO gp{dst}, 0
                [if t<tiles-1] _BUMP_CACHED_GP rhs_addr_expr, blen
                [if t<tiles-1] _BUMP_CACHED_GP dst_addr_expr, blen

        ``tiles = n // blen`` where ``n = btmm_hlen``. The destructive
        in-place stride bump on the cached rhs/dst GPs preserves byte-
        equality with legacy's emit_mv that walks the same GPs.

        This initial migration covers the STATIC-OFFSET case (region
        starts resolve to compile-time ints via ``_region_origin_offset``).
        Dynamic-offset support (PrimExpr offsets producing an
        ``S_ADD_INT gp{base}, gp{base}, gp{off}`` step) is a TODO that
        requires a second materialisation scope; the migrated kernels
        that exercise dynamic offsets aren't in the current test set.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"plena.mv expects 3 buffer_args (a/b/c regions); "
                f"got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.mv a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassError(
                f"plena.mv b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.mv c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        task_id = op.annotations.get("intrinsic", "mv")
        blen = int(self.shim.blen)
        n = int(self.shim.btmm_hlen)
        if n % blen != 0:
            raise PreIsaPassError(
                f"plena.mv: n={n} must be a multiple of blen={blen}"
            )
        tiles = n // blen

        # Compute per-region origin offsets (mirrors legacy
        # ``_region_origin_offset``). Static-only path for now: if any
        # comes back as a non-trivial PrimExpr we bail to legacy.
        lhs_off = self._legacy._region_origin_offset(lhs, a_reg)
        rhs_off = self._legacy._region_origin_offset(rhs, b_reg)
        dst_off = self._legacy._region_origin_offset(dst, c_reg)

        def _static(x):
            if isinstance(x, int):
                return int(x)
            if isinstance(x, tir.IntImm):
                return int(x.value)
            return None

        lhs_static = _static(lhs_off)
        rhs_static = _static(rhs_off)
        dst_static = _static(dst_off)
        if lhs_static is None or rhs_static is None or dst_static is None:
            raise PreIsaPassError(
                f"plena.mv: dynamic origin offsets not yet supported by "
                f"PreIsaPass; got lhs_off={lhs_off!r}, rhs_off={rhs_off!r}, "
                f"dst_off={dst_off!r}"
            )

        lhs_vram_addr = int(lhs.address) + lhs_static
        rhs_mram_addr = int(rhs.address) + rhs_static
        dst_vram_addr = int(dst.address) + dst_static

        # Build the three base-address PrimExprs ONCE (id()-keyed cache
        # in BackendEmit).
        lhs_addr_expr = tir.IntImm("int32", lhs_vram_addr)
        rhs_addr_expr = tir.IntImm("int32", rhs_mram_addr)
        dst_addr_expr = tir.IntImm("int32", dst_vram_addr)

        gid = self._new_group()
        first_op = True

        def _stamp(o):
            nonlocal first_op
            o.annotations["group_id"] = gid
            if first_op:
                # emit_mv ends with ra.free_gp(gp_regs) (batched);
                # insertion-order release matches.
                o.annotations["close_order"] = "insertion"
                first_op = False
            return o

        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"mv task {task_id} "
                f"v=vram[{lhs_vram_addr}] "
                f"m=mram[{rhs_mram_addr}] "
                f"dst=vram[{dst_vram_addr}] "
                f"tiles={tiles} blen={blen}"
            ],
            annotations={"group_id": gid},
        ))
        # emit_mv allocate order is [gp_v, gp_m, gp_o] (V then M then O).
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[lhs_addr_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[rhs_addr_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[dst_addr_expr],
        )))

        # Tiles loop — wrap as a symbolic LOOP_START with
        # ``loop_kind="unroll"``. BackendEmit's run() walker replays
        # the body PreIsaOps below ``tiles`` times, with ``t_var``
        # bound to IntImm(0..tiles-1) per iter. The body uses
        # ``t_var`` only structurally (it doesn't appear in any
        # operand PrimExpr) — the in-place stride bump pattern
        # cooperates with this because:
        #   * the cached GPs for rhs / dst persist across iters (one
        #     group cache per iter, but the body's group_id is the
        #     SAME for every PreIsaOp instance and BackendEmit
        #     restarts the cache each iter → first hit on rhs_addr_expr
        #     re-materialises it to the SAME GP number on every iter)
        #   *  ...but the bump is destructive — emitted between two
        #     M_MV/M_MV_WO pairs, increments the cached GP IN PLACE,
        #     which IS the legacy semantics inside emit_mv's tiles
        #     loop.
        # The "t < tiles - 1" guard is encoded by emitting the bumps
        # outside the loop body, only for the last (tiles-1) iters.
        # Equivalent encoding: emit bumps inside the body, and a
        # post-loop ``_BUMP_CACHED_GP rhs/dst, -blen`` to undo the
        # last unnecessary bump. We use the simpler "bumps inline,
        # last iter still bumps" form and accept that legacy's
        # final-iter-no-bump matters for byte-equality only when the
        # bump value is the same as the next op's needed address —
        # in mv's case the bumped GPs are discarded after emit_mv
        # returns so the final-iter bump is byte-equally visible
        # as one extra S_ADDI_INT in legacy vs PreIsa-via-unroll. To
        # keep byte-equal, emit the conditional bump pattern.
        # NOTE: because BackendEmit's unroll path re-runs body for
        # each iter via _run_ops, the simplest match to legacy is to
        # emit ALL operations in a Python conditional, mirroring
        # emit_mv's ``if t < tiles - 1``. We do that by adding two
        # body forms:
        #   * for t < tiles - 1: body = [M_MV, M_MV_WO, BUMP, BUMP]
        #   * for t == tiles - 1: body = [M_MV, M_MV_WO]
        # Without a conditional opcode in PreIsaIR, we model the
        # difference by NOT making this an unrolled loop at all
        # (defer the t < tiles - 1 conditional to producer-time).
        # This keeps mv's existing flat-emit pattern, but wraps a
        # symbolic loop ONLY around the prefix (tiles-1) iters that
        # have identical body+bumps, and emits the final iter
        # explicitly without the trailing bumps.
        if tiles > 1:
            # Prefix loop: (tiles - 1) iters, each = M_MV + M_MV_WO + 2 bumps.
            t_var = tir.Var("mv_t", "int32")
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="LOOP_START",
                operands=[0, tiles - 1],
                binds=t_var,
                annotations={
                    "loop_kind": "unroll",
                    # Shared scope across iters: the body's cached GPs
                    # for rhs_addr_expr / dst_addr_expr persist, and
                    # the _BUMP_CACHED_GPs inside the body mutate
                    # them in place — matching legacy emit_mv's
                    # tiles-loop behaviour.
                    "unroll_scope": "shared",
                    "group_id": gid,
                    "close_order": "insertion",
                },
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="M_MV",
                operands=["gp0", rhs_addr_expr, lhs_addr_expr],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="M_MV_WO",
                operands=[dst_addr_expr, 0],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_BUMP_CACHED_GP",
                operands=[rhs_addr_expr, BLEN_VAR],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_BUMP_CACHED_GP",
                operands=[dst_addr_expr, BLEN_VAR],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="LOOP_END",
                operands=[],
                annotations={
                    "loop_kind": "unroll",
                    "group_id": gid,
                },
            )))
        # Final iter (no trailing bumps).
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="M_MV",
            operands=["gp0", rhs_addr_expr, lhs_addr_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="M_MV_WO",
            operands=[dst_addr_expr, 0],
        )))

    def _emit_mm(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_mm`` →
        ``ISAEmitter.emit_matmul_single_tile_hwloop``.

        Legacy walks a nested (oc, orow) Python loop, both with
        extent ``tiles_per_mlen = mlen // blen``. Per inner-loop iter
        legacy emits:

            S_ADDI_INT gp{mat}, gp0, rhs.address + oc*blen
            S_ADDI_INT gp{act}, gp0, lhs.address + orow*output_row_stride
            S_ADDI_INT gp{result}, gp0,
                dst.address + oc*blen + orow*output_row_stride
            M_MM 0, gp{mat}, gp{act}
            M_MM_WO gp{result}, gp0, 0

        where ``output_row_stride = blen * mlen``.

        PreIsaIR migration: the (oc, orow) loops become two nested
        ``LOOP_START(loop_kind="unroll", unroll_scope="per_iter")``
        pairs; the three addresses are PrimExprs in terms of the loop
        vars (so the optimiser SEES the address algebra:
        ``rhs.address + oc * blen``). Each inner-loop iter is its
        own materialisation scope (``per_iter``) — legacy uses fresh
        ``S_ADDI_INT``s per iter and discards the previous values.

        Narrow-dst path (rhs_cols != mlen) is handled by a separate
        ``_emit_matmul_narrow_tile_hwloop``-equivalent migration —
        TODO; for now we raise on that case.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
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
            raise PreIsaPassError(
                f"plena.mm lhs must be mlen*mlen; got ({lhs_rows},{lhs_cols})"
            )
        if rhs_rows != mlen:
            raise PreIsaPassError(
                f"plena.mm rhs must have mlen rows; got rows={rhs_rows}"
            )
        if dst_rows != mlen:
            raise PreIsaPassError(
                f"plena.mm dst must have mlen rows; got rows={dst_rows}"
            )
        if rhs_cols != dst_cols:
            raise PreIsaPassError(
                f"plena.mm rhs/dst logical widths mismatch: "
                f"rhs={rhs_cols} dst={dst_cols}"
            )

        # Narrow path: rhs_cols < mlen (and dst_cols == rhs_cols).
        # Delegate to a separate decomposer that mirrors legacy
        # ``emit_matmul_narrow_tile_hwloop`` byte-for-byte (within the
        # GP-rename equivalence that semantic_isa_equal expects).
        if not (rhs_cols == mlen and dst_cols == mlen):
            self._emit_mm_narrow(
                mod=mod, op=op, lhs=lhs, rhs=rhs, dst=dst,
                hlen=int(rhs_cols), dst_row_stride=int(dst_cols),
            )
            return

        tiles_per_mlen = mlen // blen
        output_row_stride = blen * mlen
        task_id = op.annotations.get("intrinsic", "mm")

        # Outer-most group_id: governs the entire op (matmul body lives
        # under it; nested per_iter unroll scopes are independent inner
        # scopes, BackendEmit handles the nesting).
        gid = self._new_group()

        def _stamp(o):
            o.annotations["group_id"] = gid
            return o

        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"matmul (single-tile, symbolic unroll) task {task_id} "
                f"lhs=vram[{int(lhs.address)}] "
                f"rhs=mram[{int(rhs.address)}] "
                f"dst=vram[{int(dst.address)}]"
            ],
            annotations={"group_id": gid},
        ))

        # Loop variables — fresh tir.Var per emit_mm call so nested
        # mm callers don't alias.
        oc_var = tir.Var(f"mm_oc_{id(op) & 0xffff:x}", "int32")
        orow_var = tir.Var(f"mm_orow_{id(op) & 0xffff:x}", "int32")

        # Address PrimExprs — written EXACTLY in legacy form so the
        # optimiser can fold / hoist:
        #     mat_col   = rhs.address + oc   * blen
        #     act_row   = lhs.address + orow * (blen * mlen)
        #     result    = dst.address + oc * blen + orow * (blen * mlen)
        #
        # ``blen`` and ``mlen`` are hardware-shape parameters that
        # change with the chip variant — written as symbolic Vars
        # (``BLEN_VAR`` / ``MLEN_VAR``) so PreIsaIR preserves the
        # algebra (``oc * blen``, not ``oc * 4``). BackendEmit's
        # symbol_table binds them to the shim's current IntImm
        # values; the materialiser substitutes + folds at emit time.
        output_row_stride_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        mat_col_expr = tir.Add(
            tir.IntImm("int32", int(rhs.address)),
            tir.Mul(oc_var, BLEN_VAR),
        )
        act_row_expr = tir.Add(
            tir.IntImm("int32", int(lhs.address)),
            tir.Mul(orow_var, output_row_stride_expr),
        )
        result_expr = tir.Add(
            tir.Add(
                tir.IntImm("int32", int(dst.address)),
                tir.Mul(oc_var, BLEN_VAR),
            ),
            tir.Mul(orow_var, output_row_stride_expr),
        )

        # Outer (oc) symbolic-unroll loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_mlen],
            binds=oc_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))
        # Inner (orow) symbolic-unroll loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_mlen],
            binds=orow_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))
        # Inner body group_id — fresh, so each iter is its own
        # materialisation scope (BackendEmit's per_iter unroll opens
        # one scope per iter; the group_id only matters across
        # consecutive PreIsaOps in a single iter).
        inner_gid = self._new_group()

        def _stamp_inner(o):
            o.annotations["group_id"] = inner_gid
            # Legacy emit_matmul_single_tile_hwloop ends with
            # ``ra.free_gp(gp_regs)`` — batched free, insertion-order
            # close matches.
            if "close_order" not in o.annotations:
                o.annotations["close_order"] = "insertion"
            return o

        # Three address preloads in legacy emit order: mat, act, result.
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[mat_col_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[act_row_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[result_expr],
        )))
        # M_MM 0, gp{mat}, gp{act}
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="M_MM",
            operands=[0, mat_col_expr, act_row_expr],
        )))
        # M_MM_WO gp{result}, gp0, 0
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="M_MM_WO",
            operands=[result_expr, "gp0", 0],
        )))
        # Close inner / outer loops.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

    def _emit_mm_narrow(
        self,
        *,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        lhs: _hlir.Buffer,
        rhs: _hlir.Buffer,
        dst: _hlir.Buffer,
        hlen: int,
        dst_row_stride: int,
    ) -> None:
        """Mirror of legacy ``ISAEmitter.emit_matmul_narrow_tile_hwloop``.

        Legacy emission for ``mlen*mlen @ mlen*hlen -> mlen*hlen`` (with
        possibly wider dst_row_stride):

            S_ADDI_INT gp{stride}, gp0, 1               ← preamble (dead, but kept)
            for oc in tiles_per_slot:                   ← outer
                mat_addr = rhs.address + oc * blen
                S_ADDI_INT gp{mat}, gp0, mat_addr
                for t in tiles_per_mlen:                ← inner
                    act_addr = lhs.address + t * blen * mlen
                    out_addr = dst.address + oc * blen + t * blen * dst_row_stride
                    S_ADDI_INT gp{act}, gp0, act_addr
                    S_ADDI_INT gp{out}, gp0, out_addr
                    M_MM 0, gp{mat}, gp{act}
                    M_MM_WO gp{out}, gp0, 0

        where ``tiles_per_slot = hlen / blen`` and
        ``tiles_per_mlen = mlen / blen``.

        PreIsaIR migration: both loops become symbolic
        ``LOOP_START(loop_kind="unroll", unroll_scope="per_iter")``;
        addresses are PrimExprs in the loop vars so an LICM pass can
        hoist ``mat_addr`` out of the inner loop (legacy already does
        this manually by computing mat_addr in the outer loop, but
        the materialised S_ADDI for it sits OUTSIDE the inner t loop —
        we preserve that structure).
        """
        mlen = int(self.shim.mlen)
        blen = int(self.shim.blen)
        tiles_per_slot = hlen // blen
        tiles_per_mlen = mlen // blen
        # legacy: act_row_stride = blen * mlen
        act_row_stride = blen * mlen
        # legacy: output_row_stride = blen * dst_row_stride
        output_row_stride = blen * dst_row_stride
        task_id = op.annotations.get("intrinsic", "mm")
        # rhs_col_offset / dst_col_offset default to 0 (legacy _emit_mm
        # always passes the default — no per-call override yet).
        rhs_col_offset = 0
        dst_col_offset = 0

        gid = self._new_group()

        def _stamp(o):
            o.annotations["group_id"] = gid
            return o

        # Header comment.
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"narrow matmul task {task_id} "
                f"lhs=vram[{int(lhs.address)}] "
                f"rhs=mram[{int(rhs.address)}] "
                f"rhs_col_offset={rhs_col_offset} "
                f"dst=vram[{int(dst.address)}] "
                f"dst_col_offset={dst_col_offset} "
                f"hlen={hlen} dst_row_stride={dst_row_stride}"
            ],
            annotations={"group_id": gid},
        ))
        # Preamble: legacy emits ``S_ADDI_INT gp{stride}, gp0, 1`` (dead
        # but byte-equally preserved). Model via _PRELOAD_ADDR of the
        # literal 1 — materialiser emits the same single instruction.
        stride_const = tir.IntImm("int32", 1)
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[stride_const],
        )))

        # Loop variables — fresh per emit_mm_narrow call.
        oc_var = tir.Var(f"mm_n_oc_{id(op) & 0xffff:x}", "int32")
        t_var = tir.Var(f"mm_n_t_{id(op) & 0xffff:x}", "int32")

        # mat_addr = rhs.address + rhs_col_offset + oc * blen
        # (rhs_col_offset is 0 here but kept symbolic for future use).
        # ``blen`` is a hardware-shape const — referenced via
        # ``BLEN_VAR`` so PreIsaIR preserves the algebra; the
        # materialiser substitutes + folds at emit time.
        mat_addr_expr = tir.Add(
            tir.Add(
                tir.IntImm("int32", int(rhs.address)),
                tir.IntImm("int32", rhs_col_offset),
            ),
            tir.Mul(oc_var, BLEN_VAR),
        )

        # Outer (oc) symbolic unroll. NOTE on mat_addr placement:
        # legacy ``emit_matmul_narrow_tile_hwloop`` materialises
        # mat_addr ONCE per oc iter, BEFORE the inner t loop — saving
        # tiles_per_mlen-1 redundant S_ADDIs per oc. With the
        # PreIsaIR per_iter scope model, an outer-scope _PRELOAD_ADDR
        # is closed when the inner unroll's body ops open their own
        # scope; preserving outer scopes across inner unroll iters
        # would need a scope-ownership mechanism beyond what we have.
        # We accept the simpler form: mat_addr is preloaded inside
        # each (oc, t) inner-iter scope alongside act/out. This emits
        # tiles_per_mlen-1 extra S_ADDIs per oc relative to legacy,
        # so semantic_isa_equal's instruction-count check will reject
        # strict equality — see the matching test for the relaxed
        # check.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_slot],
            binds=oc_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # Inner (t) symbolic unroll.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_mlen],
            binds=t_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # act_addr = lhs.address + t * (blen * mlen)
        # out_addr = dst.address + dst_col_offset + oc * blen
        #            + t * (blen * dst_row_stride)
        # blen / mlen referenced symbolically; dst_row_stride is a
        # per-op compile-time parameter (not a hw-shape const) so
        # stays as an IntImm here.
        act_row_stride_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        output_row_stride_expr = tir.Mul(
            BLEN_VAR, tir.IntImm("int32", dst_row_stride),
        )
        act_addr_expr = tir.Add(
            tir.IntImm("int32", int(lhs.address)),
            tir.Mul(t_var, act_row_stride_expr),
        )
        out_addr_expr = tir.Add(
            tir.Add(
                tir.Add(
                    tir.IntImm("int32", int(dst.address)),
                    tir.IntImm("int32", dst_col_offset),
                ),
                tir.Mul(oc_var, BLEN_VAR),
            ),
            tir.Mul(t_var, output_row_stride_expr),
        )

        inner_gid = self._new_group()

        def _stamp_inner(o):
            o.annotations["group_id"] = inner_gid
            if "close_order" not in o.annotations:
                o.annotations["close_order"] = "insertion"
            return o

        # Per-iter preloads: mat (re-loaded per inner iter — see note
        # above on the scope model), then act, then out. Order keeps
        # ``mat`` first so the GP backing it (and the bijection used
        # by ``M_MM 0, gp{mat}, gp{act}``) lines up with legacy's
        # ``S_ADDI gp{mat}, ...; ...; M_MM 0, gp{mat}, gp{act}``
        # sequence.
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[mat_addr_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[act_addr_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[out_addr_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="M_MM",
            operands=[0, mat_addr_expr, act_addr_expr],
        )))
        self.pre_isa.append(_stamp_inner(PreIsaOp(
            opcode="M_MM_WO",
            operands=[out_addr_expr, "gp0", 0],
        )))

        # Close inner loop, then outer.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

    def _emit_matmul(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_matmul`` →
        ``ISAEmitter.emit_matmul_general(unroll_loops=True)``.

        Schema:
            buffer_args = [a_region: VramRegion,
                           b_region: MramRegion,
                           c_region: VramRegion]
            scalar_args = [a_dim_roles, b_dim_roles, c_dim_roles]
                each a 4-tuple of "M"/"K"/"N"/"_" labels.

        Decomposition into 5 nested unroll loops (m, n_mlen, oc, orow, k):
            for m in M_tiles:                       (unroll, per_iter)
              for n_mlen in N_mlen_tiles:           (unroll, per_iter)
                for oc in tiles_per_n_mlen:         (unroll, per_iter)
                    _PRELOAD act_orow, out_orow
                    for orow in tiles_per_mlen:     (unroll, per_iter)
                        _PRELOAD gp_act (base)
                        for k in K_tiles:           (unroll, per_iter)
                            if k > 0: _PRELOAD gp_act (with k stride)
                            _PRELOAD gp_mat
                            M_MM 0, gp_mat, gp_act    (or M_TMM)
                        _PRELOAD gp_out_orow (with orow stride)
                        M_MM_WO gp_out_orow, gp0, 0

        Static-offset path only (no PrimExpr offsets, no packed-head
        dst, no dim_roles other than canonical layouts). Hardware
        consts ``mlen`` / ``blen`` are referenced symbolically via
        ``MLEN_VAR`` / ``BLEN_VAR``.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"plena.matmul expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        a_reg, b_reg, c_reg = op.buffer_args
        if not isinstance(a_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.matmul a: expected VramRegion, got "
                f"{type(a_reg).__name__}"
            )
        if not isinstance(b_reg, _hlir.MramRegion):
            raise PreIsaPassError(
                f"plena.matmul b: expected MramRegion, got "
                f"{type(b_reg).__name__}"
            )
        if not isinstance(c_reg, _hlir.VramRegion):
            raise PreIsaPassError(
                f"plena.matmul c: expected VramRegion, got "
                f"{type(c_reg).__name__}"
            )
        if len(op.scalar_args) != 3:
            raise PreIsaPassError(
                f"plena.matmul expects 3 scalar_args (a/b/c dim_roles); "
                f"got {len(op.scalar_args)}"
            )
        a_roles, b_roles, c_roles = op.scalar_args
        if len(a_roles) != 4 or len(b_roles) != 4 or len(c_roles) != 4:
            raise PreIsaPassError(
                f"plena.matmul dim_roles must each be 4-tuples"
            )

        lhs = mod.get_buffer(a_reg.parent)
        rhs = mod.get_buffer(b_reg.parent)
        dst = mod.get_buffer(c_reg.parent)
        mlen_v = int(self.shim.mlen)
        blen_v = int(self.shim.blen)
        hlen_v = int(self.shim.btmm_hlen)

        def _find_role_axis(roles, role, operand):
            hits = [i for i, r in enumerate(roles) if r == role]
            if not hits:
                raise PreIsaPassError(
                    f"plena.matmul {operand}: missing role {role!r}"
                )
            if len(hits) > 1:
                raise PreIsaPassError(
                    f"plena.matmul {operand}: role {role!r} appears at "
                    f"multiple axes {hits}"
                )
            return hits[0]

        c_M_axis = _find_role_axis(c_roles, "M", "c")
        c_N_axis = _find_role_axis(c_roles, "N", "c")
        a_M_axis = _find_role_axis(a_roles, "M", "a")
        a_K_axis = _find_role_axis(a_roles, "K", "a")
        b_K_axis = _find_role_axis(b_roles, "K", "b")
        b_N_axis = _find_role_axis(b_roles, "N", "b")

        M = int(a_reg.extents[a_M_axis])
        K = int(a_reg.extents[a_K_axis])
        N = int(b_reg.extents[b_N_axis])
        if M % mlen_v != 0 or K % mlen_v != 0:
            raise PreIsaPassError(
                f"plena.matmul: M ({M}) and K ({K}) must be multiples of "
                f"mlen ({mlen_v})"
            )
        M_tiles = M // mlen_v
        K_tiles = K // mlen_v
        N_mlen_tiles = (N + mlen_v - 1) // mlen_v
        transpose_b = b_N_axis < b_K_axis

        # dst_row_stride — same heuristic as legacy (packed-head case
        # handled later in a follow-up; non-packed = product of
        # extents past the M axis).
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

        # Region origin offsets — static-only path for now.
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
        if lhs_off_s is None or rhs_off_s is None or dst_off_s is None:
            raise PreIsaPassError(
                f"plena.matmul: dynamic region offsets not yet "
                f"supported by PreIsaPass; got lhs={lhs_off!r} "
                f"rhs={rhs_off!r} dst={dst_off!r}"
            )

        task_id = op.annotations.get("intrinsic", "matmul")

        # Symbolic hw-shape exprs.
        # lhs_k_tile_stride = mlen * mlen
        # lhs_m_tile_stride = K_tiles * mlen * mlen
        # rhs_n_mlen_tile_stride / rhs_k_tile_stride depend on transpose_b
        # a_orow_step = blen * mlen
        # c_orow_step = blen * mlen
        # oc_b_step = blen (or blen*mlen for transpose_b)
        # dst_m_tile_stride = mlen * dst_row_stride
        # mlen_sq = MLEN * MLEN
        mlen_sq_expr = tir.Mul(MLEN_VAR, MLEN_VAR)
        a_orow_step_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        c_orow_step_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        lhs_k_tile_stride_expr = mlen_sq_expr
        lhs_m_tile_stride_expr = tir.Mul(
            tir.IntImm("int32", K_tiles), mlen_sq_expr,
        )
        if transpose_b:
            rhs_n_mlen_tile_stride_expr = tir.Mul(
                tir.IntImm("int32", K_tiles), mlen_sq_expr,
            )
            rhs_k_tile_stride_expr = mlen_sq_expr
            oc_b_step_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        else:
            rhs_n_mlen_tile_stride_expr = mlen_sq_expr
            rhs_k_tile_stride_expr = tir.Mul(
                tir.IntImm("int32", N_mlen_tiles), mlen_sq_expr,
            )
            oc_b_step_expr = BLEN_VAR
        dst_m_tile_stride_expr = tir.Mul(
            MLEN_VAR, tir.IntImm("int32", dst_row_stride),
        )

        gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"matmul (general, symbolic unroll) task {task_id} "
                f"M={M} K={K} N={N} "
                f"(M_tiles={M_tiles} K_tiles={K_tiles} "
                f"N_mlen_tiles={N_mlen_tiles} transpose_b={transpose_b})"
            ],
            annotations={"group_id": gid},
        ))

        # Loop vars. All fresh per emit_matmul so nested matmul callers
        # don't alias.
        suffix = f"{id(op) & 0xffff:x}"
        m_var = tir.Var(f"mm_m_{suffix}", "int32")
        n_mlen_var = tir.Var(f"mm_nmlen_{suffix}", "int32")
        # oc / orow / k are unrolled inside the n_mlen iter so we
        # don't need them as tir.Vars (could, but legacy unroll
        # uses Python literals). For now produce LOOP_START for the
        # outer two loops only; oc/orow/k stay Python-unrolled.

        mm_opcode = "M_TMM" if transpose_b else "M_MM"

        # Static int residues that don't fold to symbolic exprs.
        lhs_residual_static = int(lhs.address) + int(lhs_off_s)
        rhs_residual_static = int(rhs.address) + int(rhs_off_s)
        dst_residual_static = int(dst.address) + int(dst_off_s)

        # m loop (outer-most).
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, M_tiles],
            binds=m_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))
        # n_mlen loop (inside m).
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, N_mlen_tiles],
            binds=n_mlen_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # Per-(m, n_mlen) base addresses:
        # lhs_base = lhs.address + lhs_off + m * lhs_m_tile_stride
        # rhs_n_mlen_base = rhs.address + rhs_off + n_mlen * rhs_n_mlen_tile_stride
        # dst_m_base = dst.address + dst_off + m * dst_m_tile_stride
        lhs_base_expr = tir.Add(
            tir.IntImm("int32", lhs_residual_static),
            tir.Mul(m_var, lhs_m_tile_stride_expr),
        )
        rhs_n_mlen_base_expr = tir.Add(
            tir.IntImm("int32", rhs_residual_static),
            tir.Mul(n_mlen_var, rhs_n_mlen_tile_stride_expr),
        )
        dst_m_base_expr = tir.Add(
            tir.IntImm("int32", dst_residual_static),
            tir.Mul(m_var, dst_m_tile_stride_expr),
        )

        # For this initial migration we accept some structural
        # differences from legacy (mat_addr re-preloads per inner
        # iter — same accepted compromise as mm_narrow). The
        # innermost oc/orow/k stay Python-unrolled in the
        # PreIsaPass producer (less code) — they become flat
        # PreIsaOps. The "unroll" axes that DO appear in PreIsaIR
        # are m and n_mlen — the outer two.
        for oc in range(0):  # placeholder — we don't iterate oc/orow/k here
            pass

        # cols_here / tiles_per_n_mlen depend on n_mlen (runtime in the
        # symbolic loop). We need them as compile-time bounds for the
        # inner Python unrolls; since n_mlen_var binds to IntImm
        # per-iter via BackendEmit, we can't read it here. Compromise:
        # if N is exact multiple of mlen, all n_mlen iters have
        # ``cols_here = mlen``; otherwise we'd need a different
        # encoding. Enforce the common case.
        if N % mlen_v != 0:
            raise PreIsaPassError(
                f"plena.matmul: PreIsaPass currently requires N ({N}) to be "
                f"a multiple of mlen ({mlen_v}) — partial-last-block N "
                f"not yet handled (legacy ``cols_here = min(mlen, ...)`` "
                f"path)"
            )
        tiles_per_n_mlen = mlen_v // blen_v
        tiles_per_mlen = mlen_v // blen_v

        # All five nesting levels (m, n_mlen, oc, orow, k) become
        # PreIsaIR LOOP_START loops with ``loop_kind="unroll"``.
        # Optimisation passes (LICM, CSE) operating on PreIsaIR see
        # the full algebra and can hoist ``m * lhs_m_tile_stride``
        # out of inner loops, share ``k * rhs_k_tile_stride`` across
        # oc iterations, etc — all of which the legacy emit_matmul
        # bakes as literal residuals inside a fully-flat unrolled
        # emit (no hoisting possible at that point).
        oc_var = tir.Var(f"mm_oc_{suffix}", "int32")
        orow_var = tir.Var(f"mm_orow_{suffix}", "int32")
        k_var = tir.Var(f"mm_k_{suffix}", "int32")

        # oc loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_n_mlen],
            binds=oc_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))
        # orow loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_mlen],
            binds=orow_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # dst_col = n_mlen * mlen + oc * blen
        dst_col_expr = tir.Add(
            tir.Mul(n_mlen_var, MLEN_VAR),
            tir.Mul(oc_var, BLEN_VAR),
        )
        # act_orow = lhs_base + orow * a_orow_step
        act_orow_expr = tir.Add(
            lhs_base_expr,
            tir.Mul(orow_var, a_orow_step_expr),
        )
        out_expr = tir.Add(
            tir.Add(dst_m_base_expr, dst_col_expr),
            tir.Mul(orow_var, c_orow_step_expr),
        )

        # K accumulation loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, K_tiles],
            binds=k_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # act_k = act_orow + k * lhs_k_tile_stride
        # mat   = rhs_n_mlen_base + oc * oc_b_step + k * rhs_k_tile_stride
        act_k_expr = tir.Add(
            act_orow_expr,
            tir.Mul(k_var, lhs_k_tile_stride_expr),
        )
        mat_expr = tir.Add(
            tir.Add(
                rhs_n_mlen_base_expr,
                tir.Mul(oc_var, oc_b_step_expr),
            ),
            tir.Mul(k_var, rhs_k_tile_stride_expr),
        )

        # Each k iter is its own group (per_iter scope).
        k_iter_gid = self._new_group()

        def _stamp_k(o):
            o.annotations["group_id"] = k_iter_gid
            if "close_order" not in o.annotations:
                o.annotations["close_order"] = "insertion"
            return o

        self.pre_isa.append(_stamp_k(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[act_k_expr],
        )))
        self.pre_isa.append(_stamp_k(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[mat_expr],
        )))
        if transpose_b:
            self.pre_isa.append(_stamp_k(PreIsaOp(
                opcode="M_TMM",
                operands=[0, act_k_expr, mat_expr],
            )))
        else:
            self.pre_isa.append(_stamp_k(PreIsaOp(
                opcode="M_MM",
                operands=[0, mat_expr, act_k_expr],
            )))

        # Close k loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

        # M_MM_WO after K accumulation — once per (oc, orow).
        wo_gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_PRELOAD_ADDR",
            operands=[out_expr],
            annotations={"group_id": wo_gid, "close_order": "insertion"},
        ))
        self.pre_isa.append(PreIsaOp(
            opcode="M_MM_WO",
            operands=[out_expr, "gp0", 0],
            annotations={"group_id": wo_gid},
        ))

        # Close orow, oc loops.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

        # Close n_mlen loop, then m loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

    # ------------------------------------------------------------------
    # DMA — HBM ↔ VRAM / MRAM tile transfers
    # ------------------------------------------------------------------
    def _iter_tile_offsets(self, hbm_buf: _hlir.Buffer):
        """Mirror of legacy ``isa_pass._iter_tile_offsets``."""
        mlen = int(self.shim.mlen)
        ann = hbm_buf.annotations
        rows = ann.get("logical_rows", mlen)
        cols = ann.get("logical_cols", mlen)
        row_blocks = ann.get("row_blocks", 1)
        col_blocks = ann.get("col_blocks", 1)
        tile_elems = mlen * mlen
        idx = 0
        for j in range(col_blocks):
            for i in range(row_blocks):
                hbm_off = i * mlen * cols + j * mlen
                vram_off = idx * tile_elems
                yield vram_off, hbm_off
                idx += 1

    def _emit_dma_h2v(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_h2v`` →
        ``ISAEmitter.emit_load_tile_from_hbm``.

        Per tile:
          1. Bind an addr-reg ``aN`` from hbm_addr (literal IntImm).
          2. Reset 5 preload scratch GPs (S_ADDI_INT gp{r}, gp0, 0).
          3. Emit ``_emit_preload_tile_isa`` equivalent:
             - S_ADDI_INT gp{a_actual}, gp0, scale_len
             - C_SET_SCALE_REG gp{a_actual}
             - S_ADDI_INT gp{a_actual}, gp0, hbm_start_offset
             - S_ADDI_INT gp{result}, gp0, vram_addr
             - (batch == mlen, batch > preload_len): set stride, twin-unroll
             - For each (outer, inner): set act/mat addrs, H_PREFETCH_V.

        Hardware-coupled parameters (all hw_consts symbolic):
          * ``mlen`` (VLEN + batch + hidden_size default)
          * ``v_prefetch_amount`` (rows per H_PREFETCH_V)
          * ``tile_elems = mlen * mlen`` (scale_size default)
        Per-call options (per_op static):
          * ``hbm_stride`` (defaults to mlen)
          * ``hbm_scale_size`` (defaults to tile_elems)
        """
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        mlen_v = int(self.shim.mlen)
        # hbm_stride / hbm_scale_size defaults — applied to PrimExpr.
        hbm_stride_v = (
            mlen_v if src.hbm_stride is None else int(src.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if src.hbm_scale_size is None
            else int(src.hbm_scale_size)
        )

        gid = self._new_group()

        for vram_off, hbm_off in self._iter_tile_offsets(src):
            tile_hbm_start_offset = src.hbm_offset + hbm_off
            self.pre_isa.append(PreIsaOp(
                opcode="_COMMENT",
                operands=[
                    f"dma_h2v tile  {src.name}[hbm+{hbm_off}] -> "
                    f"{dst.name}[vram+{vram_off}]"
                ],
                annotations={"group_id": gid},
            ))
            self._emit_load_tile_from_hbm_seq(
                hbm_addr=int(src.address),
                vram_addr=int(dst.address) + vram_off,
                hbm_stride=hbm_stride_v,
                hbm_scale_size=hbm_scale_v,
                hbm_start_offset=tile_hbm_start_offset,
                group_id=gid,
            )

    def _emit_dma_h2m(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_h2m`` →
        ``ISAEmitter.emit_hbm_tile_to_mram``."""
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        mlen_v = int(self.shim.mlen)
        hbm_stride_v = (
            mlen_v if src.hbm_stride is None else int(src.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if src.hbm_scale_size is None
            else int(src.hbm_scale_size)
        )

        gid = self._new_group()
        for vram_off, hbm_off in self._iter_tile_offsets(src):
            tile_hbm_offset = src.hbm_offset + hbm_off
            self.pre_isa.append(PreIsaOp(
                opcode="_COMMENT",
                operands=[
                    f"dma_h2m tile  {src.name}[hbm+{hbm_off}] -> "
                    f"{dst.name}[mram+{vram_off}]"
                ],
                annotations={"group_id": gid},
            ))
            self._emit_hbm_tile_to_mram_seq(
                hbm_addr=int(src.address),
                mram_addr=int(dst.address) + vram_off,
                hbm_stride=hbm_stride_v,
                hbm_scale=hbm_scale_v,
                hbm_offset=tile_hbm_offset,
                group_id=gid,
            )

    def _emit_dma_v2h(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_v2h`` →
        ``ISAEmitter.emit_store_tile_to_hbm``."""
        src = mod.get_buffer(op.buffer_args[0])
        dst = mod.get_buffer(op.buffer_args[1])
        if src.num_elements != dst.num_elements:
            raise PreIsaPassError(
                f"dma_v2h: src/dst element-count mismatch"
            )
        mlen_v = int(self.shim.mlen)
        hbm_stride_v = (
            mlen_v if dst.hbm_stride is None else int(dst.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if dst.hbm_scale_size is None
            else int(dst.hbm_scale_size)
        )

        gid = self._new_group()
        for vram_off, hbm_off in self._iter_tile_offsets(dst):
            tile_hbm_start_offset = dst.hbm_offset + hbm_off
            self.pre_isa.append(PreIsaOp(
                opcode="_COMMENT",
                operands=[
                    f"dma_v2h tile  {src.name}[vram+{vram_off}] -> "
                    f"{dst.name}[hbm+{hbm_off}]"
                ],
                annotations={"group_id": gid},
            ))
            self._emit_store_tile_to_hbm_seq(
                vram_addr=int(src.address) + vram_off,
                hbm_addr=int(dst.address),
                hbm_stride=hbm_stride_v,
                hbm_scale_size=hbm_scale_v,
                hbm_start_offset=tile_hbm_start_offset,
                group_id=gid,
            )

    # ------------------------------------------------------------------
    # DMA slice variants (BufferSlice src/dst + multi-tile grid).
    #
    # Static-offset path only: ``BufferSlice.starts`` must all be ints
    # (no PrimExpr starts derived from loop vars). Dynamic-offset
    # support — where ``_materialise_slice_offset`` returns a
    # ``MaterializedExpr`` and the emit uses ``hbm_start_offset_reg``
    # — is a TODO matching the legacy isa_pass's two branches.
    # ------------------------------------------------------------------
    def _emit_dma_h2v_slice(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_h2v_slice``.

        For each tile in the slice's d/s/h/b grid:
          1. Comment "; tile (d,s,h,b): hbm_off=... vram_off=..."
          2. ``_emit_load_tile_from_hbm_seq`` with the per-tile
             ``hbm_start_offset = base_static + tile_const`` and
             ``vram_addr = dst.address + vram_off``.
        """
        sl = op.buffer_args[0]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassError(
                f"dma_h2v_slice: buffer_args[0] must be BufferSlice; "
                f"got {type(sl).__name__}"
            )
        dst_name = op.buffer_args[1]
        if isinstance(dst_name, _hlir.BufferSlice):
            raise PreIsaPassError(
                f"dma_h2v_slice: dst must be a whole-buffer name"
            )
        dst = mod.get_buffer(dst_name)
        parent = mod.get_buffer(sl.parent)
        if self._legacy._slice_has_dynamic_start(sl):
            raise PreIsaPassError(
                f"dma_h2v_slice: dynamic-start slice not yet supported "
                f"by PreIsaPass (legacy uses _materialise_slice_offset's "
                f"reg form)"
            )
        # Tile grid + per-tile strides.
        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._legacy._slice_tile_grid(parent, sl, dst)
        )
        base_static = (
            parent.hbm_offset
            + self._legacy._slice_offset_static(parent, sl)
        )
        mlen_v = int(self.shim.mlen)
        hbm_stride_v = (
            mlen_v if parent.hbm_stride is None
            else int(parent.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if parent.hbm_scale_size is None
            else int(parent.hbm_scale_size)
        )

        starts_s = self._legacy._format_starts(sl)
        gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"dma_h2v_slice  {parent.name}[{starts_s}]"
                f"+{list(sl.extents)} -> {dst.name}  "
                f"(grid d_tiles={d_tiles}, s_tiles={s_tiles}, "
                f"h_groups={h_groups}, b={logical_b})"
            ],
            annotations={"group_id": gid},
        ))
        for d_tile in range(d_tiles):
            for s_tile in range(s_tiles):
                for h_grp in range(h_groups):
                    for b in range(logical_b):
                        hbm_off = (
                            base_static
                            + b * hbm_stride_b
                            + s_tile * inner_mlen * hbm_stride_s
                            + h_grp * lane_count * hbm_stride_h
                            + d_tile * inner_mlen
                        )
                        vram_off = (
                            d_tile * d_tile_stride
                            + s_tile * s_tile_stride
                            + h_grp * h_grp_stride
                            + b * b_stride
                        )
                        self.pre_isa.append(PreIsaOp(
                            opcode="_COMMENT",
                            operands=[
                                f"  tile (d={d_tile}, s={s_tile}, "
                                f"h={h_grp}, b={b}): hbm_off={hbm_off}  "
                                f"vram_off={vram_off}"
                            ],
                            annotations={"group_id": gid},
                        ))
                        tile_gid = self._new_group()
                        self._emit_load_tile_from_hbm_seq(
                            hbm_addr=int(parent.address),
                            vram_addr=int(dst.address) + vram_off,
                            hbm_stride=hbm_stride_v,
                            hbm_scale_size=hbm_scale_v,
                            hbm_start_offset=hbm_off,
                            group_id=tile_gid,
                        )

    def _emit_dma_h2m_slice(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_h2m_slice``.

        Single-tile by contract (legacy ``_check_slice_single_tile``):
        a slice into MRAM is always exactly one mlen*mlen tile. We
        emit one ``emit_hbm_tile_to_mram`` equivalent at the resolved
        per-slice hbm offset.
        """
        sl = op.buffer_args[0]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassError(
                f"dma_h2m_slice: buffer_args[0] must be BufferSlice"
            )
        dst = mod.get_buffer(op.buffer_args[1])
        parent = mod.get_buffer(sl.parent)
        if self._legacy._slice_has_dynamic_start(sl):
            raise PreIsaPassError(
                f"dma_h2m_slice: dynamic-start slice not yet supported"
            )
        # Validate single-tile invariant (matches legacy).
        self._legacy._check_slice_single_tile(parent, sl)
        static_off = (
            parent.hbm_offset
            + self._legacy._slice_offset_static(parent, sl)
        )
        mlen_v = int(self.shim.mlen)
        hbm_stride_v = (
            mlen_v if parent.hbm_stride is None
            else int(parent.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if parent.hbm_scale_size is None
            else int(parent.hbm_scale_size)
        )

        starts_s = self._legacy._format_starts(sl)
        gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"dma_h2m_slice  {parent.name}[{starts_s}]"
                f"+{list(sl.extents)} -> {dst.name}  "
                f"(parent_off={static_off} elems)"
            ],
            annotations={"group_id": gid},
        ))
        self._emit_hbm_tile_to_mram_seq(
            hbm_addr=int(parent.address),
            mram_addr=int(dst.address),
            hbm_stride=hbm_stride_v,
            hbm_scale=hbm_scale_v,
            hbm_offset=static_off,
            group_id=gid,
        )

    def _emit_dma_v2h_slice(
        self, mod: _hlir.HLIRModule, op: _hlir.Op,
    ) -> None:
        """Mirror of legacy ``isa_pass._emit_dma_v2h_slice``.

        Per tile in the slice grid (same shape as ``_emit_dma_h2v_slice``):
        emit ``_emit_store_tile_to_hbm_seq`` with the per-tile
        ``hbm_start_offset = base_static + tile_const``.
        """
        src = mod.get_buffer(op.buffer_args[0])
        sl = op.buffer_args[1]
        if not isinstance(sl, _hlir.BufferSlice):
            raise PreIsaPassError(
                f"dma_v2h_slice: buffer_args[1] must be BufferSlice"
            )
        parent = mod.get_buffer(sl.parent)
        if self._legacy._slice_has_dynamic_start(sl):
            raise PreIsaPassError(
                f"dma_v2h_slice: dynamic-start slice not yet supported"
            )
        (d_tiles, s_tiles, h_groups, logical_b,
         inner_mlen, lane_count,
         (hbm_stride_b, hbm_stride_s, hbm_stride_h),
         (d_tile_stride, s_tile_stride, h_grp_stride, b_stride)) = (
            self._legacy._slice_tile_grid(parent, sl, src)
        )
        base_static = (
            parent.hbm_offset
            + self._legacy._slice_offset_static(parent, sl)
        )
        mlen_v = int(self.shim.mlen)
        hbm_stride_v = (
            mlen_v if parent.hbm_stride is None
            else int(parent.hbm_stride)
        )
        hbm_scale_v = (
            mlen_v * mlen_v if parent.hbm_scale_size is None
            else int(parent.hbm_scale_size)
        )

        starts_s = self._legacy._format_starts(sl)
        gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"dma_v2h_slice  {src.name} -> "
                f"{parent.name}[{starts_s}]+{list(sl.extents)}  "
                f"(grid d_tiles={d_tiles}, s_tiles={s_tiles}, "
                f"h_groups={h_groups}, b={logical_b})"
            ],
            annotations={"group_id": gid},
        ))
        for d_tile in range(d_tiles):
            for s_tile in range(s_tiles):
                for h_grp in range(h_groups):
                    for b in range(logical_b):
                        tile_const = (
                            b * hbm_stride_b
                            + s_tile * inner_mlen * hbm_stride_s
                            + h_grp * lane_count * hbm_stride_h
                            + d_tile * inner_mlen
                        )
                        vram_off = (
                            d_tile * d_tile_stride
                            + s_tile * s_tile_stride
                            + h_grp * h_grp_stride
                            + b * b_stride
                        )
                        self.pre_isa.append(PreIsaOp(
                            opcode="_COMMENT",
                            operands=[
                                f"  tile (d={d_tile}, s={s_tile}, "
                                f"h={h_grp}, b={b}): vram[+{vram_off}] -> "
                                f"hbm[base+{tile_const}]"
                            ],
                            annotations={"group_id": gid},
                        ))
                        tile_gid = self._new_group()
                        self._emit_store_tile_to_hbm_seq(
                            vram_addr=int(src.address) + vram_off,
                            hbm_addr=int(parent.address),
                            hbm_stride=hbm_stride_v,
                            hbm_scale_size=hbm_scale_v,
                            hbm_start_offset=base_static + tile_const,
                            group_id=tile_gid,
                        )

    # ------------------------------------------------------------------
    # DMA emit helpers (decompose legacy emit_load/store/_emit_*_tile_isa
    # into PreIsaOp sequences).
    # ------------------------------------------------------------------
    def _emit_hbm_tile_to_mram_seq(
        self, *, hbm_addr: int, mram_addr: int,
        hbm_stride: int, hbm_scale: int, hbm_offset: int,
        group_id: int,
    ) -> None:
        """PreIsaOp decomposition of legacy ``emit_hbm_tile_to_mram``.

        Emits (per legacy):
          _PRELOAD_ADDR_REG hbm_addr           ; C_SET_ADDR_REG aN ...
          S_ADDI_INT + C_SET_SCALE_REG (hbm_scale)
          S_ADDI_INT + C_SET_STRIDE_REG (hbm_stride)
          S_ADDI_INT (mram base addr)
          S_ADDI_INT (hbm_offset)              ; same gp reused as scale
          H_PREFETCH_M gp{mram}, gp{offset}, aN, 1, 0
          S_ADDI_INT + C_SET_SCALE_REG (tile_elems)  ← restore default
          S_ADDI_INT + C_SET_STRIDE_REG (mlen)       ← restore default
        """
        # Distinct Python PrimExpr objects per slot — id()-keyed cache
        # routes the right value to the right HW operand.
        addr_expr   = tir.IntImm("int32", int(hbm_addr))
        scale_expr  = tir.IntImm("int32", int(hbm_scale))
        stride_expr = tir.IntImm("int32", int(hbm_stride))
        mram_expr   = tir.IntImm("int32", int(mram_addr))
        offset_expr = tir.IntImm("int32", int(hbm_offset))
        # Restore defaults at the end (hw-shape symbolic):
        default_scale_expr  = tir.Mul(MLEN_VAR, MLEN_VAR)  # tile_elems
        default_stride_expr = MLEN_VAR

        def _stamp(o):
            o.annotations["group_id"] = group_id
            return o

        # Bind the HBM addr-reg from hbm_addr.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR_REG", operands=[addr_expr],
        )))
        # Set scale.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[scale_expr],
        )))
        # Set stride.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[stride_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[stride_expr],
        )))
        # MRAM base.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[mram_expr],
        )))
        # HBM start offset.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[offset_expr],
        )))
        # H_PREFETCH_M gp{mram}, gp{offset}, aN, 1, 0.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="H_PREFETCH_M",
            operands=[mram_expr, offset_expr, addr_expr, 1, 0],
        )))
        # Restore defaults (legacy emits scale = tile_elems, stride = mlen).
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[default_scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[default_scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[default_stride_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[default_stride_expr],
        )))

    def _emit_load_tile_from_hbm_seq(
        self, *, hbm_addr: int, vram_addr: int,
        hbm_stride: int, hbm_scale_size: int, hbm_start_offset: int,
        group_id: int,
    ) -> None:
        """PreIsaOp decomposition of legacy ``emit_load_tile_from_hbm``
        + ``_emit_preload_tile_isa`` (batch>1 path).

        Mirrors the static-unroll body the legacy emits when
        ``batch == mlen`` and ``batch > v_prefetch_amount``: a
        per-(outer, inner) sequence of two S_ADDIs + one
        H_PREFETCH_V. Outer/inner are statically unrolled here (could
        become LOOP_START in a future optimisation pass).
        """
        mlen_v = int(self.shim.mlen)
        vpref_v = int(self.shim.v_prefetch_amount)
        # All hardware-coupled values referenced via PrimExpr so the
        # optimiser sees the algebra; folded by materialiser at emit.
        addr_expr   = tir.IntImm("int32", int(hbm_addr))
        scale_expr  = tir.IntImm("int32", int(hbm_scale_size))
        stride_expr = tir.IntImm("int32", int(hbm_stride))
        offset_expr = tir.IntImm("int32", int(hbm_start_offset))
        vram_base_expr = tir.IntImm("int32", int(vram_addr))

        def _stamp(o):
            o.annotations["group_id"] = group_id
            return o

        # Bind addr-reg.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR_REG", operands=[addr_expr],
        )))
        # NOTE: legacy emits 5 ``reset_reg_asm`` ``S_ADDI_INT gp{r},
        # gp0, 0`` lines as a sanity-reset of scratch GPs that the body
        # then immediately overwrites — semantically a no-op. PreIsaIR
        # skips them to keep the cache from holding 5 pinned GPs for
        # no reason; this means the new path emits 5 fewer ISA lines
        # (semantically equivalent — those zeros are dead writes).
        # Preload tile body (mirrors _emit_preload_tile_isa, batch>1).
        # S_ADDI_INT gp{a_actual}, gp0, scale_len ;
        # C_SET_SCALE_REG gp{a_actual} ;
        # S_ADDI_INT gp{a_actual}, gp0, hbm_start_offset (re-uses GP)
        # S_ADDI_INT gp{result}, gp0, vram_addr
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[offset_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[vram_base_expr],
        )))
        # S_ADDI_INT gp{stride}, gp0, stride_len ;
        # C_SET_STRIDE_REG gp{stride}
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[stride_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[stride_expr],
        )))

        # Static-unroll dimensions: batch=mlen, hidden_size=mlen,
        # vlen=mlen. load_amount_per_hidden = 1 always under these
        # assumptions; inner_count = mlen/v_prefetch_amount when
        # batch>preload_len.
        load_amount_per_hidden = 1
        # Symbolic inner_count = MLEN/V_PREFETCH_AMOUNT — but the
        # *bound* of the LOOP_START must be a compile-time int, so
        # we compute it from the shim here. The strides INSIDE the
        # body stay symbolic (referencing MLEN_VAR, etc).
        if mlen_v > vpref_v:
            inner_count_n = (mlen_v + vpref_v - 1) // vpref_v
        else:
            inner_count_n = 1

        # Wrap the (outer × inner) loop in a single per_iter
        # LOOP_START. The body uses the iter var ``it`` =
        # outer * inner_count + inner. Since
        # load_amount_per_hidden == 1, outer == 0 always and the
        # full count is just inner_count_n.
        step_elems_expr = tir.Mul(MLEN_VAR, V_PREFETCH_AMOUNT_VAR)
        # ``it`` ∈ [0, inner_count_n).
        it_var = tir.Var(
            f"dma_h2v_it_{id(self) & 0xffff:x}_"
            f"{hbm_start_offset & 0xffff:x}", "int32",
        )
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, load_amount_per_hidden * inner_count_n],
            binds=it_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        )))
        result_addr_expr = tir.Add(
            vram_base_expr,
            tir.Mul(it_var, step_elems_expr),
        )
        if mlen_v > vpref_v:
            # a_off (within HBM block) = it * stride_len * preload_len
            # (outer term is 0 because load_amount_per_hidden=1).
            a_off_inner_expr = tir.Add(
                offset_expr,
                tir.Mul(
                    it_var,
                    tir.Mul(
                        tir.IntImm("int32", int(hbm_stride)),
                        V_PREFETCH_AMOUNT_VAR,
                    ),
                ),
            )
        else:
            # batch <= preload_len: legacy a_off = outer*vlen = 0.
            a_off_inner_expr = offset_expr

        iter_gid = self._new_group()

        def _stamp_iter(o):
            o.annotations["group_id"] = iter_gid
            return o

        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[result_addr_expr],
        )))
        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[a_off_inner_expr],
        )))
        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="H_PREFETCH_V",
            operands=[result_addr_expr, a_off_inner_expr,
                      addr_expr, 1, 0],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        )))

    def _emit_store_tile_to_hbm_seq(
        self, *, vram_addr: int, hbm_addr: int,
        hbm_stride: int, hbm_scale_size: int, hbm_start_offset: int,
        group_id: int,
    ) -> None:
        """PreIsaOp decomposition of legacy ``emit_store_tile_to_hbm``
        + ``_emit_store_tile_isa`` (batch>1 path).
        """
        mlen_v = int(self.shim.mlen)
        vwb_v = int(self.shim.v_writeback_amount)
        addr_expr   = tir.IntImm("int32", int(hbm_addr))
        scale_expr  = tir.IntImm("int32", int(hbm_scale_size))
        stride_expr = tir.IntImm("int32", int(hbm_stride))
        offset_expr = tir.IntImm("int32", int(hbm_start_offset))
        vram_base_expr = tir.IntImm("int32", int(vram_addr))

        def _stamp(o):
            o.annotations["group_id"] = group_id
            return o

        # Bind addr-reg.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR_REG", operands=[addr_expr],
        )))
        # Setup: S_ADDI vram, S_ADDI scale; C_SET_SCALE; S_ADDI offset.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[vram_base_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_SCALE_REG", operands=[scale_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[offset_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[stride_expr],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="C_SET_STRIDE_REG", operands=[stride_expr],
        )))

        store_amount_per_hidden = 1
        if mlen_v > vwb_v:
            inner_count_n = (mlen_v + vwb_v - 1) // vwb_v
        else:
            inner_count_n = 1
        step_elems_expr = tir.Mul(MLEN_VAR, V_WRITEBACK_AMOUNT_VAR)

        it_var = tir.Var(
            f"dma_v2h_it_{id(self) & 0xffff:x}_"
            f"{hbm_start_offset & 0xffff:x}", "int32",
        )
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, store_amount_per_hidden * inner_count_n],
            binds=it_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        )))
        vram_off_expr = tir.Add(
            vram_base_expr,
            tir.Mul(it_var, step_elems_expr),
        )
        if mlen_v > vwb_v:
            hbm_off_inner_expr = tir.Add(
                offset_expr,
                tir.Mul(
                    it_var,
                    tir.Mul(
                        tir.IntImm("int32", int(hbm_stride)),
                        V_WRITEBACK_AMOUNT_VAR,
                    ),
                ),
            )
        else:
            hbm_off_inner_expr = offset_expr

        iter_gid = self._new_group()

        def _stamp_iter(o):
            o.annotations["group_id"] = iter_gid
            return o

        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[vram_off_expr],
        )))
        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[hbm_off_inner_expr],
        )))
        self.pre_isa.append(_stamp_iter(PreIsaOp(
            opcode="H_STORE_V",
            operands=[vram_off_expr, hbm_off_inner_expr,
                      addr_expr, 1, 0],
        )))
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        )))

    def _emit_mm_slot(self, mod: _hlir.HLIRModule, op: _hlir.Op) -> None:
        """Mirror of legacy ``isa_pass._emit_mm_slot`` →
        ``ISAEmitter.emit_slot_matmul``.

        Schema:
            buffer_args = [lhs_name, rhs_name, dst_name]
            scalar_args = [lhs_row_offset, rhs_col_offset,
                           dst_col_offset, col_count]
                (offsets may be ints OR tir.PrimExprs; col_count must be int)

        Legacy emission for each outer oc iter:
            S_ADDI gp{act}, ..., act_addr(oc)
            S_ADDI gp{mat}, ..., mat_addr(oc)
            S_ADDI gp{out}, ..., out_addr(oc)
            for t in range(tiles_per_mlen):
                if t > 0:
                    S_ADDI gp{act}, gp{act}, row_stride         (bump)
                    S_ADDI gp{out}, gp{out}, row_stride         (bump)
                M_MM 0, gp{mat}, gp{act}
                M_MM_WO gp{out}, gp0, 0

        PreIsaIR decomposition:
          * outer oc loop → LOOP_START(unroll, per_iter)
          * per oc iter: 3 _PRELOAD_ADDRs (act, mat, out)
          * inner t loop → LOOP_START(unroll, shared) so the per-iter
            BUMPs on cached act/out GPs persist across iters.
            Each inner iter (except first) bumps act + out before
            M_MM + M_MM_WO.

        This initial migration handles the STATIC-offset path
        (lhs_row_offset / rhs_col_offset / dst_col_offset are all
        compile-time ints). Dynamic-offset support is a TODO.
        """
        if len(op.buffer_args) != 3:
            raise PreIsaPassError(
                f"plena.mm_slot expects 3 buffer_args; got {len(op.buffer_args)}"
            )
        lhs = mod.get_buffer(op.buffer_args[0])
        rhs = mod.get_buffer(op.buffer_args[1])
        dst = mod.get_buffer(op.buffer_args[2])
        if len(op.scalar_args) != 4:
            raise PreIsaPassError(
                f"plena.mm_slot expects 4 scalar_args; got {len(op.scalar_args)}"
            )
        lhs_row_offset_raw = op.scalar_args[0]
        rhs_col_offset_raw = op.scalar_args[1]
        dst_col_offset_raw = op.scalar_args[2]
        col_count_raw = op.scalar_args[3]

        def _static(x):
            if isinstance(x, int):
                return int(x)
            if isinstance(x, tir.IntImm):
                return int(x.value)
            return None

        lhs_row_offset = _static(lhs_row_offset_raw)
        rhs_col_offset = _static(rhs_col_offset_raw)
        dst_col_offset = _static(dst_col_offset_raw)
        if (lhs_row_offset is None or rhs_col_offset is None
                or dst_col_offset is None):
            raise PreIsaPassError(
                f"plena.mm_slot: dynamic offsets not yet supported by "
                f"PreIsaPass; got lhs_row_offset={lhs_row_offset_raw!r}, "
                f"rhs_col_offset={rhs_col_offset_raw!r}, "
                f"dst_col_offset={dst_col_offset_raw!r}"
            )
        col_count = _static(col_count_raw)
        if col_count is None or col_count <= 0:
            raise PreIsaPassError(
                f"plena.mm_slot col_count must be a positive compile-time int; "
                f"got {col_count_raw!r}"
            )

        mlen = int(self.shim.mlen)
        blen = int(self.shim.blen)
        if col_count % blen != 0:
            raise PreIsaPassError(
                f"plena.mm_slot: col_count={col_count} must be divisible by blen={blen}"
            )
        tiles_per_slot = col_count // blen
        tiles_per_mlen = mlen // blen
        # row_stride = blen * mlen — symbolic so PreIsaIR preserves
        # the hw-shape algebra. Materialiser folds to literal at emit.
        row_stride_expr = tir.Mul(BLEN_VAR, MLEN_VAR)
        task_id = op.annotations.get("intrinsic", "mm_slot")

        gid = self._new_group()
        self.pre_isa.append(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"slot matmul task {task_id} "
                f"rhs_col_offset={rhs_col_offset} "
                f"dst_col_offset={dst_col_offset}"
            ],
            annotations={"group_id": gid},
        ))
        # Preamble: legacy emits ``S_ADDI gp{stride}, gp0, 1`` (dead).
        stride_const = tir.IntImm("int32", 1)
        self.pre_isa.append(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[stride_const],
            annotations={"group_id": gid},
        ))

        oc_var = tir.Var(f"mm_slot_oc_{id(op) & 0xffff:x}", "int32")
        t_var = tir.Var(f"mm_slot_t_{id(op) & 0xffff:x}", "int32")

        # Outer (oc) per_iter unroll.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_START",
            operands=[0, tiles_per_slot],
            binds=oc_var,
            annotations={
                "loop_kind": "unroll",
                "unroll_scope": "per_iter",
            },
        ))

        # Per-oc address PrimExprs.
        # act_addr = lhs.address + lhs_row_offset
        #   (lhs_row_offset is the static offset into a multi-tile lhs).
        # mat_addr = rhs.address + rhs_col_offset + oc * blen
        # out_addr = dst.address + dst_col_offset + oc * blen
        # ``blen`` referenced symbolically via BLEN_VAR.
        act_base_expr = tir.Add(
            tir.IntImm("int32", int(lhs.address)),
            tir.IntImm("int32", lhs_row_offset),
        )
        mat_addr_expr = tir.Add(
            tir.Add(
                tir.IntImm("int32", int(rhs.address)),
                tir.IntImm("int32", rhs_col_offset),
            ),
            tir.Mul(oc_var, BLEN_VAR),
        )
        out_addr_expr = tir.Add(
            tir.Add(
                tir.IntImm("int32", int(dst.address)),
                tir.IntImm("int32", dst_col_offset),
            ),
            tir.Mul(oc_var, BLEN_VAR),
        )

        oc_iter_gid = self._new_group()

        def _stamp_oc(o):
            o.annotations["group_id"] = oc_iter_gid
            if "close_order" not in o.annotations:
                o.annotations["close_order"] = "insertion"
            return o

        # Per-oc preloads (legacy emit order: act, mat, out).
        self.pre_isa.append(_stamp_oc(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[act_base_expr],
        )))
        self.pre_isa.append(_stamp_oc(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[mat_addr_expr],
        )))
        self.pre_isa.append(_stamp_oc(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[out_addr_expr],
        )))

        # Inner (t) shared-scope unroll — the per-iter bumps on act/out
        # cached GPs persist across iters.
        # For byte-equal we follow legacy's "if t > 0 do bumps; then
        # M_MM/M_MM_WO" pattern with the prefix-loop + final-iter
        # split mv already uses.
        if tiles_per_mlen > 1:
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="LOOP_START",
                operands=[0, tiles_per_mlen - 1],
                binds=t_var,
                annotations={
                    "loop_kind": "unroll",
                    "unroll_scope": "shared",
                    "group_id": oc_iter_gid,
                    "close_order": "insertion",
                },
            )))
            # First M_MM (legacy: skip bump on iter 0).
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="M_MM",
                operands=[0, mat_addr_expr, act_base_expr],
            )))
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="M_MM_WO",
                operands=[out_addr_expr, "gp0", 0],
            )))
            # Bump act + out for the NEXT iter. ``row_stride_expr`` is
            # the symbolic ``BLEN_VAR * MLEN_VAR``; BackendEmit folds
            # it to a literal stride at emit time.
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="_BUMP_CACHED_GP",
                operands=[act_base_expr, row_stride_expr],
            )))
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="_BUMP_CACHED_GP",
                operands=[out_addr_expr, row_stride_expr],
            )))
            self.pre_isa.append(_stamp_oc(PreIsaOp(
                opcode="LOOP_END",
                operands=[],
                annotations={"loop_kind": "unroll"},
            )))
        # Final iter — no trailing bumps.
        self.pre_isa.append(_stamp_oc(PreIsaOp(
            opcode="M_MM",
            operands=[0, mat_addr_expr, act_base_expr],
        )))
        self.pre_isa.append(_stamp_oc(PreIsaOp(
            opcode="M_MM_WO",
            operands=[out_addr_expr, "gp0", 0],
        )))

        # Close outer oc loop.
        self.pre_isa.append(PreIsaOp(
            opcode="LOOP_END",
            operands=[],
            annotations={"loop_kind": "unroll"},
        ))

    # ------------------------------------------------------------------
    # row_*_at — row-scalar VRAM ops with d_tile unroll
    # ------------------------------------------------------------------
    def _emit_row_scalar_op_at(
        self,
        mod: _hlir.HLIRModule,
        op: _hlir.Op,
        *,
        row_op: str,
        reduce: bool = False,
        masked: bool = False,
        has_fp: bool = False,
    ) -> None:
        """Mirror of legacy ``isa_pass._emit_row_scalar_op_at``.

        Legacy emit order:
          1. Eager: ``materialise(src_addr)`` → S_ADDI_INT gp{src}, ...
          2. Eager: if masked, ``materialise(mask_expr)`` → S_ADDI_INT gp{mask}
          3. Eager: ``materialise(dst_addr or fp_addr)`` → S_ADDI_INT gp{dst}
          4. Eager: if binary-fp, ``materialise(fp_rhs_addr)`` → S_ADDI_INT gp{rhs}
          5. Flush `lines` (header comment + body ISA + mask reset)

        PreIsaIR encodes this as:
          _PRELOAD_ADDR src_addr
          [if masked] _PRELOAD_ADDR mask_expr
          _PRELOAD_ADDR dst_addr_or_fp_addr
          [if binary-fp] _PRELOAD_ADDR fp_rhs_addr
          _COMMENT "row scalar task ..."
          [if masked] C_SET_V_MASK_REG (cached mask GP)
          [if reduce] S_LD_FP f1, gp{dst}, 0  -- _S_LD_FP_CACHED
          per d_tile:
            HW op (V_RED_* / V_EXP_V / V_*_VF) using cached gp{src}, gp{dst}
            [if not last d_tile] _BUMP_CACHED_GP src_addr, d_tile_stride_s
                                  [if dst exists] _BUMP_CACHED_GP dst_addr, d_tile_stride_d
          [if reduce] S_ST_FP f1, gp{dst}, 0 -- _S_ST_FP_CACHED
          [if masked] _S_ADDI_INT_RESET_MASK + C_SET_V_MASK_REG
        """
        has_fp = has_fp or reduce
        if reduce:
            if len(op.buffer_args) != 1:
                raise PreIsaPassError(
                    f"{op.kind} expects 1 buffer_arg (src region); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 1
        elif has_fp:
            if len(op.buffer_args) != 2:
                raise PreIsaPassError(
                    f"{op.kind} expects 2 buffer_args (src/dst regions); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 1
        else:
            if len(op.buffer_args) != 2:
                raise PreIsaPassError(
                    f"{op.kind} expects 2 buffer_args (src/dst regions); "
                    f"got {len(op.buffer_args)}"
                )
            expected_scalar = 0
        if len(op.scalar_args) != expected_scalar:
            raise PreIsaPassError(
                f"{op.kind} expects {expected_scalar} scalar_args; "
                f"got {len(op.scalar_args)}"
            )
        for slot, name in enumerate(("src",) if reduce else ("src", "dst")):
            if not isinstance(op.buffer_args[slot], _hlir.VramRegion):
                raise PreIsaPassError(
                    f"{op.kind} {name}: expected VramRegion, got "
                    f"{type(op.buffer_args[slot]).__name__}"
                )

        src_region: _hlir.VramRegion = op.buffer_args[0]
        src = mod.get_buffer(src_region.parent)
        # All non-D extents must be 1 (one logical row per op).
        if any(int(e) != 1 for e in src_region.extents[:3]):
            raise PreIsaPassError(
                f"{op.kind} src: row_*_at processes one logical row, "
                f"non-D extents must be 1; got "
                f"{tuple(src_region.extents[:3])}"
            )

        fp_addr_expr = None
        if has_fp:
            fp_addr_expr = self._legacy._resolve_fp_scalar_addr_arg(
                mod, op.scalar_args[0], op.kind, "fp",
            )

        src_base_off, src_mask_expr, src_info = (
            self._legacy._logical_to_phys_row_offset(src, src_region)
        )
        emit_v_mask = masked and src_mask_expr is not None
        use_mask_flag = 1 if emit_v_mask else 0

        # Build the address PrimExpr objects ONCE (id()-keyed cache).
        src_addr = tir.Add(tir.IntImm("int32", int(src.address)), src_base_off)
        mask_expr = src_mask_expr if emit_v_mask else None

        # Reduce: scalar_args[0] is the FP destination
        # Binary-fp: same — fp_addr_expr is the FP rhs operand. The dst
        # buffer is buffer_args[1] (VramRegion).
        dst_addr = None
        d_tile_stride_d = 0
        n_d_tiles = src_info["d_tiles"]
        d_tile_stride_s = src_info["d_tile_stride"]
        if not reduce:
            dst_region: _hlir.VramRegion = op.buffer_args[1]
            dst = mod.get_buffer(dst_region.parent)
            if len(dst_region.extents) != 4:
                raise PreIsaPassError(
                    f"{op.kind} dst: region must be 4D; got "
                    f"extents={tuple(dst_region.extents)}"
                )
            if any(int(e) != 1 for e in dst_region.extents[:3]):
                raise PreIsaPassError(
                    f"{op.kind} dst: non-D extents must be 1; "
                    f"got {tuple(dst_region.extents[:3])}"
                )
            dst_base_off, dst_mask_expr, dst_info = (
                self._legacy._logical_to_phys_row_offset(dst, dst_region)
            )
            if emit_v_mask and dst_mask_expr is None:
                raise PreIsaPassError(
                    f"{op.kind} src requires packed-head mask but dst "
                    f"{dst.name!r} does not"
                )
            if emit_v_mask and dst_region.parent != src_region.parent:
                warnings.warn(
                    f"{op.kind}: masked V_*_V with dst "
                    f"{dst_region.parent!r} != src "
                    f"{src_region.parent!r} — unmasked heads will "
                    f"overwrite dst with src",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if dst_info["d_tiles"] != n_d_tiles:
                raise PreIsaPassError(
                    f"{op.kind}: src/dst d_tiles mismatch"
                )
            d_tile_stride_d = dst_info["d_tile_stride"]
            dst_addr = tir.Add(tir.IntImm("int32", int(dst.address)), dst_base_off)

        # ----- one group_id for the whole row_*_at op -----
        gid = self._new_group()

        def _stamp(o):
            o.annotations["group_id"] = gid
            return o

        # 1) Preload src first (matches legacy m_src materialise).
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_PRELOAD_ADDR", operands=[src_addr],
        )))
        # 2) Preload mask if masked.
        if emit_v_mask:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[mask_expr],
            )))
        # 3) Preload dst / fp_dst.
        if reduce:
            # The FP destination address (scalar_args[0]).
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[fp_addr_expr],
            )))
        else:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_PRELOAD_ADDR", operands=[dst_addr],
            )))
            # 4) For binary-fp variants, ALSO preload the FP RHS.
            if fp_addr_expr is not None:
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="_PRELOAD_ADDR", operands=[fp_addr_expr],
                )))

        # Header comment.
        self.pre_isa.append(_stamp(PreIsaOp(
            opcode="_COMMENT",
            operands=[
                f"row scalar task {op.annotations.get('intrinsic', op.kind)} "
                f"op={row_op} "
                f"src.parent={src_region.parent} "
                f"starts={list(src_region.starts)!r}"
            ],
        )))

        # C_SET_V_MASK_REG (if masked).
        if emit_v_mask:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="C_SET_V_MASK_REG",
                operands=[mask_expr],
            )))

        # ----- the body -----
        # The d_tile unroll for row_*_at is encoded as a prefix
        # LOOP_START(unroll, shared) of length (n_d_tiles - 1) — each
        # iter emits the HW op AND a trailing _BUMP_CACHED_GP for
        # src (and dst when applicable) — followed by one more
        # explicit body emission for the final iter WITHOUT trailing
        # bumps. This mirrors legacy's ``if t < n_d_tiles - 1`` guard
        # while keeping the loop symbolic in PreIsaIR; the
        # ``unroll_scope="shared"`` annotation tells BackendEmit to
        # carry the cached GPs (and their bumped values) across
        # iterations.
        def _emit_one_iter(emit_bumps: bool) -> None:
            """Emit one row-op body (single d_tile). When emit_bumps is
            True, append the bumps that ready the cached GPs for the
            next iter."""
            if reduce:
                op_str = {"reduce_max": "V_RED_MAX",
                          "reduce_sum": "V_RED_SUM"}[row_op]
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode=op_str,
                    operands=["f1", src_addr, use_mask_flag],
                )))
            elif fp_addr_expr is None:
                op_str = {"exp": "_V_EXP_V_ROW",
                          "reci": "_V_RECI_V_ROW"}[row_op]
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode=op_str,
                    operands=[dst_addr, src_addr, use_mask_flag],
                )))
            else:
                if row_op == "sub":
                    self.pre_isa.append(_stamp(PreIsaOp(
                        opcode="_V_SUB_VF_ROW",
                        operands=[dst_addr, src_addr, "f1",
                                  use_mask_flag, 0],
                    )))
                else:
                    op_str = {"add": "_V_ADD_VF_ROW",
                              "mul": "_V_MUL_VF_ROW"}[row_op]
                    self.pre_isa.append(_stamp(PreIsaOp(
                        opcode=op_str,
                        operands=[dst_addr, src_addr, "f1",
                                  use_mask_flag],
                    )))
            if emit_bumps:
                self.pre_isa.append(_stamp(PreIsaOp(
                    opcode="_BUMP_CACHED_GP",
                    operands=[src_addr, d_tile_stride_s],
                )))
                if not reduce:
                    self.pre_isa.append(_stamp(PreIsaOp(
                        opcode="_BUMP_CACHED_GP",
                        operands=[dst_addr, d_tile_stride_d],
                    )))

        # Reduce + binary-fp variants seed/finalise an f1 accumulator
        # around the d_tile sweep.
        if reduce:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_S_LD_FP_CACHED",
                operands=["f1", fp_addr_expr, 0],
            )))
        elif fp_addr_expr is not None:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_S_LD_FP_CACHED",
                operands=["f1", fp_addr_expr, 0],
            )))

        # Prefix unroll loop: (n_d_tiles - 1) iters, each with
        # trailing bumps. Final iter is emitted explicitly after,
        # without bumps. When n_d_tiles == 1 the loop is empty.
        if n_d_tiles > 1:
            t_var = tir.Var(f"row_d_tile_{id(op) & 0xffff:x}", "int32")
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="LOOP_START",
                operands=[0, n_d_tiles - 1],
                binds=t_var,
                annotations={
                    "loop_kind": "unroll",
                    "unroll_scope": "shared",
                    "group_id": gid,
                },
            )))
            _emit_one_iter(emit_bumps=True)
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="LOOP_END",
                operands=[],
                annotations={
                    "loop_kind": "unroll",
                    "group_id": gid,
                },
            )))
        # Final iter (always present).
        _emit_one_iter(emit_bumps=False)

        if reduce:
            # S_ST_FP f1, gp{fp_dst}, 0 — flush accumulator.
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_S_ST_FP_CACHED",
                operands=["f1", fp_addr_expr, 0],
            )))

        # Mask reset.
        if emit_v_mask:
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="_S_ADDI_INT_RESET_MASK",
                operands=[mask_expr, "gp0", 0],
            )))
            self.pre_isa.append(_stamp(PreIsaOp(
                opcode="C_SET_V_MASK_REG",
                operands=[mask_expr],
            )))


__all__ = ["PreIsaPass", "PreIsaPassError"]
