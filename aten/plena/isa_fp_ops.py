"""FPVar and FPRAM ISA helpers for IsaCompiler."""

from __future__ import annotations

from compiler.aten.isa_builder import IsaBuilder, fp, gp


class IsaFPOpsMixin:
    # =========================================================================
    # FPVar ISA helpers (address-based)
    # =========================================================================

    def _emit_fpvar_skip(self, op_name: str, count: int) -> str:
        return self._emit(IsaBuilder().comment(f"FPVar {op_name} skipped: count={count}"))

    def _fpvar_unary_asm(
        self,
        op_name: str,
        op_description: str,
        opcode: str,
        src_addr: int,
        dst_addr: int,
        count: int,
    ) -> str:
        if count <= 0:
            return self._emit_fpvar_skip(op_name, count)

        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"FPVar {op_name}: {op_description}, count={count}")
            asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)

            if self._unroll:
                for i in range(count):
                    asm.instr("S_LD_FP", fp(1), gp(gp_src), i)
                    asm.instr(opcode, fp(1), fp(1), 0)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), i)
            else:
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_LD_FP", fp(1), gp(gp_src), 0)
                asm.instr(opcode, fp(1), fp(1), 0)
                asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), 1)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), 1)
                asm.instr("C_LOOP_END", gp(gp_loop))

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _fpvar_binary_asm(
        self,
        op_name: str,
        op_description: str,
        opcode: str,
        src1_addr: int,
        src2_addr: int,
        dst_addr: int,
        count: int,
    ) -> str:
        if count <= 0:
            return self._emit_fpvar_skip(op_name, count)

        gp_regs = self._reg.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"FPVar {op_name}: {op_description}, count={count}")
            asm.instr("S_ADDI_INT", gp(gp_a), gp(0), src1_addr)
            asm.instr("S_ADDI_INT", gp(gp_b), gp(0), src2_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)

            if self._unroll:
                for i in range(count):
                    asm.instr("S_LD_FP", fp(1), gp(gp_a), i)
                    asm.instr("S_LD_FP", fp(2), gp(gp_b), i)
                    asm.instr(opcode, fp(1), fp(1), fp(2))
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), i)
            else:
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_LD_FP", fp(1), gp(gp_a), 0)
                asm.instr("S_LD_FP", fp(2), gp(gp_b), 0)
                asm.instr(opcode, fp(1), fp(1), fp(2))
                asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                asm.instr("S_ADDI_INT", gp(gp_a), gp(gp_a), 1)
                asm.instr("S_ADDI_INT", gp(gp_b), gp(gp_b), 1)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), 1)
                asm.instr("C_LOOP_END", gp(gp_loop))

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def fpvar_copy_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit_fpvar_skip("Copy", count)

        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"FPVar Copy: FPRAM[{dst_addr}:{dst_addr + count}] = FPRAM[{src_addr}:{src_addr + count}]"
            )
            asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)

            if self._unroll:
                for i in range(count):
                    asm.instr("S_LD_FP", fp(1), gp(gp_src), i)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), i)
            else:
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_LD_FP", fp(1), gp(gp_src), 0)
                asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), 1)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), 1)
                asm.instr("C_LOOP_END", gp(gp_loop))

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def fpvar_fill_from_fpram_asm(self, dst_addr: int, src_fpram_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit_fpvar_skip("Fill", count)

        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"FPVar Fill: FPRAM[{dst_addr}:{dst_addr + count}] = FPRAM[{src_fpram_addr}]")
            asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_fpram_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
            asm.instr("S_LD_FP", fp(1), gp(gp_src), 0)

            if self._unroll:
                for i in range(count):
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), i)
            else:
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), 1)
                asm.instr("C_LOOP_END", gp(gp_loop))

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def fpvar_reci_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_unary_asm("Reci", "dst = 1/src", "S_RECI_FP", src_addr, dst_addr, count)

    def fpvar_exp_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_unary_asm("Exp", "dst = exp(src)", "S_EXP_FP", src_addr, dst_addr, count)

    def fpvar_add_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_binary_asm("Add", "dst = src1 + src2", "S_ADD_FP", src1_addr, src2_addr, dst_addr, count)

    def fpvar_sub_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_binary_asm("Sub", "dst = src1 - src2", "S_SUB_FP", src1_addr, src2_addr, dst_addr, count)

    def fpvar_mul_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_binary_asm("Mul", "dst = src1 * src2", "S_MUL_FP", src1_addr, src2_addr, dst_addr, count)

    def fpvar_max_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        return self._fpvar_binary_asm("Max", "dst = max(src1, src2)", "S_MAX_FP", src1_addr, src2_addr, dst_addr, count)

    def fpvar_sum_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit_fpvar_skip("Sum", count)

        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"FPVar Sum: FPRAM[{dst_addr}] = sum(FPRAM[{src_addr}:{src_addr + count}])")
            asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
            asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))

            if self._unroll:
                for i in range(count):
                    asm.instr("S_LD_FP", fp(2), gp(gp_src), i)
                    asm.instr("S_ADD_FP", fp(1), fp(1), fp(2))
            else:
                asm.instr("C_LOOP_START", gp(gp_loop), count)
                asm.instr("S_LD_FP", fp(2), gp(gp_src), 0)
                asm.instr("S_ADD_FP", fp(1), fp(1), fp(2))
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), 1)
                asm.instr("C_LOOP_END", gp(gp_loop))

            asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def fpvar_shift_asm(
        self,
        src_addr: int,
        dst_addr: int,
        count: int,
        shift: int,
        fill_fpram_addr: int = 0,
    ) -> str:
        """
        Shift FPVar into dst.
        - shift > 0: right shift (leading positions filled)
        - shift < 0: left shift (trailing positions filled)
        """
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_fill = gp_regs
        try:
            asm = IsaBuilder().comment(
                f"FPVar Shift: dst=shift(src, shift={shift}), count={count}, fill=FPRAM[{fill_fpram_addr}]"
            )
            asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr)
            asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr)
            asm.instr("S_ADDI_INT", gp(gp_fill), gp(0), fill_fpram_addr)
            asm.instr("S_LD_FP", fp(3), gp(gp_fill), 0)

            for i in range(count):
                src_idx = i - shift
                if 0 <= src_idx < count:
                    asm.instr("S_LD_FP", fp(1), gp(gp_src), src_idx)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), i)
                else:
                    asm.instr("S_ST_FP", fp(3), gp(gp_dst), i)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    # =========================================================================
    # FPVar helpers (name-based wrappers over the address-based ISA generators)
    # =========================================================================

    def fpram_copy(self, src_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_copy_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_reci(self, src_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_reci_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_exp(self, src_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        return self.fpvar_exp_asm(self.get_fpram_addr(src_name), self.get_fpram_addr(dst_name), count)

    def fpram_add(self, src1_name: str, src2_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_add_asm(
            self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count
        )

    def fpram_sub(self, src1_name: str, src2_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_sub_asm(
            self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count
        )

    def fpram_mul(self, src1_name: str, src2_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_mul_asm(
            self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count
        )

    def fpram_max(self, src1_name: str, src2_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = min(self.get_fpram_size(src1_name), self.get_fpram_size(src2_name), self.get_fpram_size(dst_name))
        return self.fpvar_max_asm(
            self.get_fpram_addr(src1_name), self.get_fpram_addr(src2_name), self.get_fpram_addr(dst_name), count
        )

    def fpram_sum(self, src_name: str, dst_name: str, count: int | None = None) -> str:
        if count is None:
            count = self.get_fpram_size(src_name)
        return self.fpvar_sum_asm(
            self.get_fpram_addr(src_name),
            self.get_fpram_addr(dst_name),
            count,
        )

    def fpram_shift(
        self,
        src_name: str,
        dst_name: str,
        shift: int,
        count: int | None = None,
        fill_fpram_name: str | None = None,
    ) -> str:
        if count is None:
            count = min(self.get_fpram_size(src_name), self.get_fpram_size(dst_name))
        fill_addr = 0 if fill_fpram_name is None else self.get_fpram_addr(fill_fpram_name)
        return self.fpvar_shift_asm(
            src_addr=self.get_fpram_addr(src_name),
            dst_addr=self.get_fpram_addr(dst_name),
            count=count,
            shift=shift,
            fill_fpram_addr=fill_addr,
        )

    def fpram_fill_from_fpram(self, dst_name: str, src_fpram_addr: int, count: int | None = None) -> str:
        if count is None:
            count = self.get_fpram_size(dst_name)
        return self.fpvar_fill_from_fpram_asm(
            dst_addr=self.get_fpram_addr(dst_name),
            src_fpram_addr=src_fpram_addr,
            count=count,
        )


__all__ = ["IsaFPOpsMixin"]
