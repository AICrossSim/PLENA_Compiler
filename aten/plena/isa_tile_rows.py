"""Tile-row ISA helpers for IsaCompiler."""

from __future__ import annotations

from compiler.aten.isa_builder import IsaBuilder, fp, gp


class IsaTileRowMixin:
    # =========================================================================
    # Tile-row helpers (name-based)
    # =========================================================================

    def tile_row_max(
        self,
        source_matrix: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        source_addr = self.get_vram_tile_addr(source_matrix, tile_row_idx, tile_col_idx)
        return self.tile_row_max_asm(source_addr, row_map)

    def tile_row_sum(
        self,
        source_matrix: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        source_addr = self.get_vram_tile_addr(source_matrix, tile_row_idx, tile_col_idx)
        return self.tile_row_sum_asm(source_addr, row_map)

    def tile_row_exp(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_exp_asm(matrix_addr, rows)

    def tile_row_reci(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_reci_asm(matrix_addr, rows)

    def tile_row_sub_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_sub_fp_asm(matrix_addr, row_map)

    def tile_row_mul_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_mul_fp_asm(matrix_addr, row_map)

    def tile_row_add_fp(
        self,
        matrix_name: str,
        row_map: list[tuple[int, int]],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_add_fp_asm(matrix_addr, row_map)

    def tile_row_add(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_add_asm(dst_addr, src_addr, rows)

    def tile_row_sub(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_sub_asm(dst_addr, src_addr, rows)

    def tile_row_mul(
        self,
        dst_matrix: str,
        src_matrix: str,
        rows: list[int],
        dst_tile_row_idx: int = 0,
        dst_tile_col_idx: int = 0,
        src_tile_row_idx: int = 0,
        src_tile_col_idx: int = 0,
    ) -> str:
        dst_addr = self.get_vram_tile_addr(dst_matrix, dst_tile_row_idx, dst_tile_col_idx)
        src_addr = self.get_vram_tile_addr(src_matrix, src_tile_row_idx, src_tile_col_idx)
        return self.tile_row_mul_asm(dst_addr, src_addr, rows)

    def tile_row_mul_fp_broadcast(
        self,
        matrix_name: str,
        fpram_scalar_addr: int,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.tile_row_mul_fp_broadcast_asm(matrix_addr, fpram_scalar_addr, rows)

    def vram_fill_zero(
        self,
        matrix_name: str,
        rows: list[int],
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> str:
        matrix_addr = self.get_vram_tile_addr(matrix_name, tile_row_idx, tile_col_idx)
        return self.vram_fill_zero_asm(matrix_addr, rows)

    # =========================================================================
    # Tile-row ISA helpers (address-based)
    # =========================================================================

    def _arith_progression(self, values: list[int]) -> tuple[int, int, int] | None:
        """Return (start, count, step) if values form an arithmetic progression."""
        if not values:
            return None
        if len(values) == 1:
            return (values[0], 1, 0)
        step = values[1] - values[0]
        for i in range(2, len(values)):
            if values[i] - values[i - 1] != step:
                return None
        if step == 0:
            return None  # Constant sequence (step=0, count>1) would cause infinite HW loop
        return (values[0], len(values), step)

    def _row_progression(self, rows: list[int]) -> tuple[int, int, int] | None:
        return None if self._unroll else self._arith_progression(rows)

    def _emit_tile_row_reduce(
        self,
        label: str,
        source_vram_addr: int,
        row_map: list[tuple[int, int]],
        opcode: str,
        opcode_extra_args: tuple[int, ...] = (),
        clear_accumulator: bool = False,
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row {label} from VRAM[{source_vram_addr}]")
            rows = [row for row, _ in row_map]
            fp_addrs = [addr_ for _, addr_ in row_map]
            row_prog = self._row_progression(rows)
            fp_prog = self._row_progression(fp_addrs)

            if row_prog is not None and fp_prog is not None:
                row_start, row_count, row_step = row_prog
                fp_start, _, fp_step = fp_prog
                asm.instr("S_ADDI_INT", gp(gp_src), gp(0), source_vram_addr + row_start * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fp_start)
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                if clear_accumulator:
                    asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))
                asm.instr(opcode, fp(1), gp(gp_src), 0, *opcode_extra_args)
                asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), row_step * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), fp_step)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx, fpram_addr in row_map:
                    row_addr = source_vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    if clear_accumulator:
                        asm.instr("S_ADD_FP", fp(1), fp(0), fp(0))
                    asm.instr(opcode, fp(1), gp(gp_src), 0, *opcode_extra_args)
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), fpram_addr)
                    asm.instr("S_ST_FP", fp(1), gp(gp_dst), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_unary(self, label: str, opcode: str, vram_addr: int, rows: list[int]) -> str:
        gp_regs = self._reg.allocate_gp(2)
        gp_src, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row {label} on VRAM[{vram_addr}]")
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                asm.instr("S_ADDI_INT", gp(gp_src), gp(0), vram_addr + row_start * self.mlen)
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                asm.instr(opcode, gp(gp_src), gp(gp_src), 0)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), row_step * self.mlen)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_fp_scalar(
        self,
        label: str,
        opcode: str,
        vram_addr: int,
        row_map: list[tuple[int, int]],
        opcode_extra_args: tuple[int, ...] = (),
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"Tile Row {label} FP on VRAM[{vram_addr}]")
            rows = [row for row, _ in row_map]
            fp_addrs = [addr_ for _, addr_ in row_map]
            row_prog = self._row_progression(rows)
            fp_prog = self._row_progression(fp_addrs)

            if row_prog is not None and fp_prog is not None:
                row_start, row_count, row_step = row_prog
                fp_start, _, fp_step = fp_prog
                asm.instr("S_ADDI_INT", gp(gp_src), gp(0), vram_addr + row_start * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fp_start)
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                asm.instr(opcode, gp(gp_src), gp(gp_src), fp(1), *opcode_extra_args)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), row_step * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_fp), gp(gp_fp), fp_step)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx, fpram_addr in row_map:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), row_addr)
                    asm.instr("S_ADDI_INT", gp(gp_fp), gp(0), fpram_addr)
                    asm.instr("S_LD_FP", fp(1), gp(gp_fp), 0)
                    asm.instr(opcode, gp(gp_src), gp(gp_src), fp(1), *opcode_extra_args)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def _emit_tile_row_vector_op(
        self,
        label: str,
        opcode: str,
        dst_addr: int,
        src_addr: int,
        rows: list[int],
    ) -> str:
        gp_regs = self._reg.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp_regs
        try:
            assignment_op = {"Add": "+", "Sub": "-", "Mul": "*"}.get(label, label)
            asm = IsaBuilder().comment(f"Tile Row {label}: VRAM[{dst_addr}] {assignment_op}= VRAM[{src_addr}]")
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_addr + row_start * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_addr + row_start * self.mlen)
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                asm.instr(opcode, gp(gp_dst), gp(gp_dst), gp(gp_src), 0)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), row_step * self.mlen)
                asm.instr("S_ADDI_INT", gp(gp_src), gp(gp_src), row_step * self.mlen)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    dst_row_addr = dst_addr + row_idx * self.mlen
                    src_row_addr = src_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), dst_row_addr)
                    asm.instr("S_ADDI_INT", gp(gp_src), gp(0), src_row_addr)
                    asm.instr(opcode, gp(gp_dst), gp(gp_dst), gp(gp_src), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)

    def tile_row_max_asm(self, source_vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        return self._emit_tile_row_reduce("Max", source_vram_addr, row_map, "V_RED_MAX")

    def tile_row_sum_asm(self, source_vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        return self._emit_tile_row_reduce(
            "Sum",
            source_vram_addr,
            row_map,
            "V_RED_SUM",
            opcode_extra_args=(0,),
            clear_accumulator=True,
        )

    def tile_row_exp_asm(self, vram_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_unary("Exp", "V_EXP_V", vram_addr, rows)

    def tile_row_reci_asm(self, vram_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_unary("Reciprocal", "V_RECI_V", vram_addr, rows)

    def tile_row_sub_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        return self._emit_tile_row_fp_scalar("Sub", "V_SUB_VF", vram_addr, row_map, opcode_extra_args=(0, 0))

    def tile_row_mul_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        return self._emit_tile_row_fp_scalar("Mul", "V_MUL_VF", vram_addr, row_map, opcode_extra_args=(0,))

    def tile_row_add_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        return self._emit_tile_row_fp_scalar("Add", "V_ADD_VF", vram_addr, row_map, opcode_extra_args=(0,))

    def tile_row_add_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op("Add", "V_ADD_VV", dst_addr, src_addr, rows)

    def tile_row_sub_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op("Sub", "V_SUB_VV", dst_addr, src_addr, rows)

    def tile_row_mul_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        return self._emit_tile_row_vector_op("Mul", "V_MUL_VV", dst_addr, src_addr, rows)

    def tile_row_mul_fp_broadcast_asm(self, vram_addr: int, fpram_scalar_addr: int, rows: list[int]) -> str:
        row_map = [(r, fpram_scalar_addr) for r in rows]
        return self.tile_row_mul_fp_asm(vram_addr, row_map)

    def vram_fill_zero_asm(
        self,
        vram_addr: int,
        rows: list[int],
    ) -> str:
        """
        VRAM Fill Zero: fill specified rows with 0.

        For each row_idx in rows:
            VRAM[row] = 0
        """
        if not rows:
            return self._emit(IsaBuilder().comment(f"=== VRAM Fill Zero: VRAM[{vram_addr}] rows [] = 0 ==="))

        gp_regs = self._reg.allocate_gp(2)
        gp_dst, gp_loop = gp_regs
        try:
            asm = IsaBuilder().comment(f"=== VRAM Fill Zero: VRAM[{vram_addr}] rows {rows} = 0 ===")
            prog = self._row_progression(rows)

            if prog is not None:
                row_start, row_count, row_step = prog
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), vram_addr + row_start * self.mlen)
                asm.instr("C_LOOP_START", gp(gp_loop), row_count)
                asm.instr("V_MUL_VF", gp(gp_dst), gp(gp_dst), fp(0), 0)
                asm.instr("S_ADDI_INT", gp(gp_dst), gp(gp_dst), row_step * self.mlen)
                asm.instr("C_LOOP_END", gp(gp_loop))
            else:
                for row_idx in rows:
                    row_addr = vram_addr + row_idx * self.mlen
                    asm.instr("S_ADDI_INT", gp(gp_dst), gp(0), row_addr)
                    asm.instr("V_MUL_VF", gp(gp_dst), gp(gp_dst), fp(0), 0)

            return self._emit(asm)
        finally:
            self._reg.free_gp(gp_regs)


__all__ = ["IsaTileRowMixin"]
