"""Matrix movement and VRAM projection helpers for IsaCompiler."""

from __future__ import annotations

from compiler.asm_templates import preload_addr_reg_asm
from compiler.aten.isa_builder import IsaBuilder


class IsaMatrixMixin:
    def reset_mram(self) -> str:
        """
        Reset MRAM allocator, free all allocated space
        Used in scenarios where sub-blocks need to be reloaded within a for loop
        """
        self.mram_allocator.reset()
        self.loaded_sub_blocks.clear()

        return self._emit(IsaBuilder().comment("=== Reset MRAM ==="))

    def load_sub_matrix_row(
        self,
        name: str,
        row_idx: int,
        mram_start_addr: int | None = None,
    ) -> str:
        """Load entire row sub-blocks from HBM to MRAM: matrix[row_idx][:]."""
        layout = self.get_hbm_layout(name)
        num_col_blocks = layout.num_col_blocks
        block_size = self.mlen * self.mlen

        if mram_start_addr is None:
            total_size = num_col_blocks * block_size
            mram_start_addr = self.mram_allocator.allocate(f"{name}[{row_idx}][:]", total_size)

        gp_regs = self.register_allocator.allocate_gp(4)
        gp_for_addr = self.register_allocator.allocate_gp(2)
        addr_reg = self.register_allocator.allocate_addr(1)[0]

        isa_code = preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg], available_registers=gp_for_addr, addr_reg_val=[layout.hbm_base_addr]
        )

        isa_code += self.load_row_sub_matrices_asm(
            name=name, row_idx=row_idx, mram_start_addr=mram_start_addr, hbm_addr_reg=addr_reg, gp_regs=gp_regs
        )

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_gp(gp_for_addr)
        self.register_allocator.free_addr([addr_reg])

        return self._emit(isa_code)

    def load_sub_matrix_col(
        self,
        name: str,
        col_idx: int,
        mram_start_addr: int | None = None,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ) -> str:
        """
        Load entire column sub-blocks from HBM to MRAM: matrix[:][col_idx].
        Used for sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen].
        """
        layout = self.get_hbm_layout(name)
        num_row_blocks = layout.num_row_blocks
        block_size = self.mlen * self.mlen

        if mram_start_addr is None:
            effective_count = k_block_count if k_block_count is not None else num_row_blocks
            total_size = effective_count * block_size
            mram_start_addr = self.mram_allocator.allocate(f"{name}[:][{col_idx}]", total_size)

        gp_regs = self.register_allocator.allocate_gp(3)
        gp_for_addr = self.register_allocator.allocate_gp(2)
        addr_reg = self.register_allocator.allocate_addr(1)[0]

        isa_code = preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg], available_registers=gp_for_addr, addr_reg_val=[layout.hbm_base_addr]
        )

        isa_code += self.load_col_sub_matrices_asm(
            name=name,
            col_idx=col_idx,
            mram_start_addr=mram_start_addr,
            hbm_addr_reg=addr_reg,
            gp_regs=gp_regs,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_gp(gp_for_addr)
        self.register_allocator.free_addr([addr_reg])

        return self._emit(isa_code)

    def allocate_vram_matrix(
        self,
        name: str,
        rows: int,
        cols: int,
        strict: bool = True,
    ) -> int:
        """Allocate a VRAM matrix large enough to hold combined results of multiple sub-blocks. Returns the VRAM base address."""
        size = rows * cols
        vram_addr = self.vram_allocator.allocate(size, name=name)

        self.add_vram_object(
            name=name,
            shape=(rows, cols),
            vram_addr=vram_addr,
            dtype="fp32",
            kind="VRAMMatrix",
            allocate_if_none=False,
            strict=strict,
        )

        isa_code = f"; Allocate VRAM Matrix {name}: ({rows}, {cols}) at VRAM[{vram_addr}]\n"
        self._emit(isa_code)

        return vram_addr

    def _ensure_vram_matrix_layout(self, matrix_name: str):
        """Ensure a VRAM-resident tensor has a block layout in TileCompiler."""
        if matrix_name not in self:
            raise KeyError(f"Matrix '{matrix_name}' not found in symbol table")

        info = self[matrix_name]
        if info.vram_addr is None:
            raise ValueError(f"Matrix '{matrix_name}' has no VRAM address")

        try:
            self.get_vram_layout(matrix_name)
        except KeyError:
            self.register_vram_matrix(
                name=matrix_name,
                shape=info.shape,
                vram_base_addr=info.vram_addr,
            )

    def vram_block_add_to(
        self,
        src1_matrix: str,
        src1_row_idx: int,
        src1_col_idx: int,
        src2_matrix: str,
        src2_row_idx: int,
        src2_col_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
    ) -> str:
        """
        mlen x mlen block add:
            target[rt][ct] = src1[r1][c1] + src2[r2][c2]

        Source/target may be the same matrix (supports in-place overwrite).
        """
        self._ensure_vram_matrix_layout(src1_matrix)
        self._ensure_vram_matrix_layout(src2_matrix)
        self._ensure_vram_matrix_layout(target_matrix)

        gp_regs = self.register_allocator.allocate_gp(4)
        isa_code = self.vram_block_add_asm(
            src1_name=src1_matrix,
            src1_row_idx=src1_row_idx,
            src1_col_idx=src1_col_idx,
            src2_name=src2_matrix,
            src2_row_idx=src2_row_idx,
            src2_col_idx=src2_col_idx,
            target_name=target_matrix,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            gp_regs=gp_regs,
        )
        self.register_allocator.free_gp(gp_regs)

        return self._emit(isa_code)

    def vram_matrix_add(
        self,
        dst_matrix: str,
        src_matrix: str,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int | None = None,
    ) -> str:
        """
        General VRAM Matrix Addition: dst[row_offset:] += src.

        row_offsets are logical rows (not VRAM addresses); num_rows defaults
        to the source matrix's row count.
        """
        dst_info = self[dst_matrix]
        src_info = self[src_matrix]

        # Block-add path depends on TileCompiler VRAM layouts.
        self._ensure_vram_matrix_layout(dst_matrix)
        self._ensure_vram_matrix_layout(src_matrix)

        dst_addr = dst_info.vram_addr
        src_addr = src_info.vram_addr

        dst_rows, dst_cols = dst_info.shape
        src_rows, src_cols = src_info.shape

        if num_rows is None:
            num_rows = src_rows

        # Ensure column count matches
        assert dst_cols == src_cols, f"Column mismatch: dst={dst_cols}, src={src_cols}"
        assert dst_row_offset + num_rows <= dst_rows, (
            f"dst row range out of bounds: offset={dst_row_offset}, num_rows={num_rows}, dst_rows={dst_rows}"
        )
        assert src_row_offset + num_rows <= src_rows, (
            f"src row range out of bounds: offset={src_row_offset}, num_rows={num_rows}, src_rows={src_rows}"
        )
        lines = []
        lines.append(
            f"; === VRAM Matrix Add: "
            f"{dst_matrix}[{dst_row_offset}:{dst_row_offset + num_rows}] += "
            f"{src_matrix}[{src_row_offset}:{src_row_offset + num_rows}] ==="
        )
        lines.append(f"; dst shape: {dst_info.shape}, src shape: {src_info.shape}")

        # Prefer block add path so we can reuse the compact C_LOOP-based add kernel.
        block_aligned = (
            dst_cols % self.mlen == 0
            and src_cols % self.mlen == 0
            and dst_row_offset % self.mlen == 0
            and src_row_offset % self.mlen == 0
            and num_rows % self.mlen == 0
        )

        if block_aligned:
            num_row_blocks = num_rows // self.mlen
            num_col_blocks = dst_cols // self.mlen
            dst_row_block_base = dst_row_offset // self.mlen
            src_row_block_base = src_row_offset // self.mlen
            lines.append(f"; block add path: row_blocks={num_row_blocks}, col_blocks={num_col_blocks}")

            for row_block in range(num_row_blocks):
                for col_block in range(num_col_blocks):
                    gp_regs = self.register_allocator.allocate_gp(4)
                    lines.append(
                        self.vram_block_add_asm(
                            src1_name=dst_matrix,
                            src1_row_idx=dst_row_block_base + row_block,
                            src1_col_idx=col_block,
                            src2_name=src_matrix,
                            src2_row_idx=src_row_block_base + row_block,
                            src2_col_idx=col_block,
                            target_name=dst_matrix,
                            target_row_idx=dst_row_block_base + row_block,
                            target_col_idx=col_block,
                            gp_regs=gp_regs,
                        ).rstrip("\n")
                    )
                    self.register_allocator.free_gp(gp_regs)
        else:
            # Fallback for non-mlen-aligned ranges.
            gp_regs = self.register_allocator.allocate_gp(2)
            gp_dst = gp_regs[0]
            gp_src = gp_regs[1]
            num_col_blocks = dst_cols // self.mlen
            lines.append(f"; fallback row-wise path: num_rows={num_rows}, num_col_blocks={num_col_blocks}")

            for row in range(num_rows):
                dst_actual_row = dst_row_offset + row
                src_actual_row = src_row_offset + row

                for col_block in range(num_col_blocks):
                    dst_block_addr = dst_addr + col_block * dst_rows * self.mlen + dst_actual_row * self.mlen
                    src_block_addr = src_addr + col_block * src_rows * self.mlen + src_actual_row * self.mlen

                    lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_block_addr}")
                    lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_block_addr}")
                    lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            self.register_allocator.free_gp(gp_regs)

        isa_code = "\n".join(lines) + "\n"
        return self._emit(isa_code)

    def vram_sub_projection_to(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_col_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ) -> str:
        """
        Sub-block multiplication:
          target[target_row_idx][target_col_idx] = VRAM_A[vram_row_idx][:] @ MRAM_W[:][mram_col_idx].
        Target matrix must have been allocated via allocate_vram_matrix.
        """
        if target_matrix not in self:
            raise KeyError(f"Target matrix '{target_matrix}' not found. Use allocate_vram_matrix first.")

        target_info = self[target_matrix]
        target_rows, _target_cols = target_info.shape
        target_base_addr = target_info.vram_addr

        # VRAM layout: [batch, mlen, hidden/mlen], column-block major.
        # Sub-block (r, c) addr = base + c * rows * mlen + r * mlen * mlen.
        result_vram_addr = (
            target_base_addr + target_col_idx * target_rows * self.mlen + target_row_idx * self.mlen * self.mlen
        )

        gp_regs = self.register_allocator.allocate_gp(9)

        isa_code = f"; VRAM Sub Projection To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[:][{mram_col_idx}] -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
        isa_code += f"; Target VRAM addr: {result_vram_addr} (base={target_base_addr}, offset=col*{target_rows}*{self.mlen} + row*{self.mlen}*{self.mlen})\n"
        isa_code += self.vram_sub_projection_asm(
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_col_idx=mram_col_idx,
            result_vram_addr=result_vram_addr,
            gp_regs=gp_regs,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

        self.register_allocator.free_gp(gp_regs)

        return self._emit(isa_code)

    def vram_sub_projection_T_to(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_row_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
    ) -> str:
        """
        Transposed sub-block multiplication:
          target[target_row_idx][target_col_idx] = VRAM_A[vram_row_idx][:] @ MRAM_W[mram_row_idx][:]^T.

        Used by Flash Attention for S = Q @ K^T:
          Q[i][:]: (mlen, hidden_size) row sub-block
          K[j][:]: (mlen, hidden_size) row sub-block, transposed to (hidden_size, mlen)
          S[i][j]: (mlen, mlen)
        """
        if target_matrix not in self:
            raise KeyError(f"Target matrix '{target_matrix}' not found. Use allocate_vram_matrix first.")

        target_info = self[target_matrix]
        target_rows, _target_cols = target_info.shape
        target_base_addr = target_info.vram_addr

        # VRAM layout: [batch, mlen, hidden/mlen], column-block major.
        # Sub-block (r, c) addr = base + c * rows * mlen + r * mlen * mlen.
        result_vram_addr = (
            target_base_addr + target_col_idx * target_rows * self.mlen + target_row_idx * self.mlen * self.mlen
        )

        gp_regs = self.register_allocator.allocate_gp(9)

        isa_code = f"; VRAM Sub Projection T To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[{mram_row_idx}][:]^T -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
        isa_code += f"; Target VRAM addr: {result_vram_addr} (base={target_base_addr}, offset=col*{target_rows}*{self.mlen} + row*{self.mlen}*{self.mlen})\n"
        isa_code += self.vram_sub_projection_T_asm(
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_row_idx=mram_row_idx,
            result_vram_addr=result_vram_addr,
            gp_regs=gp_regs,
        )

        self.register_allocator.free_gp(gp_regs)

        return self._emit(isa_code)


__all__ = ["IsaMatrixMixin"]
