"""Matrix movement and VRAM projection helpers for IsaCompiler."""

from __future__ import annotations

from compiler.asm_templates import preload_addr_reg_asm
from compiler.asm_templates.vram_sub_projection_asm import vram_sub_projection_asm_impl
from compiler.aten.isa_builder import IsaBuilder, addr as areg, gp


class IsaMatrixMixin:
    def _emit_hbm_matrix_load(self, layout, gp_count: int, build_body) -> str:
        gp_regs = self.register_allocator.allocate_gp(gp_count)
        gp_for_addr = self.register_allocator.allocate_gp(2)
        addr_reg = self.register_allocator.allocate_addr(1)[0]
        try:
            isa_code = preload_addr_reg_asm(
                addr_reg_to_set=[addr_reg],
                available_registers=gp_for_addr,
                addr_reg_val=[layout.hbm_base_addr],
            )
            isa_code += build_body(addr_reg, gp_regs)
            return self._emit(isa_code)
        finally:
            self.register_allocator.free_gp(gp_regs)
            self.register_allocator.free_gp(gp_for_addr)
            self.register_allocator.free_addr([addr_reg])

    def reset_mram(self) -> str:
        """
        Reset MRAM allocator, free all allocated space
        Used in scenarios where sub-blocks need to be reloaded within a for loop
        """
        self.mram_allocator.reset()
        self.clear_mram_bindings()

        return self._emit(IsaBuilder().comment("=== Reset MRAM ==="))

    def _default_hbm_gp_regs(self, gp_regs: list[int] | None) -> list[int]:
        return [1, 2, 3] if gp_regs is None else gp_regs

    def _emit_hbm_prefetch_setup(self, asm: IsaBuilder, layout, gp_scale: int, gp_stride: int) -> None:
        rows, cols = layout.full_shape
        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), rows * cols)
        asm.instr("C_SET_SCALE_REG", gp(gp_scale))
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), cols)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

    def _emit_hbm_subblock_prefetch(
        self,
        asm: IsaBuilder,
        layout,
        row_idx: int,
        col_idx: int,
        mram_addr: int,
        hbm_addr_reg: int,
        gp_scale: int,
        gp_mram: int,
        comment: str | None = None,
    ) -> None:
        sub_block = layout.get_sub_block(row_idx, col_idx)
        hbm_offset = sub_block.hbm_offset
        sub_block.mram_addr = mram_addr

        asm.comment(comment if comment is not None else f"SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
        asm.instr("S_ADDI_INT", gp(gp_mram), gp(0), mram_addr)
        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), hbm_offset)
        asm.instr("H_PREFETCH_M", gp(gp_mram), gp(gp_scale), areg(hbm_addr_reg), 1, 0)

    def _emit_hbm_subblock_sequence(
        self,
        asm: IsaBuilder,
        layout,
        block_coords,
        mram_start_addr: int,
        hbm_addr_reg: int,
        gp_scale: int,
        gp_mram: int,
    ) -> None:
        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen
        for row_idx, col_idx in block_coords:
            self._emit_hbm_subblock_prefetch(
                asm,
                layout,
                row_idx,
                col_idx,
                mram_addr,
                hbm_addr_reg,
                gp_scale,
                gp_mram,
            )
            mram_addr += block_size

    def load_sub_matrix_asm(
        self,
        name: str,
        row_idx: int,
        col_idx: int,
        mram_dest_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: list[int] | None = None,
    ) -> str:
        """Emit HBM->MRAM prefetch for one mlen x mlen sub-block."""
        gp_regs = self._default_hbm_gp_regs(gp_regs)
        layout = self.hbm_matrices[name]

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix {name}[{row_idx}][{col_idx}] -> MRAM[{mram_dest_addr}]")
        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        self._emit_hbm_prefetch_setup(asm, layout, gp_scale, gp_stride)
        hbm_offset = layout.get_sub_block(row_idx, col_idx).hbm_offset
        self._emit_hbm_subblock_prefetch(
            asm,
            layout,
            row_idx,
            col_idx,
            mram_dest_addr,
            hbm_addr_reg,
            gp_scale,
            gp_mram,
            comment=f"HBM offset: {hbm_offset} (precomputed)",
        )

        return asm.render()

    def load_row_sub_matrices_asm(
        self,
        name: str,
        row_idx: int,
        mram_start_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: list[int] | None = None,
    ) -> str:
        """Emit HBM->MRAM prefetches for one block row."""
        gp_regs = self._default_hbm_gp_regs(gp_regs)
        layout = self.hbm_matrices[name]

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix Row {name}[{row_idx}][:] -> MRAM[{mram_start_addr}]")

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        self._emit_hbm_prefetch_setup(asm, layout, gp_scale, gp_stride)

        self._emit_hbm_subblock_sequence(
            asm,
            layout,
            ((row_idx, col_idx) for col_idx in range(layout.num_col_blocks)),
            mram_start_addr,
            hbm_addr_reg,
            gp_scale,
            gp_mram,
        )

        return asm.render()

    def load_col_sub_matrices_asm(
        self,
        name: str,
        col_idx: int,
        mram_start_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: list[int] | None = None,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ) -> str:
        """Emit HBM->MRAM prefetches for one block column or K-split slice."""
        gp_regs = self._default_hbm_gp_regs(gp_regs)
        layout = self.hbm_matrices[name]
        num_row_blocks = layout.num_row_blocks

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix Col {name}[:][{col_idx}] -> MRAM[{mram_start_addr}]")

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]
        self._emit_hbm_prefetch_setup(asm, layout, gp_scale, gp_stride)

        effective_count = k_block_count if k_block_count is not None else num_row_blocks
        self._emit_hbm_subblock_sequence(
            asm,
            layout,
            ((row_idx, col_idx) for row_idx in range(k_block_start, k_block_start + effective_count)),
            mram_start_addr,
            hbm_addr_reg,
            gp_scale,
            gp_mram,
        )

        return asm.render()

    def _default_projection_gp_regs(self, gp_regs: list[int] | None) -> list[int]:
        return [1, 2, 3, 4, 5, 6, 7, 8, 9] if gp_regs is None else gp_regs

    def _projection_context(self, vram_mat_name: str, vram_row_idx: int, mram_mat_name: str):
        if vram_mat_name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{vram_mat_name}' not registered")
        vram_layout = self.vram_matrices[vram_mat_name]
        return vram_layout, self.hbm_matrices[mram_mat_name], vram_layout.get_row_blocks(vram_row_idx)

    def _loaded_mram_start(self, blocks, missing_label) -> int:
        for sub_block in blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(f"SubBlock {missing_label(sub_block)} not loaded to MRAM")
        return blocks[0].mram_addr

    def _vram_sub_projection_asm_impl(
        self,
        header_lines: list[str],
        vram_row_start_addr: int,
        mram_start_addr: int,
        result_vram_addr: int,
        full_batch: int,
        num_hidden_blocks: int,
        mat_col_stride: int,
        transposed: bool,
        gp_regs: list[int],
        caller_name: str,
        unroll: bool | None = None,
    ) -> str:
        """Emit the shared projection loop after callers resolve operands."""
        do_unroll = self.unroll_loops if unroll is None else unroll
        return vram_sub_projection_asm_impl(
            mlen=self.mlen,
            blen=self.blen,
            unroll_loops=do_unroll,
            header_lines=header_lines,
            vram_row_start_addr=vram_row_start_addr,
            mram_start_addr=mram_start_addr,
            result_vram_addr=result_vram_addr,
            full_batch=full_batch,
            num_hidden_blocks=num_hidden_blocks,
            mat_col_stride=mat_col_stride,
            transposed=transposed,
            gp_regs=gp_regs,
            caller_name=caller_name,
        )

    def vram_sub_projection_asm(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_col_idx: int,
        result_vram_addr: int,
        gp_regs: list[int] | None = None,
        k_block_start: int = 0,
        k_block_count: int | None = None,
        unroll: bool | None = None,
    ) -> str:
        """Emit VRAM[row][:] @ MRAM[:][col] projection."""
        gp_regs = self._default_projection_gp_regs(gp_regs)
        vram_layout, mram_layout, vram_row_blocks = self._projection_context(
            vram_mat_name, vram_row_idx, mram_mat_name
        )
        mram_col_blocks = mram_layout.get_col_blocks(mram_col_idx)
        if k_block_count is not None:
            mram_col_blocks = mram_col_blocks[k_block_start : k_block_start + k_block_count]

        num_hidden_blocks = len(mram_col_blocks)
        if num_hidden_blocks != (k_block_count if k_block_count is not None else len(vram_row_blocks)):
            raise ValueError(
                f"Dimension mismatch: expected {k_block_count or len(vram_row_blocks)} MRAM blocks, "
                f"got {num_hidden_blocks}"
            )

        full_batch = vram_layout.full_shape[0]
        vram_row_start_addr = vram_row_blocks[k_block_start].vram_addr
        mram_col_start_addr = self._loaded_mram_start(
            mram_col_blocks,
            lambda block: f"{mram_mat_name}[{block.row_idx}][{mram_col_idx}]",
        )

        header_lines = [
            f"; VRAM Sub Projection: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[:][{mram_col_idx}]",
            f"; VRAM A[row_idx][:]: ({self.mlen}, hidden) spread across {num_hidden_blocks} column blocks",
            f"; MRAM W[:][col_idx]: (hidden, {self.mlen}) with {num_hidden_blocks} sub-blocks",
            f"; Result: ({self.mlen}, {self.mlen}) at VRAM[{result_vram_addr}]",
        ]

        return self._vram_sub_projection_asm_impl(
            header_lines=header_lines,
            vram_row_start_addr=vram_row_start_addr,
            mram_start_addr=mram_col_start_addr,
            result_vram_addr=result_vram_addr,
            full_batch=full_batch,
            num_hidden_blocks=num_hidden_blocks,
            mat_col_stride=self.blen,
            transposed=False,
            gp_regs=gp_regs,
            caller_name="vram_sub_projection_asm",
            unroll=unroll,
        )

    def vram_sub_projection_T_asm(
        self,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_row_idx: int,
        result_vram_addr: int,
        gp_regs: list[int] | None = None,
        unroll: bool | None = None,
    ) -> str:
        """Emit VRAM[row][:] @ MRAM[row][:]^T projection."""
        gp_regs = self._default_projection_gp_regs(gp_regs)
        vram_layout, mram_layout, vram_row_blocks = self._projection_context(
            vram_mat_name, vram_row_idx, mram_mat_name
        )
        mram_row_blocks = mram_layout.get_row_blocks(mram_row_idx)

        if len(vram_row_blocks) != len(mram_row_blocks):
            raise ValueError(
                f"Dimension mismatch: VRAM has {len(vram_row_blocks)} blocks, MRAM has {len(mram_row_blocks)} blocks"
            )

        num_hidden_blocks = len(vram_row_blocks)
        full_batch = vram_layout.full_shape[0]
        vram_row_start_addr = vram_row_blocks[0].vram_addr
        mram_row_start_addr = self._loaded_mram_start(
            mram_row_blocks,
            lambda block: f"{mram_mat_name}[{mram_row_idx}][{block.col_idx}]",
        )
        # M_TMM reads the weight in transposed layout, so the outer-column
        # stride is one full sub-block instead of the non-transposed blen stride.
        mat_col_stride = self.blen * self.mlen

        header_lines = [
            f"; VRAM Sub Projection T: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[{mram_row_idx}][:]^T",
            f"; VRAM A[row_idx][:]: ({self.mlen}, hidden)",
            f"; MRAM W[row_idx][:]^T: (hidden, {self.mlen})",
            f"; Result: ({self.mlen}, {self.mlen}) at VRAM[{result_vram_addr}]",
        ]

        return self._vram_sub_projection_asm_impl(
            header_lines=header_lines,
            vram_row_start_addr=vram_row_start_addr,
            mram_start_addr=mram_row_start_addr,
            result_vram_addr=result_vram_addr,
            full_batch=full_batch,
            num_hidden_blocks=num_hidden_blocks,
            mat_col_stride=mat_col_stride,
            transposed=True,
            gp_regs=gp_regs,
            caller_name="vram_sub_projection_T_asm",
            unroll=unroll,
        )

    def vram_block_add_asm(
        self,
        src1_name: str,
        src1_row_idx: int,
        src1_col_idx: int,
        src2_name: str,
        src2_row_idx: int,
        src2_col_idx: int,
        target_name: str,
        target_row_idx: int,
        target_col_idx: int,
        gp_regs: list[int] | None = None,
    ) -> str:
        """
        Add two mlen x mlen blocks and write to any target block:
            target[target_row_idx][target_col_idx] =
                src1[src1_row_idx][src1_col_idx] + src2[src2_row_idx][src2_col_idx]
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4]
        if len(gp_regs) < 4:
            raise ValueError(f"Need at least 4 GP regs, got {len(gp_regs)}")

        src1_block = self.get_vram_sub_block(src1_name, src1_row_idx, src1_col_idx)
        src2_block = self.get_vram_sub_block(src2_name, src2_row_idx, src2_col_idx)
        target_block = self.get_vram_sub_block(target_name, target_row_idx, target_col_idx)

        gp_dst = gp_regs[0]
        gp_src1 = gp_regs[1]
        gp_src2 = gp_regs[2]
        gp_loop = gp_regs[3]

        lines = [
            f"; VRAM Block Add: {target_name}[{target_row_idx}][{target_col_idx}] = "
            f"{src1_name}[{src1_row_idx}][{src1_col_idx}] + {src2_name}[{src2_row_idx}][{src2_col_idx}]"
        ]

        if self.unroll_loops:
            for i in range(self.mlen):
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {target_block.vram_addr + i * self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_src1}, gp0, {src1_block.vram_addr + i * self.mlen}")
                lines.append(f"S_ADDI_INT gp{gp_src2}, gp0, {src2_block.vram_addr + i * self.mlen}")
                lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_src1}, gp{gp_src2}, 0")
        else:
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {target_block.vram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_src1}, gp0, {src1_block.vram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_src2}, gp0, {src2_block.vram_addr}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {self.mlen}")
            lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_src1}, gp{gp_src2}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src1}, gp{gp_src1}, {self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src2}, gp{gp_src2}, {self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")

        return "\n".join(lines) + "\n"

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

        return self._emit_hbm_matrix_load(
            layout,
            3,
            lambda addr_reg, gp_regs: self.load_row_sub_matrices_asm(
                name=name,
                row_idx=row_idx,
                mram_start_addr=mram_start_addr,
                hbm_addr_reg=addr_reg,
                gp_regs=gp_regs,
            ),
        )

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

        return self._emit_hbm_matrix_load(
            layout,
            3,
            lambda addr_reg, gp_regs: self.load_col_sub_matrices_asm(
                name=name,
                col_idx=col_idx,
                mram_start_addr=mram_start_addr,
                hbm_addr_reg=addr_reg,
                gp_regs=gp_regs,
                k_block_start=k_block_start,
                k_block_count=k_block_count,
            ),
        )

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
        """Ensure a VRAM-resident tensor has a block layout."""
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

        # Block-add path depends on registered VRAM block layouts.
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

    def _target_tile_addr(self, target_matrix: str, target_row_idx: int, target_col_idx: int) -> tuple[int, int, int]:
        if target_matrix not in self:
            raise KeyError(f"Target matrix '{target_matrix}' not found. Use allocate_vram_matrix first.")

        target_info = self[target_matrix]
        target_rows, _target_cols = target_info.shape
        target_base_addr = target_info.vram_addr
        result_vram_addr = (
            target_base_addr + target_col_idx * target_rows * self.mlen + target_row_idx * self.mlen * self.mlen
        )
        return result_vram_addr, target_base_addr, target_rows

    def _emit_vram_sub_projection_to(
        self,
        *,
        transposed: bool,
        vram_mat_name: str,
        vram_row_idx: int,
        mram_mat_name: str,
        mram_idx: int,
        target_matrix: str,
        target_row_idx: int,
        target_col_idx: int,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ) -> str:
        result_vram_addr, target_base_addr, target_rows = self._target_tile_addr(
            target_matrix, target_row_idx, target_col_idx
        )
        gp_regs = self.register_allocator.allocate_gp(9)

        if transposed:
            isa_code = f"; VRAM Sub Projection T To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[{mram_idx}][:]^T -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
            asm = self.vram_sub_projection_T_asm(
                vram_mat_name=vram_mat_name,
                vram_row_idx=vram_row_idx,
                mram_mat_name=mram_mat_name,
                mram_row_idx=mram_idx,
                result_vram_addr=result_vram_addr,
                gp_regs=gp_regs,
            )
        else:
            isa_code = f"; VRAM Sub Projection To: {vram_mat_name}[{vram_row_idx}][:] @ {mram_mat_name}[:][{mram_idx}] -> {target_matrix}[{target_row_idx}][{target_col_idx}]\n"
            asm = self.vram_sub_projection_asm(
                vram_mat_name=vram_mat_name,
                vram_row_idx=vram_row_idx,
                mram_mat_name=mram_mat_name,
                mram_col_idx=mram_idx,
                result_vram_addr=result_vram_addr,
                gp_regs=gp_regs,
                k_block_start=k_block_start,
                k_block_count=k_block_count,
            )
        isa_code += f"; Target VRAM addr: {result_vram_addr} (base={target_base_addr}, offset=col*{target_rows}*{self.mlen} + row*{self.mlen}*{self.mlen})\n"
        isa_code += asm

        self.register_allocator.free_gp(gp_regs)
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
        return self._emit_vram_sub_projection_to(
            transposed=False,
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_idx=mram_col_idx,
            target_matrix=target_matrix,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

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
        return self._emit_vram_sub_projection_to(
            transposed=True,
            vram_mat_name=vram_mat_name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_mat_name,
            mram_idx=mram_row_idx,
            target_matrix=target_matrix,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )


__all__ = ["IsaMatrixMixin"]
