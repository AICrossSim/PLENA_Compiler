"""Tile/block lowering helpers for the ATen PLENA compiler."""

from __future__ import annotations

from compiler.asm_templates.vram_sub_projection_asm import vram_sub_projection_asm_impl
from compiler.aten.isa_builder import IsaBuilder, addr as areg, gp
from compiler.aten.plena.constants import BLEN, MLEN
from compiler.aten.plena.memory import (
    FPRAMAllocator,
    FPRAMObjectLayout,
    MRAMAllocator,
    MatrixBlockLayout,
    MemoryObjectInfo,
    SubMatrixInfo,
    VRAMAllocator,
    VRAMMatrixBlockLayout,
    VRAMSubMatrixInfo,
)


class TileCompiler:
    """Sub-matrix layout manager and ISA emitter for tiled PLENA memory ops."""

    def __init__(self, mlen: int = MLEN, blen: int = BLEN, unroll_loops: bool = False):
        self.mlen = mlen
        self.blen = blen
        self.unroll_loops = unroll_loops

        # Layout tables
        self.hbm_matrices: dict[str, MatrixBlockLayout] = {}
        self.vram_matrices: dict[str, VRAMMatrixBlockLayout] = {}
        self.fpram_matrices: dict[str, FPRAMObjectLayout] = {}
        # Memory Allocators
        self.vram_allocator = VRAMAllocator()
        self.mram_allocator = MRAMAllocator()
        self.fpram_allocator = FPRAMAllocator()

        # Currently loaded sub-blocks in MRAM
        self.loaded_sub_blocks: dict[str, SubMatrixInfo] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.hbm_matrices or name in self.vram_matrices or name in self.fpram_matrices

    def __getitem__(self, name: str) -> MemoryObjectInfo:
        if name not in self:
            raise KeyError(f"Object '{name}' not found")
        info = MemoryObjectInfo(name=name, kind="Unknown")
        hbm_layout = self.hbm_matrices.get(name)
        vram_layout = self.vram_matrices.get(name)
        fpram_layout = self.fpram_matrices.get(name)

        if hbm_layout is not None:
            rows, cols = hbm_layout.full_shape
            info.shape = hbm_layout.full_shape
            info.size = rows * cols
            info.hbm_addr = hbm_layout.hbm_base_addr
            info.hbm_size = hbm_layout.hbm_size
            info.kind = "Matrix"

        if vram_layout is not None:
            rows, cols = vram_layout.full_shape
            info.shape = vram_layout.full_shape
            info.size = rows * cols
            info.vram_addr = vram_layout.vram_base_addr
            info.kind = "VRAMMatrix" if hbm_layout is None else "Batch"

        if fpram_layout is not None:
            info.shape = (1, fpram_layout.size)
            info.size = fpram_layout.size
            info.fpram_addr = fpram_layout.fpram_addr
            info.fpram_size = fpram_layout.size
            info.kind = "FPRAMObject"

        return info

    def get_hbm_layout(self, name: str) -> MatrixBlockLayout:
        """Read HBM matrix layout by name."""
        if name not in self.hbm_matrices:
            raise KeyError(f"HBM matrix '{name}' not found")
        return self.hbm_matrices[name]

    def get_vram_layout(self, name: str) -> VRAMMatrixBlockLayout:
        """Read VRAM matrix layout by name."""
        if name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not found")
        return self.vram_matrices[name]

    def get_fpram_layout(self, name: str) -> FPRAMObjectLayout:
        """Read FPRAM object layout by name."""
        if name not in self.fpram_matrices:
            raise KeyError(f"FPRAM object '{name}' not found")
        return self.fpram_matrices[name]

    # ==========================================================================
    # Unified Object Management APIs
    # ==========================================================================

    def add_hbm_object(
        self,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int,
        dtype: str = "fp16",
        kind: str = "HBMObject",
        real_data_ratio: float = 1.125,
        strict: bool = False,
    ) -> MemoryObjectInfo:
        del dtype, kind
        self.register_matrix(
            name=name,
            shape=shape,
            hbm_base_addr=hbm_addr,
            real_data_ratio=real_data_ratio,
            strict=strict,
        )
        return self[name]

    def free_hbm_object(self, name: str, strict: bool = True) -> MemoryObjectInfo | None:
        if name not in self.hbm_matrices:
            if strict:
                raise KeyError(f"HBM object '{name}' not found")
            return None
        info = self[name]
        self.hbm_matrices.pop(name, None)
        return info

    def add_vram_object(
        self,
        name: str,
        shape: tuple[int, int],
        vram_addr: int | None = None,
        dtype: str = "fp16",
        kind: str = "VRAMObject",
        allocate_if_none: bool = True,
        strict: bool = True,
    ) -> MemoryObjectInfo:
        rows, cols = shape
        size = rows * cols
        if vram_addr is None:
            if not allocate_if_none:
                raise ValueError("vram_addr is None and allocate_if_none is False")
            vram_addr = self.vram_allocator.allocate(size=size, name=name)
        del dtype, kind
        self.register_vram_matrix(
            name=name,
            shape=shape,
            vram_base_addr=vram_addr,
            strict=strict,
        )
        return self[name]

    def free_vram_object(self, name: str, strict: bool = True) -> MemoryObjectInfo | None:
        if name not in self.vram_matrices:
            if strict:
                raise KeyError(f"VRAM object '{name}' not found")
            return None
        info = self[name]
        self.vram_allocator.free(name, strict=strict)
        self.vram_matrices.pop(name, None)
        return info

    def add_fpram_object(
        self,
        name: str,
        size: int,
        dtype: str = "fp16",
        kind: str = "FPRAMObject",
    ) -> MemoryObjectInfo:
        fpram_addr = self.fpram_allocator.allocate(name, size)
        self.fpram_matrices[name] = FPRAMObjectLayout(
            name=name,
            fpram_addr=fpram_addr,
            size=size,
            dtype=dtype,
            kind=kind,
        )
        return self[name]

    def free_fpram_object(self, name: str, strict: bool = True) -> MemoryObjectInfo | None:
        if name not in self.fpram_matrices:
            if strict:
                raise KeyError(f"FPRAM object '{name}' not found")
            return None
        info = self[name]
        self.fpram_allocator.free(name, strict=strict)
        self.fpram_matrices.pop(name, None)
        return info

    def register_matrix(
        self,
        name: str,
        shape: tuple[int, int],
        hbm_base_addr: int,
        real_data_ratio: float = 1.125,
        strict: bool = True,
    ) -> MatrixBlockLayout:
        """Register an HBM matrix and derive its mlen block layout."""
        rows, cols = shape

        if strict:
            if rows % self.mlen != 0:
                raise ValueError(f"Matrix rows ({rows}) must be multiple of mlen ({self.mlen})")
            if cols % self.mlen != 0:
                raise ValueError(f"Matrix cols ({cols}) must be multiple of mlen ({self.mlen})")

        size = rows * cols
        hbm_size = int(size * real_data_ratio)

        layout = MatrixBlockLayout(
            name=name, full_shape=shape, block_size=self.mlen, hbm_base_addr=hbm_base_addr, hbm_size=hbm_size
        )

        self.hbm_matrices[name] = layout
        return layout

    # ==========================================================================
    # VRAM Sub-matrix Management
    # ==========================================================================

    def register_vram_matrix(
        self,
        name: str,
        shape: tuple[int, int],
        vram_base_addr: int,
        strict: bool = True,
    ) -> VRAMMatrixBlockLayout:
        """
        Register a large matrix in VRAM and automatically block it.

        Args:
            name: matrix name
            shape: full shape (batch, hidden_size)
            vram_base_addr: VRAM base address

        Returns:
            VRAMMatrixBlockLayout object
        """
        batch, hidden = shape

        if strict:
            if batch % self.mlen != 0:
                raise ValueError(f"VRAM matrix batch ({batch}) must be multiple of mlen ({self.mlen})")
            if hidden % self.mlen != 0:
                raise ValueError(f"VRAM matrix hidden ({hidden}) must be multiple of mlen ({self.mlen})")

        layout = VRAMMatrixBlockLayout(name=name, full_shape=shape, vram_base_addr=vram_base_addr, block_size=self.mlen)

        self.vram_matrices[name] = layout
        return layout

    def get_vram_sub_block(self, name: str, row_idx: int, col_idx: int) -> VRAMSubMatrixInfo:
        """Get VRAM sub-block information"""
        if name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_sub_block(row_idx, col_idx)

    # ==========================================================================
    # ISA Generation: Load Sub Matrix
    # ==========================================================================

    def _default_hbm_gp_regs(self, gp_regs: list[int] | None) -> list[int]:
        return [1, 2, 3] if gp_regs is None else gp_regs

    def _emit_hbm_prefetch_setup(self, asm: IsaBuilder, layout: MatrixBlockLayout, gp_scale: int, gp_stride: int) -> None:
        rows, cols = layout.full_shape
        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), rows * cols)
        asm.instr("C_SET_SCALE_REG", gp(gp_scale))
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), cols)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

    def _emit_hbm_subblock_prefetch(
        self,
        asm: IsaBuilder,
        name: str,
        layout: MatrixBlockLayout,
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

        self.loaded_sub_blocks[f"{name}[{row_idx}][{col_idx}]"] = sub_block

    def _emit_hbm_subblock_sequence(
        self,
        asm: IsaBuilder,
        name: str,
        layout: MatrixBlockLayout,
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
                asm, name, layout, row_idx, col_idx, mram_addr, hbm_addr_reg, gp_scale, gp_mram
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
            name,
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
            name,
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
            name,
            layout,
            ((row_idx, col_idx) for row_idx in range(k_block_start, k_block_start + effective_count)),
            mram_start_addr,
            hbm_addr_reg,
            gp_scale,
            gp_mram,
        )

        return asm.render()

    # ==========================================================================
    # ISA Generation: Sub Projection
    # ==========================================================================

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
        # K-split: slice to only the loaded k-chunk
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
        # NOTE: For M_TMM (transposed matmul), the MRAM outer-column stride is
        # blen * mlen (full sub-block size) because M_TMM reads the weight in
        # transposed layout. This differs from the non-transposed path which uses
        # blen. This is intentional for the M_TMM addressing contract.
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

        Source/target can be the same matrix (supports in-place overwrite).
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

        lines = []
        lines.append(
            f"; VRAM Block Add: {target_name}[{target_row_idx}][{target_col_idx}] = "
            f"{src1_name}[{src1_row_idx}][{src1_col_idx}] + {src2_name}[{src2_row_idx}][{src2_col_idx}]"
        )

        # One V_ADD_VV processes one row (mlen elements). Use C_LOOP to reduce ISA size.
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

    def reset(self):
        """Reset manager state."""
        self.hbm_matrices.clear()
        self.vram_matrices.clear()
        self.fpram_matrices.clear()
        self.vram_allocator.reset()
        self.mram_allocator.reset()
        self.fpram_allocator.reset()
        self.loaded_sub_blocks.clear()

__all__ = ["TileCompiler"]
