"""Tile/block lowering helpers for the ATen PLENA compiler."""

from __future__ import annotations

import math

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
    """
    Tile compiler / sub-matrix manager.

    Core Functions:
    1. Register large matrices as blocked matrices.
    2. Support sub-block indexing: matrix[row_idx][col_idx] or matrix[row_idx][:].
    3. Pre-calculate all addresses at compiler phase.
    4. Generate ISA code for load_sub_matrix and sub_projection.

    Key Constraints:
    - Minimum block size is 64x64 (MLEN).
    - Matrix must be loaded into MRAM before participating in computation.
    - HBM and VRAM/MRAM use different storage formats (row-major vs column-block major).
    """

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

        # Pre-calculated address cache
        self._address_cache: dict[str, int] = {}

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

    def get(self, name: str, default: MemoryObjectInfo | None = None) -> MemoryObjectInfo | None:
        try:
            return self[name]
        except KeyError:
            return default

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
        """
        Register a large matrix and automatically block it.

        Args:
            name: matrix name
            shape: full shape (rows, cols)
            hbm_base_addr: HBM base address
            real_data_ratio: HBM storage ratio (MXFP8 = 1.125)
            strict: if False, skip mlen-alignment check (for raw HBM access)

        Returns:
            MatrixBlockLayout object
        """
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

    def get_sub_block(self, name: str, row_idx: int, col_idx: int) -> SubMatrixInfo:
        """Get sub-block information."""
        if name not in self.hbm_matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.hbm_matrices[name].get_sub_block(row_idx, col_idx)

    def get_row_blocks(self, name: str, row_idx: int) -> list[SubMatrixInfo]:
        """Get all sub-blocks in a row: matrix[row_idx][:]"""
        if name not in self.hbm_matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.hbm_matrices[name].get_row_blocks(row_idx)

    def get_col_blocks(self, name: str, col_idx: int) -> list[SubMatrixInfo]:
        """Get all sub-blocks in a column: matrix[:][col_idx]"""
        if name not in self.hbm_matrices:
            raise KeyError(f"Matrix '{name}' not registered")
        return self.hbm_matrices[name].get_col_blocks(col_idx)

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

    def get_vram_row_blocks(self, name: str, row_idx: int) -> list[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a row of VRAM matrix: matrix[row_idx][:]"""
        if name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_row_blocks(row_idx)

    def get_vram_col_blocks(self, name: str, col_idx: int) -> list[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a column of VRAM matrix: matrix[:][col_idx]"""
        if name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{name}' not registered")
        return self.vram_matrices[name].get_col_blocks(col_idx)

    # ==========================================================================
    # Address Calculation (Core - Pre-calculated during compiler phase)
    # ==========================================================================

    def compute_hbm_offset(self, name: str, row_idx: int, col_idx: int) -> int:
        """
        Compute HBM offset for sub-block (in elements, not bytes).

        HBM row-major: sub-block (r, c) starts at r*block_size*full_cols + c*block_size.
        """
        layout = self.hbm_matrices[name]
        sub_block = layout.get_sub_block(row_idx, col_idx)
        return sub_block.hbm_offset

    def compute_absolute_hbm_addr(self, name: str, row_idx: int, col_idx: int) -> int:
        """
        Calculate absolute HBM address of sub-block (in elements).

        Returns:
            Absolute HBM address = base + offset
        """
        layout = self.hbm_matrices[name]
        offset = self.compute_hbm_offset(name, row_idx, col_idx)
        return layout.hbm_base_addr + offset

    # ==========================================================================
    # ISA Generation: Load Sub Matrix
    # ==========================================================================

    def load_sub_matrix_asm(
        self,
        name: str,
        row_idx: int,
        col_idx: int,
        mram_dest_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: list[int] | None = None,
    ) -> str:
        """
        Generate ISA code for loading sub-matrix from HBM to MRAM.

        HBM is row-major; H_PREFETCH_M loads mlen x mlen blocks into MRAM.
        SCALE = full matrix element count; STRIDE = full column width (row stride).
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]

        layout = self.hbm_matrices[name]
        sub_block = layout.get_sub_block(row_idx, col_idx)

        hbm_offset = sub_block.hbm_offset
        sub_block.mram_addr = mram_dest_addr

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix {name}[{row_idx}][{col_idx}] -> MRAM[{mram_dest_addr}]")
        asm.comment(f"HBM offset: {hbm_offset} (precomputed)")

        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), full_size)
        asm.instr("C_SET_SCALE_REG", gp(gp_scale))
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), full_cols)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

        asm.instr("S_ADDI_INT", gp(gp_mram), gp(0), mram_dest_addr)
        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), hbm_offset)

        asm.instr("H_PREFETCH_M", gp(gp_mram), gp(gp_scale), areg(hbm_addr_reg), 1, 0)

        block_key = f"{name}[{row_idx}][{col_idx}]"
        self.loaded_sub_blocks[block_key] = sub_block

        return asm.render()

    def load_row_sub_matrices_asm(
        self,
        name: str,
        row_idx: int,
        mram_start_addr: int,
        hbm_addr_reg: int = 1,
        gp_regs: list[int] | None = None,
    ) -> str:
        """
        Generate ISA code for loading all sub-blocks in a row: matrix[row_idx][:]

        SCALE and STRIDE set once for all sub-blocks in the row.
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]

        layout = self.hbm_matrices[name]
        num_col_blocks = layout.num_col_blocks

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix Row {name}[{row_idx}][:] -> MRAM[{mram_start_addr}]")

        # Set SCALE and STRIDE once for all sub-blocks
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), full_size)
        asm.instr("C_SET_SCALE_REG", gp(gp_scale))
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), full_cols)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen

        for col_idx in range(num_col_blocks):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset

            sub_block.mram_addr = mram_addr

            asm.comment(f"SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            asm.instr("S_ADDI_INT", gp(gp_mram), gp(0), mram_addr)
            asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), hbm_offset)
            asm.instr("H_PREFETCH_M", gp(gp_mram), gp(gp_scale), areg(hbm_addr_reg), 1, 0)

            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block

            mram_addr += block_size

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
        """
        Generate ISA code for loading all sub-blocks in a column: matrix[:][col_idx].

        Used for sub_projection: A @ W[:, col_idx*mlen:(col_idx+1)*mlen].
        k_block_start/k_block_count select a K-split slice of the column.
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3]

        layout = self.hbm_matrices[name]
        num_row_blocks = layout.num_row_blocks

        asm = IsaBuilder()
        asm.comment(f"Load SubMatrix Col {name}[:][{col_idx}] -> MRAM[{mram_start_addr}]")

        # Set SCALE and STRIDE once for all sub-blocks
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), full_size)
        asm.instr("C_SET_SCALE_REG", gp(gp_scale))
        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), full_cols)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen

        effective_count = k_block_count if k_block_count is not None else num_row_blocks
        for row_idx in range(k_block_start, k_block_start + effective_count):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset

            sub_block.mram_addr = mram_addr

            asm.comment(f"SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            asm.instr("S_ADDI_INT", gp(gp_mram), gp(0), mram_addr)
            asm.instr("S_ADDI_INT", gp(gp_scale), gp(0), hbm_offset)
            asm.instr("H_PREFETCH_M", gp(gp_mram), gp(gp_scale), areg(hbm_addr_reg), 1, 0)

            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block

            mram_addr += block_size

        return asm.render()

    # ==========================================================================
    # ISA Generation: Sub Projection
    # ==========================================================================

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
        """
        Shared implementation kernel for vram_sub_projection_asm and
        vram_sub_projection_T_asm.

        Parameters resolved by the caller before this point:
            header_lines      -- comment lines already assembled by the caller
            vram_row_start_addr -- VRAM address of the first activation block
            mram_start_addr   -- MRAM address of the first weight block
            result_vram_addr  -- VRAM destination address for the (mlen, mlen) result
            full_batch        -- full batch dimension of the activation VRAM matrix
            num_hidden_blocks -- number of K-blocks to accumulate over
            mat_col_stride    -- MRAM outer-column stride (blen for M_MM, blen*mlen for M_TMM)
            transposed        -- True → emit M_TMM with (act, mat) operand order;
                                 False → emit M_MM with (mat, act) operand order
            gp_regs           -- list of at least 9 GP register indices
            caller_name       -- used in error messages only
            unroll            -- override instance unroll_loops flag (None = use instance default)

        Returns:
            ISA code string.
        """
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
        """
        Generate ISA for VRAM sub-block × MRAM sub-matrix multiply.

        Computes: result = VRAM_A[row_idx][:] @ MRAM_W[:][col_idx]

        VRAM_A[row_idx][:] is (mlen, hidden_size) spread across multiple (mlen, mlen) column blocks.
        MRAM_W[:][col_idx] is (hidden_size, mlen) already loaded into MRAM.
        Result is (mlen, mlen) written to VRAM.

        Loop structure (outer=output cols by blen, middle=output rows by blen, inner=K accumulation).
        k_block_start/k_block_count select a K-split chunk.
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        if vram_mat_name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{vram_mat_name}' not registered")
        vram_layout = self.vram_matrices[vram_mat_name]

        mram_layout = self.hbm_matrices[mram_mat_name]

        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
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

        for sub_block in mram_col_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(f"SubBlock {mram_mat_name}[{sub_block.row_idx}][{mram_col_idx}] not loaded to MRAM")

        full_batch = vram_layout.full_shape[0]
        vram_row_start_addr = vram_row_blocks[k_block_start].vram_addr
        mram_col_start_addr = mram_col_blocks[0].mram_addr

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
        """
        Generate ISA for VRAM sub-block × MRAM sub-matrix transposed multiply.

        Computes: result = VRAM_A[row_idx][:] @ MRAM_W[row_idx][:]^T

        VRAM_A[row_idx][:] is (mlen, hidden_size).
        MRAM_W[row_idx][:] is (mlen, hidden_size); transposed to (hidden_size, mlen).
        Result is (mlen, mlen) written to VRAM.

        Uses M_TMM instruction. mat_col_stride = blen*mlen (transposed addressing).
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        if vram_mat_name not in self.vram_matrices:
            raise KeyError(f"VRAM matrix '{vram_mat_name}' not registered")
        vram_layout = self.vram_matrices[vram_mat_name]

        mram_layout = self.hbm_matrices[mram_mat_name]

        vram_row_blocks = vram_layout.get_row_blocks(vram_row_idx)
        mram_row_blocks = mram_layout.get_row_blocks(mram_row_idx)

        if len(vram_row_blocks) != len(mram_row_blocks):
            raise ValueError(
                f"Dimension mismatch: VRAM has {len(vram_row_blocks)} blocks, MRAM has {len(mram_row_blocks)} blocks"
            )

        num_hidden_blocks = len(vram_row_blocks)

        for sub_block in mram_row_blocks:
            if sub_block.mram_addr is None:
                raise RuntimeError(f"SubBlock {mram_mat_name}[{mram_row_idx}][{sub_block.col_idx}] not loaded to MRAM")

        full_batch = vram_layout.full_shape[0]
        vram_row_start_addr = vram_row_blocks[0].vram_addr
        mram_row_start_addr = mram_row_blocks[0].mram_addr
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

    # ==========================================================================
    # High-level Interface: Complete Sub-block Computation
    # ==========================================================================

    def compute_sub_matmul(
        self,
        a_name: str,
        a_row_idx: int | slice,
        b_name: str,
        b_col_idx: int | slice,
        result_name: str,
        transpose_b: bool = False,
    ) -> tuple[str, int]:
        """Not yet implemented. Use vram_sub_projection_asm or vram_sub_projection_T_asm."""
        raise NotImplementedError(
            "compute_sub_matmul is not yet implemented. Use vram_sub_projection_asm "
            "or vram_sub_projection_T_asm for matrix multiplication."
        )

    # ==========================================================================
    # Format Conversion: HBM <-> VRAM
    # ==========================================================================

    def load_activation_with_format_convert_asm(
        self,
        name: str,
        hbm_base_addr: int,
        batch: int,
        hidden_size: int,
        vram_dest_addr: int,
        hbm_addr_reg: int = 0,
        gp_regs: list[int] | None = None,
    ) -> str:
        """
        Load activation from HBM to VRAM with format conversion.

        HBM layout: [batch, hidden_size] row-major (element[b,h] at hbm_base + b*hidden_size + h).
        VRAM layout: [batch, mlen, hidden/mlen] column-block major.
            element[b,h]: vram_base + (h//mlen)*batch*mlen + b*mlen + (h%mlen).

        H_PREFETCH_V loads mlen elements per call; columns loaded in blocks of mlen.
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5]

        asm = IsaBuilder()
        asm.comment(f"Load Activation with Format Convert: {name}")
        asm.comment(f"HBM[{hbm_base_addr}]: [batch={batch}, hidden={hidden_size}] row-major")
        asm.comment(f"VRAM[{vram_dest_addr}]: [batch, mlen, hidden/mlen] column-block major")

        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        _gp_outer = gp_regs[3]
        _gp_inner = gp_regs[4]

        num_col_blocks = hidden_size // self.mlen
        preload_len = 4  # load 4 rows per H_PREFETCH_V call

        total_size = batch * hidden_size
        asm.instr("S_ADDI_INT", gp(gp_hbm_offset), gp(0), total_size)
        asm.instr("C_SET_SCALE_REG", gp(gp_hbm_offset))

        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), hidden_size)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

        for col_block in range(num_col_blocks):
            asm.comment(f"Column block {col_block}")

            hbm_offset = col_block * self.mlen
            vram_addr = vram_dest_addr + col_block * batch * self.mlen

            asm.instr("S_ADDI_INT", gp(gp_hbm_offset), gp(0), hbm_offset)
            asm.instr("S_ADDI_INT", gp(gp_vram), gp(0), vram_addr)

            for batch_block in range(math.ceil(batch / preload_len)):
                actual_batch_offset = batch_block * preload_len * hidden_size
                actual_vram_offset = batch_block * preload_len * self.mlen

                asm.instr("S_ADDI_INT", gp(gp_hbm_offset), gp(0), hbm_offset + actual_batch_offset)
                asm.instr("S_ADDI_INT", gp(gp_vram), gp(0), vram_addr + actual_vram_offset)
                asm.instr("H_PREFETCH_V", gp(gp_vram), gp(gp_hbm_offset), areg(hbm_addr_reg), 1, 0)

        return asm.render()

    def store_activation_with_format_convert_asm(
        self,
        name: str,
        vram_src_addr: int,
        batch: int,
        hidden_size: int,
        hbm_dest_addr: int,
        hbm_addr_reg: int = 0,
        gp_regs: list[int] | None = None,
    ) -> str:
        """
        Store activation from VRAM to HBM with format conversion.

        VRAM layout: [batch, mlen, hidden/mlen] column-block major.
        HBM layout: [batch, hidden_size] row-major.

        H_STORE_V stores mlen elements per call; columns stored in blocks of mlen.
        """
        if gp_regs is None:
            gp_regs = [1, 2, 3, 4, 5]

        asm = IsaBuilder()
        asm.comment(f"Store Activation with Format Convert: {name}")
        asm.comment(f"VRAM[{vram_src_addr}]: [batch, mlen, hidden/mlen] column-block major")
        asm.comment(f"HBM[{hbm_dest_addr}]: [batch={batch}, hidden={hidden_size}] row-major")

        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        _gp_outer = gp_regs[3]
        _gp_inner = gp_regs[4]

        num_col_blocks = hidden_size // self.mlen
        store_amount = 4  # store 4 rows per H_STORE_V call

        asm.instr("S_ADDI_INT", gp(gp_stride), gp(0), hidden_size)
        asm.instr("C_SET_STRIDE_REG", gp(gp_stride))

        for col_block in range(num_col_blocks):
            asm.comment(f"Column block {col_block}")

            hbm_offset = col_block * self.mlen
            vram_addr = vram_src_addr + col_block * batch * self.mlen

            for batch_block in range(math.ceil(batch / store_amount)):
                actual_batch_offset = batch_block * store_amount * hidden_size
                actual_vram_offset = batch_block * store_amount * self.mlen

                asm.instr("S_ADDI_INT", gp(gp_hbm_offset), gp(0), hbm_offset + actual_batch_offset)
                asm.instr("S_ADDI_INT", gp(gp_vram), gp(0), vram_addr + actual_vram_offset)
                asm.instr("H_STORE_V", gp(gp_vram), gp(gp_hbm_offset), areg(hbm_addr_reg), 0)

        return asm.render()

    # ==========================================================================
    # Pre-calculated Address Table Generation
    # ==========================================================================

    def generate_address_table(self, name: str) -> dict[str, int]:
        """Generate complete address table for a matrix (for debugging and verification)."""
        if name not in self.hbm_matrices:
            raise KeyError(f"Matrix '{name}' not registered")

        layout = self.hbm_matrices[name]
        addr_table = {}

        for (r, c), sub_block in layout.sub_blocks.items():
            key = f"{name}[{r}][{c}]"
            addr_table[f"{key}_hbm_offset"] = sub_block.hbm_offset
            addr_table[f"{key}_hbm_abs"] = layout.hbm_base_addr + sub_block.hbm_offset
            if sub_block.mram_addr is not None:
                addr_table[f"{key}_mram"] = sub_block.mram_addr

        return addr_table

    def print_address_table(self, name: str):
        """Print address table for a matrix."""
        addr_table = self.generate_address_table(name)
        print(f"Address Table for {name}:")
        for key, addr in sorted(addr_table.items()):
            print(f"  {key}: {addr}")

    # ==========================================================================
    # Helper Methods
    # ==========================================================================

    def get_loaded_block_addr(self, name: str, row_idx: int, col_idx: int) -> int:
        """Get MRAM address of a loaded sub-block."""
        block_key = f"{name}[{row_idx}][{col_idx}]"
        if block_key not in self.loaded_sub_blocks:
            raise KeyError(f"SubBlock {block_key} not loaded")
        return self.loaded_sub_blocks[block_key].mram_addr

    def is_block_loaded(self, name: str, row_idx: int, col_idx: int) -> bool:
        """Check whether a sub-block has been loaded into MRAM."""
        block_key = f"{name}[{row_idx}][{col_idx}]"
        return block_key in self.loaded_sub_blocks

    def reset(self):
        """Reset manager state."""
        self.hbm_matrices.clear()
        self.vram_matrices.clear()
        self.fpram_matrices.clear()
        self.vram_allocator.reset()
        self.mram_allocator.reset()
        self.fpram_allocator.reset()
        self.loaded_sub_blocks.clear()
        self._address_cache.clear()

    def print_table(self):
        """Print unified managed object table."""
        print("=" * 95)
        print("Managed Object Table")
        print("=" * 95)
        print(
            f"{'Name':<20} {'Kind':<12} {'Shape':<15} {'HBM Addr':<10} {'HBM Size':<10} {'VRAM Addr':<12} {'FPRAM':<12} {'Size':<8}"
        )
        print("-" * 95)
        names = sorted(set(self.hbm_matrices) | set(self.vram_matrices) | set(self.fpram_matrices))
        for name in names:
            info = self[name]
            shape_str = f"({info.shape[0]}, {info.shape[1]})"
            vram_str = f"{info.vram_addr}" if info.vram_addr is not None else "None"
            fpram_str = f"{info.fpram_addr}/{info.fpram_size}" if info.fpram_addr is not None else "None"
            print(
                f"{name:<20} {info.kind:<12} {shape_str:<15} "
                f"{info.hbm_addr:<10} {info.hbm_size:<10} {vram_str:<12} {fpram_str:<12} {info.size:<8}"
            )
        print("=" * 95)

    def print_layout(self, name: str):
        """Print block layout for a matrix."""
        if name not in self.hbm_matrices:
            print(f"Matrix '{name}' not registered")
            return

        layout = self.hbm_matrices[name]
        print(f"Matrix: {name}")
        print(f"  Full shape: {layout.full_shape}")
        print(f"  Block size: {layout.block_size}")
        print(f"  Blocks: {layout.num_row_blocks} x {layout.num_col_blocks}")
        print(f"  HBM base: {layout.hbm_base_addr}")
        print("  Sub blocks:")
        for (r, c), sub in layout.sub_blocks.items():
            loaded = "LOADED" if sub.mram_addr is not None else ""
            print(f"    [{r}][{c}]: hbm_off={sub.hbm_offset}, mram={sub.mram_addr} {loaded}")

__all__ = ["TileCompiler"]
