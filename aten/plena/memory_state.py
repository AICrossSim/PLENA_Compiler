"""Memory layout state for the ATen PLENA compiler."""

from __future__ import annotations

from compiler.aten.plena.constants import BLEN, MLEN
from compiler.aten.plena.memory import (
    FPRAMAllocator,
    FPRAMObjectLayout,
    MRAMAllocator,
    MatrixBlockLayout,
    MemoryObjectInfo,
    VRAMAllocator,
    VRAMMatrixBlockLayout,
    VRAMSubMatrixInfo,
)


class MemoryStateMixin:
    """Sub-matrix layout manager for PLENA HBM, VRAM, MRAM, and FPRAM state."""

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

    def clear_mram_bindings(self) -> None:
        """Clear cached MRAM addresses on all HBM sub-blocks."""
        for layout in self.hbm_matrices.values():
            for sub_block in layout.sub_blocks.values():
                sub_block.mram_addr = None

    def reset(self):
        """Reset manager state."""
        self.clear_mram_bindings()
        self.hbm_matrices.clear()
        self.vram_matrices.clear()
        self.fpram_matrices.clear()
        self.vram_allocator.reset()
        self.mram_allocator.reset()
        self.fpram_allocator.reset()


__all__ = ["MemoryStateMixin"]
