"""Memory layouts and allocators for the ATen PLENA compiler path."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from compiler.aten.plena.constants import MLEN


# ==============================================================================
# Virtual Memory Manager
# ==============================================================================


@dataclass
class MemoryBlock:
    name: str  # Allocation name (e.g., "W[0][1]" or "activation_A")
    addr: int  # Starting address
    size: int  # Block size (number of elements)


class VirtualMemoryManager:
    """Best-fit reuse plus bump allocation for PLENA virtual memories."""

    def __init__(self, total_size: int, alignment: int = MLEN, mem_name: str = "Memory"):
        self.total_size = total_size
        self.alignment = alignment
        self.mem_name = mem_name
        self.next_bump = 0  # Bump allocation pointer

        # Two core stacks
        self.used_stack: list[MemoryBlock] = []
        self.free_stack: list[MemoryBlock] = []

    def _align(self, value: int) -> int:
        """Align value to alignment"""
        return ((value + self.alignment - 1) // self.alignment) * self.alignment

    def allocate(self, name: str, size: int) -> int:
        """Allocate by best-fit reuse first, then bump allocation."""
        aligned_size = self._align(size)

        best = min(
            ((block.size - aligned_size, i) for i, block in enumerate(self.free_stack) if block.size >= aligned_size),
            default=None,
        )

        if best is not None:
            _, best_idx = best
            reused_block = self.free_stack.pop(best_idx)

            # If block is larger than needed, split remaining part and return to free_stack
            if reused_block.size > aligned_size:
                remaining = MemoryBlock(
                    name="<fragment>", addr=reused_block.addr + aligned_size, size=reused_block.size - aligned_size
                )
                self.free_stack.append(remaining)

            new_block = MemoryBlock(name=name, addr=reused_block.addr, size=aligned_size)
            self.used_stack.append(new_block)
            return new_block.addr

        aligned_addr = self._align(self.next_bump)

        if self.total_size > 0 and aligned_addr + aligned_size > self.total_size:
            raise MemoryError(
                f"{self.mem_name} overflow: need {aligned_size} at addr {aligned_addr}, "
                f"total_size={self.total_size}, "
                f"used={len(self.used_stack)} blocks, "
                f"free={len(self.free_stack)} blocks"
            )

        new_block = MemoryBlock(name=name, addr=aligned_addr, size=aligned_size)
        self.used_stack.append(new_block)
        self.next_bump = aligned_addr + aligned_size
        return aligned_addr

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        """Move an allocation from used_stack to reusable free_stack."""
        for i, block in enumerate(self.used_stack):
            if block.name == name:
                freed = self.used_stack.pop(i)
                self.free_stack.append(freed)
                self._coalesce_free_stack()
                return freed

        if strict:
            raise KeyError(
                f"{self.mem_name}: allocation '{name}' not found in used_stack. "
                f"Current used: {[b.name for b in self.used_stack]}"
            )
        return None

    def mark_used(self, addr: int, size: int, name: str) -> None:
        """Register a pre-known occupied range and advance bump past it."""
        aligned_size = self._align(size)
        block = MemoryBlock(name=name, addr=addr, size=aligned_size)
        self.used_stack.append(block)
        # Advance bump pointer past this region if it would otherwise overlap.
        end = addr + aligned_size
        if self.next_bump < end:
            self.next_bump = end

    def _coalesce_free_stack(self):
        """Merge adjacent free blocks by address."""
        if len(self.free_stack) <= 1:
            return

        blocks = sorted(self.free_stack, key=lambda b: b.addr)
        merged: list[MemoryBlock] = [blocks[0]]
        for block in blocks[1:]:
            prev = merged[-1]
            if prev.addr + prev.size == block.addr:
                merged[-1] = MemoryBlock(
                    name="<merged>",
                    addr=prev.addr,
                    size=prev.size + block.size,
                )
            else:
                merged.append(block)

        self.free_stack = merged

    def reset(self):
        """Reset manager"""
        self.next_bump = 0
        self.used_stack.clear()
        self.free_stack.clear()


# ==============================================================================
# Sub-matrix Information
# ==============================================================================


@dataclass
class SubMatrixInfo:
    """Metadata for sub-matrices"""

    parent_name: str  # Parent matrix name
    row_idx: int  # Sub-block row index
    col_idx: int  # Sub-block column index
    shape: tuple[int, int]  # Sub-block shape (typically 64x64)

    # Pre-calculated addresses (computed during compiler phase, used directly at runtime)
    hbm_offset: int = 0  # Offset in HBM (in elements)
    mram_addr: int | None = None  # Address in MRAM (if loaded)


@dataclass
class MatrixBlockLayout:
    """
    Block layout information for large matrices.

    HBM storage: [rows, cols] row-major contiguous, stride=cols per row.
    MRAM storage: [batch, mlen, hidden/mlen] column-block major.
    """

    name: str
    full_shape: tuple[int, int]  # Full matrix shape (rows, cols)
    block_size: int = MLEN  # Sub-block size (default 64)

    num_row_blocks: int = 0
    num_col_blocks: int = 0

    # HBM Address Information
    hbm_base_addr: int = 0
    hbm_size: int = 0  # Size after applying real_data_ratio (MXFP8 = 1.125)

    # Sub-block map: (row_idx, col_idx) -> SubMatrixInfo
    sub_blocks: dict[tuple[int, int], SubMatrixInfo] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize block information"""
        rows, cols = self.full_shape
        self.num_row_blocks = math.ceil(rows / self.block_size)
        self.num_col_blocks = math.ceil(cols / self.block_size)

        # Create information for all sub-blocks (pre-calculate addresses)
        for r in range(self.num_row_blocks):
            for c in range(self.num_col_blocks):
                # HBM offset (row-major): sub-block (r,c) starts at r*block_size*cols + c*block_size
                hbm_offset = r * self.block_size * cols + c * self.block_size

                sub_info = SubMatrixInfo(
                    parent_name=self.name,
                    row_idx=r,
                    col_idx=c,
                    shape=(self.block_size, self.block_size),
                    hbm_offset=hbm_offset,
                    mram_addr=None,
                )
                self.sub_blocks[(r, c)] = sub_info

    def get_sub_block(self, row_idx: int, col_idx: int) -> SubMatrixInfo:
        """Get specified sub-block"""
        if (row_idx, col_idx) not in self.sub_blocks:
            raise IndexError(f"Sub block [{row_idx}][{col_idx}] out of range")
        return self.sub_blocks[(row_idx, col_idx)]

    def get_row_blocks(self, row_idx: int) -> list[SubMatrixInfo]:
        """Get all sub-blocks in a row"""
        return [self.sub_blocks[(row_idx, c)] for c in range(self.num_col_blocks)]

    def get_col_blocks(self, col_idx: int) -> list[SubMatrixInfo]:
        """Get all sub-blocks in a column"""
        return [self.sub_blocks[(r, col_idx)] for r in range(self.num_row_blocks)]


# ==============================================================================
# VRAM Sub-matrix Information
# ==============================================================================


@dataclass
class VRAMSubMatrixInfo:
    """Metadata for sub-matrices in VRAM"""

    parent_name: str  # Parent matrix name
    row_idx: int  # Sub-block row index (batch dimension)
    col_idx: int  # Sub-block column index (hidden dimension)
    shape: tuple[int, int]  # Sub-block shape (typically mlen x mlen)

    # Pre-calculated VRAM address
    vram_addr: int = 0


@dataclass
class VRAMMatrixBlockLayout:
    """
    Block layout for large matrices in VRAM.

    VRAM storage format: [batch, mlen, hidden/mlen] column-block major.
    - Batch dimension contiguous within each column block.
    - Column blocks laid out sequentially.

    Column block c base address: vram_base + c * batch * mlen.
    Sub-block (r, c) offset within column block: r * mlen * mlen.
    """

    name: str
    full_shape: tuple[int, int]  # Full matrix shape (batch, hidden_size)
    vram_base_addr: int  # VRAM base address
    block_size: int = MLEN  # Sub-block size (default 64)

    num_row_blocks: int = 0  # Number of blocks in batch dimension
    num_col_blocks: int = 0  # Number of blocks in hidden dimension

    # Sub-block map: (row_idx, col_idx) -> VRAMSubMatrixInfo
    sub_blocks: dict[tuple[int, int], VRAMSubMatrixInfo] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize block information"""
        batch, hidden = self.full_shape
        self.num_row_blocks = math.ceil(batch / self.block_size)
        self.num_col_blocks = math.ceil(hidden / self.block_size)

        # VRAM column-block major address calculation:
        # col block c base = vram_base + c * batch * mlen
        # row sub-block r offset within column block = r * mlen * mlen
        for r in range(self.num_row_blocks):
            for c in range(self.num_col_blocks):
                col_block_base = self.vram_base_addr + c * batch * self.block_size
                row_offset = r * self.block_size * self.block_size
                vram_addr = col_block_base + row_offset

                sub_info = VRAMSubMatrixInfo(
                    parent_name=self.name,
                    row_idx=r,
                    col_idx=c,
                    shape=(self.block_size, self.block_size),
                    vram_addr=vram_addr,
                )
                self.sub_blocks[(r, c)] = sub_info

    def get_sub_block(self, row_idx: int, col_idx: int) -> VRAMSubMatrixInfo:
        """Get specified sub-block"""
        if (row_idx, col_idx) not in self.sub_blocks:
            raise IndexError(f"VRAM sub block [{row_idx}][{col_idx}] out of range")
        return self.sub_blocks[(row_idx, col_idx)]

    def get_row_blocks(self, row_idx: int) -> list[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a row (A[row_idx][:])"""
        return [self.sub_blocks[(row_idx, c)] for c in range(self.num_col_blocks)]

    def get_col_blocks(self, col_idx: int) -> list[VRAMSubMatrixInfo]:
        """Get all sub-blocks in a column"""
        return [self.sub_blocks[(r, col_idx)] for r in range(self.num_row_blocks)]


# ==============================================================================
# Unified Memory Object Metadata
# ==============================================================================


@dataclass
class MemoryObjectInfo:
    """Unified metadata for objects managed across HBM / VRAM / FPRAM."""

    name: str
    kind: str
    dtype: str = "fp16"
    shape: tuple[int, int] = (0, 0)
    size: int = 0
    hbm_addr: int = -1
    hbm_size: int = 0
    vram_addr: int | None = None
    fpram_addr: int | None = None
    fpram_size: int = 0


@dataclass
class FPRAMObjectLayout:
    """FPRAM object layout."""

    name: str
    fpram_addr: int
    size: int
    dtype: str = "fp16"
    kind: str = "FPRAMObject"


# ==============================================================================
# Allocators
# ==============================================================================


class MemoryAllocatorBase:
    """Shared wrapper over VirtualMemoryManager for compiler address spaces."""

    def __init__(self, total_size: int, alignment: int, mem_name: str):
        self.total_size = total_size
        self.alignment = alignment
        self._vmm = VirtualMemoryManager(total_size=total_size, alignment=alignment, mem_name=mem_name)

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @next_free.setter
    def next_free(self, value: int):
        self._validate_next_free(value)
        self._vmm.next_bump = value

    def _validate_next_free(self, value: int) -> None:
        del value

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        return self._vmm.free(name, strict=strict)

    def reset(self):
        self._vmm.reset()


class MRAMAllocator(MemoryAllocatorBase):
    """
    Matrix RAM address allocator.

    Each sub-block is mlen x mlen = 4096 elements; aligned to mlen*mlen.
    Default total_size=16384 holds 4 sub-blocks (MAX_K_TILES=4).
    """

    def __init__(self, total_size: int = MLEN * MLEN * 4):
        super().__init__(total_size=total_size, alignment=MLEN * MLEN, mem_name="MRAM")

    def allocate(self, name: str, size: int) -> int:
        return self._vmm.allocate(name, size)


class VRAMAllocator(MemoryAllocatorBase):
    """VRAM address allocator with MLEN-aligned best-fit reuse + bump allocation."""

    def __init__(self, alignment: int = MLEN, total_size: int = 0):
        super().__init__(total_size=total_size, alignment=alignment, mem_name="VRAM")

    def allocate(self, size: int, name: str = "") -> int:
        if not name:
            raise ValueError("VRAMAllocator.allocate() requires name for subsequent free.")
        return self._vmm.allocate(name, size)


class FPRAMAllocator(MemoryAllocatorBase):
    """
    Floating Point RAM Allocator (based on VirtualMemoryManager).

    FPRAM stores scalar FP values (f16), accessed via S_LD_FP / S_ST_FP.
    Uses the same strategy as VRAM/MRAM allocator:
    - used_stack + free_stack
    - allocate: best-fit reuse first, then bump
    - free: move block to free_stack, supports out-of-order free

    Fixed slot conventions (must not be overwritten by dynamic allocations):
      slot 0 = 0.0 (gp0/f0 reserved as hardware zero)
      slot 1 = attn_scale
      slot 2 = -inf (online softmax)
      slot 3 = eps (rms_norm/layer_norm)
      slot 4 = 1/hidden_size (rms_norm/layer_norm)
      slot 5 = 1.0 (FFN SiLU sigmoid denominator; im2col fp_one_reg)

    Hardware: 1024 f16 elements (configurable via total_size).
    """

    def __init__(self, total_size: int = 1024):
        """
        Args:
            total_size: Total FP RAM size (default 1024, matching hardware fpsram)
        """
        super().__init__(total_size=total_size, alignment=1, mem_name="FPRAM")
        self.allocations: dict[str, tuple[int, int]] = {}

    def _validate_next_free(self, value: int) -> None:
        if value < 0 or value > self.total_size:
            raise ValueError(f"next_free out of range: {value}, expected [0, {self.total_size}]")

    def allocate(self, name: str, size: int) -> int:
        """Allocate FP RAM space (best-fit + bump)."""
        if size <= 0:
            raise ValueError(f"FPRAM allocation size must be > 0, got {size}")
        if name in self.allocations:
            raise KeyError(f"FPRAM name '{name}' already allocated")

        addr = self._vmm.allocate(name, size)
        self.allocations[name] = (addr, size)
        return addr

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        """Free a block and move it to free_stack (same as VirtualMemoryManager)."""
        freed = self._vmm.free(name, strict=strict)
        if freed is not None:
            self.allocations.pop(name, None)
        return freed

    def reset(self):
        """Reset allocator"""
        self._vmm.reset()
        self.allocations.clear()
