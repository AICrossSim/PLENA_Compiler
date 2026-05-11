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
    """Memory Block Information"""

    name: str  # Allocation name (e.g., "W[0][1]" or "activation_A")
    addr: int  # Starting address
    size: int  # Block size (number of elements)

    def __repr__(self) -> str:
        return f"MemBlock({self.name}, addr={self.addr}, size={self.size})"


class VirtualMemoryManager:
    """
    Virtual Memory Manager

    Core Design:
    - used_stack: Allocated and in-use memory blocks
    - free_stack: Freed and reusable memory blocks

    Workflow:
    1. allocate(name, size): Allocate memory
       - Prioritize best-fit search for reusable blocks in free_stack
       - If not found, use bump allocation from the end
    2. free(name): Free memory
       - Move block from used_stack to free_stack
       - Address can be reused by subsequent allocate calls
       - Throws exception if not found when strict=True, returns None when strict=False

    VRAM/MRAM storage format: (batch_size, mlen, hidden_size/mlen), column-block major.
    mlen=64, blen=4. Alignment depends on storage hierarchy.
    """

    def __init__(self, total_size: int, alignment: int = MLEN, mem_name: str = "Memory"):
        """
        Args:
            total_size: Total memory size (number of elements)
            alignment: alignment granularity (VRAM uses MLEN=64, MRAM uses MLEN*MLEN=4096)
            mem_name: Memory name, for debugging information (e.g., "VRAM" or "MRAM")
        """
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
        """
        Allocate memory.

        Strategy:
        1. Best-fit from free_stack (reusable block with least waste).
        2. If no suitable block, bump allocation.

        Returns:
            Allocated starting address.

        Raises:
            MemoryError: Insufficient memory.
        """
        aligned_size = self._align(size)

        # Strategy 1: best-fit from free_stack
        best_idx = None
        best_waste = float("inf")

        for i, block in enumerate(self.free_stack):
            if block.size >= aligned_size:
                waste = block.size - aligned_size
                if waste < best_waste:
                    best_waste = waste
                    best_idx = i

        if best_idx is not None:
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

        # Strategy 2: Bump allocation
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
        """
        Free memory: move block from used_stack to free_stack.

        Args:
            name: Name of allocation to free
            strict: Throws KeyError if not found when strict=True, returns None when strict=False

        Returns:
            Freed memory block, returns None if strict=False and not found
        """
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
        """
        Register a pre-known address range as occupied without bump allocation.

        Used for prestaged VRAM tensors that are already in VRAM before program
        execution (e.g. Q pre-loaded by the test harness).  Advances next_bump
        past the end of this region so subsequent bump allocations do not collide.

        Args:
            addr: Start address of the pre-occupied region.
            size: Number of elements in the region.
            name: Name for tracking/free.
        """
        aligned_size = self._align(size)
        block = MemoryBlock(name=name, addr=addr, size=aligned_size)
        self.used_stack.append(block)
        # Advance bump pointer past this region if it would otherwise overlap.
        end = addr + aligned_size
        if self.next_bump < end:
            self.next_bump = end

    def _coalesce_free_stack(self):
        """
        Merge adjacent free blocks by address to reduce long-term fragmentation.
        """
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

    def is_allocated(self, name: str) -> bool:
        """Check if a name is in used_stack"""
        return any(b.name == name for b in self.used_stack)

    def get_block(self, name: str) -> MemoryBlock | None:
        """Get memory block with specified name from used_stack"""
        for block in self.used_stack:
            if block.name == name:
                return block
        return None

    def get_used_size(self) -> int:
        """Get total used size"""
        return sum(b.size for b in self.used_stack)

    def get_free_size(self) -> int:
        """Get total reusable size"""
        return sum(b.size for b in self.free_stack)

    def reset(self):
        """Reset manager"""
        self.next_bump = 0
        self.used_stack.clear()
        self.free_stack.clear()

    def print_status(self):
        """Print memory status"""
        print(f"=== {self.mem_name} Virtual Memory Status ===")
        print(f"Total size: {self.total_size}")
        print(f"Bump pointer: {self.next_bump}")
        print(f"Used blocks ({len(self.used_stack)}):")
        for b in self.used_stack:
            print(f"  {b}")
        print(f"Free blocks ({len(self.free_stack)}):")
        for b in self.free_stack:
            print(f"  {b}")
        total_used = self.get_used_size()
        total_free = self.get_free_size()
        if self.total_size > 0:
            available = self.total_size - self.next_bump + total_free
            print(
                f"Summary: used={total_used}, free={total_free}, "
                f"bump={self.next_bump}, available={available}/{self.total_size}"
            )
        else:
            print(f"Summary: used={total_used}, free={total_free}, bump={self.next_bump} (unlimited mode)")

    def __repr__(self) -> str:
        return (
            f"VirtualMemoryManager({self.mem_name}, "
            f"used={len(self.used_stack)}, free={len(self.free_stack)}, "
            f"bump={self.next_bump}/{self.total_size})"
        )


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

    def __repr__(self) -> str:
        mram_str = f"{self.mram_addr}" if self.mram_addr is not None else "None"
        return (
            f"SubMatrix({self.parent_name}[{self.row_idx}][{self.col_idx}], "
            f"shape={self.shape}, hbm_off={self.hbm_offset}, mram={mram_str})"
        )


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

    def __repr__(self) -> str:
        return (
            f"VRAMSubMatrix({self.parent_name}[{self.row_idx}][{self.col_idx}], "
            f"shape={self.shape}, vram={self.vram_addr})"
        )


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

    def get_row_vram_addrs(self, row_idx: int) -> list[int]:
        """Get list of VRAM addresses for all sub-blocks in a row"""
        return [block.vram_addr for block in self.get_row_blocks(row_idx)]


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
# MRAM Allocator
# ==============================================================================


class MRAMAllocator:
    """
    Matrix RAM address allocator (based on VirtualMemoryManager).

    Each sub-block is mlen x mlen = 4096 elements; aligned to mlen*mlen.
    Default total_size=16384 holds 4 sub-blocks (MAX_K_TILES=4).
    Supports virtual free/reuse: freed blocks move to free_stack and are
    preferred by subsequent allocations.
    """

    def __init__(self, total_size: int = MLEN * MLEN * 4):
        """
        Args:
            total_size: Total MRAM size (default 16384, can hold 4 64x64 matrix blocks)
        """
        self.total_size = total_size
        self._vmm = VirtualMemoryManager(
            total_size=total_size,
            alignment=MLEN * MLEN,  # aligned to one sub-block size
            mem_name="MRAM",
        )

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @property
    def used_stack(self) -> list[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> list[MemoryBlock]:
        return self._vmm.free_stack

    def allocate(self, name: str, size: int) -> int:
        """Allocate MRAM space (prioritize reusing freed blocks)."""
        return self._vmm.allocate(name, size)

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        """Free specified allocation: move from used_stack to free_stack."""
        return self._vmm.free(name, strict=strict)

    def is_allocated(self, name: str) -> bool:
        """Check if a name is allocated"""
        return self._vmm.is_allocated(name)

    def reset(self):
        """Reset allocator"""
        self._vmm.reset()

    def print_status(self):
        """Print memory status"""
        self._vmm.print_status()


class VRAMAllocator:
    """
    VRAM address allocator (based on VirtualMemoryManager).

    VRAM supports best-fit reuse + bump allocation, same as MRAM/FPRAM allocators.
    Alignment defaults to MLEN to match VRAM storage format requirements.
    """

    def __init__(self, alignment: int = MLEN, total_size: int = 0):
        self.alignment = alignment
        self._vmm = VirtualMemoryManager(total_size=total_size, alignment=alignment, mem_name="VRAM")

    @property
    def next_free(self) -> int:
        return self._vmm.next_bump

    @next_free.setter
    def next_free(self, value: int):
        self._vmm.next_bump = value

    @property
    def used_stack(self) -> list[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> list[MemoryBlock]:
        return self._vmm.free_stack

    def allocate(self, size: int, name: str = "") -> int:
        if not name:
            raise ValueError("VRAMAllocator.allocate() requires name for subsequent free.")
        return self._vmm.allocate(name, size)

    def free(self, name: str, strict: bool = True) -> MemoryBlock | None:
        return self._vmm.free(name, strict=strict)

    def is_allocated(self, name: str) -> bool:
        return self._vmm.is_allocated(name)

    def reset(self):
        self._vmm.reset()

    def print_status(self):
        self._vmm.print_status()

    def __repr__(self) -> str:
        return (
            f"VRAMAllocator(next_free={self.next_free}, alignment={self.alignment}, "
            f"used={len(self.used_stack)}, free={len(self.free_stack)})"
        )


class FPRAMAllocator:
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
        self.total_size = total_size
        self._vmm = VirtualMemoryManager(
            total_size=total_size,
            alignment=1,
            mem_name="FPRAM",
        )
        self.allocations: dict[str, tuple[int, int]] = {}
        self._snapshots: dict[int, tuple[int, list[MemoryBlock], list[MemoryBlock], dict[str, tuple[int, int]]]] = {}
        self._next_snapshot_id = 1

    @property
    def next_free(self) -> int:
        """Compatibility alias: next bump pointer."""
        return self._vmm.next_bump

    @next_free.setter
    def next_free(self, value: int):
        if value < 0 or value > self.total_size:
            raise ValueError(f"next_free out of range: {value}, expected [0, {self.total_size}]")
        self._vmm.next_bump = value

    @property
    def used_stack(self) -> list[MemoryBlock]:
        return self._vmm.used_stack

    @property
    def free_stack(self) -> list[MemoryBlock]:
        return self._vmm.free_stack

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

    def save_state(self) -> int:
        """
        Save current allocator state and return a snapshot token.
        """
        sid = self._next_snapshot_id
        self._next_snapshot_id += 1
        self._snapshots[sid] = (
            self._vmm.next_bump,
            [MemoryBlock(b.name, b.addr, b.size) for b in self._vmm.used_stack],
            [MemoryBlock(b.name, b.addr, b.size) for b in self._vmm.free_stack],
            dict(self.allocations),
        )
        return sid

    def restore_state(self, snapshot: int):
        """Restore allocator state from snapshot token."""
        if snapshot not in self._snapshots:
            raise KeyError(f"Unknown FPRAM snapshot id: {snapshot}")
        next_bump, used_stack, free_stack, allocations = self._snapshots[snapshot]
        self._vmm.next_bump = next_bump
        self._vmm.used_stack = [MemoryBlock(b.name, b.addr, b.size) for b in used_stack]
        self._vmm.free_stack = [MemoryBlock(b.name, b.addr, b.size) for b in free_stack]
        self.allocations = dict(allocations)

    def reset(self):
        """Reset allocator"""
        self._vmm.reset()
        self.allocations.clear()
        self._snapshots.clear()
        self._next_snapshot_id = 1


