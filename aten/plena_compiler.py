"""PlenaCompiler — unified single-class PLENA ISA compiler.

Contains the full inheritance chain: TileCompiler (memory bookkeeping) →
DeveloperCompiler (ISA emission, FP/FPRAM ops, interrupts) → PlenaCompiler
(user-facing DSL). Tensor proxy classes (TensorVar, InputVar, VRAMMatrixVar,
FPVar) and the unified Tensor type union / TensorKind enum are re-exported
from the same module.

Previously aliased as ``PLENAProgram``; that alias has been retired.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
import math

import os
from collections.abc import Callable
from functools import wraps

from compiler.asm_templates import (
    preload_act_asm,
    reset_reg_asm,
    preload_addr_reg_asm,
    store_act_asm,
    rms_norm_asm,
    layer_norm_asm,
    rope_asm,
)
from compiler.asm_templates.vram_sub_projection_asm import vram_sub_projection_asm_impl


MLEN = 64  # Minimum matrix block size
BLEN = 4  # Vector tile size
IMM2_BOUND = 2**18


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


# ==============================================================================
# Sub Matrix Manager
# ==============================================================================


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

        lines = []
        lines.append(f"; Load SubMatrix {name}[{row_idx}][{col_idx}] -> MRAM[{mram_dest_addr}]")
        lines.append(f"; HBM offset: {hbm_offset} (precomputed)")

        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_dest_addr}")
        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")

        lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")

        block_key = f"{name}[{row_idx}][{col_idx}]"
        self.loaded_sub_blocks[block_key] = sub_block

        return "\n".join(lines) + "\n"

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

        lines = []
        lines.append(f"; Load SubMatrix Row {name}[{row_idx}][:] -> MRAM[{mram_start_addr}]")

        # Set SCALE and STRIDE once for all sub-blocks
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen

        for col_idx in range(num_col_blocks):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset

            sub_block.mram_addr = mram_addr

            lines.append(f"; SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")

            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block

            mram_addr += block_size

        return "\n".join(lines) + "\n"

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

        lines = []
        lines.append(f"; Load SubMatrix Col {name}[:][{col_idx}] -> MRAM[{mram_start_addr}]")

        # Set SCALE and STRIDE once for all sub-blocks
        full_size = layout.full_shape[0] * layout.full_shape[1]
        full_cols = layout.full_shape[1]

        gp_scale = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_mram = gp_regs[2]

        lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {full_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_scale}")
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {full_cols}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        mram_addr = mram_start_addr
        block_size = self.mlen * self.mlen

        effective_count = k_block_count if k_block_count is not None else num_row_blocks
        for row_idx in range(k_block_start, k_block_start + effective_count):
            sub_block = layout.get_sub_block(row_idx, col_idx)
            hbm_offset = sub_block.hbm_offset

            sub_block.mram_addr = mram_addr

            lines.append(f"; SubBlock [{row_idx}][{col_idx}]: HBM offset = {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_mram}, gp0, {mram_addr}")
            lines.append(f"S_ADDI_INT gp{gp_scale}, gp0, {hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_mram}, gp{gp_scale}, a{hbm_addr_reg}, 1, 0")

            block_key = f"{name}[{row_idx}][{col_idx}]"
            self.loaded_sub_blocks[block_key] = sub_block

            mram_addr += block_size

        return "\n".join(lines) + "\n"

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

        lines = []
        lines.append(f"; Load Activation with Format Convert: {name}")
        lines.append(f"; HBM[{hbm_base_addr}]: [batch={batch}, hidden={hidden_size}] row-major")
        lines.append(f"; VRAM[{vram_dest_addr}]: [batch, mlen, hidden/mlen] column-block major")

        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        _gp_outer = gp_regs[3]
        _gp_inner = gp_regs[4]

        num_col_blocks = hidden_size // self.mlen
        preload_len = 4  # load 4 rows per H_PREFETCH_V call

        total_size = batch * hidden_size
        lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {total_size}")
        lines.append(f"C_SET_SCALE_REG gp{gp_hbm_offset}")

        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        for col_block in range(num_col_blocks):
            lines.append(f"; Column block {col_block}")

            hbm_offset = col_block * self.mlen
            vram_addr = vram_dest_addr + col_block * batch * self.mlen

            lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset}")
            lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr}")

            for batch_block in range(math.ceil(batch / preload_len)):
                actual_batch_offset = batch_block * preload_len * hidden_size
                actual_vram_offset = batch_block * preload_len * self.mlen

                lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset + actual_batch_offset}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr + actual_vram_offset}")
                lines.append(f"H_PREFETCH_V gp{gp_vram}, gp{gp_hbm_offset}, a{hbm_addr_reg}, 1, 0")

        return "\n".join(lines) + "\n"

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

        lines = []
        lines.append(f"; Store Activation with Format Convert: {name}")
        lines.append(f"; VRAM[{vram_src_addr}]: [batch, mlen, hidden/mlen] column-block major")
        lines.append(f"; HBM[{hbm_dest_addr}]: [batch={batch}, hidden={hidden_size}] row-major")

        gp_hbm_offset = gp_regs[0]
        gp_stride = gp_regs[1]
        gp_vram = gp_regs[2]
        _gp_outer = gp_regs[3]
        _gp_inner = gp_regs[4]

        num_col_blocks = hidden_size // self.mlen
        store_amount = 4  # store 4 rows per H_STORE_V call

        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, {hidden_size}")
        lines.append(f"C_SET_STRIDE_REG gp{gp_stride}")

        for col_block in range(num_col_blocks):
            lines.append(f"; Column block {col_block}")

            hbm_offset = col_block * self.mlen
            vram_addr = vram_src_addr + col_block * batch * self.mlen

            for batch_block in range(math.ceil(batch / store_amount)):
                actual_batch_offset = batch_block * store_amount * hidden_size
                actual_vram_offset = batch_block * store_amount * self.mlen

                lines.append(f"S_ADDI_INT gp{gp_hbm_offset}, gp0, {hbm_offset + actual_batch_offset}")
                lines.append(f"S_ADDI_INT gp{gp_vram}, gp0, {vram_addr + actual_vram_offset}")
                lines.append(f"H_STORE_V gp{gp_vram}, gp{gp_hbm_offset}, a{hbm_addr_reg}, 0")

        return "\n".join(lines) + "\n"

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


# ==============================================================================
# Example Usage
# ==============================================================================


class RegisterAllocator:
    """Register Allocator: Manages address registers and GP registers"""

    def __init__(self, start_gp: int = 1, start_addr: int = 0, start_fp: int = 1):
        # HW OPERAND_WIDTH = 4 bits → gp0-gp15; gp0 reserved as constant 0.
        self.gp_registers = list(range(start_gp, 16))
        self.addr_registers = list(range(start_addr, 8))
        # f0 reserved as constant 0 (writing to f0 is a no-op for V_RED_MAX/V_RED_SUM).
        self.fp_registers = list(range(start_fp, 8))
        self.used_gp = []
        self.used_addr = []
        self.used_fp = []

    def allocate_gp(self, count: int = 1) -> list[int]:
        if len(self.gp_registers) < count:
            raise RuntimeError(f"Not enough GP registers available. Need {count}, have {len(self.gp_registers)}")

        allocated = self.gp_registers[:count]
        self.gp_registers = self.gp_registers[count:]
        self.used_gp.extend(allocated)
        return allocated

    def allocate_addr(self, count: int = 1) -> list[int]:
        if len(self.addr_registers) < count:
            raise RuntimeError(f"Not enough address registers available. Need {count}, have {len(self.addr_registers)}")

        allocated = self.addr_registers[:count]
        self.addr_registers = self.addr_registers[count:]
        self.used_addr.extend(allocated)
        return allocated

    def free_gp(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_gp:
                self.used_gp.remove(reg)
                self.gp_registers.append(reg)
        self.gp_registers.sort()

    def free_addr(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_addr:
                self.used_addr.remove(reg)
                self.addr_registers.append(reg)
        self.addr_registers.sort()

    def allocate_fp(self, count: int = 1) -> list[int]:
        if len(self.fp_registers) < count:
            raise RuntimeError(f"Not enough FP registers available. Need {count}, have {len(self.fp_registers)}")

        # Reverse allocation: prefer high-numbered regs to avoid conflicts with legacy hardcoded forward-allocation.
        allocated = list(reversed(self.fp_registers[-count:]))
        self.fp_registers = self.fp_registers[:-count]
        self.used_fp.extend(allocated)
        return allocated

    def free_fp(self, registers: list[int]):
        for reg in registers:
            if reg in self.used_fp:
                self.used_fp.remove(reg)
                self.fp_registers.append(reg)
        # Keep sorted so allocate_fp's tail-slice continues to return descending IDs.
        self.fp_registers.sort()


class DeveloperCompiler(TileCompiler):
    """
    Developer Compiler: Compiles high-level IR to ISA.

    Owns symbol_table, register_allocator, and the InterruptManager.
    Sub-matrix / memory management is inherited from TileCompiler; the
    legacy ``self.tile_compiler`` accessor is preserved as a property
    returning ``self`` for a handful of remaining external callers.
    """

    _ONLINE_SOFTMAX_FPSRAM_BASE = 10

    class InterruptManager:
        """
        Interrupt Manager — manages execution timing only.
        Actual handlers live on DeveloperCompiler as ``_handle_k_start``,
        ``_handle_k_prefetch_done``, ``_handle_s_tile_done``, ``_handle_k_end``.
        """

        def __init__(self, compiler: DeveloperCompiler):
            self.compiler = compiler
            self.enabled = False

            self._k_count = 0
            self._tile_count = 0

            self.current_matrix: str = ""
            self.current_activation: str = ""
            self._mlen = compiler.mlen
            self._blen = compiler.blen
            self._batch = compiler.mlen

            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        @property
        def tile_compiler(self):
            return self.compiler.tile_compiler

        @property
        def k_count(self) -> int:
            return self._k_count

        @property
        def tile_count(self) -> int:
            return self._tile_count

        @property
        def batch(self) -> int:
            return self._batch

        @property
        def out_features(self) -> int:
            if self.current_matrix and self.current_matrix in self:
                info = self[self.current_matrix]
                return info.shape[0]
            return self._mlen

        @property
        def hidden_size(self) -> int:
            if self.current_matrix and self.current_matrix in self:
                info = self[self.current_matrix]
                return info.shape[1]
            return self._mlen

        @property
        def k_block(self) -> int:
            return self._k_block_idx

        @property
        def q_block(self) -> int:
            return self._q_block_idx

        @property
        def s_tile_address(self) -> int:
            return self._s_tile_address

        @property
        def mlen(self) -> int:
            return self._mlen

        @property
        def blen(self) -> int:
            return self._blen

        def reset(self):
            """Reset counters (does not clear current_matrix)."""
            self._k_count = 0
            self._tile_count = 0
            self._q_block_idx = 0
            self._k_block_idx = 0
            self._s_tile_address = 0

        def enable(self):
            self.enabled = True

        def disable(self):
            self.enabled = False

        def trigger_k_start(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_start()

        def trigger_k_prefetch_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_k_prefetch_done()
            self._k_count += 1
            return result

        def trigger_s_tile_done(self) -> str:
            if not self.enabled:
                return ""
            result = self.compiler._handle_s_tile_done()
            self._tile_count += 1
            return result

        def trigger_k_end(self) -> str:
            if not self.enabled:
                return ""
            return self.compiler._handle_k_end()

    def __init__(self, mlen: int = 64, blen: int = 4, real_data_ratio: float = 1.125, unroll_loops: bool = False):
        # TileCompiler.__init__ sets mlen, blen, unroll_loops, the HBM/VRAM/FPRAM
        # matrices and allocators, loaded_sub_blocks, and _address_cache.
        super().__init__(mlen=mlen, blen=blen, unroll_loops=unroll_loops)
        self.real_data_ratio = real_data_ratio
        self.register_allocator = RegisterAllocator()
        self.generated_code = ""
        self.interrupt = self.InterruptManager(self)

    # Back-compat shim: older callers (and a couple of external ops modules)
    # reach into ``compiler.tile_compiler`` directly. After the merge,
    # DeveloperCompiler *is* the TileCompiler, so the property just returns
    # ``self``.
    @property
    def tile_compiler(self) -> DeveloperCompiler:
        return self

    # Interrupt handler placeholders (overridden by flash-attention passes).

    def _handle_k_start(self) -> str:
        return ""

    def _handle_k_prefetch_done(self) -> str:
        return ""

    def _handle_s_tile_done(self) -> str:
        return ""

    def _handle_k_end(self) -> str:
        return ""

    # =========================================================================
    # Flash Attention Implementation
    # =========================================================================

    def _online_softmax_asm(
        self,
        mlen: int,
        s_address: int,
        m_start_address: int,
        scale: float = 1.0,
    ) -> str:
        """
        Online Softmax Computation.

        Per row of S:
          1. m_curr = max(S[row], m_old)
          2. m_res = exp(m_old - m_curr)              # used to update O downstream
          3. S'[row] = S[row] - m_curr
          4. P[row] = exp(S'[row])
          5. l_new = l_old * m_res + sum(P[row])

        FP SRAM layout (from m_start_address):
          [0, mlen):        m_old / m_curr
          [mlen, 2*mlen):   m_res = exp(m_old - m_curr)
          [2*mlen, 3*mlen): l_old / l_new
        """
        gp_regs = self.register_allocator.allocate_gp(4)
        gp_s = gp_regs[0]
        gp_m_addr = gp_regs[1]
        gp_m_res_addr = gp_regs[2]
        gp_l_addr = gp_regs[3]

        # Fixed FP register allocation for online softmax pipeline.
        # These registers are shared across _online_softmax_asm, _scale_o_asm,
        # and _final_scaling_asm — they MUST remain consistent across all three.
        # WARNING: Do not use f1-f6 in any code that calls these methods.
        fp_m_old = 1  # f1: m_old value
        fp_m_res = 2  # f2: exp(m_old - m_curr)
        fp_l_old = 3  # f3: l_old value
        fp_sum_p = 4  # f4: sum(P)
        fp_scale = 5  # f5: scale factor
        fp_row_max = 6  # f6: current row max (temporary)

        lines = []
        lines.append("; === Online Softmax ===")

        # Set address registers
        lines.append(f"S_ADDI_INT gp{gp_s}, gp0, {s_address}")
        lines.append(f"S_ADDI_INT gp{gp_m_addr}, gp0, {m_start_address}")
        lines.append(f"S_ADDI_INT gp{gp_m_res_addr}, gp{gp_m_addr}, {mlen}")
        lines.append(f"S_ADDI_INT gp{gp_l_addr}, gp{gp_m_res_addr}, {mlen}")

        # scale factor is pre-loaded at FP SRAM addr 1 by the flash-attention driver.
        if scale != 1.0:
            lines.append(f"S_LD_FP f{fp_scale}, gp0, 1")

        for row in range(mlen):
            lines.append(f"; Row {row}")

            lines.append(f"S_LD_FP f{fp_m_old}, gp{gp_m_addr}, {row}")
            lines.append(f"S_ADD_FP f{fp_m_res}, f{fp_m_old}, f0")

            if scale != 1.0:
                lines.append(f"V_MUL_VF gp{gp_s}, gp{gp_s}, f{fp_scale}, 0")

            lines.append(f"V_RED_MAX f{fp_row_max}, gp{gp_s}, 0")

            # m_curr = max(row_max, m_old) — online softmax must retain the running max.
            lines.append(f"S_MAX_FP f{fp_m_old}, f{fp_row_max}, f{fp_m_old}")

            lines.append(f"S_SUB_FP f{fp_m_res}, f{fp_m_res}, f{fp_m_old}")
            lines.append(f"S_EXP_FP f{fp_m_res}, f{fp_m_res}, 0")

            lines.append(f"S_ST_FP f{fp_m_res}, gp{gp_m_res_addr}, {row}")
            lines.append(f"S_ST_FP f{fp_m_old}, gp{gp_m_addr}, {row}")

            lines.append(f"V_SUB_VF gp{gp_s}, gp{gp_s}, f{fp_m_old}, 0, 0")
            lines.append(f"V_EXP_V gp{gp_s}, gp{gp_s}, 0, 0")

            lines.append(f"S_LD_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

            lines.append(f"S_ADD_FP f{fp_sum_p}, f0, f0")
            lines.append(f"V_RED_SUM f{fp_sum_p}, gp{gp_s}, 0, 0")

            lines.append(f"S_MUL_FP f{fp_l_old}, f{fp_l_old}, f{fp_m_res}")
            lines.append(f"S_ADD_FP f{fp_l_old}, f{fp_l_old}, f{fp_sum_p}")

            lines.append(f"S_ST_FP f{fp_l_old}, gp{gp_l_addr}, {row}")

            lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {mlen}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _pv_multiply_asm(
        self,
        mlen: int,
        blen: int,
        head_dim: int,
        p_address: int,
        v_hbm_offset_reg: int,
        v_hbm_offset: int,
        pv_address: int,
    ) -> str:
        """
        Compute PV = P @ V via M_MM.

        P:  (mlen, mlen)     in VRAM   (softmax output)
        V:  (mlen, head_dim) in HBM    (prefetched into MSRAM in mlen-wide column blocks)
        PV: (mlen, head_dim) in VRAM

        M_MM computes one (blen, mlen) @ (mlen, blen) -> (blen, blen) in a single op
        (K=mlen done in one shot). For head_dim > mlen, V is split into head_dim/mlen
        column blocks; the outer loop iterates blocks, middle loop iterates blen-wide
        V columns within a block, inner loop iterates blen-wide P rows.
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(5)
        gp_p = gp_regs[0]
        gp_v = gp_regs[1]
        gp_pv = gp_regs[2]
        gp_hbm = gp_regs[3]
        gp_stride = gp_regs[4]

        num_v_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === PV Multiply (P @ V) using M_MM ===")
        lines.append(f"; P: ({mlen}, {mlen}) @ V: ({mlen}, {head_dim}) -> PV: ({mlen}, {head_dim})")
        lines.append("; M_MM: (blen, mlen) @ (mlen, blen) -> (blen, blen), K=mlen in one shot")
        lines.append(f"; V split into {num_v_col_blocks} column blocks of width {mlen}")
        lines.append("; Storage layout: (batch, mlen, hidden/mlen), column-block major")

        # STRIDE was set to mlen by the flash-attention driver — do not overwrite it here.
        # M_MM_WO requires a nonzero stride reg (gp0=0 would be interpreted as stride=1).
        # With column-block-major storage, consecutive rows within a column block are
        # adjacent, so the writeback stride = 1.
        lines.append(f"S_ADDI_INT gp{gp_stride}, gp0, 1")

        for v_col_block in range(num_v_col_blocks):
            lines.append(
                f"; --- V column block {v_col_block} (columns {v_col_block * mlen} to {(v_col_block + 1) * mlen - 1}) ---"
            )

            # Prefetch V[:, v_col_block*mlen:(v_col_block+1)*mlen] (mlen × mlen) to MSRAM.
            # V is row-major in HBM: V[row, col] at offset row*head_dim + col, so the
            # column-block base offset = v_hbm_offset + v_col_block * mlen (elements).
            v_block_hbm_offset = v_hbm_offset + v_col_block * mlen
            lines.append(f"S_ADDI_INT gp{gp_v}, gp0, 0")
            lines.append(f"S_ADDI_INT gp{gp_hbm}, gp0, {v_block_hbm_offset}")
            lines.append(f"H_PREFETCH_M gp{gp_v}, gp{gp_hbm}, a{v_hbm_offset_reg}, 1, 1")

            # mat_offset constraint: < mlen and a multiple of blen.
            for v_col in range(mlen // blen):
                lines.append(f"; V column {v_col_block * mlen + v_col * blen}")

                v_msram_offset = v_col * blen
                lines.append(f"S_ADDI_INT gp{gp_v}, gp0, {v_msram_offset}")

                for p_row in range(mlen // blen):
                    p_row_addr = p_address + p_row * blen * mlen
                    lines.append(f"S_ADDI_INT gp{gp_p}, gp0, {p_row_addr}")

                    lines.append(f"M_MM 0, gp{gp_v}, gp{gp_p}")

                    # PV[row, col] addr = base + col_block * mlen * mlen + row * mlen + col_in_block
                    # with row = p_row * blen and col_in_block = v_col * blen.
                    pv_offset = v_col_block * mlen * mlen + p_row * blen * mlen + v_col * blen
                    lines.append(f"S_ADDI_INT gp{gp_pv}, gp0, {pv_address + pv_offset}")
                    lines.append(f"M_MM_WO gp{gp_pv}, gp{gp_stride}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _scale_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        m_res_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Scale each row of O by m_res: O[row] *= m_res[row]."""
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_m_res = gp_regs[0]
        gp_o = gp_regs[1]
        fp_m_res = 1

        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Scale O by m_res ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        lines.append(f"S_ADDI_INT gp{gp_m_res}, gp0, {m_res_address}")

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_m_res}, gp{gp_m_res}, {row}")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_m_res}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _add_pv_to_o_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        pv_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """Accumulate PV into O: O[row] += PV[row]."""
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_o = gp_regs[0]
        gp_pv = gp_regs[1]

        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Add PV to O ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        for row in range(mlen):
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                pv_addr = pv_address + col_block * mlen * mlen + row * mlen

                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"S_ADDI_INT gp{gp_pv}, gp0, {pv_addr}")
                lines.append(f"V_ADD_VV gp{gp_o}, gp{gp_o}, gp{gp_pv}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _final_scaling_asm(
        self,
        mlen: int,
        head_dim: int,
        seq_len: int,
        l_address: int,
        o_address: int,
        row_offset: int = 0,
    ) -> str:
        """
        Final scaling: O[row] /= l[row].

        V_MUL_VF processes mlen elements at a time; when head_dim > mlen,
        each row is split into head_dim // mlen mlen-wide blocks.
        """
        assert head_dim % mlen == 0, f"head_dim ({head_dim}) must be multiple of mlen ({mlen})"

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_l = gp_regs[0]
        gp_o = gp_regs[1]
        fp_l = 1

        num_col_blocks = head_dim // mlen

        lines = []
        lines.append("; === Final Scaling O = O / l ===")
        lines.append(f"; head_dim = {head_dim}, {num_col_blocks} mlen-blocks per row")
        lines.append("; Storage layout: (seq_len, mlen, head_dim/mlen), column-block major")
        lines.append(f"; seq_len = {seq_len}, row_offset = {row_offset}")

        lines.append(f"S_ADDI_INT gp{gp_l}, gp0, {l_address}")

        for row in range(mlen):
            lines.append(f"S_LD_FP f{fp_l}, gp{gp_l}, {row}")
            lines.append(f"S_RECI_FP f{fp_l}, f{fp_l}, 0")
            actual_row = row_offset + row

            for col_block in range(num_col_blocks):
                o_addr = o_address + col_block * seq_len * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_o}, gp0, {o_addr}")
                lines.append(f"V_MUL_VF gp{gp_o}, gp{gp_o}, f{fp_l}, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_fpsram_asm(
        self,
        start_address: int,
        count: int,
        value_address: int,  # FP SRAM slot: 0 = zero, 2 = -inf
    ) -> str:
        """Reset a region of FP SRAM to the value at value_address."""
        gp_regs = self.register_allocator.allocate_gp(1)
        gp_addr = gp_regs[0]

        lines = []
        lines.append(f"; Reset FP SRAM [{start_address}, {start_address + count})")

        lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {start_address}")
        # Use f1 for FP scalar - FP registers don't go through GP allocator
        lines.append(f"S_LD_FP f1, gp0, {value_address}")

        for i in range(count):
            lines.append(f"S_ST_FP f1, gp{gp_addr}, {i}")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def _reset_vram_asm(
        self,
        start_address: int,
        rows: int,
        cols: int,
        total_rows: int,
        mlen: int = 64,
        row_offset: int = 0,
    ) -> str:
        """
        Reset a region of VRAM to zero.

        V_MUL_VF processes mlen elements at a time; when cols > mlen, each
        row is split into cols // mlen mlen-wide blocks.
        """
        gp_regs = self.register_allocator.allocate_gp(1)
        gp_addr = gp_regs[0]

        num_col_blocks = (cols + mlen - 1) // mlen

        lines = []
        lines.append(f"; Reset VRAM rows [{row_offset}, {row_offset + rows}) of matrix at {start_address}")
        lines.append(f"; {rows} rows x {cols} cols, {num_col_blocks} blocks per row")
        lines.append("; Storage layout: (total_rows, mlen, cols/mlen), column-block major")
        lines.append(f"; total_rows = {total_rows}, row_offset = {row_offset}")

        for row in range(rows):
            actual_row = row_offset + row
            for col_block in range(num_col_blocks):
                addr = start_address + col_block * total_rows * mlen + actual_row * mlen
                lines.append(f"S_ADDI_INT gp{gp_addr}, gp0, {addr}")
                lines.append(f"V_MUL_VF gp{gp_addr}, gp{gp_addr}, f0, 0")

        self.register_allocator.free_gp(gp_regs)
        return "\n".join(lines) + "\n"

    def load_batch(
        self,
        hbm_object_name: str,
        vram_object_name: str,
        vlen: int = 64,
        preload_len: int = 4,
    ) -> str:
        """
        Load a Batch tensor from HBM to VRAM.

        HBM storage is MXFP (1 scale per 8 elements), so HBM actual size =
        logical size * real_data_ratio = 1.125. VRAM stores only the vector
        data (no scale), so VRAM size = logical size.

        Order (matters): allocate VRAM → register in symbol table → emit ISA.
        """
        hbm_layout = self.get_hbm_layout(hbm_object_name)
        h, w = hbm_layout.full_shape
        hbm_addr = hbm_layout.hbm_base_addr
        size = h * w
        vram_base = self.vram_allocator.allocate(size, name=vram_object_name)
        self.add_vram_object(
            name=vram_object_name,
            shape=(h, w),
            vram_addr=vram_base,
            dtype="fp16",
            kind="Batch",
            allocate_if_none=False,
            strict=False,
        )

        addr_reg = self.register_allocator.allocate_addr(1)[0]
        gp_regs_for_addr = self.register_allocator.allocate_gp(1)

        isa_code = f"; Load_Batch {hbm_object_name} -> {vram_object_name}\n"
        isa_code += f"; HBM[{hbm_addr}] → VRAM[{vram_base}], shape=({h}, {w})\n"

        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[addr_reg], available_registers=gp_regs_for_addr, addr_reg_val=[hbm_addr]
        )

        # preload_act_asm requires 5 GP registers: [a_actual, stride, result, outer_loop, inner_loop].
        gp_regs_for_preload = self.register_allocator.allocate_gp(5)
        isa_code += reset_reg_asm(alive_registers=gp_regs_for_preload)

        isa_code += preload_act_asm(
            vlen=vlen,
            preload_len=preload_len,
            batch=h,
            hidden_size=w,
            alive_registers=gp_regs_for_preload,
            act_vram_offset=vram_base,
            activation_offset_reg=addr_reg,
            stride_size=w,
        )

        self.register_allocator.free_gp(gp_regs_for_addr)
        self.register_allocator.free_gp(gp_regs_for_preload)
        self.register_allocator.free_addr([addr_reg])

        self.generated_code += isa_code

        return isa_code

    def store_to_hbm(
        self,
        tensor_name: str,
        hbm_addr: int | None = None,
        hbm_object_name: str | None = None,
        hbm_addr_reg: int | None = None,
        vlen: int = 64,
        precision: int = 0,  # 0 = Activation, 1 = KeyValue
        store_amount: int = 4,  # HBM_V_Writeback_Amount
    ) -> str:
        """
        Write tensor from VRAM back to HBM.

        Used to spill computed intermediates (e.g., K) from VRAM to HBM so
        downstream ops (e.g., QK^T) can read them from HBM. Emits
        ``store_act_asm`` for tensors of any supported size.
        """
        if tensor_name not in self:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")

        tensor_info = self[tensor_name]

        # Batch and VRAMMatrix share the same VRAM storage layout.
        if tensor_info.kind not in ("Batch", "VRAMMatrix"):
            raise ValueError(
                f"Tensor '{tensor_name}' must be Batch or VRAMMatrix to store from VRAM, got {tensor_info.kind}"
            )

        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address to store")

        if hbm_addr is None:
            if tensor_info.hbm_addr >= 0:
                hbm_addr = tensor_info.hbm_addr
            else:
                raise ValueError(f"Tensor '{tensor_name}' has no HBM address. Please specify hbm_addr.")

        batch_size = tensor_info.shape[0]
        hidden_size = tensor_info.shape[1]

        isa_code = f"; Store {tensor_name} from VRAM to HBM\n"
        isa_code += f"; VRAM[{tensor_info.vram_addr}] -> HBM[{hbm_addr}], shape=({batch_size}, {hidden_size})\n"

        gp_regs = self.register_allocator.allocate_gp(5)

        if hbm_addr_reg is None:
            addr_regs = self.register_allocator.allocate_addr(1)
            hbm_addr_reg = addr_regs[0]
            need_free_addr = True
        else:
            addr_regs = []
            need_free_addr = False

        try:
            gp_regs_for_addr = self.register_allocator.allocate_gp(2)
            isa_code += preload_addr_reg_asm(
                addr_reg_to_set=[hbm_addr_reg], available_registers=gp_regs_for_addr, addr_reg_val=[hbm_addr]
            )
            self.register_allocator.free_gp(gp_regs_for_addr)

            isa_code += store_act_asm(
                vlen=vlen,
                batch=batch_size,
                hidden_size=hidden_size,
                alive_registers=gp_regs,
                act_vram_offset=tensor_info.vram_addr,
                hbm_addr_reg=hbm_addr_reg,
                stride_size=hidden_size,
                store_amount=store_amount,
            )

            if tensor_info.hbm_addr < 0 or tensor_info.hbm_addr != hbm_addr:
                tensor_info.hbm_addr = hbm_addr
                # HBM stores the MXFP-expanded size (logical size × real_data_ratio).
                size = batch_size * hidden_size
                tensor_info.hbm_size = int(size * self.real_data_ratio)
        finally:
            self.register_allocator.free_gp(gp_regs)
            if need_free_addr:
                self.register_allocator.free_addr(addr_regs)

        if hbm_object_name is not None:
            self.add_hbm_object(
                name=hbm_object_name,
                hbm_addr=hbm_addr,
                shape=(batch_size, hidden_size),
            )

        self.generated_code += isa_code

        return isa_code

    def normalize(
        self,
        tensor_name: str,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> str:
        """
        Normalize a VRAM tensor in-place.

        Supports:
        - mode="rms":   RMSNorm
        - mode="layer": LayerNorm

        Args:
            tensor_name: Tensor name in symbol table (must have VRAM address)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: self.mlen)
            scratchpad_vram_addr: scratchpad VRAM address (default: auto-allocate temporary space)
        """
        if tensor_name not in self:
            raise KeyError(f"Tensor '{tensor_name}' not found in symbol table")

        tensor_info = self[tensor_name]
        if tensor_info.vram_addr is None:
            raise ValueError(f"Tensor '{tensor_name}' has no VRAM address")

        batch_size, hidden_dim = tensor_info.shape
        if vlen is None:
            vlen = self.mlen
        if hidden_dim % vlen != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by vlen ({vlen}) for normalization_asm")

        mode = mode.lower()
        if mode not in ("rms", "layer"):
            raise ValueError(f"Unsupported normalization mode: {mode}. Expected 'rms' or 'layer'.")

        gp_regs = self.register_allocator.allocate_gp(4)

        temp_scratchpad_name = None
        if scratchpad_vram_addr is None:
            temp_scratchpad_name = f"__norm_scratch__{tensor_name}__{len(self.generated_code)}"
            scratchpad_vram_addr = self.vram_allocator.allocate(vlen, name=temp_scratchpad_name)

        try:
            isa_code = f"; Normalize ({mode}) {tensor_name}, shape=({batch_size}, {hidden_dim})\n"
            if mode == "rms":
                isa_code += rms_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )
            else:
                isa_code += layer_norm_asm(
                    _eps_offset=eps_offset,
                    reci_hid_offset=reci_hid_offset,
                    alive_registers=gp_regs,
                    activation_base_address=tensor_info.vram_addr,
                    scratchpad_base_address=scratchpad_vram_addr,
                    vlen=vlen,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                )

            self.generated_code += isa_code
            return isa_code
        finally:
            # Always release allocated GP registers used by normalization template.
            self.register_allocator.free_gp(gp_regs)
            if temp_scratchpad_name is not None:
                self.vram_allocator.free(temp_scratchpad_name, strict=False)

    def rope(
        self,
        x_name: str,
        x_rot_name: str,
        cos_name: str,
        sin_name: str,
    ) -> str:
        """Apply RoPE in-place: x = x * cos + rotate_half(x) * sin

        All four tensors must already be in VRAM with the same shape (seq_len, head_dim).
        x_rot must be preloaded by the caller as rotate_half(x).
        """
        x_info = self[x_name]
        xrot_info = self[x_rot_name]
        cos_info = self[cos_name]
        sin_info = self[sin_name]

        if x_info.vram_addr is None:
            raise ValueError(f"Tensor '{x_name}' has no VRAM address")

        seq_len, head_dim = x_info.shape
        vlen = self.mlen

        if head_dim % vlen != 0:
            raise ValueError(f"head_dim ({head_dim}) must be divisible by vlen ({vlen}) for rope")

        gp_regs = self.register_allocator.allocate_gp(5)

        scratch_name = f"__rope_scratch__{x_name}__{len(self.generated_code)}"
        scratch_addr = self.vram_allocator.allocate(vlen, name=scratch_name)

        try:
            isa_code = rope_asm(
                alive_registers=gp_regs,
                x_base_address=x_info.vram_addr,
                x_rot_base_address=xrot_info.vram_addr,
                cos_base_address=cos_info.vram_addr,
                sin_base_address=sin_info.vram_addr,
                scratchpad_base_address=scratch_addr,
                vlen=vlen,
                seq_len=seq_len,
                head_dim=head_dim,
            )
            self.generated_code += isa_code
            return isa_code
        finally:
            self.register_allocator.free_gp(gp_regs)
            self.vram_allocator.free(scratch_name, strict=False)

    def get_code(self) -> str:
        """Get all accumulated generated ISA code"""
        return self.generated_code

    def reset(self):
        """Reset compiler state (clear code, but retain symbol table)"""
        self.generated_code = ""
        self.register_allocator = RegisterAllocator()
        # Call TileCompiler.reset() explicitly since the merged class shadows it.
        TileCompiler.reset(self)

    def print_symbol_table(self):
        """Print symbol table"""
        self.print_table()

    def get_symbol_table(self):
        """Get managed object table view."""
        return self

    def get_tensor_info(self, name: str):
        """Get unified tensor/object info by name."""
        return self[name]

    def add_hbm_object(
        self,
        name: str,
        hbm_addr: int,
        shape: tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Register an HBM object and build its HBM layout.

        Wraps ``TileCompiler.add_hbm_object`` with a different positional
        parameter order ``(name, hbm_addr, shape, ...)`` that all DeveloperCompiler
        callers use.
        """
        return TileCompiler.add_hbm_object(
            self,
            name=name,
            shape=shape,
            hbm_addr=hbm_addr,
            real_data_ratio=real_data_ratio,
        )

    def free_hbm_object(self, name: str, strict: bool = False):
        """Free an HBM object by name (defaults to non-strict)."""
        return TileCompiler.free_hbm_object(self, name, strict=strict)

    def get_vram_addr(self, name: str) -> int:
        """Get VRAM base address of an object."""
        info = self.get_tensor_info(name)
        if info.vram_addr is None:
            raise ValueError(f"Object '{name}' has no VRAM address")
        return info.vram_addr

    def get_vram_tile_addr(
        self,
        name: str,
        tile_row_idx: int = 0,
        tile_col_idx: int = 0,
    ) -> int:
        """
        Get VRAM address of a specific tile (sub-block) in a VRAM matrix.

        Args:
            name: matrix name
            tile_row_idx: tile row index (0-based)
            tile_col_idx: tile col index (0-based)
        """
        self._ensure_vram_matrix_layout(name)
        sub = self.get_vram_sub_block(name, tile_row_idx, tile_col_idx)
        return sub.vram_addr

    def ensure_hbm_sub_matrix(
        self,
        name: str,
        hbm_addr: int,
        shape: tuple[int, int],
        real_data_ratio: float = 1.125,
    ):
        """Ensure HBM matrix layout exists."""
        if name in self.hbm_matrices:
            return
        self.add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=real_data_ratio,
        )

    def ensure_vram_matrix_layout(self, name: str, shape: tuple[int, int]):
        """Ensure VRAM matrix layout exists for an already allocated VRAM object."""
        if name in self.vram_matrices:
            return
        vram_addr = self.get_vram_addr(name)
        self.add_vram_object(
            name=name,
            shape=shape,
            vram_addr=vram_addr,
            allocate_if_none=False,
        )

    def free_vram_object(self, name: str, strict: bool = False):
        """Free a VRAM object by name (defaults to non-strict)."""
        return TileCompiler.free_vram_object(self, name, strict=strict)

    # =========================================================================
    # FP Register & FPRAM Management (inlined from former FPRAMCompiler).
    # All state lives on self (register_allocator, fpram_allocator, etc.).
    # =========================================================================

    @property
    def _reg(self) -> RegisterAllocator:
        """Shorthand for self.register_allocator (used by FPVar ISA helpers)."""
        return self.register_allocator

    @property
    def _unroll(self) -> bool:
        """Shorthand for self.unroll_loops."""
        return self.unroll_loops

    def _emit(self, isa_code: str) -> str:
        """Append ISA text to the output buffer and return it."""
        self.generated_code += isa_code
        return isa_code

    # ------------------------------------------------------------------
    # FP Register management
    # ------------------------------------------------------------------

    def allocate_fp_reg(self, count: int = 1) -> list[int]:
        """Allocate FP registers (f0-f7)."""
        return self._reg.allocate_fp(count)

    def free_fp_reg(self, registers: list[int]):
        """Free FP registers."""
        self._reg.free_fp(registers)

    # ------------------------------------------------------------------
    # FPRAM address-space management
    # ------------------------------------------------------------------

    def allocate_fpram(self, name: str, size: int) -> int:
        """Allocate FPRAM space, returns base address."""
        info = self.add_fpram_object(name=name, size=size)
        if info.fpram_addr is None:
            raise RuntimeError(f"Failed to allocate FPRAM for '{name}'")
        return info.fpram_addr

    def free_fpram(self, name: str, strict: bool = True):
        """Free FPRAM object by name."""
        return self.free_fpram_object(name, strict=strict)

    def save_fpram_state(self) -> int:
        """Save FPRAM allocator snapshot."""
        return self.fpram_allocator.save_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM allocator snapshot."""
        self.fpram_allocator.restore_state(snapshot)

    def list_fpram_allocations(self) -> list[str]:
        """List currently allocated FPRAM object names."""
        return list(self.fpram_allocator.allocations.keys())

    def get_fpram_addr(self, name: str) -> int:
        """Get FPRAM base address from object name."""
        return self.get_fpram_layout(name).fpram_addr

    def get_fpram_size(self, name: str) -> int:
        """Get FPRAM allocation size from object name."""
        return self.get_fpram_layout(name).size

    # =========================================================================
    # FPVar ISA helpers (address-based)
    # =========================================================================

    def fpvar_copy_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Copy skipped: count={count}\n")
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Copy: FPRAM[{dst_addr}:{dst_addr + count}] = FPRAM[{src_addr}:{src_addr + count}]"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_src}, {i}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_fill_from_fpram_asm(self, dst_addr: int, src_fpram_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Fill skipped: count={count}\n")
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Fill: FPRAM[{dst_addr}:{dst_addr + count}] = FPRAM[{src_fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_fpram_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_reci_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Reci skipped: count={count}\n")
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Reci: dst = 1/src, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_src}, {i}")
                lines.append("S_RECI_FP f1, f1, 0")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
            lines.append("S_RECI_FP f1, f1, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_exp_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Exp skipped: count={count}\n")
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Exp: dst = exp(src), count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_src}, {i}")
                lines.append("S_EXP_FP f1, f1, 0")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_src}, 0")
            lines.append("S_EXP_FP f1, f1, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_add_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Add skipped: count={count}\n")
        gp = self._reg.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Add: dst = src1 + src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_a}, {i}")
                lines.append(f"S_LD_FP f2, gp{gp_b}, {i}")
                lines.append("S_ADD_FP f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
            lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
            lines.append("S_ADD_FP f1, f1, f2")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
            lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_sub_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Sub skipped: count={count}\n")
        gp = self._reg.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Sub: dst = src1 - src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_a}, {i}")
                lines.append(f"S_LD_FP f2, gp{gp_b}, {i}")
                lines.append("S_SUB_FP f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
            lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
            lines.append("S_SUB_FP f1, f1, f2")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
            lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_mul_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Mul skipped: count={count}\n")
        gp = self._reg.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Mul: dst = src1 * src2, count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_a}, {i}")
                lines.append(f"S_LD_FP f2, gp{gp_b}, {i}")
                lines.append("S_MUL_FP f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
            lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
            lines.append("S_MUL_FP f1, f1, f2")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
            lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_max_asm(self, src1_addr: int, src2_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Max skipped: count={count}\n")
        gp = self._reg.allocate_gp(4)
        gp_a, gp_b, gp_dst, gp_loop = gp
        lines = [f"; FPVar Max: dst = max(src1, src2), count={count}"]
        lines.append(f"S_ADDI_INT gp{gp_a}, gp0, {src1_addr}")
        lines.append(f"S_ADDI_INT gp{gp_b}, gp0, {src2_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f1, gp{gp_a}, {i}")
                lines.append(f"S_LD_FP f2, gp{gp_b}, {i}")
                lines.append("S_MAX_FP f1, f1, f2")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f1, gp{gp_a}, 0")
            lines.append(f"S_LD_FP f2, gp{gp_b}, 0")
            lines.append("S_MAX_FP f1, f1, f2")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_a}, gp{gp_a}, 1")
            lines.append(f"S_ADDI_INT gp{gp_b}, gp{gp_b}, 1")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

    def fpvar_sum_asm(self, src_addr: int, dst_addr: int, count: int) -> str:
        if count <= 0:
            return self._emit(f"; FPVar Sum skipped: count={count}\n")
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; FPVar Sum: FPRAM[{dst_addr}] = sum(FPRAM[{src_addr}:{src_addr + count}])"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append("S_ADD_FP f1, f0, f0")
        if self._unroll:
            for i in range(count):
                lines.append(f"S_LD_FP f2, gp{gp_src}, {i}")
                lines.append("S_ADD_FP f1, f1, f2")
        else:
            lines.append(f"C_LOOP_START gp{gp_loop}, {count}")
            lines.append(f"S_LD_FP f2, gp{gp_src}, 0")
            lines.append("S_ADD_FP f1, f1, f2")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, 1")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

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
        gp = self._reg.allocate_gp(3)
        gp_src, gp_dst, gp_fill = gp
        lines = [f"; FPVar Shift: dst=shift(src, shift={shift}), count={count}, fill=FPRAM[{fill_fpram_addr}]"]
        lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr}")
        lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr}")
        lines.append(f"S_ADDI_INT gp{gp_fill}, gp0, {fill_fpram_addr}")
        lines.append(f"S_LD_FP f3, gp{gp_fill}, 0")

        for i in range(count):
            src_idx = i - shift
            if 0 <= src_idx < count:
                lines.append(f"S_LD_FP f1, gp{gp_src}, {src_idx}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, {i}")
            else:
                lines.append(f"S_ST_FP f3, gp{gp_dst}, {i}")

        self._reg.free_gp(gp)
        return self._emit("\n".join(lines) + "\n")

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

    def tile_row_max_asm(self, source_vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; Tile Row Max from VRAM[{source_vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = None if self.unroll_loops else self._arith_progression(rows)
        fp_prog = None if self.unroll_loops else self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {source_vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_RED_MAX f1, gp{gp_src}, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = source_vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_RED_MAX f1, gp{gp_src}, 0")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sum_asm(self, source_vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_dst, gp_loop = gp
        lines = [f"; Tile Row Sum from VRAM[{source_vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = None if self.unroll_loops else self._arith_progression(rows)
        fp_prog = None if self.unroll_loops else self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {source_vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append("S_ADD_FP f1, f0, f0")
            lines.append(f"V_RED_SUM f1, gp{gp_src}, 0, 0")
            lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = source_vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append("S_ADD_FP f1, f0, f0")
                lines.append(f"V_RED_SUM f1, gp{gp_src}, 0, 0")
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {fpram_addr}")
                lines.append(f"S_ST_FP f1, gp{gp_dst}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_exp_asm(self, vram_addr: int, rows: list[int]) -> str:
        gp = self.register_allocator.allocate_gp(2)
        gp_src, gp_loop = gp
        lines = [f"; Tile Row Exp on VRAM[{vram_addr}]"]
        prog = None if self.unroll_loops else self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_EXP_V gp{gp_src}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_EXP_V gp{gp_src}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_reci_asm(self, vram_addr: int, rows: list[int]) -> str:
        gp = self.register_allocator.allocate_gp(2)
        gp_src, gp_loop = gp
        lines = [f"; Tile Row Reciprocal on VRAM[{vram_addr}]"]
        prog = None if self.unroll_loops else self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_RECI_V gp{gp_src}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"V_RECI_V gp{gp_src}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sub_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Sub FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = None if self.unroll_loops else self._arith_progression(rows)
        fp_prog = None if self.unroll_loops else self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, 0, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_SUB_VF gp{gp_src}, gp{gp_src}, f1, 0, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_mul_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Mul FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = None if self.unroll_loops else self._arith_progression(rows)
        fp_prog = None if self.unroll_loops else self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_MUL_VF gp{gp_src}, gp{gp_src}, f1, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_MUL_VF gp{gp_src}, gp{gp_src}, f1, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_add_fp_asm(self, vram_addr: int, row_map: list[tuple[int, int]]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_src, gp_fp, gp_loop = gp
        lines = [f"; Tile Row Add FP on VRAM[{vram_addr}]"]
        rows = [r for r, _ in row_map]
        fp_addrs = [a for _, a in row_map]
        row_prog = None if self.unroll_loops else self._arith_progression(rows)
        fp_prog = None if self.unroll_loops else self._arith_progression(fp_addrs)
        if row_prog is not None and fp_prog is not None:
            row_start, row_count, row_step = row_prog
            fp_start, _, fp_step = fp_prog
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fp_start}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
            lines.append(f"V_ADD_VF gp{gp_src}, gp{gp_src}, f1, 0")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_fp}, gp{gp_fp}, {fp_step}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx, fpram_addr in row_map:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {row_addr}")
                lines.append(f"S_ADDI_INT gp{gp_fp}, gp0, {fpram_addr}")
                lines.append(f"S_LD_FP f1, gp{gp_fp}, 0")
                lines.append(f"V_ADD_VF gp{gp_src}, gp{gp_src}, f1, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_add_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Add: VRAM[{dst_addr}] += VRAM[{src_addr}]"]
        prog = None if self.unroll_loops else self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_ADD_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_sub_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Sub: VRAM[{dst_addr}] -= VRAM[{src_addr}]"]
        prog = None if self.unroll_loops else self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_SUB_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_SUB_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def tile_row_mul_asm(self, dst_addr: int, src_addr: int, rows: list[int]) -> str:
        gp = self.register_allocator.allocate_gp(3)
        gp_dst, gp_src, gp_loop = gp
        lines = [f"; Tile Row Mul: VRAM[{dst_addr}] *= VRAM[{src_addr}]"]
        prog = None if self.unroll_loops else self._arith_progression(rows)
        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {dst_addr + row_start * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {src_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_MUL_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"S_ADDI_INT gp{gp_src}, gp{gp_src}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                d = dst_addr + row_idx * self.mlen
                s = src_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {d}")
                lines.append(f"S_ADDI_INT gp{gp_src}, gp0, {s}")
                lines.append(f"V_MUL_VV gp{gp_dst}, gp{gp_dst}, gp{gp_src}, 0")
        self.register_allocator.free_gp(gp)
        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

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
            isa_code = f"; === VRAM Fill Zero: VRAM[{vram_addr}] rows [] = 0 ===\n"
            self.generated_code += isa_code
            return isa_code

        gp_regs = self.register_allocator.allocate_gp(2)
        gp_dst, gp_loop = gp_regs

        lines = []
        lines.append(f"; === VRAM Fill Zero: VRAM[{vram_addr}] rows {rows} = 0 ===")
        prog = None if self.unroll_loops else self._arith_progression(rows)

        if prog is not None:
            row_start, row_count, row_step = prog
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {vram_addr + row_start * self.mlen}")
            lines.append(f"C_LOOP_START gp{gp_loop}, {row_count}")
            lines.append(f"V_MUL_VF gp{gp_dst}, gp{gp_dst}, f0, 0")
            lines.append(f"S_ADDI_INT gp{gp_dst}, gp{gp_dst}, {row_step * self.mlen}")
            lines.append(f"C_LOOP_END gp{gp_loop}")
        else:
            for row_idx in rows:
                row_addr = vram_addr + row_idx * self.mlen
                lines.append(f"S_ADDI_INT gp{gp_dst}, gp0, {row_addr}")
                lines.append(f"V_MUL_VF gp{gp_dst}, gp{gp_dst}, f0, 0")

        self.register_allocator.free_gp(gp_regs)

        isa_code = "\n".join(lines) + "\n"
        self.generated_code += isa_code
        return isa_code

    def reset_mram(self) -> str:
        """
        Reset MRAM allocator, free all allocated space
        Used in scenarios where sub-blocks need to be reloaded within a for loop
        """
        self.mram_allocator.reset()
        self.loaded_sub_blocks.clear()

        isa_code = "; === Reset MRAM ===\n"
        self.generated_code += isa_code
        return isa_code

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

        self.generated_code += isa_code
        return isa_code

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

        self.generated_code += isa_code
        return isa_code

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
        self.generated_code += isa_code

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

        self.generated_code += isa_code
        return isa_code

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
        self.generated_code += isa_code
        return isa_code

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

        self.generated_code += isa_code
        return isa_code

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

        self.generated_code += isa_code
        return isa_code

    # =========================================================================
    # Expanded Flash Attention Operations
    # =========================================================================

    def init_online_softmax(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """
        Initialize Online Softmax state for Q block q_idx:
          m_old = -inf (FP SRAM), l = 0 (FP SRAM), O_row = 0 (VRAM).
        """
        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_old_addr = fp_sram_start
        l_addr = fp_sram_start + 2 * self.mlen  # skip m_res region

        o_info = self[o_matrix]
        o_vram_addr = o_info.vram_addr
        row_offset = q_idx * self.mlen

        isa_code = f"; === Init Online Softmax for Q block {q_idx} ===\n"

        isa_code += self._reset_fpsram_asm(m_old_addr, self.mlen, 2)  # slot 2 = -inf
        isa_code += self._reset_fpsram_asm(l_addr, self.mlen, 0)  # slot 0 = 0.0
        isa_code += self._reset_vram_asm(
            start_address=o_vram_addr,
            rows=self.mlen,
            cols=head_dim,
            total_rows=seq_len,
            mlen=self.mlen,
            row_offset=row_offset,
        )

        self.generated_code += isa_code
        return isa_code

    def online_softmax_block(
        self,
        s_block_matrix: str,
        scale: float,
    ) -> str:
        """
        Run Online Softmax on one S block.
          Input:   S_block (mlen × mlen) in VRAM
          Output:  P (mlen × mlen) in-place in VRAM
          Updates: m_old, m_res, l in FP SRAM
          ``scale`` is the QK^T scaling factor (typically 1/sqrt(d)).
        """
        s_info = self[s_block_matrix]
        s_address = s_info.vram_addr

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_start_address = fp_sram_start

        isa_code = f"; === Online Softmax Block {s_block_matrix} ===\n"
        isa_code += self._online_softmax_asm(
            mlen=self.mlen, s_address=s_address, m_start_address=m_start_address, scale=scale
        )

        self.generated_code += isa_code
        return isa_code

    def compute_pv(
        self,
        s_block_matrix: str,
        v_sub_matrix: str,
        k_idx: int,
        pv_matrix: str,
        head_dim: int,
    ) -> str:
        """
        Compute PV = P @ V[k_idx].

        P lives in s_block_matrix (softmax result); V is prefetched from
        HBM; PV is written to VRAM via pv_matrix.
        """
        s_info = self[s_block_matrix]
        p_address = s_info.vram_addr

        pv_info = self[pv_matrix]
        pv_address = pv_info.vram_addr

        v_layout = self.get_hbm_layout(v_sub_matrix)
        v_hbm_offset = k_idx * self.mlen * head_dim

        isa_code = f"; === Compute PV = P @ V[k_idx={k_idx}] ===\n"

        addr_regs = self.register_allocator.allocate_addr(1)
        v_hbm_reg = addr_regs[0]
        gp_regs = self.register_allocator.allocate_gp(2)

        from compiler.asm_templates import preload_addr_reg_asm

        isa_code += preload_addr_reg_asm(
            addr_reg_to_set=[v_hbm_reg], available_registers=gp_regs, addr_reg_val=[v_layout.hbm_base_addr]
        )

        isa_code += self._pv_multiply_asm(
            mlen=self.mlen,
            blen=self.blen,
            head_dim=head_dim,
            p_address=p_address,
            v_hbm_offset_reg=v_hbm_reg,
            v_hbm_offset=v_hbm_offset,
            pv_address=pv_address,
        )

        self.register_allocator.free_gp(gp_regs)
        self.register_allocator.free_addr(addr_regs)

        self.generated_code += isa_code
        return isa_code

    def scale_o_row(
        self,
        o_matrix: str,
        q_idx: int,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """Scale the current row block of O by m_res: O[q_idx] *= m_res."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        m_res_addr = fp_sram_start + self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Scale O[q_idx={q_idx}] by m_res ===\n"
        isa_code += self._scale_o_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=seq_len,
            m_res_address=m_res_addr,
            o_address=o_address,
            row_offset=row_offset,
        )

        self.generated_code += isa_code
        return isa_code

    def final_scale_o(
        self,
        q_idx: int,
        o_matrix: str,
        seq_len: int,
        head_dim: int,
    ) -> str:
        """Final scaling: O[q_idx] /= l."""
        o_info = self[o_matrix]
        o_address = o_info.vram_addr

        fp_sram_start = self._ONLINE_SOFTMAX_FPSRAM_BASE
        l_addr = fp_sram_start + 2 * self.mlen

        row_offset = q_idx * self.mlen

        isa_code = f"; === Final Scale O for Q block {q_idx} ===\n"
        isa_code += self._final_scaling_asm(
            mlen=self.mlen,
            head_dim=head_dim,
            seq_len=seq_len,
            l_address=l_addr,
            o_address=o_address,
            row_offset=row_offset,
        )

        self.generated_code += isa_code
        return isa_code


# Example Usage


class _DeveloperView:
    """
    Back-compat proxy for legacy ``prog._compiler.X(...)`` call sites.

    PlenaCompiler now inherits DeveloperCompiler rather than composing it, so
    for call sites that still expect to reach the low-level DeveloperCompiler
    API (e.g., ``allocate_fpram(name=..., size=...)`` returning an int), we
    expose a proxy whose attribute lookup resolves callables on
    DeveloperCompiler directly (bypassing any PlenaCompiler overrides with
    colliding names). Non-callable attributes (e.g., ``generated_code``,
    ``vram_allocator``) fall through to the underlying instance unchanged.
    """

    __slots__ = ("_inst",)

    def __init__(self, inst: PlenaCompiler):
        object.__setattr__(self, "_inst", inst)

    def __getattr__(self, name: str):
        cls_attr = getattr(DeveloperCompiler, name, None)
        if cls_attr is not None and callable(cls_attr):
            return cls_attr.__get__(self._inst, DeveloperCompiler)
        return getattr(self._inst, name)

    def __setattr__(self, name: str, value):
        setattr(self._inst, name, value)


# ============================================================================
# TensorVar Proxy Object Hierarchy
# ============================================================================


class TensorVar:
    """
    Tensor proxy object base class

    All tensor variables inherit from this class.
    Supports __matmul__ (`@`) operator, which automatically dispatches to appropriate PlenaCompiler methods.

    Dual naming:
    - display_name: User-visible name (e.g., "temp", "Q", "S")
    - internal_name: System internal name (e.g., "my_func_0/temp"), used for symbol table and ISA generation
    """

    def __init__(
        self,
        program: PlenaCompiler,
        internal_name: str,
        kind: str,
        shape: tuple[int, int],
        display_name: str | None = None,
    ):
        self._program = program
        self.internal_name = internal_name  # System internal name (with scope prefix), used by symbol table
        self.display_name = display_name if display_name is not None else internal_name  # User-visible name
        self.kind = kind  # "input", "batch", "matrix", "vram_matrix"
        self.shape = shape

    @property
    def name(self) -> str:
        """Compatibility property: returns internal_name for internal system use"""
        return self.internal_name

    def __matmul__(self, other):
        """A @ B: Dispatch to appropriate computation based on operand types"""
        return self._program._dispatch_matmul(self, other)

    def __repr__(self):
        if self.display_name != self.internal_name:
            return (
                f"{self.__class__.__name__}(display={self.display_name!r}, "
                f"internal={self.internal_name!r}, shape={self.shape})"
            )
        return f"{self.__class__.__name__}({self.display_name!r}, shape={self.shape})"


class InputVar(TensorVar):
    """
    Input variable: tensor declared in HBM

    Not yet loaded to VRAM; needs to be loaded via load_batch / load_matrix.

    If ``prestaged_vram_addr`` is not None the tensor is assumed to be already
    present in VRAM at that byte address.  ``load_batch`` will register it at
    that address without emitting any HBM→VRAM prefetch instructions.
    """

    def __init__(
        self,
        program: PlenaCompiler,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int,
        hbm_size: int,
        display_name: str | None = None,
        prestaged_vram_addr: int | None = None,
    ):
        super().__init__(program, name, "input", shape, display_name=display_name)
        self.hbm_addr = hbm_addr
        self.hbm_size = hbm_size
        self.prestaged_vram_addr = prestaged_vram_addr


class FPVar:
    """
    FP variable: maps to a contiguous region in FPRAM

    Declared via prog.fp_var("scale", size=1), automatically allocates FPRAM space.
    Provides .address for ISA generation (S_LD_FP / S_ST_FP).

    Usage:
        scale = prog.fp_var("scale", size=1)
        m_old = prog.fp_var("m_old", size=64)

        scale.address   # -> FPRAM address (int)
        scale.size      # -> number of elements
        scale[3]        # -> address + 3 (element offset)
    """

    def __init__(
        self, program: PlenaCompiler, internal_name: str, address: int, size: int, display_name: str | None = None
    ):
        self._program = program
        self.internal_name = internal_name
        self.display_name = display_name if display_name is not None else internal_name
        self.address = address
        self.size = size

    @property
    def name(self) -> str:
        return self.internal_name

    def __getitem__(self, idx: int) -> int:
        """Element offset: fp_var[i] -> address + i"""
        if idx < 0 or idx >= self.size:
            raise IndexError(f"FPVar '{self.display_name}' index {idx} out of range [0, {self.size})")
        return self.address + idx

    def __repr__(self):
        return f"FPVar({self.display_name!r}, addr={self.address}, size={self.size})"


class VRAMMatrixVar(TensorVar):
    """
    VRAM matrix variable: large matrix allocated via alloc

    Used to store intermediate results (e.g., S block, PV, O).
    Supports sub-block indexed writes: `O[r][c] = ...`
    """

    def __init__(self, program: PlenaCompiler, name: str, shape: tuple[int, int], display_name: str | None = None):
        super().__init__(program, name, "vram_matrix", shape, display_name=display_name)


# ============================================================================
# PlenaCompiler Main Class
# ============================================================================


class PlenaCompiler(DeveloperCompiler):
    """
    PLENA High-level Compiler Interface.

    Inherits the ISA-emission machinery from DeveloperCompiler and layers a
    Pythonic DSL on top. All operations are eagerly evaluated — ISA code is
    generated immediately upon call.
    """

    def __init__(self, mlen: int = 64, blen: int = 4, real_data_ratio: float = 1.125, unroll_loops: bool = False):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM storage ratio (MXFP8 format = 1.125)
            unroll_loops: If True, unroll sub-projection loops at ASM-gen time to
                          eliminate C_LOOP_START/END overhead. Overridden by the
                          ATEN_UNROLL env var ("1"=True, "0"=False).
        """
        _env_unroll = os.environ.get("ATEN_UNROLL", "")
        if _env_unroll == "1":
            unroll_loops = True
        elif _env_unroll == "0":
            unroll_loops = False
        super().__init__(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio, unroll_loops=unroll_loops)

        # HBM address auto-allocation
        self._next_hbm_addr: int = 0
        self._hbm_free_blocks: list[tuple[int, int]] = []  # (addr, size)

        # Variable registries
        self._inputs: dict[str, InputVar] = {}
        self._tensors: dict[str, TensorVar] = {}
        self._fp_vars: dict[str, FPVar] = {}
        self._functions: dict[str, Callable] = {}
        self._registered_hbm_sub_matrices: dict[str, bool] = {}
        self._registered_vram_sub_matrices: dict[str, bool] = {}

        self._result_tensor: TensorVar | None = None

        # Auto-generated name counter
        self._auto_name_counter: int = 0

        # Function scope namespace
        # Push a prefix on each function call (e.g., "linear_0/"), pop on exit
        # _auto_name will automatically add current scope prefix, avoiding name conflicts when calling the same function multiple times
        self._scope_stack: list[str] = []
        self._function_call_counters: dict[str, int] = {}  # func_name -> call count

    # ========================================================================
    # Property Access
    # ========================================================================

    # mlen / blen are instance attributes inherited from TileCompiler.__init__.

    @property
    def compiler(self) -> PlenaCompiler:
        """Legacy accessor — returns self now that PlenaCompiler is the compiler."""
        return self

    @property
    def _compiler(self) -> _DeveloperView:
        """Back-compat shim for legacy ``prog._compiler.X(...)`` call sites.
        Returns a proxy that resolves callables against DeveloperCompiler
        directly so callers reach the low-level API regardless of any
        PlenaCompiler override with the same name."""
        return _DeveloperView(self)

    @property
    def symbol_table(self):
        """Access symbol table."""
        return self.get_symbol_table()

    # ========================================================================
    # Input Declaration
    # ========================================================================

    def input(
        self,
        name: str,
        shape: tuple[int, int],
        hbm_addr: int | None = None,
        prestaged_vram_addr: int | None = None,
    ) -> InputVar:
        """
        Declare an input tensor (in HBM).

        Args:
            name: tensor name
            shape: (height, width)
            hbm_addr: HBM address (None = auto-allocate)
            prestaged_vram_addr: If an int, the tensor is assumed to be already
                present in VRAM at this byte address.  A subsequent call to
                ``load_batch`` will register it at that address without emitting
                any HBM→VRAM prefetch instructions.  If None (default), the
                normal HBM→VRAM load path is used.

        Returns:
            InputVar proxy object
        """
        h, w = shape
        size = h * w
        hbm_size = int(size * self.real_data_ratio)

        if hbm_addr is None:
            hbm_addr = self._allocate_hbm(hbm_size)

        var = InputVar(self, name, shape, hbm_addr, hbm_size, prestaged_vram_addr=prestaged_vram_addr)
        self._inputs[name] = var
        super().add_hbm_object(
            name=name,
            hbm_addr=hbm_addr,
            shape=shape,
            real_data_ratio=self.real_data_ratio,
        )
        return var

    # ========================================================================
    # Load Operations
    # ========================================================================

    def load_batch(
        self,
        input_var: InputVar,
        name: str | None = None,
    ) -> VRAMMatrixVar:
        """
        Load tensor from HBM to VRAM (Batch type).

        When ``input_var.prestaged_vram_addr`` is set the tensor is assumed to
        be already resident in VRAM at that address.  No HBM→VRAM prefetch
        instructions are emitted; the tensor is simply registered in the symbol
        table at the given address.

        Args:
            input_var: source InputVar
            name: result name (None = use input name)

        Returns:
            VRAMMatrixVar proxy object
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Expected InputVar, got {type(input_var)}")

        display_name = name if name is not None else input_var.display_name
        internal_name = self._scoped_name(display_name)

        if input_var.prestaged_vram_addr is not None:
            # Prestaged path: tensor is already in VRAM — register without ISA.
            h, w = input_var.shape
            vram_addr = input_var.prestaged_vram_addr
            # Tell the VRAM allocator that this region is occupied so subsequent
            # allocations don't collide with it.
            self.vram_allocator._vmm.mark_used(vram_addr, h * w, name=internal_name)
            super().add_vram_object(
                name=internal_name,
                shape=(h, w),
                vram_addr=vram_addr,
                dtype="fp16",
                kind="Batch",
                allocate_if_none=False,
                strict=False,
            )
        else:
            # Normal path: emit HBM → VRAM prefetch ISA.
            super().load_batch(
                hbm_object_name=input_var.name, vram_object_name=internal_name, vlen=self.mlen, preload_len=4
            )

        var = VRAMMatrixVar(self, internal_name, input_var.shape, display_name=display_name)
        self._tensors[internal_name] = var
        return var

    # ========================================================================
    # Store Operations
    # ========================================================================

    def store(self, tensor_var, name: str | None = None, hbm_addr: int | None = None) -> InputVar:
        """
        Write tensor from VRAM back to HBM.

        Returns:
            InputVar proxy object (can be loaded back later)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Store requires VRAMMatrixVar, got {type(tensor_var)}")

        display_name = name if name is not None else f"{tensor_var.display_name}_stored"
        internal_name = self._scoped_name(display_name)

        if hbm_addr is None:
            h, w = tensor_var.shape
            size = h * w
            hbm_size = int(size * self.real_data_ratio)
            hbm_addr = self._allocate_hbm(hbm_size)
        else:
            h, w = tensor_var.shape
            hbm_size = int(h * w * self.real_data_ratio)

        super().store_to_hbm(
            tensor_name=tensor_var.name,  # internal name for symbol table lookup
            hbm_addr=hbm_addr,
            hbm_object_name=internal_name,
            vlen=self.mlen,
        )

        var = InputVar(self, internal_name, tensor_var.shape, hbm_addr, hbm_size, display_name=display_name)
        self._inputs[internal_name] = var
        return var

    # ========================================================================
    # VRAM Matrix Allocation
    # ========================================================================

    def alloc(self, name: str, rows: int, cols: int, strict: bool = True) -> VRAMMatrixVar:
        """
        Allocate a VRAM matrix.

        Used to store intermediate results (e.g., S block, PV, O).
        Within function scope, names are automatically prefixed to avoid conflicts.

        Args:
            name: matrix name (user-visible)
            rows: number of rows
            cols: number of columns
            strict: if False, skip mlen-alignment checks (for small scratch matrices)

        Returns:
            VRAMMatrixVar proxy object
        """
        display_name = name
        internal_name = self._scoped_name(name)
        super().allocate_vram_matrix(name=internal_name, rows=rows, cols=cols, strict=strict)

        var = VRAMMatrixVar(self, internal_name, (rows, cols), display_name=display_name)
        self._tensors[internal_name] = var
        return var

    def free_tensor(self, tensor_var: TensorVar):
        """
        Free a tensor in VRAM, reclaiming space for subsequent allocations.

        Freed space can be reused by new alloc() or other operations.
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"Can only free VRAMMatrixVar, got {type(tensor_var)}")

        super().free_vram_object(tensor_var.name, strict=False)
        # Keep sub-matrix registration state consistent after free.
        self._registered_vram_sub_matrices[tensor_var.name] = False

    def free_input(self, input_var: InputVar):
        """
        Free an InputVar bookkeeping and recycle its HBM range for future auto-allocation.

        Notes:
        - This only affects PlenaCompiler's address management state.
        - If a freed input is referenced again later, caller is responsible for correctness.
        """
        if not isinstance(input_var, InputVar):
            raise TypeError(f"Can only free InputVar, got {type(input_var)}")

        super().free_hbm_object(input_var.name, strict=False)
        self._registered_hbm_sub_matrices[input_var.name] = False
        self._recycle_hbm(input_var.hbm_addr, input_var.hbm_size)
        self._inputs.pop(input_var.name, None)

    def free_fp_var(self, fp_var: FPVar):
        """
        Free an FPVar and return its block to FPRAM free pool.
        """
        if not isinstance(fp_var, FPVar):
            raise TypeError(f"Can only free FPVar, got {type(fp_var)}")
        self.free_fpram(fp_var.name, strict=True)

    # ========================================================================
    # Normalization Operations
    # ========================================================================

    def norm(
        self,
        tensor_var: TensorVar,
        mode: str = "rms",
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> TensorVar:
        """
        Normalize tensor in-place.

        Args:
            tensor_var: tensor to normalize (must have VRAM backing, e.g., VRAMMatrixVar)
            mode: "rms" or "layer"
            eps_offset: FPRAM address of epsilon
            reci_hid_offset: FPRAM address of 1/hidden_dim
            vlen: vector length (default: program mlen)
            scratchpad_vram_addr: optional scratchpad VRAM address

        Returns:
            The same tensor_var (in-place operation)
        """
        if not isinstance(tensor_var, VRAMMatrixVar):
            raise TypeError(f"norm requires VRAMMatrixVar, got {type(tensor_var)}")

        super().normalize(
            tensor_name=tensor_var.name,
            mode=mode,
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )
        return tensor_var

    def rms_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> TensorVar:
        """RMS normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="rms",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )

    def layer_norm(
        self,
        tensor_var: TensorVar,
        eps_offset: int = 1,
        reci_hid_offset: int = 2,
        vlen: int | None = None,
        scratchpad_vram_addr: int | None = None,
    ) -> TensorVar:
        """Layer normalization (in-place)."""
        return self.norm(
            tensor_var=tensor_var,
            mode="layer",
            eps_offset=eps_offset,
            reci_hid_offset=reci_hid_offset,
            vlen=vlen,
            scratchpad_vram_addr=scratchpad_vram_addr,
        )

    # ========================================================================
    # FP Variable (FPRAM)
    # ========================================================================

    def allocate_fpram(
        self,
        internal_name: str,
        size: int = 1,
        display_name: str | None = None,
    ) -> FPVar:
        """
        Allocate FPRAM with explicit internal name and return FPVar proxy.
        """
        if size <= 0:
            raise ValueError(f"FPRAM allocation size must be positive, got {size}")

        address = super().allocate_fpram(internal_name, size)
        var = FPVar(
            self,
            internal_name,
            address,
            size,
            display_name=display_name if display_name is not None else internal_name,
        )
        self._fp_vars[internal_name] = var
        return var

    def free_fpram(self, internal_name: str, strict: bool = True):
        """
        Free FPRAM allocation by internal name.
        """
        super().free_fpram(internal_name, strict=strict)
        self._fp_vars.pop(internal_name, None)

    def fp_var(self, name: str, size: int = 1) -> FPVar:
        """
        Declare an FP variable in FPRAM.

        Allocates a contiguous region in FPRAM and returns an FPVar proxy.
        Within function scope, names are automatically prefixed.

        Args:
            name: variable name
            size: number of f16 elements to allocate (default 1)

        Returns:
            FPVar proxy object (use .address for ISA generation)

        Example:
            scale = prog.fp_var("scale")          # 1 element
            m_old = prog.fp_var("m_old", size=64) # 64 elements
            prog.compiler  # access compiler for ISA if needed
        """
        display_name = name
        internal_name = self._scoped_name(name)

        return self.allocate_fpram(
            internal_name=internal_name,
            size=size,
            display_name=display_name,
        )

    def save_fpram_state(self) -> int:
        """Save FPRAM allocator snapshot"""
        return super().save_fpram_state()

    def restore_fpram_state(self, snapshot: int):
        """Restore FPRAM allocator snapshot"""
        super().restore_fpram_state(snapshot)
        # Remove FPVar proxies that are no longer allocated in allocator.
        allocations = set(super().list_fpram_allocations())
        to_remove = [n for n in self._fp_vars if n not in allocations]
        for n in to_remove:
            del self._fp_vars[n]

    # ========================================================================
    # FPRAM Tile Operations
    # ========================================================================

    def _resolve_fpram_addr(self, addr_or_var: int | FPVar, offset: int = 0) -> int:
        if isinstance(addr_or_var, FPVar):
            if offset < 0 or offset >= addr_or_var.size:
                raise ValueError(
                    f"FPVar offset out of range: offset={offset}, size={addr_or_var.size}, var={addr_or_var.name}"
                )
            return addr_or_var.address + offset
        if not isinstance(addr_or_var, int):
            raise TypeError(f"Expected int or FPVar, got {type(addr_or_var)}")
        return addr_or_var + offset

    def _resolve_rows(
        self,
        row_idx: int | None = None,
        rows: list[int] | None = None,
    ) -> list[int]:
        if row_idx is not None and rows is not None:
            raise ValueError("Provide either row_idx or rows, not both")
        if rows is not None:
            return rows
        if row_idx is not None:
            return [row_idx]
        return list(range(self.mlen))

    def tile_row_max(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Max: reduce a single row to max, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)

        Example:
            m = prog.fp_var("m", size=1)
            S = prog.alloc("S", 64, 64)
            for row in range(64):
                prog.tile_row_max(m, S, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(target_fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_max(
            source_matrix=source.name,
            row_map=row_map,
        )

    def tile_row_sum(
        self,
        target_fpram_addr: int | FPVar,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        target_offset: int = 0,
        target_base_offset: int = 0,
    ):
        """
        Tile Row Sum: reduce a single row to sum, store to FPRAM address.

        Args:
            target_fpram_addr: FPRAM address or FPVar to write result
            source: VRAM tile (mlen x mlen)
            row_idx: single row index (legacy path)
            rows: multiple row indices
            target_offset: element offset when target_fpram_addr is FPVar
            target_base_offset: base offset for multi-row writes (contiguous)
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [target_offset]
        else:
            offsets = [target_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(target_fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_sum(source.name, row_map)

    def tile_row_exp(
        self,
        source: VRAMMatrixVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Exp: in-place exp on specified rows.

        For each row i: source[i, :] = exp(source[i, :])
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        super().tile_row_exp(source.name, resolved_rows)

    def tile_row_reci(
        self,
        source: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Reciprocal: in-place 1/x on specified rows.

        For each row i: source[i, :] = 1.0 / source[i, :]
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_reci(source.name, rows)

    def tile_row_sub_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Sub FP: subtract FPRAM scalar from a single row.

        For row i: source[i, :] = source[i, :] - FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_sub_fp(source.name, row_map)

    def tile_row_mul_fp(
        self,
        source: VRAMMatrixVar,
        fpram_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
        fpram_base_offset: int = 0,
    ):
        """
        Tile Row Mul FP: multiply a single row by FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_addr]
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        if len(resolved_rows) == 1:
            offsets = [fpram_offset]
        else:
            offsets = [fpram_base_offset + i for i in range(len(resolved_rows))]
        row_map = [(r, self._resolve_fpram_addr(fpram_addr, off)) for r, off in zip(resolved_rows, offsets)]
        super().tile_row_mul_fp(source.name, row_map)

    def tile_row_add_fp(
        self,
        source: VRAMMatrixVar,
        fp_var: FPVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Add FP: add FPRAM scalar to each specified row.

        For each row i: source[i, :] = source[i, :] + fp_var[i]
        """
        if rows is None:
            rows = list(range(self.mlen))
        row_map = [(r, fp_var[r]) for r in rows]
        super().tile_row_add_fp(source.name, row_map)

    def tile_row_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Add: dst[i, :] += src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_add(dst.name, src.name, rows)

    def tile_row_sub(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Sub: dst[i, :] -= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_sub(dst.name, src.name, rows)

    def tile_row_mul(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        Tile Row Mul: dst[i, :] *= src[i, :] for specified rows.
        """
        if rows is None:
            rows = list(range(self.mlen))
        super().tile_row_mul(dst.name, src.name, rows)

    def fpvar_reci(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Reciprocal: compute 1/x for FPRAM scalar array.

        For each element i: dst[i] = 1.0 / src[i]

        Args:
            src: source FPVar
            dst: destination FPVar
            count: number of elements (default: min(src.size, dst.size))

        Example:
            l = prog.fp_var("l", size=64)
            inv_l = prog.fp_var("inv_l", size=64)
            prog.fpvar_reci(l, inv_l)  # inv_l = 1/l
        """
        if count is None:
            count = min(src.size, dst.size)
        if count > src.size or count > dst.size:
            raise ValueError(f"count={count} exceeds FPVar size: src.size={src.size}, dst.size={dst.size}")
        super().fpram_reci(src.name, dst.name, count)

    def fpvar_max(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Max: element-wise max for FPRAM scalar arrays.

        For each element i: dst[i] = max(src1[i], src2[i])

        Example:
            m_new = prog.fp_var("m_new", size=64)
            prog.fpvar_max(m_old, row_max, m_new)  # m_new = max(m_old, row_max)
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_max(src1.name, src2.name, dst.name, count)

    def fpvar_sub(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Subtract: element-wise subtraction for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] - src2[i]

        Example:
            diff = prog.fp_var("diff", size=64)
            prog.fpvar_sub(m_old, m_new, diff)  # diff = m_old - m_new
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_sub(src1.name, src2.name, dst.name, count)

    def fpvar_exp(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Exp: element-wise exp for FPRAM scalar array.

        For each element i: dst[i] = exp(src[i])

        Example:
            m_res = prog.fp_var("m_res", size=64)
            prog.fpvar_exp(diff, m_res)  # m_res = exp(diff)
        """
        if count is None:
            count = min(src.size, dst.size)
        super().fpram_exp(src.name, dst.name, count)

    def fpvar_mul(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Multiply: element-wise multiplication for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] * src2[i]

        Example:
            result = prog.fp_var("result", size=64)
            prog.fpvar_mul(l_old, m_res, result)  # result = l_old * m_res
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_mul(src1.name, src2.name, dst.name, count)

    def fpvar_add(
        self,
        src1: FPVar,
        src2: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Add: element-wise addition for FPRAM scalar arrays.

        For each element i: dst[i] = src1[i] + src2[i]

        Example:
            l_new = prog.fp_var("l_new", size=64)
            prog.fpvar_add(l_old, sum_p, l_new)  # l_new = l_old + sum_p
        """
        if count is None:
            count = min(src1.size, src2.size, dst.size)
        super().fpram_add(src1.name, src2.name, dst.name, count)

    def fpvar_copy(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Copy: copy FPRAM scalar array.

        For each element i: dst[i] = src[i]

        Example:
            m_old_saved = prog.fp_var("m_old_saved", size=64)
            prog.fpvar_copy(m_old, m_old_saved)  # backup m_old
        """
        if count is None:
            count = min(src.size, dst.size)
        super().fpram_copy(src.name, dst.name, count)

    def fpvar_sum(
        self,
        src: FPVar,
        dst: FPVar,
        count: int | None = None,
    ):
        """
        FPVar Sum: reduction sum of src into dst[0] (via compiler FPRAM op).
        """
        if count is None:
            count = src.size
        super().fpram_sum(src.name, dst.name, count)

    def fpvar_shift(
        self,
        src: FPVar,
        dst: FPVar,
        shift: int,
        count: int | None = None,
        fill: FPVar | None = None,
    ):
        """
        FPVar Shift: shift src into dst, filling out-of-range slots with fill (default FPRAM zero).
        """
        if count is None:
            count = min(src.size, dst.size)
        fill_name = None if fill is None else fill.name
        super().fpram_shift(
            src_name=src.name,
            dst_name=dst.name,
            shift=shift,
            count=count,
            fill_fpram_name=fill_name,
        )

    def tile_row_mul_fp_broadcast(
        self,
        source: VRAMMatrixVar,
        fpram_scalar_addr: int | FPVar,
        row_idx: int | None = None,
        rows: list[int] | None = None,
        fpram_offset: int = 0,
    ):
        """
        Tile Row Mul FP Broadcast: multiply a single row by a FPRAM scalar.

        For row i: source[i, :] = source[i, :] * FPRAM[fpram_scalar_addr]

        Args:
            source: VRAM tile (mlen x mlen)
            fpram_scalar_addr: FPRAM address or FPVar of the scalar
            row_idx: single row index (legacy path)
            rows: multiple row indices

        Example:
            scale_fp = prog.fp_var("scale", size=1)
            for row in range(64):
                prog.tile_row_mul_fp_broadcast(S, scale_fp, rows=list(range(64)))
        """
        resolved_rows = self._resolve_rows(row_idx=row_idx, rows=rows)
        scalar_addr = self._resolve_fpram_addr(fpram_scalar_addr, fpram_offset)
        super().tile_row_mul_fp_broadcast(source.name, scalar_addr, resolved_rows)

    def fpvar_fill_from_fpram(
        self,
        dst: FPVar,
        src_fpram_addr: int,
        count: int | None = None,
    ):
        """
        FPVar Fill from FPRAM: fill all elements with a value from FPRAM.

        For each element i: dst[i] = FPRAM[src_fpram_addr]

        Args:
            dst: destination FPVar
            src_fpram_addr: source FPRAM address (e.g., 0 for 0.0, 2 for -inf)
            count: number of elements (default: dst.size)

        Example:
            m_old = prog.fp_var("m_old", size=64)
            prog.fpvar_fill_from_fpram(m_old, 2)  # fill with -inf from address 2
        """
        if count is None:
            count = dst.size
        super().fpram_fill_from_fpram(dst.name, src_fpram_addr, count)

    def vram_fill_zero(
        self,
        matrix: VRAMMatrixVar,
        rows: list[int] | None = None,
    ):
        """
        VRAM Fill Zero: fill specified rows with 0.

        Args:
            matrix: VRAM matrix
            rows: which rows to fill (default: all rows)

        Example:
            O = prog.alloc("O", 128, 128)
            prog.vram_fill_zero(O, rows=range(64, 128))  # zero out second half
        """
        if rows is None:
            rows = list(range(matrix.shape[0]))
        super().vram_fill_zero(matrix.name, rows)

    def _ensure_hbm_sub_matrix_registered(self, input_var: InputVar):
        """Ensure an HBM input is registered in compiler sub-matrix manager."""
        if (
            input_var.name in self._registered_hbm_sub_matrices
            and self._registered_hbm_sub_matrices[input_var.name] is True
        ):
            return
        h, w = input_var.shape
        super().ensure_hbm_sub_matrix(
            name=input_var.name,
            hbm_addr=input_var.hbm_addr,
            shape=(h, w),
            real_data_ratio=self.real_data_ratio,
        )
        self._registered_hbm_sub_matrices[input_var.name] = True

    def _ensure_vram_sub_matrix_registered(self, matrix_var: VRAMMatrixVar):
        """Ensure a VRAM matrix is registered in compiler sub-matrix manager."""
        if (
            matrix_var.name in self._registered_vram_sub_matrices
            and self._registered_vram_sub_matrices[matrix_var.name] is True
        ):
            return
        super().ensure_vram_matrix_layout(
            name=matrix_var.name,
            shape=matrix_var.shape,
        )
        self._registered_vram_sub_matrices[matrix_var.name] = True

    def vram_sub_projection_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_col_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
        k_block_start: int = 0,
        k_block_count: int | None = None,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[:][mram_col_idx]
        Supports K-split: k_block_start/k_block_count select a subset of K tiles.
        """
        if not isinstance(vram_matrix, VRAMMatrixVar):
            raise TypeError(f"vram_matrix must be VRAMMatrixVar, got {type(vram_matrix)}")
        if not isinstance(mram_input, InputVar):
            raise TypeError(f"mram_input must be InputVar, got {type(mram_input)}")
        if not isinstance(target, VRAMMatrixVar):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            super().reset_mram()
        super().load_sub_matrix_col(
            name=mram_input.name,
            col_idx=mram_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )
        super().vram_sub_projection_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_col_idx=mram_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
            k_block_start=k_block_start,
            k_block_count=k_block_count,
        )

    def vram_sub_projection_T_to(
        self,
        vram_matrix: VRAMMatrixVar,
        vram_row_idx: int,
        mram_input: InputVar,
        mram_row_idx: int,
        target: VRAMMatrixVar,
        target_row_idx: int,
        target_col_idx: int,
        auto_reset_mram: bool = True,
    ):
        """
        target[target_row_idx][target_col_idx] = vram_matrix[vram_row_idx][:] @ mram_input[mram_row_idx][:]^T
        """
        if not isinstance(vram_matrix, VRAMMatrixVar):
            raise TypeError(f"vram_matrix must be VRAMMatrixVar, got {type(vram_matrix)}")
        if not isinstance(mram_input, InputVar):
            raise TypeError(f"mram_input must be InputVar, got {type(mram_input)}")
        if not isinstance(target, VRAMMatrixVar):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        self._ensure_vram_sub_matrix_registered(vram_matrix)
        self._ensure_hbm_sub_matrix_registered(mram_input)
        if auto_reset_mram:
            super().reset_mram()
        super().load_sub_matrix_row(name=mram_input.name, row_idx=mram_row_idx)
        super().vram_sub_projection_T_to(
            vram_mat_name=vram_matrix.name,
            vram_row_idx=vram_row_idx,
            mram_mat_name=mram_input.name,
            mram_row_idx=mram_row_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    # ========================================================================
    # RoPE (1D Positional Encoding)
    # ========================================================================

    def rope(
        self,
        x_var: VRAMMatrixVar,
        x_rot_var: VRAMMatrixVar,
        cos_var: VRAMMatrixVar,
        sin_var: VRAMMatrixVar,
    ) -> VRAMMatrixVar:
        """Apply Rotary Position Embedding in-place: x = x * cos + rotate_half(x) * sin

        x_rot_var must already be in VRAM as rotate_half(x), preloaded by caller.
        Returns x_var (modified in-place).
        """
        super().rope(
            x_name=x_var.name,
            x_rot_name=x_rot_var.name,
            cos_name=cos_var.name,
            sin_name=sin_var.name,
        )
        return x_var

    # ========================================================================
    # VRAM Matrix Addition
    # ========================================================================

    def vram_add(
        self,
        dst: VRAMMatrixVar,
        src: VRAMMatrixVar,
        dst_row_offset: int = 0,
        src_row_offset: int = 0,
        num_rows: int | None = None,
    ):
        """VRAM matrix add: dst[row_offset:] += src"""
        super().vram_matrix_add(
            dst_matrix=dst.name,
            src_matrix=src.name,
            dst_row_offset=dst_row_offset,
            src_row_offset=src_row_offset,
            num_rows=num_rows,
        )

    def vram_block_add_to(
        self,
        src1: TensorVar,
        src1_row_idx: int,
        src1_col_idx: int,
        src2: TensorVar,
        src2_row_idx: int,
        src2_col_idx: int,
        target: TensorVar,
        target_row_idx: int,
        target_col_idx: int,
    ):
        """
        mlen x mlen block add:
            target[target_row_idx][target_col_idx] =
                src1[src1_row_idx][src1_col_idx] + src2[src2_row_idx][src2_col_idx]

        Supports writing back to the same matrix/block (in-place overwrite).
        """
        allowed = (VRAMMatrixVar,)
        if not isinstance(src1, allowed):
            raise TypeError(f"src1 must be VRAMMatrixVar, got {type(src1)}")
        if not isinstance(src2, allowed):
            raise TypeError(f"src2 must be VRAMMatrixVar, got {type(src2)}")
        if not isinstance(target, allowed):
            raise TypeError(f"target must be VRAMMatrixVar, got {type(target)}")

        super().vram_block_add_to(
            src1_matrix=src1.name,
            src1_row_idx=src1_row_idx,
            src1_col_idx=src1_col_idx,
            src2_matrix=src2.name,
            src2_row_idx=src2_row_idx,
            src2_col_idx=src2_col_idx,
            target_matrix=target.name,
            target_row_idx=target_row_idx,
            target_col_idx=target_col_idx,
        )

    # ========================================================================
    # Flash Attention Operations
    # ========================================================================

    def init_online_softmax(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Initialize Online Softmax state: m=-inf, l=0, O_row=0"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().init_online_softmax(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def online_softmax_block(self, s_block: VRAMMatrixVar, scale: float):
        """Perform Online Softmax on S block"""
        super().online_softmax_block(
            s_block_matrix=s_block.name,
            scale=scale,
        )

    def compute_pv(
        self,
        s_block: VRAMMatrixVar,
        v_input: InputVar,
        k_idx: int,
        pv_matrix: VRAMMatrixVar,
        head_dim: int,
    ):
        """Compute PV = P @ V[k_idx] where P is stored in s_block."""
        if not isinstance(s_block, VRAMMatrixVar):
            raise TypeError(f"s_block must be VRAMMatrixVar, got {type(s_block)}")
        if not isinstance(v_input, InputVar):
            raise TypeError(f"v_input must be InputVar, got {type(v_input)}")
        if not isinstance(pv_matrix, VRAMMatrixVar):
            raise TypeError(f"pv_matrix must be VRAMMatrixVar, got {type(pv_matrix)}")

        self._ensure_hbm_sub_matrix_registered(v_input)
        super().compute_pv(
            s_block_matrix=s_block.name,
            v_sub_matrix=v_input.name,
            k_idx=k_idx,
            pv_matrix=pv_matrix.name,
            head_dim=head_dim,
        )

    def scale_o_row(self, o_matrix: VRAMMatrixVar, q_idx: int):
        """Scale current row block of O by m_res"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().scale_o_row(
            o_matrix=o_matrix.name,
            q_idx=q_idx,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    def final_scale_o(self, q_idx: int, o_matrix: VRAMMatrixVar):
        """Final scaling: O[q_idx] = O[q_idx] / l"""
        o_info = super().get_tensor_info(o_matrix.name)
        seq_len, head_dim = o_info.shape

        super().final_scale_o(
            q_idx=q_idx,
            o_matrix=o_matrix.name,
            seq_len=seq_len,
            head_dim=head_dim,
        )

    # ========================================================================
    # Function Decorator
    # ========================================================================

    def function(self, func: Callable) -> Callable:
        """
        Decorator: Define reusable functions.

        Each invocation generates fresh ISA code (eager evaluation).
        Internally allocated tensors are auto-freed on exit unless returned.

        Scoping: intermediate tensors get a call-index prefix to avoid name
        collisions across repeated calls (e.g., "linear_0/proj_1", "linear_1/proj_1").
        Nested functions compose prefixes: "two_layer_0/linear_0/proj_1".
        """
        func_name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            call_idx = self._function_call_counters.get(func_name, 0)
            self._function_call_counters[func_name] = call_idx + 1

            scope = f"{func_name}_{call_idx}/"
            self._scope_stack.append(scope)

            self.generated_code += f"; === Enter {func_name} (call #{call_idx}) ===\n"

            # Snapshot: record existing tensors before function execution
            tensors_before = set(self._tensors.keys())
            inputs_before = set(self._inputs.keys())
            fp_vars_before = set(self._fp_vars.keys())

            try:
                result = func(*args, **kwargs)

                # Auto-free: free locally allocated tensors that are not returned
                return_names = set()
                return_fp_names = set()
                if isinstance(result, TensorVar):
                    return_names.add(result.internal_name)
                elif isinstance(result, FPVar):
                    return_fp_names.add(result.internal_name)
                elif isinstance(result, (tuple, list)):
                    for r in result:
                        if isinstance(r, TensorVar):
                            return_names.add(r.internal_name)
                        elif isinstance(r, FPVar):
                            return_fp_names.add(r.internal_name)

                for name in set(self._tensors.keys()) - tensors_before:
                    if name not in return_names:
                        tensor = self._tensors[name]
                        if isinstance(tensor, VRAMMatrixVar):
                            self.free_tensor(tensor)
                            self._registered_vram_sub_matrices[tensor.name] = False

                for name in set(self._inputs.keys()) - inputs_before:
                    if name not in return_names:
                        self.free_input(self._inputs[name])

                local_fp_names = sorted(
                    set(self._fp_vars.keys()) - fp_vars_before,
                    key=lambda n: self._fp_vars[n].address,
                    reverse=True,
                )
                for name in local_fp_names:
                    if name in return_fp_names:
                        continue
                    fp_var = self._fp_vars.get(name)
                    if fp_var is not None:
                        self.free_fp_var(fp_var)
            finally:
                self._scope_stack.pop()
                self.generated_code += f"; === Exit {func_name} (call #{call_idx}) ===\n"

            return result

        self._functions[func_name] = wrapper
        wrapper._plena_function = True
        wrapper._plena_name = func_name
        return wrapper

    # ========================================================================
    # Result Marking
    # ========================================================================

    def result(self, tensor_var: TensorVar):
        """Mark output result tensor."""
        self._result_tensor = tensor_var

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get generated ISA code string."""
        return super().get_code()

    def print_symbol_table(self):
        """Print symbol table"""
        super().print_symbol_table()

    def get_symbol_table(self):
        """Get symbol table"""
        return super().get_symbol_table()

    # ========================================================================
    # Operator Dispatch (internal)
    # ========================================================================

    def _dispatch_matmul(self, left: TensorVar, right) -> TensorVar:
        raise TypeError("@ operator is no longer supported in PlenaCompiler. Use explicit program APIs instead.")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _scoped_name(self, name: str) -> str:
        """
        Apply current scope prefix to a name.

        - Top-level alloc("temp"):                    -> "temp"
        - Inside linear call 0, alloc("temp"):        -> "linear_0/temp"
        - Nested two_layer->linear, alloc("temp"):    -> "two_layer_0/linear_0/temp"
        """
        if not self._scope_stack:
            return name
        scope_prefix = "".join(self._scope_stack)
        return f"{scope_prefix}{name}"

    def _allocate_hbm(self, hbm_size: int) -> int:
        """Allocate HBM range, preferring previously freed blocks."""
        best_idx = None
        best_waste = None
        for i, (addr, size) in enumerate(self._hbm_free_blocks):
            if size >= hbm_size:
                waste = size - hbm_size
                if best_waste is None or waste < best_waste:
                    best_idx = i
                    best_waste = waste

        if best_idx is not None:
            addr, block_size = self._hbm_free_blocks.pop(best_idx)
            # Return excess fragment to free list
            excess = block_size - hbm_size
            if excess > 0:
                self._hbm_free_blocks.append((addr + hbm_size, excess))
            return addr

        addr = self._next_hbm_addr
        m = self.mlen
        self._next_hbm_addr = ((addr + hbm_size + m - 1) // m) * m
        return addr

    def _recycle_hbm(self, hbm_addr: int, hbm_size: int):
        """Recycle an HBM range for future auto-allocation."""
        if hbm_size <= 0:
            return
        self._hbm_free_blocks.append((hbm_addr, hbm_size))

    def _auto_name(self, prefix: str = "t") -> str:
        """
        Generate a unique scoped name.

        - Top-level:          "__proj_1"
        - linear call 0:      "linear_0/__proj_1"
        - nested:             "two_layer_0/linear_0/__proj_1"
        """
        self._auto_name_counter += 1
        scope_prefix = "".join(self._scope_stack)
        return f"{scope_prefix}__{prefix}_{self._auto_name_counter}"

    def __repr__(self):
        num_inputs = len(self._inputs)
        num_tensors = len(self._tensors)
        num_functions = len(self._functions)
        code_len = len(super().get_code().splitlines())
        return (
            f"PlenaCompiler(mlen={self.mlen}, blen={self.blen}, "
            f"inputs={num_inputs}, tensors={num_tensors}, "
            f"functions={num_functions}, isa_lines={code_len})"
        )


class TensorKind(Enum):
    """Identifies which memory the tensor lives in / which proxy backs it."""

    HBM = "hbm"  # legacy: InputVar
    VRAM = "vram"  # legacy: VRAMMatrixVar
    FPRAM = "fpram"  # legacy: FPVar


# Type alias: a "Tensor" is any of the legacy proxy objects.  Callers can use
# this annotation without worrying about which specific backing allocator
# a given tensor lives in.
Tensor = TensorVar | InputVar | VRAMMatrixVar | FPVar


def tensor_kind(tensor: Tensor) -> TensorKind:
    """Return the backing storage of a tensor proxy."""
    if isinstance(tensor, FPVar):
        return TensorKind.FPRAM
    if isinstance(tensor, VRAMMatrixVar):
        return TensorKind.VRAM
    if isinstance(tensor, InputVar):
        return TensorKind.HBM
    if isinstance(tensor, TensorVar):
        # Generic TensorVar without a specific backing — classify by ``kind``.
        kind = getattr(tensor, "kind", "")
        if kind in ("vram_matrix", "batch", "matrix"):
            return TensorKind.VRAM
        if kind == "input":
            return TensorKind.HBM
    raise TypeError(f"Unknown tensor type: {type(tensor).__name__}")


# =============================================================================
# Unified dataclass aliases for the three overlapping "Info" and three
# overlapping "Layout" types in tile_compiler.py.
# =============================================================================


# ``TensorInfo`` is the union of the three Info dataclasses.  Callers can
# import ``TensorInfo`` and use it as an annotation; at runtime the object
# will be whichever specific Info subtype ``TileCompiler`` constructed.
TensorInfo = MemoryObjectInfo | SubMatrixInfo | VRAMSubMatrixInfo

# ``TileLayout`` is the union of the three Layout dataclasses.
TileLayout = MatrixBlockLayout | VRAMMatrixBlockLayout | FPRAMObjectLayout


# =============================================================================
# Public exports.
# =============================================================================


__all__ = [
    "DeveloperCompiler",
    "FPRAMAllocator",
    "FPRAMObjectLayout",
    "FPVar",
    "InputVar",
    "MRAMAllocator",
    "MatrixBlockLayout",
    "MemoryBlock",
    "MemoryObjectInfo",
    "PlenaCompiler",
    "RegisterAllocator",
    "SubMatrixInfo",
    "Tensor",
    "TensorInfo",
    "TensorKind",
    "TensorVar",
    "TileCompiler",
    "TileLayout",
    "VRAMAllocator",
    "VRAMMatrixBlockLayout",
    "VRAMMatrixVar",
    "VRAMSubMatrixInfo",
    "VirtualMemoryManager",
    "tensor_kind",
]
