"""User-facing PLENA compiler program builder."""

from __future__ import annotations

import os

from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.program_attention import ProgramAttentionMixin
from compiler.aten.plena.program_fp_tile_ops import ProgramFPTileOpsMixin
from compiler.aten.plena.program_matrix_ops import ProgramMatrixOpsMixin
from compiler.aten.plena.program_tensors import ProgramTensorMixin
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar


# ============================================================================
# PlenaCompiler Main Class
# ============================================================================


class PlenaCompiler(
    ProgramTensorMixin,
    ProgramFPTileOpsMixin,
    ProgramMatrixOpsMixin,
    ProgramAttentionMixin,
    IsaCompiler,
):
    """
    PLENA High-level Compiler Interface.

    Inherits the ISA-emission machinery from IsaCompiler and layers typed
    program-builder helpers on top. Operations eagerly emit ISA text.
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
        self._registered_hbm_sub_matrices: dict[str, bool] = {}
        self._registered_vram_sub_matrices: dict[str, bool] = {}

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get generated ISA code string."""
        return super().get_code()

    @property
    def _compiler(self) -> PlenaCompiler:
        """Compatibility alias for simulator testbench callers."""
        return self

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _scoped_name(self, name: str) -> str:
        return name

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


__all__ = ["PlenaCompiler"]
