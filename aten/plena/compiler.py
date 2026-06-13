"""User-facing PLENA compiler program builder."""

from __future__ import annotations

import os
from pathlib import Path

from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.program_attention import ProgramAttentionMixin
from compiler.aten.plena.program_fp_tile_ops import ProgramFPTileOpsMixin
from compiler.aten.plena.program_matrix_ops import ProgramMatrixOpsMixin
from compiler.aten.plena.program_tensors import ProgramTensorMixin
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar
from compiler.utils.load_config import load_toml_config


def _find_plena_settings_toml() -> Path | None:
    env_path = os.environ.get("PLENA_SETTINGS_TOML")
    if env_path:
        return Path(env_path)

    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for base in candidates:
        path = base / "plena_settings.toml"
        if path.exists():
            return path
    return None


def _behavior_config_value(key: str, default: int) -> int:
    settings_path = _find_plena_settings_toml()
    if settings_path is None or not settings_path.exists():
        return default

    try:
        config = load_toml_config(settings_path, "CONFIG", mode="BEHAVIOR")
    except Exception:
        return default

    value = config.get(key, {})
    if isinstance(value, dict):
        value = value.get("value", default)
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


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

    def __init__(
        self,
        mlen: int = 64,
        blen: int = 4,
        real_data_ratio: float = 1.125,
        unroll_loops: bool = False,
        mram_tile_capacity: int = 4,
        hbm_v_prefetch_amount: int | None = None,
        hbm_v_writeback_amount: int | None = None,
    ):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM storage ratio (MXFP8 format = 1.125)
            mram_tile_capacity: Number of mlen x mlen tiles that fit in MRAM.
            hbm_v_prefetch_amount: H_PREFETCH_V transfer count. Defaults to
                          BEHAVIOR.CONFIG.HBM_V_Prefetch_Amount in
                          PLENA_SETTINGS_TOML / plena_settings.toml.
            hbm_v_writeback_amount: H_STORE_V transfer count. Defaults to
                          BEHAVIOR.CONFIG.HBM_V_Writeback_Amount in
                          PLENA_SETTINGS_TOML / plena_settings.toml.
            unroll_loops: If True, unroll sub-projection and attention helper loops
                          at ASM-gen time to eliminate C_LOOP_START/END overhead.
                          Overridden by the ATEN_OPS_UNROLL env var ("1"=True, "0"=False).
        """
        _env_unroll = os.environ.get("ATEN_OPS_UNROLL", "")
        if _env_unroll == "1":
            unroll_loops = True
        elif _env_unroll == "0":
            unroll_loops = False
        super().__init__(
            mlen=mlen,
            blen=blen,
            real_data_ratio=real_data_ratio,
            unroll_loops=unroll_loops,
            mram_tile_capacity=mram_tile_capacity,
        )
        if hbm_v_prefetch_amount is None:
            hbm_v_prefetch_amount = _behavior_config_value("HBM_V_Prefetch_Amount", 4)
        if hbm_v_writeback_amount is None:
            hbm_v_writeback_amount = _behavior_config_value("HBM_V_Writeback_Amount", 4)
        if hbm_v_prefetch_amount <= 0:
            raise ValueError(f"hbm_v_prefetch_amount must be > 0, got {hbm_v_prefetch_amount}")
        if hbm_v_writeback_amount <= 0:
            raise ValueError(f"hbm_v_writeback_amount must be > 0, got {hbm_v_writeback_amount}")
        self.hbm_v_prefetch_amount = hbm_v_prefetch_amount
        self.hbm_v_writeback_amount = hbm_v_writeback_amount
        self.hlen = _behavior_config_value("HLEN", mlen)
        self.broadcast_amount = _behavior_config_value("BROADCAST_AMOUNT", max(1, mlen // max(1, self.hlen)))

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
        """Allocate HBM range, preferring previously freed blocks.

        Large allocations (>= mlen*mlen) are aligned to mlen*mlen because the
        Rust emulator's continous_write_delayed requires it (src/main.rs:155).
        Small allocations only need mlen alignment, preserving sliced-test layout.
        """
        m = self.mlen
        tile_bytes = m * m
        # Only pad to mlen*mlen at large tile sizes where the Rust emulator's
        # continous_write_delayed (main.rs:155) requires tile-index alignment.
        # At MLEN=64/128 the HBM layout must match create_mem_for_sim's
        # sequential write order, which does not insert gaps.
        needs_tile_align = m >= 256

        best_idx = None
        best_waste = None
        for i, (addr, size) in enumerate(self._hbm_free_blocks):
            aligned_addr = ((addr + tile_bytes - 1) // tile_bytes) * tile_bytes if needs_tile_align else addr
            aligned_waste = aligned_addr - addr
            effective_size = size - aligned_waste
            if effective_size >= hbm_size:
                waste = effective_size - hbm_size
                if best_waste is None or waste < best_waste:
                    best_idx = i
                    best_waste = waste

        if best_idx is not None:
            addr, block_size = self._hbm_free_blocks.pop(best_idx)
            if needs_tile_align:
                aligned_addr = ((addr + tile_bytes - 1) // tile_bytes) * tile_bytes
                if aligned_addr > addr:
                    self._hbm_free_blocks.append((addr, aligned_addr - addr))
            else:
                aligned_addr = addr
            excess = block_size - (aligned_addr - addr) - hbm_size
            if excess > 0:
                self._hbm_free_blocks.append((aligned_addr + hbm_size, excess))
            return aligned_addr

        addr = self._next_hbm_addr
        if needs_tile_align:
            addr = ((addr + tile_bytes - 1) // tile_bytes) * tile_bytes
        self._next_hbm_addr = ((addr + hbm_size + m - 1) // m) * m
        if needs_tile_align:
            self._next_hbm_addr = ((self._next_hbm_addr + tile_bytes - 1) // tile_bytes) * tile_bytes
        return addr

    def _recycle_hbm(self, hbm_addr: int, hbm_size: int):
        """Recycle an HBM range for future auto-allocation."""
        if hbm_size <= 0:
            return
        self._hbm_free_blocks.append((hbm_addr, hbm_size))


__all__ = ["PlenaCompiler"]
