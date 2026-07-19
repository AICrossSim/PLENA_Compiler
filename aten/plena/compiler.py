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
        # Most simulator-generated settings files only carry a TRANSACTIONAL
        # section.  Falling straight back to a hard-coded default here can make
        # compiler address increments disagree with the emulator's DMA amount:
        # e.g. codegen advances four rows while H_PREFETCH_V writes eight.  Use
        # TRANSACTIONAL as the authoritative fallback so both sides consume the
        # same transfer contract.
        if key not in config:
            config = load_toml_config(settings_path, "CONFIG", mode="TRANSACTIONAL")
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
        emission_mode: str = "asm",
        cost_strict_raw: bool = False,
        packed_attention_schedule: str = "direct-first-block-v1",
        vector_scalar_schedule: str = "compiler-v1",
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
            emission_mode=emission_mode,
            cost_strict_raw=cost_strict_raw,
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
        if packed_attention_schedule not in {"direct-first-block-v1", "legacy"}:
            raise ValueError(
                "packed_attention_schedule must be 'direct-first-block-v1' or "
                f"'legacy', got {packed_attention_schedule!r}"
            )
        self.packed_attention_schedule = packed_attention_schedule
        if vector_scalar_schedule not in {"compiler-v1", "legacy"}:
            raise ValueError(
                "vector_scalar_schedule must be 'compiler-v1' or 'legacy', got "
                f"{vector_scalar_schedule!r}"
            )
        self.vector_scalar_schedule = vector_scalar_schedule
        self._vector_scalar_stats: dict[str, int] = {
            "segmented_norm_square_ops_elided": 0,
            "segmented_norm_copy_ops_elided": 0,
            "segmented_norm_constant_loads_elided": 0,
            "inactive_norm_rows_elided": 0,
            "redundant_valid_masks_elided": 0,
            "valid_mask_build_count": 0,
            "rms_norm_address_loads_elided": 0,
            "rms_norm_nops_elided": 0,
        }
        self._packed_attention_stats: dict[str, int] = {
            "softmax_first_block_specialized_count": 0,
            "softmax_first_block_specialized_rows": 0,
            "softmax_state_initializations_elided": 0,
            "softmax_state_initialization_rows_elided": 0,
            "temporary_o_matrices_elided": 0,
            "direct_o_lane_updates": 0,
            "qk_compute_count": 0,
            "ideal_qk_compute_count": 0,
            "pv_compute_count": 0,
            "kv_tile_load_count": 0,
            "ideal_kv_tile_load_count": 0,
        }

        # HBM address auto-allocation
        self._next_hbm_addr: int = 0
        self._hbm_free_blocks: list[tuple[int, int]] = []  # (addr, size)

        # Variable registries
        self._inputs: dict[str, InputVar] = {}
        self._tensors: dict[str, TensorVar] = {}
        self._fp_vars: dict[str, FPVar] = {}
        self._registered_hbm_sub_matrices: dict[str, bool] = {}
        self._registered_vram_sub_matrices: dict[str, bool] = {}

    def packed_attention_stats(self) -> dict[str, int | float | str]:
        """Return compiler-observed packed-attention work and reuse factors."""

        stats: dict[str, int | float | str] = {
            "packed_attention_schedule": self.packed_attention_schedule,
            **self._packed_attention_stats,
        }
        qk_ideal = self._packed_attention_stats["ideal_qk_compute_count"]
        kv_ideal = self._packed_attention_stats["ideal_kv_tile_load_count"]
        stats["qk_recompute_factor"] = (
            self._packed_attention_stats["qk_compute_count"] / qk_ideal
            if qk_ideal
            else 0.0
        )
        stats["kv_reload_factor"] = (
            self._packed_attention_stats["kv_tile_load_count"] / kv_ideal
            if kv_ideal
            else 0.0
        )
        return stats

    def record_vector_scalar_stats(self, values: dict[str, int]) -> None:
        """Accumulate metadata emitted by shared normalization/mask plans."""

        for key, value in values.items():
            if key not in self._vector_scalar_stats:
                self._vector_scalar_stats[key] = 0
            self._vector_scalar_stats[key] += int(value)

    def vector_scalar_stats(self) -> dict[str, int | str]:
        return {
            "vector_scalar_schedule": self.vector_scalar_schedule,
            "valid_mask_scope": "program" if self._vector_scalar_stats["valid_mask_build_count"] else "none",
            **self._vector_scalar_stats,
        }

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get generated ISA code string."""
        return super().get_code()

    def compile_cost_trace(self):
        """Return the algebraic trace collected in cost/both emission mode."""
        return super().get_cost_trace()

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
