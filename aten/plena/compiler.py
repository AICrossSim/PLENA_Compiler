"""User-facing PLENA compiler program builder."""

from __future__ import annotations

import os
from pathlib import Path

from compiler.aten.execution_trace import (
    CompilationArtifact,
    TensorTraceMetadata,
    build_execution_trace,
    build_request_memory_trace,
)
from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.program_attention import ProgramAttentionMixin
from compiler.aten.plena.program_fp_tile_ops import ProgramFPTileOpsMixin
from compiler.aten.plena.program_matrix_ops import ProgramMatrixOpsMixin
from compiler.aten.plena.program_tensors import ProgramTensorMixin
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar
from compiler.asm_templates._imm import legalize_immediates
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
        hbm_row_width: int = 256,
        hbm_element_width: int = 8,
        hbm_block_size: int = 8,
        hbm_scale_width: int = 8,
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
            hbm_row_width=hbm_row_width,
            hbm_element_width=hbm_element_width,
            hbm_block_size=hbm_block_size,
            hbm_scale_width=hbm_scale_width,
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
        self._trace_hbm_tensors: dict[str, TensorTraceMetadata] = {}

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get the generated ISA, with over-wide immediates legalised."""
        return legalize_immediates(super().get_code())

    def compile_with_trace(self) -> CompilationArtifact:
        """Return ISA with algebraic execution and exact request-memory traces."""

        assembly = self.compile()

        def trace_metadata(layout) -> TensorTraceMetadata:
            return TensorTraceMetadata(
                name=layout.name,
                hbm_address=layout.hbm_base_addr,
                precision_mode=layout.precision_role,
                element_bits=layout.hbm_element_width,
                block_size=layout.hbm_block_size,
                scale_bits=layout.hbm_scale_width,
                physical_shape=tuple(layout.physical_shape),
                element_plane_bytes=layout.element_plane_bytes,
                hbm_size=layout.hbm_size,
            )

        active_tensors = {
            layout.name: trace_metadata(layout)
            for layout in self.hbm_matrices.values()
        }
        tensors = tuple(
            (self._trace_hbm_tensors | active_tensors).values()
        )
        trace = build_execution_trace(
            assembly,
            mlen=self.mlen,
            blen=self.blen,
            vlen=self.mlen,
            hlen=self.hlen,
            vector_prefetch_amount=self.hbm_v_prefetch_amount,
            vector_store_amount=self.hbm_v_writeback_amount,
            default_element_bits=self.hbm_element_width,
            default_block_size=self.hbm_block_size,
            default_scale_bits=self.hbm_scale_width,
            tensors=tensors,
        )
        request_memory = build_request_memory_trace(
            assembly,
            trace,
            vector_prefetch_amount=self.hbm_v_prefetch_amount,
            vector_store_amount=self.hbm_v_writeback_amount,
            tensors=tensors,
        )
        return CompilationArtifact(
            assembly=assembly,
            execution_trace=trace,
            request_memory=request_memory,
        )

    def _remember_trace_tensor(self, name: str) -> None:
        """Retain HBM metadata even when allocation state is later recycled."""

        if not hasattr(self, "_trace_hbm_tensors"):
            return
        layout = self.hbm_matrices[name]
        self._trace_hbm_tensors[name] = TensorTraceMetadata(
            name=layout.name,
            hbm_address=layout.hbm_base_addr,
            precision_mode=layout.precision_role,
            element_bits=layout.hbm_element_width,
            block_size=layout.hbm_block_size,
            scale_bits=layout.hbm_scale_width,
            physical_shape=tuple(layout.physical_shape),
            element_plane_bytes=layout.element_plane_bytes,
            hbm_size=layout.hbm_size,
        )

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
        """Allocate an HBM range, preferring previously freed blocks.

        The bump cursor advances by exactly the row-aligned footprint
        `hbm_tensor_size` reports, because that is what the stager writes: it
        lays tensors down back to back as `[elements][scales]` pairs, each plane
        padded only to an HBM row. Any additional padding here puts every
        following tensor at an address the stager never wrote to, and the
        weights come back as whatever the neighbouring tensor's bytes decode to.
        """
        best_idx = None
        best_waste = None
        for i, (_addr, size) in enumerate(self._hbm_free_blocks):
            if size >= hbm_size:
                waste = size - hbm_size
                if best_waste is None or waste < best_waste:
                    best_idx = i
                    best_waste = waste

        if best_idx is not None:
            addr, block_size = self._hbm_free_blocks.pop(best_idx)
            excess = block_size - hbm_size
            if excess > 0:
                self._hbm_free_blocks.append((addr + hbm_size, excess))
            return addr

        addr = self._next_hbm_addr
        self._next_hbm_addr = addr + hbm_size
        return addr

    def _recycle_hbm(self, hbm_addr: int, hbm_size: int):
        """Recycle an HBM range for future auto-allocation."""
        if hbm_size <= 0:
            return
        self._hbm_free_blocks.append((hbm_addr, hbm_size))


__all__ = ["CompilationArtifact", "PlenaCompiler"]
