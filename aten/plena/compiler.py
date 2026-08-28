"""User-facing PLENA compiler program builder."""

from __future__ import annotations

import os
from pathlib import Path

from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.program_attention import ProgramAttentionMixin
from compiler.aten.plena.program_fp_tile_ops import ProgramFPTileOpsMixin
from compiler.aten.plena.program_moe_shared import ProgramMoeSharedMixin
from compiler.aten.plena.program_routed_moe import ProgramRoutedMoeMixin
from compiler.aten.plena.program_mamba_common import ProgramMambaCommonMixin
from compiler.aten.plena.program_matrix_ops import ProgramMatrixOpsMixin
from compiler.aten.plena.program_ssd import ProgramSSDMixin
from compiler.aten.plena.program_ssm_recurrent import ProgramSSMRecurrentMixin
from compiler.aten.plena.program_kda_common import ProgramKdaCommonMixin
from compiler.aten.plena.program_kda_chunk import ProgramKdaChunkMixin
from compiler.aten.plena.program_kda_conv import ProgramKdaConvMixin
from compiler.aten.plena.program_kda_gates import ProgramKdaGatesMixin
from compiler.aten.plena.program_kda_layer import ProgramKdaLayerMixin
from compiler.aten.plena.program_kda_prefill import ProgramKdaPrefillMixin
from compiler.aten.plena.program_kda_mixer import ProgramKdaMixerMixin
from compiler.aten.plena.program_kda_recurrent import ProgramKdaRecurrentMixin
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


#: Config sections `plena_settings.toml` actually ships, in the order they are
#: searched after `BEHAVIOR`.
_MACHINE_MODES = ("TRANSACTIONAL", "ANALYTIC")


def _config_section(mode: str) -> dict:
    settings_path = _find_plena_settings_toml()
    if settings_path is None or not settings_path.exists():
        return {}
    try:
        return load_toml_config(settings_path, "CONFIG", mode=mode) or {}
    except Exception:
        return {}


def _section_value(section: dict, key: str) -> int | None:
    value = section.get(key)
    if isinstance(value, dict):
        value = value.get("value")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _config_mode_for_mlen(mlen: int) -> str | None:
    """The shipped section whose ``MLEN`` is the one being compiled for.

    ``plena_settings.toml`` has no ``[BEHAVIOR]`` table -- it ships ``[MODE]``,
    ``[ANALYTIC.*]`` and ``[TRANSACTIONAL.*]`` -- so a reader that asks for
    ``BEHAVIOR`` finds nothing and every value silently falls back to whatever
    default the caller passed. That is what this function replaces.

    Matching on ``MLEN`` rather than on ``[MODE].active`` is deliberate.
    ``active`` selects which machine the *simulator* models; it does not say
    what the compiler is emitting for, and the two are routinely different --
    the shipped file has ``active = "analytic"`` (MLEN 2048) while every
    program on this branch is compiled at ``mlen`` 64 and executed by the
    transactional emulator, whose section declares MLEN 64. Reading ``active``
    would hand a 64-wide compilation the 2048-wide machine's numbers.

    Returns ``None`` when no section matches, which is the case for the test
    shapes (``mlen`` 8) and means "keep the caller's default".
    """
    for mode in _MACHINE_MODES:
        if _section_value(_config_section(mode), "MLEN") == mlen:
            return mode
    return None


def _behavior_config_value(key: str, default: int, mlen: int | None = None) -> int:
    """``key`` from ``[BEHAVIOR]``, else from the section matching ``mlen``.

    ``BEHAVIOR`` is searched first so that a file which does define it keeps
    overriding the machine sections; nothing ships one today.
    """
    parsed = _section_value(_config_section("BEHAVIOR"), key)
    if parsed is not None:
        return parsed
    if mlen is not None:
        mode = _config_mode_for_mlen(mlen)
        if mode is not None:
            parsed = _section_value(_config_section(mode), key)
            if parsed is not None:
                return parsed
    return default


def _derive_mram_tile_capacity(mlen: int) -> int | None:
    """``MATRIX_SRAM_SIZE // mlen`` for the machine ``mlen`` names.

    The emulator's ``MatrixSram::new(tile_size = MLEN, depth =
    MATRIX_SRAM_SIZE)`` keeps ``depth / tile_size`` tiles, so this is the same
    arithmetic on the same two numbers. Returns ``None`` when no section
    matches ``mlen``, and raises when one does but cannot hold a single tile --
    a compiler that split the contraction into zero-tile chunks would emit an
    infinite program, so there is no sensible fallback.
    """
    mode = _config_mode_for_mlen(mlen)
    if mode is None:
        return None
    size = _section_value(_config_section(mode), "MATRIX_SRAM_SIZE")
    if size is None:
        return None
    tiles = size // mlen
    if tiles < 1:
        raise ValueError(
            f"[{mode}] MATRIX_SRAM_SIZE {size} holds {size / mlen:.3f} tiles of "
            f"mlen {mlen}; a projection cannot be split into fewer than one "
            f"tile per chunk. Pass mram_tile_capacity explicitly to override, "
            f"or fix the configuration."
        )
    return tiles


# ============================================================================
# PlenaCompiler Main Class
# ============================================================================


class PlenaCompiler(
    ProgramTensorMixin,
    ProgramFPTileOpsMixin,
    ProgramMatrixOpsMixin,
    ProgramRoutedMoeMixin,
    # After ProgramRoutedMoeMixin: the shared-expert emitters call into the routed
    # substrate (moe_expert_activation_v0, moe_materialize_route_weights_*).
    ProgramMoeSharedMixin,
    ProgramAttentionMixin,
    # After ProgramMatrixOpsMixin and ProgramFPTileOpsMixin: the Mamba emitters
    # are built on vram_add/vram_mul/vram_fill_zero and the tile_row_* family.
    # ProgramSSDMixin and ProgramSSMRecurrentMixin must come after
    # ProgramMambaCommonMixin -- both call its mamba_row_* / mamba_block_copy
    # primitives and its stage-marker vocabulary.
    ProgramMambaCommonMixin,
    ProgramSSDMixin,
    ProgramSSMRecurrentMixin,
    # After ProgramSSMRecurrentMixin and ProgramMambaCommonMixin: the KDA
    # emitters reuse pin_hbm_region for state residency and mamba_row_copy /
    # mamba_block_copy / mamba_rsqrt_fpram for the row primitives. KDA is a
    # second consumer of that substrate, not a second copy of it.
    ProgramKdaCommonMixin,
    # After ProgramKdaCommonMixin: the recurrence uses its layout helpers and
    # stage vocabulary.
    ProgramKdaRecurrentMixin,
    ProgramKdaConvMixin,
    ProgramKdaGatesMixin,
    # After ProgramKdaGatesMixin: the chunk primitives use its key-block helpers.
    ProgramKdaChunkMixin,
    # Last of the chunk path: it calls the chunk primitives and the matrix ops.
    ProgramKdaPrefillMixin,
    # Assembles the layer around the mixer; needs the projections.
    ProgramKdaLayerMixin,
    # Last of the KDA mixins: it calls all the others.
    ProgramKdaMixerMixin,
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
        mram_tile_capacity: int | None = None,
        hbm_v_prefetch_amount: int | None = None,
        hbm_v_writeback_amount: int | None = None,
    ):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM storage ratio (MXFP8 format = 1.125)
            mram_tile_capacity: Number of mlen x mlen tiles that fit in MRAM.
                          Defaults to MATRIX_SRAM_SIZE // mlen from the config
                          section whose MLEN is `mlen` -- the same arithmetic
                          the emulator's MatrixSram does -- and to 4 when no
                          section matches. It was an unconditional 4 until
                          2026-08-28, which is 16x below the transactional
                          machine's 64 and cost Kimi K3's input projection a
                          factor of 2.79; see
                          `test_the_mram_tile_default_costs_the_projection_a_factor_of_three`.
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
        if mram_tile_capacity is None:
            mram_tile_capacity = _derive_mram_tile_capacity(mlen) or 4
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
            hbm_v_prefetch_amount = _behavior_config_value("HBM_V_Prefetch_Amount", 4, mlen)
        if hbm_v_writeback_amount is None:
            hbm_v_writeback_amount = _behavior_config_value("HBM_V_Writeback_Amount", 4, mlen)
        if hbm_v_prefetch_amount <= 0:
            raise ValueError(f"hbm_v_prefetch_amount must be > 0, got {hbm_v_prefetch_amount}")
        if hbm_v_writeback_amount <= 0:
            raise ValueError(f"hbm_v_writeback_amount must be > 0, got {hbm_v_writeback_amount}")
        self.hbm_v_prefetch_amount = hbm_v_prefetch_amount
        self.hbm_v_writeback_amount = hbm_v_writeback_amount
        self.hlen = _behavior_config_value("HLEN", mlen, mlen)
        self.broadcast_amount = _behavior_config_value(
            "BROADCAST_AMOUNT", max(1, mlen // max(1, self.hlen)), mlen
        )

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
