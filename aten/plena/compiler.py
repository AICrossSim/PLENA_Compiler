"""User-facing PLENA compiler program builder."""

from __future__ import annotations

import os
from pathlib import Path

from compiler.aten.plena.isa_compiler import IsaCompiler
from compiler.aten.plena.attention_pipeline_plan import GQATimingProfile
from compiler.aten.plena.program_attention import ProgramAttentionMixin
from compiler.aten.plena.program_fp_tile_ops import ProgramFPTileOpsMixin
from compiler.aten.plena.program_matrix_ops import ProgramMatrixOpsMixin
from compiler.aten.plena.program_tensors import ProgramTensorMixin
from compiler.aten.plena.vars import FPVar, InputVar, TensorVar
from compiler.aten.agu import (
    AGU_MODE_LEGACY,
    AGU_MODE_LOOP_V1,
    optimize_agu_assembly,
)
from compiler.aten.plena.native_layout import (
    COMPACT_STATS_LANE_POLICY_AUTO_V1,
    COMPACT_STATS_LANE_POLICY_FIXED_16_V1,
    COMPACT_STATS_LANE_TIERS,
    FP_CONSTANT_NUM_DEFAULT,
    PACKED_QK_SCHEDULES,
    PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
    SOFTMAX_STATE_SCHEDULES,
    SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
    build_softmax_state_layout,
)
from compiler.aten.plena.kv_residency import MATRIX_SRAM_POLICIES
from compiler.aten.moe import (
    MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2,
    MOE_LOWERING_SCHEDULES,
)
from compiler.asm_templates.ffn_address_plan import (
    FFN_ADDRESS_SCHEDULES,
    FFN_ADDRESS_SCHEDULE_LEGACY,
    FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
)
from compiler.asm_templates.ffn_projection_plan import (
    FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1,
    FFN_PROJECTION_SCHEDULES,
)
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
        hbm_m_prefetch_amount: int | None = None,
        hbm_v_prefetch_amount: int | None = None,
        hbm_v_writeback_amount: int | None = None,
        emission_mode: str = "asm",
        cost_strict_raw: bool = False,
        cost_trace_granularity: str = "detailed",
        cost_address_generation_mode: str = "legacy",
        packed_attention_schedule: str = "direct-first-block-v1",
        softmax_state_schedule: str = SOFTMAX_STATE_SCHEDULE_STREAMED_V2,
        packed_qk_schedule: str = PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1,
        vector_scalar_schedule: str = "rtl-v5",
        compact_stats_lanes: int | None = None,
        selector_schedule: str = "legacy",
        reduction_output_mode: str = "accumulate-v1",
        gqa_pipeline_schedule: str | None = None,
        gqa_timing_calibration: str | Path | None = None,
        address_generation_mode: str = AGU_MODE_LOOP_V1,
        ffn_address_schedule: str = FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
        ffn_projection_schedule: str = FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
        fp_sram_depth: int | None = None,
        fp_constant_num: int = FP_CONSTANT_NUM_DEFAULT,
        kv_residency_policy: str = "raw-tiles",
        moe_lowering_schedule: str = MOE_LOWERING_SCHEDULE_COMPACT_ROUTE_V2,
    ):
        """
        Args:
            mlen: Matrix tile size (default 64)
            blen: Vector tile size (default 4)
            real_data_ratio: HBM storage ratio (MXFP8 format = 1.125)
            mram_tile_capacity: Number of mlen x mlen tiles that fit in MRAM.
            hbm_m_prefetch_amount: H_PREFETCH_M transfer count. Defaults to
                          BEHAVIOR.CONFIG.HBM_M_Prefetch_Amount in
                          PLENA_SETTINGS_TOML / plena_settings.toml.
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
            cost_trace_granularity=cost_trace_granularity,
            cost_address_generation_mode=cost_address_generation_mode,
        )
        if hbm_m_prefetch_amount is None:
            hbm_m_prefetch_amount = _behavior_config_value("HBM_M_Prefetch_Amount", mlen)
        if hbm_v_prefetch_amount is None:
            hbm_v_prefetch_amount = _behavior_config_value("HBM_V_Prefetch_Amount", 4)
        if hbm_v_writeback_amount is None:
            hbm_v_writeback_amount = _behavior_config_value("HBM_V_Writeback_Amount", 4)
        if hbm_m_prefetch_amount <= 0:
            raise ValueError(f"hbm_m_prefetch_amount must be > 0, got {hbm_m_prefetch_amount}")
        if hbm_v_prefetch_amount <= 0:
            raise ValueError(f"hbm_v_prefetch_amount must be > 0, got {hbm_v_prefetch_amount}")
        if hbm_v_writeback_amount <= 0:
            raise ValueError(f"hbm_v_writeback_amount must be > 0, got {hbm_v_writeback_amount}")
        self.hbm_m_prefetch_amount = hbm_m_prefetch_amount
        self.hbm_v_prefetch_amount = hbm_v_prefetch_amount
        self.hbm_v_writeback_amount = hbm_v_writeback_amount
        self.hlen = _behavior_config_value("HLEN", mlen)
        self.broadcast_amount = _behavior_config_value("BROADCAST_AMOUNT", max(1, mlen // max(1, self.hlen)))
        if kv_residency_policy not in {"raw-tiles", *MATRIX_SRAM_POLICIES}:
            raise ValueError(
                f"kv_residency_policy must be 'raw-tiles' or one of {MATRIX_SRAM_POLICIES}, got {kv_residency_policy!r}"
            )
        self.kv_residency_policy = kv_residency_policy
        if moe_lowering_schedule not in MOE_LOWERING_SCHEDULES:
            raise ValueError(
                f"moe_lowering_schedule must be one of {sorted(MOE_LOWERING_SCHEDULES)}, got {moe_lowering_schedule!r}"
            )
        self.moe_lowering_schedule = moe_lowering_schedule
        if packed_attention_schedule not in {"direct-first-block-v1", "legacy"}:
            raise ValueError(
                "packed_attention_schedule must be 'direct-first-block-v1' or "
                f"'legacy', got {packed_attention_schedule!r}"
            )
        self.packed_attention_schedule = packed_attention_schedule
        if softmax_state_schedule not in SOFTMAX_STATE_SCHEDULES:
            raise ValueError(
                "softmax_state_schedule must be one of "
                f"{sorted(SOFTMAX_STATE_SCHEDULES)}, got "
                f"{softmax_state_schedule!r}"
            )
        if packed_qk_schedule not in PACKED_QK_SCHEDULES:
            raise ValueError(
                f"packed_qk_schedule must be one of {sorted(PACKED_QK_SCHEDULES)}, got {packed_qk_schedule!r}"
            )
        self.softmax_state_schedule = softmax_state_schedule
        self.packed_qk_schedule = packed_qk_schedule
        self.fp_constant_num = int(fp_constant_num)
        self.fp_sram_depth = (
            int(fp_sram_depth) if fp_sram_depth is not None else _behavior_config_value("FP_SRAM_DEPTH", 0)
        )
        if vector_scalar_schedule not in {
            "rtl-v5",
            "rtl-v4",
            "rtl-v3",
            "rtl-v2",
            "compiler-v1",
            "legacy",
        }:
            raise ValueError(
                "vector_scalar_schedule must be 'rtl-v5', 'rtl-v4', 'rtl-v3', "
                "'rtl-v2', 'compiler-v1', or 'legacy', got "
                f"{vector_scalar_schedule!r}"
            )
        self.vector_scalar_schedule = vector_scalar_schedule
        if compact_stats_lanes is None:
            compact_stats_lanes = (
                next(
                    tier
                    for tier in COMPACT_STATS_LANE_TIERS
                    if tier >= min(64, max(1, mlen // max(1, self.hlen)))
                )
                if vector_scalar_schedule == "rtl-v5"
                else 16
            )
        if compact_stats_lanes not in COMPACT_STATS_LANE_TIERS:
            raise ValueError(
                "compact_stats_lanes must be one of "
                f"{COMPACT_STATS_LANE_TIERS}, got {compact_stats_lanes}"
            )
        if vector_scalar_schedule != "rtl-v5" and compact_stats_lanes != 16:
            raise ValueError(
                "non-rtl-v5 schedules require fixed compact_stats_lanes=16"
            )
        self.compact_stats_lanes = int(compact_stats_lanes)
        self.compact_stats_lane_policy = (
            COMPACT_STATS_LANE_POLICY_AUTO_V1
            if vector_scalar_schedule == "rtl-v5"
            else COMPACT_STATS_LANE_POLICY_FIXED_16_V1
        )
        if selector_schedule not in {"hoisted-v1", "legacy"}:
            raise ValueError(f"selector_schedule must be 'hoisted-v1' or 'legacy', got {selector_schedule!r}")
        if reduction_output_mode not in {"overwrite-v1", "accumulate-v1"}:
            raise ValueError(
                f"reduction_output_mode must be 'overwrite-v1' or 'accumulate-v1', got {reduction_output_mode!r}"
            )
        self.selector_schedule = selector_schedule
        self.reduction_output_mode = reduction_output_mode
        if gqa_pipeline_schedule is None:
            # Keep the low-level builder backward compatible. Native decoder
            # frontends select row-interleaved-v1 explicitly when rtl-v3 is in
            # use; direct unit-test/program-builder callers remain row-serial.
            gqa_pipeline_schedule = "row-serial"
        if gqa_pipeline_schedule not in {"row-interleaved-v1", "row-serial"}:
            raise ValueError(
                f"gqa_pipeline_schedule must be 'row-interleaved-v1' or 'row-serial', got {gqa_pipeline_schedule!r}"
            )
        if gqa_pipeline_schedule == "row-interleaved-v1" and vector_scalar_schedule not in {
            "rtl-v3",
            "rtl-v4",
            "rtl-v5",
        }:
            raise ValueError(
                "gqa_pipeline_schedule='row-interleaved-v1' requires "
                "vector_scalar_schedule='rtl-v3', 'rtl-v4', or 'rtl-v5'"
            )
        self.gqa_pipeline_schedule = gqa_pipeline_schedule
        if address_generation_mode not in {
            AGU_MODE_LEGACY,
            AGU_MODE_LOOP_V1,
        }:
            raise ValueError(
                "address_generation_mode must be 'loop-agu-v1' or 'legacy', got "
                f"{address_generation_mode!r}"
            )
        self.address_generation_mode = address_generation_mode
        if ffn_address_schedule not in FFN_ADDRESS_SCHEDULES:
            raise ValueError(
                f"ffn_address_schedule must be one of {FFN_ADDRESS_SCHEDULES}, got {ffn_address_schedule!r}"
            )
        self.ffn_address_schedule = ffn_address_schedule
        if ffn_projection_schedule not in FFN_PROJECTION_SCHEDULES:
            raise ValueError(
                f"ffn_projection_schedule must be one of {FFN_PROJECTION_SCHEDULES}, got {ffn_projection_schedule!r}"
            )
        # Historical callers commonly select only the legacy address mode.
        # Preserve that complete compatibility path instead of creating the
        # unsupported affine-loop/legacy-address hybrid.
        if (
            ffn_address_schedule == FFN_ADDRESS_SCHEDULE_LEGACY
            and ffn_projection_schedule == FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2
        ):
            ffn_projection_schedule = FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1
        self.ffn_projection_schedule = ffn_projection_schedule
        self._ffn_address_stats: dict[str, int] = {
            "ffn_dead_k_pointer_updates_elided": 0,
            "ffn_dead_prefetch_updates_elided": 0,
            "ffn_dead_output_updates_elided": 0,
            "ffn_invariant_stride_loads": 0,
            "ffn_large_immediate_chunks_avoided": 0,
            "ffn_residual_address_opcodes": 0,
            "ffn_address_cycles_before": 0,
            "ffn_address_cycles_after": 0,
            "ffn_schedule_fallback_count": 0,
        }
        self._ffn_projection_metadata: dict[str, object] = {
            "ffn_loop_plan_version": None,
            "ffn_explicit_loop_depth": 0,
            "ffn_agu_streams_by_axis": {},
            "ffn_schedule_guard_status": "not_evaluated",
            "ffn_schedule_fallback_reason": None,
            "ffn_legacy_template_bypassed": False,
        }
        self._agu_metadata: dict[str, object] = {}
        self.gqa_timing_calibration = gqa_timing_calibration
        self.gqa_timing_profile = (
            GQATimingProfile.load(gqa_timing_calibration) if gqa_pipeline_schedule == "row-interleaved-v1" else None
        )
        self._vector_scalar_stats: dict[str, object] = {
            "segmented_norm_square_ops_elided": 0,
            "segmented_norm_copy_ops_elided": 0,
            "segmented_norm_constant_loads_elided": 0,
            "inactive_norm_rows_elided": 0,
            "redundant_valid_masks_elided": 0,
            "valid_mask_build_count": 0,
            "rms_norm_address_loads_elided": 0,
            "rms_norm_nops_elided": 0,
            "segment_reductions_emitted": 0,
            "segment_reduction_levels_elided": 0,
            "scalar_moves_emitted": 0,
            "scalar_rsqrt_emitted": 0,
            "multi_segment_reductions_emitted": 0,
            "single_segment_reductions_elided": 0,
            "compact_stats_lane_loads": 0,
            "compact_stats_lane_stores": 0,
            "compact_stats_lane_loads_before": 0,
            "compact_stats_lane_stores_before": 0,
            "compact_lane_selectors_before": 0,
            "compact_lane_selectors_remaining": 0,
            "segment_broadcast_ops": 0,
            "scalar_modulo_schedule_width": 0,
            "compact_stat_simd_ops": 0,
            "compact_scalar_chain_ops_elided": 0,
            "compact_lane_selectors_elided": 0,
            "selector_loads_before": 0,
            "selector_loads_hoisted": 0,
            "selector_setup_instructions": 0,
            "neutral_accumulator_setups_before": 0,
            "neutral_accumulator_setups_elided": 0,
            "compact_stats_lane_policy": self.compact_stats_lane_policy,
            "compact_stats_lanes": self.compact_stats_lanes,
            "compact_stats_required_segments": 0,
            "compact_stats_utilization": 0.0,
            "compact_stats_fallback_reason": None,
        }
        self._packed_attention_stats: dict[str, int] = {
            "softmax_first_block_specialized_count": 0,
            "softmax_first_block_specialized_rows": 0,
            "softmax_state_initializations_elided": 0,
            "softmax_state_initialization_rows_elided": 0,
            "softmax_m_moves_elided": 0,
            "softmax_l_moves_elided": 0,
            "softmax_m_stores_elided": 0,
            "m_res_stores_elided": 0,
            "m_res_loads_elided": 0,
            "m_res_streamed_rows": 0,
            "temporary_o_matrices_elided": 0,
            "direct_o_lane_updates": 0,
            "qk_compute_count": 0,
            "ideal_qk_compute_count": 0,
            "pv_compute_count": 0,
            "kv_tile_load_count": 0,
            "ideal_kv_tile_load_count": 0,
            "kv_cache_hits": 0,
            "kv_cache_misses": 0,
            "full_q_tiles": 0,
            "q_tail_rows": 0,
            "tail_bmm_occurrences": 0,
            "tail_full_width_work_cycles": 0,
        }
        self._gqa_pipeline_stats: dict[str, int | str | bool] = {
            "softmax_first_block_pipeline_width": 0,
            "softmax_recurrent_pipeline_width": 0,
            "o_scale_pipeline_width": 0,
            "o_shift_ring_width": 0,
            "interleaved_softmax_rows": 0,
            "interleaved_o_rows": 0,
            "gqa_kv_double_buffered": False,
            "gqa_dma_overlap_eligible_occurrences": 0,
            "gqa_pipeline_fallback_reason": "none",
            "arithmetic_opcode_count_delta": 0,
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
            "softmax_state_schedule": self.softmax_state_schedule,
            "packed_qk_schedule": self.packed_qk_schedule,
            "gqa_pipeline_schedule": self.gqa_pipeline_schedule,
            "gqa_timing_artifact": (str(self.gqa_timing_profile.path) if self.gqa_timing_profile is not None else None),
            "gqa_timing_artifact_sha256": (
                self.gqa_timing_profile.sha256 if self.gqa_timing_profile is not None else None
            ),
            **self._packed_attention_stats,
            **self._gqa_pipeline_stats,
        }
        qk_ideal = self._packed_attention_stats["ideal_qk_compute_count"]
        kv_ideal = self._packed_attention_stats["ideal_kv_tile_load_count"]
        stats["qk_recompute_factor"] = self._packed_attention_stats["qk_compute_count"] / qk_ideal if qk_ideal else 0.0
        stats["kv_reload_factor"] = self._packed_attention_stats["kv_tile_load_count"] / kv_ideal if kv_ideal else 0.0
        residency = getattr(self, "_kv_residency_plan_metadata", None)
        if residency is not None:
            stats.update(
                {
                    "requested_kv_residency_fraction": residency["requested_residency_fraction"],
                    "realized_kv_residency_fraction": residency["realized_residency_fraction"],
                    "resident_kv_blocks": residency["resident_prefix_blocks"],
                    "streamed_kv_blocks": residency["streaming_blocks"],
                    "resident_stream_slot_tiles": (2 if residency["streaming_blocks"] else 0),
                    "peak_live_tiles": residency["peak_live_tiles"],
                    "average_live_tiles": residency["average_live_tiles"],
                    "tile_utilization": residency["tile_utilization"],
                    "matrix_sram_tiles": residency["matrix_sram_tiles"],
                    "matrix_sram_policy": residency["policy"],
                    "kv_cache_fidelity": "exact_compiler_schedule_single_chip",
                }
            )
        q_tail_rows = self._packed_attention_stats["q_tail_rows"]
        stats["q_tail_utilization"] = q_tail_rows / self.mlen if q_tail_rows else 1.0
        stats["tail_isa_limitation"] = "active_row_bmm_unavailable" if q_tail_rows else "none"
        state_layout = getattr(self, "_softmax_state_layout", None)
        if state_layout is not None:
            stats.update(state_layout.metadata())
        stats["qk_broadcast_reuse_factor"] = stats["qk_recompute_factor"]
        stats["broadcast_rtl_validation_status"] = (
            "broadcast_rtl_unvalidated"
            if self.packed_qk_schedule == PACKED_QK_SCHEDULE_BROADCAST_K_MAJOR_V1
            else "not_applicable"
        )
        return stats

    def configure_softmax_state_layout(self, active_broadcast_heads: int):
        """Install and validate the Scalar FP SRAM layout used by attention."""

        layout = build_softmax_state_layout(
            mlen=self.mlen,
            active_broadcast_heads=active_broadcast_heads,
            schedule=self.softmax_state_schedule,
            fp_constant_num=self.fp_constant_num,
        )
        if self.fp_sram_depth and self.fp_sram_depth < layout.required_depth:
            raise ValueError(
                f"FP_SRAM_DEPTH={self.fp_sram_depth} is smaller than the "
                f"{self.packed_qk_schedule}/{self.softmax_state_schedule} "
                f"requirement {layout.required_depth}"
            )
        self._softmax_state_layout = layout
        return layout

    def record_gqa_pipeline_stats(self, values: dict[str, int | str | bool]) -> None:
        """Accumulate pipeline-lowering metadata without affecting ISA."""

        max_fields = {
            "softmax_first_block_pipeline_width",
            "softmax_recurrent_pipeline_width",
            "o_scale_pipeline_width",
            "o_shift_ring_width",
        }
        for key, value in values.items():
            if key in max_fields:
                self._gqa_pipeline_stats[key] = max(int(self._gqa_pipeline_stats.get(key, 0)), int(value))
            elif key == "gqa_kv_double_buffered":
                self._gqa_pipeline_stats[key] = bool(self._gqa_pipeline_stats.get(key, False)) or bool(value)
            elif key == "gqa_pipeline_fallback_reason":
                if value and value != "none":
                    self._gqa_pipeline_stats[key] = str(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                self._gqa_pipeline_stats[key] = int(self._gqa_pipeline_stats.get(key, 0)) + value
            else:
                self._gqa_pipeline_stats[key] = value

    def record_vector_scalar_stats(self, values: dict[str, object]) -> None:
        """Accumulate metadata emitted by shared normalization/mask plans."""

        max_fields = {
            "scalar_modulo_schedule_width",
            "compact_stats_lanes",
            "compact_stats_required_segments",
            "compact_stats_utilization",
        }
        stable_fields = {
            "compact_stats_lane_policy",
        }
        for key, value in values.items():
            if key not in self._vector_scalar_stats:
                self._vector_scalar_stats[key] = 0
            if key in stable_fields:
                existing = self._vector_scalar_stats.get(key)
                if existing not in {None, value}:
                    raise ValueError(
                        f"inconsistent vector/scalar metadata {key}: "
                        f"{existing!r} vs {value!r}"
                    )
                self._vector_scalar_stats[key] = value
            elif key == "compact_stats_fallback_reason":
                if value:
                    self._vector_scalar_stats[key] = str(value)
            elif key in max_fields:
                self._vector_scalar_stats[key] = max(
                    float(self._vector_scalar_stats.get(key, 0)),
                    float(value),
                )
                if key != "compact_stats_utilization":
                    self._vector_scalar_stats[key] = int(
                        self._vector_scalar_stats[key]
                    )
            elif isinstance(value, bool):
                self._vector_scalar_stats[key] = bool(
                    self._vector_scalar_stats.get(key, False)
                ) or value
            elif isinstance(value, int):
                self._vector_scalar_stats[key] = (
                    int(self._vector_scalar_stats.get(key, 0)) + value
                )
            else:
                self._vector_scalar_stats[key] = value

    def vector_scalar_stats(self) -> dict[str, object]:
        return {
            "vector_scalar_schedule": self.vector_scalar_schedule,
            "selector_schedule": self.selector_schedule,
            "reduction_output_mode": self.reduction_output_mode,
            "valid_mask_scope": "program" if self._vector_scalar_stats["valid_mask_build_count"] else "none",
            **self._vector_scalar_stats,
        }

    def record_ffn_address_stats(self, values: dict[str, int | str]) -> None:
        """Accumulate FFN pointer-liveness metadata from the shared plan."""

        for key, value in values.items():
            if key == "ffn_address_schedule":
                continue
            self._ffn_address_stats[key] = self._ffn_address_stats.get(key, 0) + int(value)

    def ffn_address_stats(self) -> dict[str, int | str]:
        return {
            "ffn_address_schedule": self.ffn_address_schedule,
            "ffn_projection_schedule": self.ffn_projection_schedule,
            **self._ffn_address_stats,
            **self._ffn_projection_metadata,
        }

    def record_ffn_projection_metadata(self, values: dict[str, object]) -> None:
        """Record structural FFN loop-plan and guard provenance."""

        self._ffn_projection_metadata.update(values)

    # ========================================================================
    # Compilation
    # ========================================================================

    def compile(self) -> str:
        """Get generated ISA code string."""
        code, metadata = optimize_agu_assembly(
            super().get_code(),
            mode=self.address_generation_mode,
        )
        self._agu_metadata = metadata
        return code

    def agu_metadata(self) -> dict[str, object]:
        """Return metadata from the most recent assembly finalization."""
        return dict(self._agu_metadata)

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
