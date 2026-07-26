from __future__ import annotations

from collections import Counter

import pytest

from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    _build_layout,
    _finalize_energy_action_lineage,
    _scale_trace,
    clear_cost_trace_cache,
    compile_native_decoder_cost_trace,
    load_cost_model_config,
)
from compiler.aten.cost_emitter import (
    CostSink,
    CostTrace,
    EnergyAction,
    ParallelKernelCensusEntry,
    ScheduleRepeat,
    parallel_kernel_lineage_id,
)
from compiler.aten.isa_builder import IsaBuilder, gp
from compiler.aten.model_extract import ModelConfig


def _qwen3_32b() -> ModelConfig:
    return ModelConfig(
        hidden_size=5120,
        inter_dim=25600,
        num_heads=64,
        num_kv_heads=8,
        head_dim=128,
        eps=1e-6,
        rope_theta=1_000_000.0,
        vocab_size=151936,
        model_type="qwen3",
    )


def _target_hardware() -> CompilerCostHardware:
    return CompilerCostHardware(
        mlen=128,
        blen=128,
        vlen=128,
        hlen=128,
        broadcast_amount=8,
        mram_tile_capacity=16,
        hbm_m_prefetch_amount=128,
        hbm_v_prefetch_amount=128,
        hbm_v_writeback_amount=128,
        hbm_channels=128,
    )


def _small_ratio_six_qwen3() -> ModelConfig:
    return ModelConfig(
        hidden_size=192,
        inter_dim=384,
        num_heads=12,
        num_kv_heads=2,
        head_dim=16,
        eps=1e-6,
        rope_theta=10_000.0,
        vocab_size=1024,
        model_type="qwen3",
    )


def _small_hardware(*, hlen: int, mram_tile_capacity: int) -> CompilerCostHardware:
    return CompilerCostHardware(
        mlen=128,
        blen=64,
        vlen=128,
        hlen=hlen,
        broadcast_amount=6,
        mram_tile_capacity=mram_tile_capacity,
        hbm_m_prefetch_amount=128,
        hbm_v_prefetch_amount=128,
        hbm_v_writeback_amount=128,
        hbm_channels=32,
    )


def _tiny_packed_qwen3() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        inter_dim=64,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        eps=1e-6,
        rope_theta=10_000.0,
        vocab_size=128,
        model_type="qwen3",
    )


def _tiny_packed_hardware() -> CompilerCostHardware:
    return CompilerCostHardware(
        mlen=16,
        blen=4,
        vlen=16,
        hlen=4,
        broadcast_amount=2,
        mram_tile_capacity=8,
        hbm_m_prefetch_amount=16,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
        hbm_channels=8,
    )


def _arithmetic_opcodes(trace) -> dict[str, int]:
    return {
        opcode: count
        for opcode, count in trace.dynamic_opcodes.items()
        if opcode.startswith("V_")
        or (opcode.startswith("S_") and opcode.endswith("_FP"))
    }


def test_affine_ffn_projection_preserves_matrix_and_dma_work() -> None:
    common = dict(
        model_config=_tiny_packed_qwen3(),
        hardware_config=_tiny_packed_hardware(),
        seq_len=8,
        batch_size=1,
        num_layers=1,
        address_generation_mode="loop-agu-v1",
        ffn_address_schedule="live-stride-v1",
        use_cache=False,
    )

    compatibility = compile_native_decoder_cost_trace(
        **common,
        ffn_projection_schedule="legacy-auto-v1",
    )
    affine = compile_native_decoder_cost_trace(
        **common,
        ffn_projection_schedule="affine-loop-v2",
    )

    for opcode in ("M_MM", "M_MM_WO", "V_ADD_VV", "H_PREFETCH_M"):
        assert affine.stages["layer/ffn"].dynamic_opcodes[opcode] == (
            compatibility.stages["layer/ffn"].dynamic_opcodes[opcode]
        )
    assert [event.to_dict() for event in affine.memory_events] == [
        event.to_dict() for event in compatibility.memory_events
    ]

    affine_metadata = affine.metadata["ffn_address_optimization"]
    assert affine_metadata["ffn_loop_plan_version"] == "ffn-affine-loop-ir-v2"
    assert affine_metadata["ffn_explicit_loop_depth"] <= 4
    assert max(affine_metadata["ffn_agu_streams_by_axis"].values()) <= 2
    assert affine_metadata["ffn_schedule_guard_status"] == (
        "structural_invariants_passed"
    )
    assert affine_metadata["ffn_legacy_template_bypassed"] is True
    assert affine_metadata["ffn_address_cycles_after"] <= (
        compatibility.metadata["ffn_address_optimization"][
            "ffn_address_cycles_after"
        ]
    )


def test_gqa_row_interleaving_preserves_arithmetic_and_dma_work() -> None:
    hardware = _tiny_packed_hardware()
    hardware = CompilerCostHardware(
        **{**hardware.__dict__, "mram_tile_capacity": 2}
    )
    common = dict(
        model_config=_tiny_packed_qwen3(),
        hardware_config=hardware,
        seq_len=39,
        batch_size=1,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        softmax_state_schedule="sram-v1",
        packed_qk_schedule="head-major-v1",
        use_cache=False,
    )

    serial = compile_native_decoder_cost_trace(
        **common, gqa_pipeline_schedule="row-serial"
    )
    interleaved = compile_native_decoder_cost_trace(
        **common, gqa_pipeline_schedule="row-interleaved-v1"
    )

    assert _arithmetic_opcodes(interleaved) == _arithmetic_opcodes(serial)
    for opcode in ("M_MM", "M_MM_WO", "M_BTMM", "M_BMM_WO"):
        assert interleaved.dynamic_opcodes[opcode] == serial.dynamic_opcodes[opcode]
    for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
        assert interleaved.dynamic_opcodes[opcode] == serial.dynamic_opcodes[opcode]
    assert len(interleaved.memory_events) == len(serial.memory_events)
    assert sum(event.multiplicity for event in interleaved.memory_events) == sum(
        event.multiplicity for event in serial.memory_events
    )
    metadata = interleaved.metadata["packed_attention"]
    assert metadata["softmax_first_block_pipeline_width"] == 3
    assert metadata["softmax_recurrent_pipeline_width"] == 2
    assert metadata["o_scale_pipeline_width"] == 8
    assert metadata["o_shift_ring_width"] == 16
    assert metadata["gqa_kv_double_buffered"] is True
    assert metadata["gqa_dma_overlap_eligible_occurrences"] > 0
    assert metadata["arithmetic_opcode_count_delta"] == 0
    assert not interleaved.schedule_unavailable_reasons


def test_affine_summary_matches_detailed_ideal_work() -> None:
    hardware = CompilerCostHardware(
        **{**_tiny_packed_hardware().__dict__, "mram_tile_capacity": 2}
    )
    common = dict(
        model_config=_tiny_packed_qwen3(),
        hardware_config=hardware,
        seq_len=39,
        batch_size=1,
        num_layers=1,
        vector_scalar_schedule="rtl-v4",
        selector_schedule="hoisted-v1",
        reduction_output_mode="overwrite-v1",
        address_generation_mode="loop-agu-v1",
        use_cache=False,
    )

    detailed = compile_native_decoder_cost_trace(
        **common, cost_trace_granularity="detailed"
    )
    summary = compile_native_decoder_cost_trace(
        **common, cost_trace_granularity="affine-block-summary-v1"
    )

    assert summary.static_opcodes == detailed.static_opcodes
    assert summary.dynamic_opcodes == detailed.dynamic_opcodes
    assert {
        name: (stage.static_opcodes, stage.dynamic_opcodes)
        for name, stage in summary.stages.items()
    } == {
        name: (stage.static_opcodes, stage.dynamic_opcodes)
        for name, stage in detailed.stages.items()
    }
    assert sum(event.multiplicity for event in summary.memory_events) == sum(
        event.multiplicity for event in detailed.memory_events
    )
    def aggregate_actions(trace):
        result = Counter()
        for action in trace.energy_actions:
            key = (
                action.stage,
                action.component,
                action.action,
                action.precision,
                action.active_lanes,
                action.total_lanes,
                action.active_bits,
                action.segment_log2,
                action.segment_count,
                action.activity_fidelity,
            )
            result[(key, "count")] += action.count
            result[(key, "busy_cycles")] += action.busy_cycles
            result[(key, "bytes")] += action.bytes
        return result

    assert aggregate_actions(summary) == aggregate_actions(detailed)
    assert summary.metadata["ordered_schedule_available"] is False
    assert summary.metadata["materialized_block_pair_count"] == 0


def test_parallel_kernel_census_covers_every_compute_opcode() -> None:
    trace = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        seq_len=8,
        batch_size=1,
        num_layers=2,
        use_cache=False,
    )
    compute_count = sum(
        count
        for opcode, count in trace.dynamic_opcodes.items()
        if not opcode.startswith("H_")
    )
    assert trace.schema_version == 7
    assert trace.metadata["parallel_kernel_census_coverage"] == 1.0
    assert sum(entry.count for entry in trace.parallel_kernel_census) == (
        compute_count
    )
    assert all(entry.tp_semantics for entry in trace.parallel_kernel_census)
    assert all(entry.cp_semantics for entry in trace.parallel_kernel_census)
    assert {
        entry.kernel for entry in trace.parallel_kernel_census
    } >= {
        "attention_q_projection",
        "attention_qk_softmax_pv",
        "dense_ffn_projection",
    }


def test_gqa_double_buffer_reports_capacity_fallback() -> None:
    hardware = _tiny_packed_hardware()
    hardware = CompilerCostHardware(
        **{**hardware.__dict__, "mram_tile_capacity": 1}
    )

    trace = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        hardware,
        seq_len=39,
        batch_size=1,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        gqa_pipeline_schedule="row-interleaved-v1",
        use_cache=False,
    )

    metadata = trace.metadata["packed_attention"]
    assert metadata["gqa_kv_double_buffered"] is False
    assert metadata["gqa_pipeline_fallback_reason"] == "mram_tile_capacity_lt_2"


def test_broadcast_k_major_reuses_qk_and_preserves_pv_count() -> None:
    common = dict(
        model_config=_tiny_packed_qwen3(),
        hardware_config=_tiny_packed_hardware(),
        seq_len=7,
        batch_size=4,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        address_generation_mode="legacy",
        use_cache=False,
    )
    head_major = compile_native_decoder_cost_trace(
        **common,
        softmax_state_schedule="sram-v1",
        packed_qk_schedule="head-major-v1",
    )
    k_major = compile_native_decoder_cost_trace(
        **common,
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
    )

    assert head_major.dynamic_opcodes["M_BTMM"] == 2 * k_major.dynamic_opcodes["M_BTMM"]
    assert head_major.dynamic_opcodes["M_BMM_WO"] == 2 * k_major.dynamic_opcodes["M_BMM_WO"]
    assert (
        head_major.metadata["packed_attention"]["pv_compute_count"]
        == k_major.metadata["packed_attention"]["pv_compute_count"]
    )
    metadata = k_major.metadata["packed_attention"]
    assert metadata["qk_recompute_factor"] == 1.0
    assert metadata["softmax_m_moves_elided"] > 0
    assert metadata["softmax_l_moves_elided"] > 0
    assert metadata["softmax_m_stores_elided"] > 0
    assert k_major.metadata["broadcast_rtl_validated"] is False
    assert (
        k_major.metadata["broadcast_rtl_validation_status"]
        == "broadcast_rtl_unvalidated"
    )


def test_broadcast_k_major_streams_recurrent_state_and_reuses_nonresident_kv() -> None:
    hardware = CompilerCostHardware(
        **{**_tiny_packed_hardware().__dict__, "mram_tile_capacity": 2}
    )
    common = dict(
        model_config=_tiny_packed_qwen3(),
        hardware_config=hardware,
        seq_len=39,
        batch_size=2,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        gqa_pipeline_schedule="row-interleaved-v1",
        address_generation_mode="legacy",
        use_cache=False,
    )
    head_major = compile_native_decoder_cost_trace(
        **common,
        softmax_state_schedule="sram-v1",
        packed_qk_schedule="head-major-v1",
    )
    k_major = compile_native_decoder_cost_trace(
        **common,
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
    )

    head_stats = head_major.metadata["packed_attention"]
    k_stats = k_major.metadata["packed_attention"]
    assert k_stats["m_res_stores_elided"] > 0
    assert k_stats["m_res_loads_elided"] == k_stats["m_res_stores_elided"]
    assert k_stats["m_res_streamed_rows"] == k_stats["m_res_stores_elided"]
    assert k_stats["softmax_m_stores_elided"] > 0
    assert k_stats["pv_compute_count"] == head_stats["pv_compute_count"]
    assert k_stats["qk_compute_count"] * 2 == head_stats["qk_compute_count"]
    assert k_stats["kv_tile_load_count"] < head_stats["kv_tile_load_count"]
    assert k_stats["gqa_kv_double_buffered"] is True
    assert head_major.metadata["broadcast_rtl_validation_status"] == "not_applicable"


def test_broadcast_k_major_uses_partial_resident_kv_prefix() -> None:
    hardware = CompilerCostHardware(
        **{**_tiny_packed_hardware().__dict__, "mram_tile_capacity": 4}
    )
    trace = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        hardware,
        seq_len=39,
        batch_size=2,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
        address_generation_mode="legacy",
        use_cache=False,
    )

    stats = trace.metadata["packed_attention"]
    schedule = trace.metadata["attention_schedule"]
    assert schedule["resident_prefix_blocks"] == 1
    assert schedule["streaming_blocks"] == 2
    assert stats["resident_kv_blocks"] == 1
    assert stats["streamed_kv_blocks"] == 2
    # Per batch/KV head: 2 resident loads plus 2 and 4 streamed loads for
    # causal q-blocks 1 and 2. There are two batches and two KV heads.
    assert stats["kv_tile_load_count"] == 8 * 4
    assert stats["ideal_kv_tile_load_count"] == 6 * 4
    assert stats["kv_cache_hits"] == 6 * 4
    assert stats["kv_cache_misses"] == stats["kv_tile_load_count"]
    assert stats["qk_compute_count"] == stats["ideal_qk_compute_count"]


def test_streaming_policy_overrides_spare_matrix_sram_capacity() -> None:
    hardware = CompilerCostHardware(
        **{
            **_tiny_packed_hardware().__dict__,
            "mram_tile_capacity": 4,
            "kv_residency_policy": "streaming",
        }
    )
    trace = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        hardware,
        seq_len=39,
        batch_size=2,
        num_layers=1,
        vector_scalar_schedule="rtl-v3",
        softmax_state_schedule="streamed-v2",
        packed_qk_schedule="broadcast-k-major-v1",
        address_generation_mode="legacy",
        use_cache=False,
    )

    stats = trace.metadata["packed_attention"]
    schedule = trace.metadata["attention_schedule"]
    assert schedule["policy"] == "streaming"
    assert schedule["resident_prefix_blocks"] == 0
    assert stats["resident_kv_blocks"] == 0
    assert stats["streamed_kv_blocks"] == 3
    assert stats["kv_tile_load_count"] == 12 * 4
    assert stats["kv_cache_hits"] == 0


def test_cost_model_config_parses_qwen3_moe_fields() -> None:
    model, layers = load_cost_model_config(
        {
            "model_type": "qwen3_moe",
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_hidden_layers": 94,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "norm_topk_prob": True,
            "decoder_sparse_step": 2,
            "mlp_only_layers": [0, 4],
        }
    )

    assert layers == 94
    assert model.inter_dim == 1536
    assert model.dense_inter_dim == 12288
    assert model.moe_inter_dim == 1536
    assert model.num_experts == 128
    assert model.experts_per_token == 8
    assert model.norm_topk_prob is True
    assert model.decoder_sparse_step == 2
    assert model.mlp_only_layers == (0, 4)
    assert model.is_moe_layer(0) is False
    assert model.is_moe_layer(1) is True
    assert model.is_moe_layer(4) is False


def test_scale_trace_repeats_complete_decoder_layer_in_program_order() -> None:
    program = IsaBuilder()
    program.stage("global/setup", IsaBuilder().instr("S_ADD_INT", gp(1), gp(2), gp(3)))
    program.stage("layer/attention", IsaBuilder().instr("M_MM", gp(1), gp(2)))
    program.stage("layer/ffn", IsaBuilder().instr("V_ADD_VV", gp(1), gp(2), gp(3), 0))
    program.stage("global/final", IsaBuilder().instr("S_SUB_INT", gp(1), gp(2), gp(3)))
    sink = CostSink()
    sink.emit(program)

    scaled = _scale_trace(sink.finish(), 3)

    assert scaled.dynamic_opcodes == {
        "S_ADD_INT": 1,
        "M_MM": 3,
        "V_ADD_VV": 3,
        "S_SUB_INT": 1,
    }
    assert len(scaled.schedule.children) == 3
    layer_repeat = scaled.schedule.children[1]
    assert isinstance(layer_repeat, ScheduleRepeat)
    assert layer_repeat.name == "decoder_layer"
    assert layer_repeat.count == 3
    assert [child.opcode for child in layer_repeat.body.children] == ["M_MM", "V_ADD_VV"]


def test_qwen3_target_cost_trace_matches_transactional_instruction_profile() -> None:
    clear_cost_trace_cache()
    trace = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        seq_len=482,
        batch_size=16,
        num_layers=1,
        vector_scalar_schedule="rtl-v2",
        address_generation_mode="legacy",
        ffn_address_schedule="legacy",
    )

    # Learned decoder/QK/final norm weights are now part of the native Qwen3
    # schedule, so this profile intentionally differs from the pre-norm trace.
    assert trace.static_instruction_count == 56_702_159
    assert trace.dynamic_opcodes["M_MM"] == 1_919_488
    assert trace.dynamic_opcodes["M_BTMM"] == 10_240
    assert trace.dynamic_opcodes["H_PREFETCH_M"] == 394_177
    assert trace.dynamic_opcodes["S_MAP_V_FP"] == 1024
    assert trace.dynamic_opcodes.get("V_SHIFT_V", 0) == 0
    assert trace.dynamic_opcodes["V_RED_SUM_SEG"] > 0
    # seq_len > VLEN takes the tiled attention fallback, so only Q/K norm can
    # use a segment reduction in this particular profile.
    assert trace.dynamic_opcodes["V_RED_MAX"] > 0
    assert trace.dynamic_opcodes["S_MV_FP"] > 0
    assert trace.dynamic_opcodes["S_RSQRT_FP"] > 0
    assert trace.metadata["cost_cache_hit"] is False
    assert trace.metadata["packed_attention"]["packed_attention_schedule"] == (
        "direct-first-block-v1"
    )
    assert trace.metadata["packed_attention"][
        "softmax_first_block_specialized_count"
    ] == 4096
    dma_counts = Counter()
    for event in trace.memory_events:
        dma_counts[event.transfer.opcode] += event.multiplicity
    assert dma_counts == {
        opcode: trace.dynamic_opcodes[opcode]
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
    }
    assert {event.transfer.precision for event in trace.memory_events} >= {
        "weight",
        "matrix_kv",
        "activation",
    }
    assert [event.stream_index for event in trace.memory_events] == list(
        range(len(trace.memory_events))
    )
    assert all(event.transfer.geometry_fidelity == "exact" for event in trace.memory_events)
    assert all(event.transfer.source for event in trace.memory_events)
    assert trace.metadata["dma_coverage"] == {
        "geometry_fidelity": "exact",
        "stream_count": len(trace.memory_events),
        "parallel_kernel_lineage_coverage": (
            sum(
                event.multiplicity
                for event in trace.memory_events
                if event.parallel_kernel is not None
            )
            / sum(event.multiplicity for event in trace.memory_events)
        ),
        "layer_parallel_kernel_lineage_coverage": 1.0,
        "layer_dynamic_occurrences": sum(
            event.multiplicity
            for event in trace.memory_events
            if event.stage.startswith("layer/")
        ),
        "global_unclassified_occurrences": sum(
            event.multiplicity
            for event in trace.memory_events
            if event.parallel_kernel is None
        ),
        "dynamic_opcodes": {
            opcode: trace.dynamic_opcodes[opcode]
            for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
        },
    }

    cached = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        seq_len=482,
        batch_size=16,
        num_layers=64,
        vector_scalar_schedule="rtl-v2",
        address_generation_mode="legacy",
        ffn_address_schedule="legacy",
    )
    assert cached.metadata["cost_cache_hit"] is True
    assert cached.dynamic_opcodes["M_MM"] == 64 * 1_919_488

    cached.dynamic_opcodes["M_MM"] = 0
    untouched = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        seq_len=482,
        batch_size=16,
        num_layers=1,
        vector_scalar_schedule="rtl-v2",
    )
    assert untouched.dynamic_opcodes["M_MM"] == 1_919_488


def test_energy_action_lineage_rejects_missing_structural_family() -> None:
    entry = ParallelKernelCensusEntry(
        stage="layer/ffn",
        kernel="dense_ffn_projection",
        opcode="M_MM",
        count=1,
        tp_semantics="ffn_projection_tiled",
        cp_semantics="token_partitioned",
        ep_semantics="none",
    )
    lineage = parallel_kernel_lineage_id(entry.to_dict())
    trace = CostTrace(
        energy_actions=[
            EnergyAction(
                stage="layer/ffn",
                component="matrix",
                action="array_compute",
                count=1,
                precision="M_MM",
                parallel_kernel=lineage,
            )
        ],
        parallel_kernel_census=[entry],
    )

    with pytest.raises(ValueError, match="structural-family coverage"):
        _finalize_energy_action_lineage(trace)


def test_qwen3_target_default_vector_scalar_schedule_is_rtl_v3() -> None:
    clear_cost_trace_cache()
    common = {
        "seq_len": 482,
        "batch_size": 16,
        "num_layers": 1,
        "use_cache": False,
    }
    default = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        **common,
    )
    explicit = compile_native_decoder_cost_trace(
        _qwen3_32b(),
        _target_hardware(),
        vector_scalar_schedule="rtl-v3",
        **common,
    )

    assert default.metadata["vector_scalar_schedule"] == "rtl-v3"
    assert default.metadata["packed_attention"]["gqa_pipeline_schedule"] == (
        "row-interleaved-v1"
    )
    assert default.static_opcodes == explicit.static_opcodes
    assert default.dynamic_opcodes == explicit.dynamic_opcodes
    assert default.dynamic_opcodes["V_RED_SUM_SEGS"] > 0
    assert default.dynamic_opcodes["S_LD_VLANE_FP"] > 0
    assert default.dynamic_opcodes["S_ST_VLANE_FP"] > 0


def test_packed_attention_optimized_schedule_preserves_compute_and_dma_geometry() -> None:
    common = {
        "seq_len": 39,
        "batch_size": 2,
        "num_layers": 1,
        "softmax_state_schedule": "sram-v1",
        "packed_qk_schedule": "head-major-v1",
        "use_cache": False,
    }
    legacy = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        packed_attention_schedule="legacy",
        **common,
    )
    optimized = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        packed_attention_schedule="direct-first-block-v1",
        **common,
    )

    legacy_packed = legacy.metadata["packed_attention"]
    optimized_packed = optimized.metadata["packed_attention"]
    for field in (
        "qk_compute_count",
        "ideal_qk_compute_count",
        "pv_compute_count",
        "kv_tile_load_count",
        "ideal_kv_tile_load_count",
    ):
        assert optimized_packed[field] == legacy_packed[field]
    for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
        assert optimized.dynamic_opcodes[opcode] == legacy.dynamic_opcodes[opcode]

    assert optimized_packed["softmax_first_block_specialized_count"] > 0
    assert optimized_packed["softmax_state_initializations_elided"] > 0
    assert optimized_packed["temporary_o_matrices_elided"] > 0
    assert optimized_packed["direct_o_lane_updates"] > 0
    assert sum(optimized.dynamic_opcodes.values()) < sum(legacy.dynamic_opcodes.values())
    assert optimized.dynamic_opcodes["S_MAX_FP"] < legacy.dynamic_opcodes["S_MAX_FP"]
    assert optimized.dynamic_opcodes["S_EXP_FP"] < legacy.dynamic_opcodes["S_EXP_FP"]


def test_packed_attention_schedule_is_part_of_cost_trace_cache_key() -> None:
    clear_cost_trace_cache()
    common = {
        "seq_len": 7,
        "batch_size": 1,
        "num_layers": 1,
        "softmax_state_schedule": "sram-v1",
        "packed_qk_schedule": "head-major-v1",
    }
    optimized = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        packed_attention_schedule="direct-first-block-v1",
        **common,
    )
    legacy = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        packed_attention_schedule="legacy",
        **common,
    )
    optimized_cached = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        packed_attention_schedule="direct-first-block-v1",
        **common,
    )

    assert optimized.metadata["cost_cache_hit"] is False
    assert legacy.metadata["cost_cache_hit"] is False
    assert optimized_cached.metadata["cost_cache_hit"] is True
    assert optimized.static_instruction_count != legacy.static_instruction_count


def test_vector_scalar_schedule_is_shared_with_cost_trace_and_cache_key() -> None:
    clear_cost_trace_cache()
    common = {
        "seq_len": 7,
        "batch_size": 4,
        "num_layers": 1,
    }
    optimized = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        vector_scalar_schedule="compiler-v1",
        **common,
    )
    legacy = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        vector_scalar_schedule="legacy",
        **common,
    )
    optimized_cached = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        vector_scalar_schedule="compiler-v1",
        **common,
    )

    assert optimized.metadata["cost_cache_hit"] is False
    assert legacy.metadata["cost_cache_hit"] is False
    assert optimized_cached.metadata["cost_cache_hit"] is True
    stats = optimized.metadata["vector_scalar_optimization"]
    assert stats["vector_scalar_schedule"] == "compiler-v1"
    assert stats["segmented_norm_square_ops_elided"] > 0
    assert stats["segmented_norm_constant_loads_elided"] > 0
    assert stats["inactive_norm_rows_elided"] > 0
    # Power-of-two aligned batch slots make this a physically full tile; the
    # block-diagonal causal mask covers inactive slot tails directly, so no
    # separate valid-column mask exists to build or count as redundant.
    assert stats["redundant_valid_masks_elided"] == 0
    assert stats["valid_mask_build_count"] == 0
    assert stats["valid_mask_scope"] == "none"
    for opcode in ("M_MM", "M_BTMM", "H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
        assert optimized.dynamic_opcodes[opcode] == legacy.dynamic_opcodes[opcode]
    assert optimized.dynamic_instruction_count < legacy.dynamic_instruction_count


def test_rtl_v2_cost_trace_keeps_segment_width_variants() -> None:
    from compiler.aten.cost_emitter import schedule_instruction_variants

    trace = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        seq_len=7,
        batch_size=4,
        num_layers=1,
        vector_scalar_schedule="rtl-v2",
        softmax_state_schedule="sram-v1",
        packed_qk_schedule="head-major-v1",
        use_cache=False,
    )
    variants = schedule_instruction_variants(
        trace.schedule,
        opcodes={"V_RED_SUM_SEG", "V_RED_MAX_SEG"},
    )

    assert sum(
        count for (opcode, _), count in variants.items() if opcode == "V_RED_SUM_SEG"
    ) == trace.dynamic_opcodes["V_RED_SUM_SEG"]
    assert sum(
        count for (opcode, _), count in variants.items() if opcode == "V_RED_MAX_SEG"
    ) == trace.dynamic_opcodes["V_RED_MAX_SEG"]
    # HLEN=4 segmented Q/K norm and slot-width=8 softmax coexist in one trace.
    assert {
        int(args[-1]) for (_, args) in variants if args
    } == {2, 3}


def test_rtl_v3_cost_trace_uses_segment_parallel_norm_without_changing_matrix_or_dma() -> None:
    from compiler.aten.cost_emitter import schedule_instruction_variants

    common = {
        "seq_len": 7,
        "batch_size": 4,
        "num_layers": 1,
        "softmax_state_schedule": "sram-v1",
        "packed_qk_schedule": "head-major-v1",
        "use_cache": False,
    }
    rtl_v2 = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        vector_scalar_schedule="rtl-v2",
        **common,
    )
    rtl_v3 = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        vector_scalar_schedule="rtl-v3",
        **common,
    )

    assert rtl_v3.metadata["vector_scalar_schedule"] == "rtl-v3"
    stats = rtl_v3.metadata["vector_scalar_optimization"]
    assert stats["scalar_modulo_schedule_width"] == 8
    assert stats["multi_segment_reductions_emitted"] > 0
    assert stats["compact_stats_lane_loads"] > 0
    assert stats["compact_stats_lane_stores"] > 0
    assert stats["segment_broadcast_ops"] > 0
    for opcode in (
        "V_RED_SUM_SEGS",
        "V_MUL_VSEG",
        "S_LD_VLANE_FP",
        "S_ST_VLANE_FP",
    ):
        assert rtl_v3.dynamic_opcodes[opcode] > 0

    # Online softmax remains on the calibrated single-segment path.  Only the
    # packed Q/K normalization changes to a compact multi-segment reduction.
    assert rtl_v3.dynamic_opcodes["V_RED_SUM_SEG"] > 0
    assert rtl_v3.dynamic_opcodes["V_RED_MAX_SEG"] > 0
    multi_variants = schedule_instruction_variants(
        rtl_v3.schedule,
        opcodes={"V_RED_SUM_SEGS"},
    )
    assert {int(args[-1]) for (_, args) in multi_variants if args} == {2}

    for opcode in ("M_MM", "M_MM_WO", "M_BTMM", "M_BMM_WO"):
        assert rtl_v3.dynamic_opcodes[opcode] == rtl_v2.dynamic_opcodes[opcode]
    for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
        assert rtl_v3.dynamic_opcodes[opcode] == rtl_v2.dynamic_opcodes[opcode]


def test_long_context_valid_mask_is_program_scoped_and_not_layer_scaled() -> None:
    common = {
        "seq_len": 39,
        "batch_size": 2,
        "vector_scalar_schedule": "compiler-v1",
        "use_cache": False,
    }
    one_layer = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        num_layers=1,
        **common,
    )
    four_layers = compile_native_decoder_cost_trace(
        _tiny_packed_qwen3(),
        _tiny_packed_hardware(),
        num_layers=4,
        **common,
    )

    stats = one_layer.metadata["vector_scalar_optimization"]
    assert stats["redundant_valid_masks_elided"] == 0
    assert stats["valid_mask_build_count"] == 1
    assert stats["valid_mask_scope"] == "program"
    one_global = one_layer.stages["global/valid_col_mask"]
    four_global = four_layers.stages["global/valid_col_mask"]
    assert four_global.static_opcodes == one_global.static_opcodes
    assert four_global.dynamic_opcodes == one_global.dynamic_opcodes
    assert four_layers.schedule_unavailable_reasons == Counter()


def test_cost_frontend_rejects_head_dimension_tiling() -> None:
    hardware = _target_hardware()
    hardware = CompilerCostHardware(**{**hardware.__dict__, "hlen": 64})

    try:
        compile_native_decoder_cost_trace(
            _qwen3_32b(), hardware, seq_len=128, batch_size=1, num_layers=1
        )
    except ValueError as error:
        assert "head-dimension tiling" in str(error)
        assert "HLEN=64" in str(error)
        assert "head_dim=128" in str(error)
    else:
        raise AssertionError("HLEN < head_dim should be rejected")


@pytest.mark.parametrize(
    "mlen,blen,pack_factor,attention_groups",
    [(512, 64, 1, 16), (1024, 128, 2, 8), (2048, 256, 4, 4)],
)
def test_cost_frontend_uses_shared_compact_native_layout(
    mlen: int,
    blen: int,
    pack_factor: int,
    attention_groups: int,
) -> None:
    hardware = CompilerCostHardware(
        mlen=mlen,
        blen=blen,
        vlen=mlen,
        hlen=128,
        broadcast_amount=8,
        mram_tile_capacity=16,
        hbm_m_prefetch_amount=mlen,
        hbm_v_prefetch_amount=blen,
        hbm_v_writeback_amount=blen,
        hbm_channels=128,
    )
    layout = _build_layout(
        _qwen3_32b(),
        hardware,
        seq_len=482,
        batch_size=16,
        layer_idx=0,
        native_layout_mode="compact",
    )

    assert layout.sequence_packing.batch_pack_factor == pack_factor
    assert layout.sequence_packing.attention_group_count == attention_groups
    assert layout.compile_seq_rows == 8192
    assert layout.head_packing.total_q_dim == 8192
    assert layout.head_packing.heads_per_storage_block == {
        512: 4,
        1024: 8,
        2048: 16,
    }[mlen]
    assert layout.head_packing.hardware_broadcast_amount == {
        512: 4,
        1024: 8,
        2048: 16,
    }[mlen]


@pytest.mark.parametrize(
    (
        "seq_len",
        "batch_size",
        "hlen",
        "mram_tile_capacity",
        "physical_broadcast",
        "chunks_per_kv",
        "tail_heads",
        "kv_resident",
    ),
    [
        (128, 1, 16, 16, 6, 1, 0, True),
        (64, 1, 32, 16, 4, 2, 2, True),
        (257, 2, 64, 2, 2, 3, 0, False),
    ],
)
def test_cost_frontend_covers_general_packed_gqa_schedules(
    seq_len: int,
    batch_size: int,
    hlen: int,
    mram_tile_capacity: int,
    physical_broadcast: int,
    chunks_per_kv: int,
    tail_heads: int,
    kv_resident: bool,
) -> None:
    trace = compile_native_decoder_cost_trace(
        _small_ratio_six_qwen3(),
        _small_hardware(hlen=hlen, mram_tile_capacity=mram_tile_capacity),
        seq_len=seq_len,
        batch_size=batch_size,
        num_layers=3,
        use_cache=False,
    )

    schedule = trace.metadata["attention_schedule"]
    assert schedule["active_head_dim"] == 16
    assert schedule["head_slot_dim"] == hlen
    assert schedule["physical_broadcast"] == physical_broadcast
    assert schedule["chunks_per_kv"] == chunks_per_kv
    assert schedule["tail_heads"] == tail_heads
    assert schedule["q_blocks"] == (seq_len + 127) // 128
    assert schedule["kv_resident"] is kv_resident
    assert schedule["looped_batch"] is (batch_size > 1)
    assert trace.dynamic_opcodes["M_MM"] == 3 * trace.metadata["one_layer_dynamic_opcodes"]["M_MM"]

    dma_counts = Counter()
    for event in trace.memory_events:
        dma_counts[event.transfer.opcode] += event.multiplicity
    assert dma_counts == {
        opcode: trace.dynamic_opcodes[opcode]
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
    }
