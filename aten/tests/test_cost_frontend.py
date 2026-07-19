from __future__ import annotations

from collections import Counter

import pytest

from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    _build_layout,
    _scale_trace,
    clear_cost_trace_cache,
    compile_native_decoder_cost_trace,
    load_cost_model_config,
)
from compiler.aten.cost_emitter import CostSink, ScheduleRepeat
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
    )

    # Learned decoder/QK/final norm weights are now part of the native Qwen3
    # schedule, so this profile intentionally differs from the pre-norm trace.
    assert trace.static_instruction_count == 56_702_207
    assert trace.dynamic_opcodes["M_MM"] == 1_919_488
    assert trace.dynamic_opcodes["M_BTMM"] == 10_240
    assert trace.dynamic_opcodes["H_PREFETCH_M"] == 394_177
    assert trace.dynamic_opcodes["S_MAP_V_FP"] == 1024
    assert trace.dynamic_opcodes.get("V_SHIFT_V", 0) == 0
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
    )
    assert untouched.dynamic_opcodes["M_MM"] == 1_919_488


def test_packed_attention_optimized_schedule_preserves_compute_and_dma_geometry() -> None:
    common = {
        "seq_len": 39,
        "batch_size": 2,
        "num_layers": 1,
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
    assert stats["redundant_valid_masks_elided"] > 0
    assert stats["valid_mask_build_count"] == 0
    assert stats["valid_mask_scope"] == "none"
    for opcode in ("M_MM", "M_BTMM", "H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"):
        assert optimized.dynamic_opcodes[opcode] == legacy.dynamic_opcodes[opcode]
    assert optimized.dynamic_instruction_count < legacy.dynamic_instruction_count


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
