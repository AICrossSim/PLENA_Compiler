from __future__ import annotations

from collections import Counter

import pytest

from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    _scale_trace,
    clear_cost_trace_cache,
    compile_native_decoder_cost_trace,
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

    assert trace.static_instruction_count == 52_354_031
    assert trace.dynamic_opcodes["M_MM"] == 1_919_488
    assert trace.dynamic_opcodes["M_BTMM"] == 10_240
    assert trace.dynamic_opcodes["H_PREFETCH_M"] == 394_177
    assert trace.dynamic_opcodes["S_MAP_V_FP"] == 128
    assert trace.dynamic_opcodes.get("V_SHIFT_V", 0) == 0
    assert trace.metadata["cost_cache_hit"] is False
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
