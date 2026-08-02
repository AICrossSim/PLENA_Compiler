import pytest

from compiler.aten.cost_frontend import (
    CompilerHardwareSpec,
    DecoderModelSpec,
    RoutingHistogram,
    compile_dense_decoder_trace,
    compile_routed_moe_trace,
)
from compiler.aten.program_sink import COST_TRACE_GRANULARITY_SUMMARY


def _dma_occurrences(result):
    return sum(event.multiplicity for event in result.trace.dma_events)


def _dense_model():
    return DecoderModelSpec(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        num_hidden_layers=2,
        model_type="llama",
    )


def test_dense_shape_frontend_uses_real_compiler_schedule():
    result = compile_dense_decoder_trace(
        _dense_model(),
        CompilerHardwareSpec(mlen=64, blen=4),
        seq_len=16,
        batch_size=1,
        num_layers=1,
        include_assembly=True,
    )
    counts = result.trace.dynamic_opcode_counts
    assert counts["M_MM"] > 0
    assert counts["V_RED_SUM"] > 0
    assert counts["H_PREFETCH_M"] > 0
    assert result.trace.dma_events
    assert all(event.stage for event in result.trace.dma_events)
    assert result.assembly is not None


def test_routing_histogram_is_required_and_conservative():
    with pytest.raises(ValueError, match="expected"):
        RoutingHistogram((1, 1), token_count=2, top_k=2)


def test_tiny_routed_moe_frontend_has_router_expert_and_combine_stages():
    model = DecoderModelSpec(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        model_type="qwen3_moe",
        num_experts=32,
        experts_per_token=4,
        moe_intermediate_size=64,
    )
    routing = RoutingHistogram.balanced(token_count=1, top_k=4, num_experts=32)
    result = compile_routed_moe_trace(
        model,
        CompilerHardwareSpec(mlen=64, blen=4),
        routing,
    )
    stages = {instruction.stage for instruction in result.trace.instructions}
    assert "decoder/moe/router" in stages
    assert "decoder/moe/expert" in stages
    assert "decoder/moe/combine" in stages
    assert result.trace.dynamic_opcode_counts["V_TOPK"] == 1


def test_production_routes_must_use_summary_backend():
    model = DecoderModelSpec(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        model_type="gpt_oss",
        num_experts=2,
        experts_per_token=1,
        moe_intermediate_size=64,
    )
    routing = RoutingHistogram((5000, 0), token_count=5000, top_k=1)
    with pytest.raises(ValueError, match="summary mode"):
        compile_routed_moe_trace(
            model,
            CompilerHardwareSpec(mlen=64, blen=4),
            routing,
        )


@pytest.mark.parametrize("seq_len", [16, 96])
def test_dense_summary_is_opcode_and_dma_exact(seq_len):
    hardware = CompilerHardwareSpec(mlen=64, blen=4, mram_tile_capacity=2)
    detailed = compile_dense_decoder_trace(
        _dense_model(),
        hardware,
        seq_len=seq_len,
        num_layers=1,
        cost_trace_granularity="detailed",
    )
    summary = compile_dense_decoder_trace(
        _dense_model(),
        hardware,
        seq_len=seq_len,
        num_layers=1,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )

    assert summary.trace.dynamic_opcode_counts == detailed.trace.dynamic_opcode_counts
    assert _dma_occurrences(summary) == _dma_occurrences(detailed)
    assert summary.trace.metadata["materialized_dynamic_instructions"] == 0
    assert summary.trace.metadata["ordered_schedule_available"] is False


def test_dense_summary_materializes_model_layer_repeat():
    hardware = CompilerHardwareSpec(mlen=64, blen=4, mram_tile_capacity=2)
    one = compile_dense_decoder_trace(
        _dense_model(),
        hardware,
        seq_len=16,
        num_layers=1,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )
    three = compile_dense_decoder_trace(
        _dense_model(),
        hardware,
        seq_len=16,
        num_layers=3,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )

    def stage_counts(result, prefix):
        counts = {}
        for item in result.trace.instructions:
            if item.stage.startswith(prefix):
                counts[item.opcode] = counts.get(item.opcode, 0) + item.multiplicity
        return counts

    assert stage_counts(three, "decoder/layer") == {
        opcode: 3 * count
        for opcode, count in stage_counts(one, "decoder/layer").items()
    }
    assert stage_counts(three, "global/setup") == stage_counts(one, "global/setup")
    assert three.trace.metadata["layer_scaling_required"] is False


def test_dense_detailed_rejects_implicit_multilayer_expansion():
    with pytest.raises(ValueError, match="use summary mode"):
        compile_dense_decoder_trace(
            _dense_model(),
            CompilerHardwareSpec(mlen=64, blen=4),
            seq_len=16,
            num_layers=2,
            cost_trace_granularity="detailed",
        )


def test_routed_moe_summary_matches_detailed_without_route_objects():
    model = DecoderModelSpec(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        model_type="qwen3_moe",
        num_experts=32,
        experts_per_token=4,
        moe_intermediate_size=64,
    )
    hardware = CompilerHardwareSpec(mlen=32, blen=4, mram_tile_capacity=2)
    routing = RoutingHistogram.balanced(token_count=5, top_k=4, num_experts=32)
    detailed = compile_routed_moe_trace(model, hardware, routing)
    summary = compile_routed_moe_trace(
        model,
        hardware,
        routing,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )

    assert summary.trace.dynamic_opcode_counts == detailed.trace.dynamic_opcode_counts
    assert _dma_occurrences(summary) == _dma_occurrences(detailed)
    assert summary.trace.metadata["route_object_count"] == 0
    assert summary.trace.metadata["expert_template_count"] > 0


def test_routed_moe_summary_chunks_to_main_fpram_capacity():
    model = DecoderModelSpec(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=64,
        model_type="qwen3_moe",
        num_experts=32,
        experts_per_token=4,
        moe_intermediate_size=64,
    )
    routing = RoutingHistogram.skewed(
        token_count=121,
        top_k=4,
        num_experts=32,
        hot_fraction=0.5,
    )
    summary = compile_routed_moe_trace(
        model,
        CompilerHardwareSpec(mlen=64, blen=4),
        routing,
        cost_trace_granularity=COST_TRACE_GRANULARITY_SUMMARY,
    )

    assert summary.trace.metadata["route_object_count"] == 0
    assert summary.trace.metadata["expert_chunk_count"] > model.num_experts
    assert summary.trace.metadata["route_storage_mode"] == "streamed-histogram-scratch"
    assert summary.trace.metadata["max_routes_per_chunk"] == 120
