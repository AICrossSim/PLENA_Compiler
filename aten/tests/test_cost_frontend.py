import pytest

from compiler.aten.cost_frontend import (
    CompilerHardwareSpec,
    DecoderModelSpec,
    RoutingHistogram,
    compile_dense_decoder_trace,
    compile_routed_moe_trace,
)


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
