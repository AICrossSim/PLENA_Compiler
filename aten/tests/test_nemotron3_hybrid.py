import json

from aten.mamba.scheduler import MambaScheduleConfig, SchedulePhase
from aten.nemotron3.scheduler import (
    NEMOTRON3_PATTERN,
    HybridLayerType,
    Nemotron3Architecture,
    Nemotron3HybridScheduler,
)


def _trace(*, phase: SchedulePhase, sequence_length: int = 1, decode_tokens: int = 1):
    return Nemotron3HybridScheduler(
        MambaScheduleConfig(
            phase=phase,
            sequence_length=sequence_length,
            decode_tokens=decode_tokens,
            chunk_size=128,
        )
    ).build()


def test_architecture_uses_the_official_52_layer_pattern() -> None:
    arch = Nemotron3Architecture()

    assert len(NEMOTRON3_PATTERN) == 52
    assert NEMOTRON3_PATTERN.count("M") == 23
    assert NEMOTRON3_PATTERN.count("E") == 23
    assert NEMOTRON3_PATTERN.count("*") == 6
    assert arch.attention_head_dim == 128
    assert arch.kv_heads == 2


def test_decode_interleaves_all_layer_families_in_model_order() -> None:
    trace = _trace(phase=SchedulePhase.DECODE)
    body = [event for event in trace.events if event.stage == "block_rms_norm"]

    assert len(body) == 52
    assert "".join(
        {
            HybridLayerType.MAMBA: "M",
            HybridLayerType.MOE: "E",
            HybridLayerType.ATTENTION: "*",
        }[event.layer_type]
        for event in body
    ) == NEMOTRON3_PATTERN
    assert trace.count("block_residual") == 52
    assert trace.count("mamba_in_projection") == 23
    assert trace.count("x_state_step") == 23
    assert trace.count("moe_router_topk") == 23
    assert trace.count("moe_shared_expert") == 23
    assert trace.count("attention_qkv_projection") == 6
    assert trace.count("attention_qk_softmax_pv") == 6


def test_prefill_chunks_each_mamba_layer_without_changing_model_order() -> None:
    trace = _trace(phase=SchedulePhase.PREFILL, sequence_length=257)

    assert trace.count("x_state_prefill") == 23 * 3
    assert trace.count("block_rms_norm") == 52
    assert trace.count("moe_routed_experts") == 23
    assert trace.count("attention_out_projection") == 6


def test_hybrid_trace_json_exposes_implementation_boundaries() -> None:
    rendered = json.loads(
        json.dumps(_trace(phase=SchedulePhase.DECODE).to_dict())
    )

    assert rendered["scope"] == "full_52_layer_body_trace"
    assert rendered["architecture"]["layer_counts"] == {
        "mamba": 23,
        "moe": 23,
        "attention": 6,
    }
    implementations = rendered["summary"]["implementation_counts"]
    assert implementations["existing_isa"] > 0
    assert implementations["existing_plena_service"] > 0
    assert implementations["l_scatter_m_v1"] == 23
    assert rendered["mamba_residency"] == {
        "cache_policy": "none",
        "capacity_bytes": 0,
        "state_cache_entries": 0,
        "source": None,
        "target": None,
        "resident_state_keys": [],
    }
