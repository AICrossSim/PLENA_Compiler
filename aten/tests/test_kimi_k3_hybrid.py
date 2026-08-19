from __future__ import annotations

from aten.kda.scheduler import KdaScheduleConfig
from aten.kimi3.scheduler import KimiK3Architecture, KimiK3HybridScheduler
from aten.mamba.scheduler import SchedulePhase


def test_kimi_architecture_matches_pinned_93_layer_config() -> None:
    arch = KimiK3Architecture()
    assert len(arch.kda_layers) == 69
    assert len(arch.mla_layers) == 24
    assert set(arch.kda_layers).isdisjoint(arch.mla_layers)
    assert sorted((*arch.kda_layers, *arch.mla_layers)) == list(range(93))
    assert arch.dense_layers == (0,)
    assert len(arch.moe_layers) == 92
    assert arch.attn_res_capture_layers == (0, 12, 24, 36, 48, 60, 72, 84)


def test_decode_interleaves_kda_mla_moe_and_attn_res_in_model_order() -> None:
    trace = KimiK3HybridScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    assert trace.count("input_rms_norm") == 93
    assert trace.count("kda_qkv_projection") == 69
    assert trace.count("mla_q_low_rank_projection") == 24
    assert trace.count("dense_situ_ffn") == 1
    assert trace.count("latent_moe_router_top16") == 92
    assert trace.count("attn_res_capture_prefix") == 8
    assert trace.count("attn_res_before_mixer") == 92
    assert trace.count("attn_res_before_ffn") == 93
    assert trace.count("output_attn_res") == 1

    input_norms = [event for event in trace.events if event.stage == "input_rms_norm"]
    assert [event.layer_id for event in input_norms] == list(range(93))
    rendered = trace.to_dict()
    assert rendered["scope"] == "full_93_layer_text_backbone_structural_trace"
    assert rendered["limits"][0] == "KDA events have existing-ISA physical lowering"


def test_prefill_preserves_full_model_order_while_chunking_only_kda() -> None:
    trace = KimiK3HybridScheduler(
        KdaScheduleConfig(
            phase=SchedulePhase.PREFILL,
            sequence_length=33,
            chunk_size=16,
        )
    ).build()
    assert trace.count("kda_qkv_projection") == 69 * 3
    assert trace.count("mla_q_low_rank_projection") == 24
    assert trace.count("latent_moe_router_top16") == 92
    assert trace.count("input_rms_norm") == 93
