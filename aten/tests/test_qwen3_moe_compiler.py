"""Tests for the Qwen3-MoE static-index native compiler path."""

from __future__ import annotations

from collections import Counter
import math
from types import SimpleNamespace

import pytest
import torch

from compiler.aten.cost_frontend import (
    CompilerCostHardware,
    clear_cost_trace_cache,
    compile_native_decoder_cost_trace,
)
from compiler.aten.model_extract import (
    LayerWeights,
    MoeExpertWeights,
    TensorMoeExpertProvider,
    extract_layer_weights,
    extract_model_config,
)
from compiler.aten.moe import (
    FixedBalancedRoutingSummary,
    MoeRoutingPlan,
    StaticRoute,
    derive_static_routing_plan,
)
from compiler.aten.plena.compiler import PlenaCompiler
from compiler.aten.plena_frontend import compile_native_hf_decoder
from compiler.aten.qwen3_moe_partial import load_qwen3_moe_partial_model


class _ProceduralExperts:
    def __init__(self, *, num_experts: int, hidden: int, intermediate: int):
        self.num_experts = num_experts
        self.hidden_size = hidden
        self.intermediate_size = intermediate
        self.materialized: list[int] = []

    def materialize(self, expert_id: int) -> MoeExpertWeights:
        self.materialized.append(expert_id)
        generator = torch.Generator().manual_seed(1000 + expert_id)
        scale = 0.03
        return MoeExpertWeights(
            w_gate=torch.randn(
                self.hidden_size, self.intermediate_size, generator=generator
            )
            * scale,
            w_up=torch.randn(
                self.hidden_size, self.intermediate_size, generator=generator
            )
            * scale,
            w_down=torch.randn(
                self.intermediate_size, self.hidden_size, generator=generator
            )
            * scale,
        )


def _linear(in_features: int, out_features: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    module = SimpleNamespace()
    module.weight = torch.randn(out_features, in_features, generator=generator) * 0.03
    return module


def make_tiny_qwen3_moe():
    hidden = 128
    heads = 8
    kv_heads = 2
    head_dim = 16
    intermediate = 64
    num_experts = 8
    provider = _ProceduralExperts(
        num_experts=num_experts, hidden=hidden, intermediate=intermediate
    )
    # Non-unity weights make the tests exercise the learned decoder and Q/K
    # RMSNorm multiply paths instead of passing through an identity wrapper.
    norm = lambda width: SimpleNamespace(
        weight=torch.linspace(0.75, 1.25, width), eps=1e-6
    )
    attention = SimpleNamespace(
        q_proj=_linear(hidden, heads * head_dim, 1),
        k_proj=_linear(hidden, kv_heads * head_dim, 2),
        v_proj=_linear(hidden, kv_heads * head_dim, 3),
        o_proj=_linear(heads * head_dim, hidden, 4),
        q_norm=norm(head_dim),
        k_norm=norm(head_dim),
    )
    experts = SimpleNamespace(_plena_expert_provider=provider)
    mlp = SimpleNamespace(gate=_linear(hidden, num_experts, 5), experts=experts)
    layer = SimpleNamespace(
        self_attn=attention,
        mlp=mlp,
        input_layernorm=norm(hidden),
        post_attention_layernorm=norm(hidden),
    )
    config = SimpleNamespace(
        hidden_size=hidden,
        intermediate_size=256,
        moe_intermediate_size=intermediate,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        rms_norm_eps=1e-6,
        rope_theta=1.0e6,
        vocab_size=1024,
        model_type="qwen3_moe",
    )
    model = SimpleNamespace(
        config=config,
        layers=[layer],
        norm=norm(hidden),
        _test_expert_provider=provider,
    )
    return model


def _dynamic_opcode_histogram(asm: str) -> Counter[str]:
    dynamic: Counter[str] = Counter()
    loop_stack: list[int] = []
    for raw_line in asm.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        opcode = line.split(maxsplit=1)[0]
        dynamic[opcode] += math.prod(loop_stack)
        if opcode == "C_LOOP_START":
            loop_stack.append(int(line.rsplit(",", 1)[1]))
        elif opcode == "C_LOOP_END":
            loop_stack.pop()
    assert not loop_stack
    return dynamic


def test_qwen3_moe_config_and_lazy_extraction():
    model = make_tiny_qwen3_moe()
    config = extract_model_config(model)
    assert config.inter_dim == 64
    assert config.dense_inter_dim == 256
    assert config.moe_inter_dim == 64
    assert config.num_experts == 8
    assert config.experts_per_token == 2
    assert config.is_moe_layer(0)

    weights = extract_layer_weights(model.layers[0], config)
    assert weights.w_router.shape == (128, 8)
    assert model._test_expert_provider.materialized == []
    selected = weights.with_active_experts({2, 5})
    assert sorted(selected.active_experts) == [2, 5]
    assert model._test_expert_provider.materialized == [2, 5]


def test_static_route_plan_is_stable_and_hash_checked():
    probabilities = torch.tensor(
        [[0.05, 0.30, 0.10, 0.25, 0.05, 0.05, 0.15, 0.05]]
    )
    plan = derive_static_routing_plan(
        probabilities,
        active_physical_rows=[0],
        num_experts=8,
        experts_per_token=2,
        max_routes=16,
    )
    assert [route.expert_id for route in plan.routes] == [1, 3]
    restored = MoeRoutingPlan.from_dict(plan.as_dict())
    assert restored == plan
    assert restored.routing_plan_hash == plan.routing_plan_hash


def test_fixed_balanced_routing_summary_is_compact_and_deterministic():
    summary = FixedBalancedRoutingSummary.build(
        num_tokens=7_712, num_experts=128, experts_per_token=8
    )
    assert summary.route_count == 61_696
    assert summary.active_expert_ids == tuple(range(128))
    assert set(summary.routes_per_expert.values()) == {482}
    assert set(summary.padded_bucket_rows(64).values()) == {512}
    assert not hasattr(summary, "routes")
    assert summary.routing_summary_hash == FixedBalancedRoutingSummary.build(
        num_tokens=7_712, num_experts=128, experts_per_token=8
    ).routing_summary_hash
    uneven = FixedBalancedRoutingSummary.build(
        num_tokens=5, num_experts=7, experts_per_token=2
    )
    assert sum(uneven.routes_per_expert.values()) == 10
    assert max(uneven.routes_per_expert.values()) - min(
        uneven.routes_per_expert.values()
    ) == 1


def test_fixed_balanced_cost_trace_matches_explicit_round_robin_plan():
    model = make_tiny_qwen3_moe()
    config = extract_model_config(model)
    hardware = CompilerCostHardware(
        mlen=64,
        blen=4,
        vlen=64,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        hbm_m_prefetch_amount=64,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
    )
    routes = tuple(
        StaticRoute(
            token_index=token,
            physical_row=token,
            rank=rank,
            expert_id=(token * config.experts_per_token + rank)
            % config.num_experts,
        )
        for token in range(8)
        for rank in range(config.experts_per_token)
    )
    plan = MoeRoutingPlan(
        num_experts=config.num_experts,
        experts_per_token=config.experts_per_token,
        routes=routes,
    )
    explicit = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=8,
        moe_routing_plan=plan,
        use_cache=False,
    )
    aggregate = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=8,
        moe_routing_mode="fixed-balanced",
        use_cache=False,
    )
    assert aggregate.dynamic_opcodes == explicit.dynamic_opcodes
    assert {
        stage: cost.dynamic_opcodes
        for stage, cost in aggregate.stages.items()
    } == {
        stage: cost.dynamic_opcodes for stage, cost in explicit.stages.items()
    }
    assert [event.to_dict() for event in aggregate.memory_events] == [
        event.to_dict() for event in explicit.memory_events
    ]
    assert aggregate.metadata["route_count"] == 16
    assert aggregate.metadata["route_count_per_layer"] == 16
    assert aggregate.metadata["decoder_route_count"] == 16
    assert aggregate.metadata["routing_fidelity"] == "fixed_balanced_histogram"
    assert aggregate.metadata["exact_token_addresses"] is False
    assert aggregate.metadata["materialized_route_count"] == 0
    assert explicit.metadata["materialized_route_count"] == 16
    assert aggregate.metadata["host_selected_indices"] is False
    assert explicit.metadata["host_selected_indices"] is True
    with pytest.raises(ValueError, match="cannot be combined"):
        compile_native_decoder_cost_trace(
            config,
            hardware,
            seq_len=8,
            moe_routing_mode="fixed-balanced",
            moe_routing_plan=plan,
            use_cache=False,
        )


def test_fused_expert_provider_splits_gate_up_without_materializing_all_experts():
    hidden, intermediate, experts = 3, 2, 2
    gate_up = torch.arange(
        experts * 2 * intermediate * hidden, dtype=torch.float32
    ).reshape(experts, 2 * intermediate, hidden)
    down = torch.arange(
        experts * hidden * intermediate, dtype=torch.float32
    ).reshape(experts, hidden, intermediate)
    provider = TensorMoeExpertProvider(
        gate_up_proj=gate_up,
        down_proj=down,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    selected = provider.materialize(1)
    assert torch.equal(selected.w_gate, gate_up[1, :intermediate].T)
    assert torch.equal(selected.w_up, gate_up[1, intermediate:].T)
    assert torch.equal(selected.w_down, down[1].T)


def test_hybrid_dense_layer_uses_dense_intermediate_width():
    model = make_tiny_qwen3_moe()
    hidden = model.config.hidden_size
    dense_intermediate = model.config.intermediate_size
    model.config.mlp_only_layers = [0]
    model.layers[0].mlp = SimpleNamespace(
        gate_proj=_linear(hidden, dense_intermediate, 11),
        up_proj=_linear(hidden, dense_intermediate, 12),
        down_proj=_linear(dense_intermediate, hidden, 13),
    )

    config = extract_model_config(model)
    assert config.is_moe_layer(0) is False
    weights = extract_layer_weights(model.layers[0], config)
    assert isinstance(weights, LayerWeights)
    assert weights.w_gate.shape == (hidden, dense_intermediate)

    result = compile_native_hf_decoder(
        model,
        seq_len=1,
        batch_size=1,
        num_layers=1,
        mlen=64,
        blen=4,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        seed=7,
        reference_backend="scheduled",
    )
    assert result["info"]["selected_layer_type"] == "dense"
    assert result["info"]["inter_dim"] == dense_intermediate
    assert result["info"]["moe_routing_mode"] is None

    hardware = CompilerCostHardware(
        mlen=64,
        blen=4,
        vlen=64,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        hbm_m_prefetch_amount=64,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
    )
    trace = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=1,
        layer_idx=0,
        num_layers=1,
        use_cache=False,
    )
    assert trace.metadata["workload"]["inter_dim"] == dense_intermediate
    assert trace.metadata["moe_routing_mode"] is None
    with pytest.raises(ValueError, match="hybrid Qwen3-MoE dense-layer"):
        compile_native_decoder_cost_trace(
            config,
            hardware,
            seq_len=1,
            layer_idx=0,
            num_layers=2,
            use_cache=False,
        )


def test_tiny_qwen3_moe_compile_emits_existing_isa_only():
    model = make_tiny_qwen3_moe()
    result = compile_native_hf_decoder(
        model,
        seq_len=8,
        batch_size=1,
        num_layers=1,
        mlen=64,
        blen=4,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        seed=7,
        reference_backend="scheduled",
        moe_routing_mode="static-indices",
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
    )
    info = result["info"]
    assert info["moe_routing_mode"] == "static-indices"
    assert info["route_count"] == 16
    assert 1 <= info["active_expert_count"] <= 8
    assert info["host_selected_indices"] is True
    assert info["runtime_computed_route_weights"] is True
    assert info["excluded_runtime_operation"] == "arg_topk"
    assert len(model._test_expert_provider.materialized) == info["active_expert_count"]
    assert torch.isfinite(result["golden_output"]).all()
    isa = result["isa"]
    assert "S_MAP_V_FP" in isa
    assert "Normalize 2 selected MoE route weights" in isa
    assert "V_ADD_VV" in isa
    assert "W_input_norm_0" in result["input_tensors"]
    assert "W_post_attn_norm_0" in result["input_tensors"]
    assert "W_q_norm_0" in result["input_tensors"]
    assert "W_k_norm_0" in result["input_tensors"]
    assert "W_final_norm" in result["input_tensors"]
    assert all(opcode not in isa for opcode in ("V_GATHER", "V_TOPK", "V_SCATTER"))

    trace = compile_native_decoder_cost_trace(
        extract_model_config(model),
        CompilerCostHardware(
            mlen=64,
            blen=4,
            vlen=64,
            hlen=16,
            broadcast_amount=4,
            mram_tile_capacity=4,
            hbm_m_prefetch_amount=64,
            hbm_v_prefetch_amount=4,
            hbm_v_writeback_amount=4,
            hbm_channels=128,
        ),
        seq_len=8,
        batch_size=1,
        num_layers=1,
        moe_routing_plan=result["moe_routing_plan"],
        use_cache=False,
    )
    assert trace.dynamic_opcodes == _dynamic_opcode_histogram(isa)
    dma_counts = Counter()
    for event in trace.memory_events:
        dma_counts[event.transfer.opcode] += event.multiplicity
    assert dma_counts == {
        opcode: trace.dynamic_opcodes[opcode]
        for opcode in ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")
    }
    assert trace.metadata["routing_plan_hash"] == info["routing_plan_hash"]
    assert trace.metadata["active_expert_ids"] == info["active_expert_ids"]
    assert trace.metadata["expert_bucket_rows"] == info["expert_bucket_rows"]
    assert {
        "layer/moe/router",
        "layer/moe/dispatch",
        "layer/moe/experts",
        "layer/moe/combine",
    } <= trace.stages.keys()


def test_moe_cost_frontend_requires_explicit_plan_and_opt_in_layer_repeat():
    model = make_tiny_qwen3_moe()
    config = extract_model_config(model)
    hardware = CompilerCostHardware(
        mlen=64,
        blen=4,
        vlen=64,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        hbm_m_prefetch_amount=64,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
    )
    with pytest.raises(ValueError, match="explicit moe_routing_plan"):
        compile_native_decoder_cost_trace(
            config, hardware, seq_len=1, num_layers=1, use_cache=False
        )

    probabilities = torch.tensor(
        [[0.30, 0.25, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03]]
    )
    plan = derive_static_routing_plan(
        probabilities,
        active_physical_rows=[0],
        num_experts=8,
        experts_per_token=2,
        max_routes=16,
    )
    with pytest.raises(ValueError, match="repeat-static-plan"):
        compile_native_decoder_cost_trace(
            config,
            hardware,
            seq_len=1,
            num_layers=2,
            moe_routing_plan=plan,
            use_cache=False,
        )

    clear_cost_trace_cache()
    one_layer = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=1,
        num_layers=1,
        moe_routing_plan=plan,
    )
    cached = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=1,
        num_layers=1,
        moe_routing_plan=plan,
    )
    assert cached.metadata["cost_cache_hit"] is True

    other_plan = derive_static_routing_plan(
        torch.tensor([[0.03, 0.04, 0.30, 0.25, 0.15, 0.10, 0.08, 0.05]]),
        active_physical_rows=[0],
        num_experts=8,
        experts_per_token=2,
        max_routes=16,
    )
    different_route = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=1,
        num_layers=1,
        moe_routing_plan=other_plan,
    )
    assert different_route.metadata["cost_cache_hit"] is False
    assert different_route.metadata["config_hash"] != one_layer.metadata["config_hash"]

    repeated = compile_native_decoder_cost_trace(
        config,
        hardware,
        seq_len=1,
        num_layers=2,
        moe_routing_plan=plan,
        moe_layer_scaling="repeat-static-plan",
    )
    assert repeated.metadata["layer_scaling_fidelity"] == (
        "approximate_repeated_static_plan"
    )
    for stage_name, stage in one_layer.stages.items():
        multiplier = 2 if stage_name.startswith("layer/") else 1
        assert repeated.stages[stage_name].dynamic_opcodes == Counter(
            {opcode: count * multiplier for opcode, count in stage.dynamic_opcodes.items()}
        )


def test_vector_prefetch_codegen_uses_runtime_transfer_amount():
    prog = PlenaCompiler(
        mlen=128,
        blen=8,
        hbm_v_prefetch_amount=8,
        hbm_v_writeback_amount=8,
    )
    source = prog.input("prefetch_source", shape=(128, 128))
    prog.load_batch(source, name="prefetched")
    isa = prog.compile()

    # One H_PREFETCH_V writes amount*MLEN logical elements.  Both destination
    # and source offsets must advance by that exact span, not the old four-row
    # default that caused adjacent VRAM allocations to be overwritten.
    assert "S_ADDI_INT gp4, gp4, 1024" in isa
    assert "S_ADDI_INT gp3, gp3, 1024" in isa
    assert "C_LOOP_START gp6, 16" in isa


def test_partial_safetensors_loader_reads_only_selected_expert(tmp_path):
    from safetensors.torch import save_file

    layer_idx = 3
    hidden, heads, kv_heads, head_dim = 8, 2, 1, 4
    num_experts, intermediate = 4, 3
    prefix = f"model.layers.{layer_idx}"
    names = {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(heads * head_dim, hidden),
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv_heads * head_dim, hidden),
        f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, heads * head_dim),
        f"{prefix}.self_attn.q_norm.weight": torch.randn(head_dim),
        f"{prefix}.self_attn.k_norm.weight": torch.randn(head_dim),
        f"{prefix}.mlp.gate.weight": torch.randn(num_experts, hidden),
        f"{prefix}.mlp.experts.gate_up_proj": torch.arange(
            num_experts * 2 * intermediate * hidden, dtype=torch.float32
        ).reshape(num_experts, 2 * intermediate, hidden),
        f"{prefix}.mlp.experts.down_proj": torch.arange(
            num_experts * hidden * intermediate, dtype=torch.float32
        ).reshape(num_experts, hidden, intermediate),
        f"{prefix}.input_layernorm.weight": torch.randn(hidden),
        f"{prefix}.post_attention_layernorm.weight": torch.randn(hidden),
        "model.norm.weight": torch.randn(hidden),
    }
    shard = "model-00001-of-00001.safetensors"
    save_file(names, tmp_path / shard)
    (tmp_path / "config.json").write_text(
        __import__("json").dumps(
            {
                "model_type": "qwen3_moe",
                "hidden_size": hidden,
                "intermediate_size": 16,
                "moe_intermediate_size": intermediate,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": head_dim,
                "num_experts": num_experts,
                "num_experts_per_tok": 2,
                "norm_topk_prob": True,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1.0e6,
                "vocab_size": 32,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        __import__("json").dumps(
            {"weight_map": {name: shard for name in names}}
        )
    )

    model = load_qwen3_moe_partial_model(tmp_path, layer_idx=layer_idx)
    provider = model._plena_expert_provider
    assert provider.materialized == []
    expert = provider.materialize(2)
    assert provider.materialized == [2]
    assert expert.w_gate.shape == (hidden, intermediate)
    assert expert.w_up.shape == (hidden, intermediate)
    assert expert.w_down.shape == (intermediate, hidden)
    assert torch.equal(
        expert.w_gate,
        names[f"{prefix}.mlp.experts.gate_up_proj"][2, :intermediate].T,
    )
    assert model._plena_source_layer_idx == layer_idx
