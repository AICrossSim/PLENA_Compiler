from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from compiler.aten.model_extract import (
    ModelConfig,
    MoeLayerWeights,
    extract_layer_weights,
    extract_model_config,
)
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena_frontend import compile_native_hf_decoder
from compiler.aten.qwen3_moe_runtime import (
    QWEN3_FULL_LAYER_DATAFLOW_STAGES,
    QWEN3_MOE_MODEL_ID,
    QWEN3_MOE_MODEL_REVISION,
    QWEN3_MOE_ROUTER_PRECISION,
    QWEN3_MOE_RUNTIME_STAGES,
    RouterPrecisionContract,
    Qwen3RawBf16ExpertBankLayout,
    audit_qwen3_30b_a3b_full_model_frontend,
    build_qwen3_30b_a3b_decode_plan,
    qwen3_moe_cpu_reference,
    qwen3_moe_tail_isa_provenance_receipt,
    qwen3_post_attention_moe_cpu_reference,
    qwen3_moe_router_cpu_reference,
    qwen3_route_isa_provenance_receipt,
    qwen3_router_linear_provenance_receipt,
    qwen3_tiny_single_layer_transaction_provenance_receipt,
    pack_qwen3_raw_bf16_expert_hbm,
)


def _target_config(**overrides) -> ModelConfig:
    config = ModelConfig(
        hidden_size=2048,
        inter_dim=768,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        eps=1e-6,
        rope_theta=10_000_000.0,
        vocab_size=151936,
        model_type="qwen3_moe",
        qk_norm=True,
        num_hidden_layers=48,
        dense_inter_dim=6144,
        moe_inter_dim=768,
        num_experts=128,
        experts_per_token=8,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=(),
    )
    return replace(config, **overrides)


def _build_plan(**overrides):
    arguments = {
        "config": _target_config(),
        "layer_count": 48,
        "batch_size": 4,
        "key_precision": "MXINT4",
        "value_precision": "MXINT4",
    }
    arguments.update(overrides)
    return build_qwen3_30b_a3b_decode_plan(**arguments)


def test_exact_target_plan_covers_all_sparse_layers_and_stages():
    plan = _build_plan()

    assert len(plan.layers) == 48
    assert [layer.layer_index for layer in plan.layers] == list(range(48))
    assert all(_target_config().is_moe_layer(index) for index in range(48))
    assert all(layer.stages == QWEN3_MOE_RUNTIME_STAGES for layer in plan.layers)
    assert plan.expected_assignments_per_layer == 32
    assert plan.expected_assignments_total == 48 * 32
    receipt = plan.as_dict()
    assert receipt["compiled_layer_count"] == 48
    assert receipt["layers"][0]["shared_experts"] == 0
    assert receipt["router_execution_contract"]["narrow_vector_profile"] == "analytic_only"
    frontend = receipt["full_frontend_contract"]
    assert frontend["layer_count"] == 48
    assert frontend["ordered_layer_dataflow"] == list(
        QWEN3_FULL_LAYER_DATAFLOW_STAGES
    )
    assert len(frontend["layers"]) == 48
    assert frontend["routed_moe"]["all_expert_banks_addressable_per_layer"] == 128
    assert frontend["routed_moe"]["dense_fallback_forbidden"] is True
    assert frontend["runtime_route_substrate_available"] is True
    assert (
        frontend["routed_moe"][
            "tiny_hidden64_single_layer_emulator_transaction_parity"
        ]
        is True
    )
    assert frontend["end_to_end_lowering_available"] is False
    assert frontend["compiler_pipeline_valid"] is False
    assert frontend["emulator_valid"] is False
    assert frontend["rtl_valid"] is False
    assert frontend["timing_calibrated"] is False
    assert frontend["publication_rankable"] is False
    assert len(frontend["contract_sha256"]) == 64
    capacity = receipt["route_sram_capacity"]
    assert capacity["schedule"] == "whole_batch"
    assert capacity["entries"] == 1024
    assert capacity["bytes"] == 4096
    assert capacity["scores_per_token"] == 8
    assert capacity["max_resident_routed_tokens"] == 128
    assert capacity["required_entries"] == 32
    assert capacity["token_serial_reuse_implemented"] is False
    validity = receipt["study_validity"]
    assert validity["compiler_substrate_valid"] is True
    assert validity["emulator_router_linear_fixture_valid"] is True
    assert validity["single_layer_runtime_moe_lowering_available"] is True
    assert validity["tiny_hidden64_single_layer_emulator_transaction_parity"] is True
    assert validity["tiny_single_layer_transaction"][
        "compiler_generated_binary_emulator_parity_valid"
    ] is True
    assert validity["single_layer_emulator_transaction_parity"] is False
    assert validity["emulator_fp32_route_path_available"] is True
    assert validity["compiler_pipeline_valid"] is False
    assert validity["emulator_valid"] is False
    assert validity["rtl_valid"] is False
    assert validity["timing_calibrated"] is False
    assert validity["publication_rankable"] is False
    assert validity["blockers"] == [
        "full_model_moe_frontend_not_wired",
        "complete_runtime_moe_layer_emulator_transaction_parity_not_verified",
        "independent_bf16_router_precision_switch_missing",
        "router_and_fp32_route_isa_timing_uncalibrated_and_rtl_unsupported",
    ]
    assert (
        validity["router_linear"][
            "transformers_5_5_cpu_fixture_parity_valid"
        ]
        is True
    )
    with pytest.raises(RuntimeError, match="study execution is blocked"):
        plan.require_study_executable()


def test_assignment_conservation_is_checked_for_every_layer():
    plan = _build_plan()
    counts = {index: 32 for index in range(48)}

    receipt = plan.validate_runtime_assignment_counts(counts)
    assert receipt["conserved"] is True
    assert receipt["actual_total"] == 48 * 32
    counts[17] = 31
    with pytest.raises(ValueError, match="route conservation"):
        plan.validate_runtime_assignment_counts(counts)
    with pytest.raises(ValueError, match="cover layers"):
        plan.validate_runtime_assignment_counts([32] * 47)


def test_whole_batch_route_sram_capacity_is_fail_closed():
    boundary = _build_plan(batch_size=128).as_dict()["route_sram_capacity"]
    assert boundary["required_entries"] == boundary["entries"] == 1024
    assert boundary["capacity_valid"] is True

    with pytest.raises(ValueError, match="exceeds FP32 route SRAM"):
        _build_plan(batch_size=129)
    with pytest.raises(ValueError, match="exceeds FP32 route SRAM"):
        _build_plan(batch_size=256)


def test_route_isa_receipt_pins_fp32_lifetime_and_claim_boundary():
    receipt = qwen3_route_isa_provenance_receipt()

    assert receipt["abi"] == "V_TOPK@0x37+V_MUL_ROUTE_F32@0x38"
    assert receipt["route_sram_dtype"] == "FP32"
    assert receipt["route_sram_entries"] == 1024
    assert receipt["route_sram_bytes"] == 4096
    assert receipt["route_scores_per_token"] == 8
    assert receipt["resident_route_schedule"] == "whole_batch"
    assert receipt["max_resident_routed_tokens"] == 128
    assert receipt["route_multiply_dtype"] == "FP32"
    assert receipt["output_cast_dtype"] == "BF16"
    assert receipt["functional_emulator_implementation_available"] is True
    assert receipt["timing_calibrated"] is False
    assert receipt["rtl_valid"] is False
    assert receipt["publication_rankable"] is False
    assert len(receipt["contract_sha256"]) == 64


def test_router_linear_receipt_pins_exact_emulator_abi_and_claim_boundary():
    receipt = qwen3_router_linear_provenance_receipt()

    assert receipt["abi"] == "V_ROUTER_LINEAR_BF16@0x36"
    assert receipt["input_dtype"] == "BF16"
    assert receipt["weight_dtype"] == "BF16"
    assert receipt["output_dtype"] == "BF16"
    assert receipt["output_cast_count"] == 1
    assert receipt["policies"] == {
        "0": {"hidden_size": 64},
        "1": {"hidden_size": 2048},
    }
    assert receipt["transformers_5_5_cpu_fixture_parity_valid"] is True
    assert receipt["router_to_topk_transformers_5_5_fixture_parity_valid"] is True
    assert receipt["hidden64_fixture_hash"] == "b678a0d7fe63550a"
    assert receipt["hidden2048_fixture_hash"] == "81e382ff292ffd8e"
    assert receipt["timing_calibrated"] is False
    assert receipt["rtl_valid"] is False
    assert receipt["publication_rankable"] is False
    assert len(receipt["contract_sha256"]) == 64

@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_count": 1}, "all 48 layers"),
        ({"routing_mode": "static_indices"}, "host/static"),
        ({"dense_fallback": True}, "dense fallback"),
        ({"active_expert_ids": tuple(range(8))}, "active-expert-only"),
        ({"shared_expert_count": 1}, "no shared experts"),
        ({"value_precision": "MXINT8"}, "canonical K=V"),
    ],
)
def test_target_plan_rejects_shortcuts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _build_plan(**overrides)


def test_target_plan_rejects_dense_or_partial_architectures():
    for config in (
        replace(_target_config(), model_type="qwen3"),
        replace(_target_config(), decoder_sparse_step=2),
        replace(_target_config(), mlp_only_layers=(0,)),
        replace(_target_config(), num_experts=1),
    ):
        with pytest.raises(ValueError, match="not the exact"):
            _build_plan(config=config)


def test_contract_emits_every_layer_stage_marker():
    compiler = PlenaCompiler(mlen=64, blen=4)
    plan = _build_plan(batch_size=1)
    compiler.emit_qwen3_moe_decode_contract(plan)
    assembly = compiler.compile()

    assert "scope=compiler_substrate_only" in assembly
    assert "compiler_pipeline_valid=false emulator_valid=false" in assembly
    assert "@qwen3_frontend_contract scope=contract_only layers=48" in assembly
    assert (
        f"sha256={plan.full_frontend_contract_receipt()['contract_sha256']}"
        in assembly
    )
    assert assembly.count("@qwen3_frontend_layer layer=") == 48
    assert assembly.count("expert_banks=128 topk=8") == 48
    for layer_index in range(48):
        for stage in QWEN3_MOE_RUNTIME_STAGES:
            assert assembly.count(f"@stage={stage} layer={layer_index} ") == 1
    assert assembly.count("@stage=non_moe decode stack complete") == 1


def test_executable_router_rejects_narrow_global_vector_format():
    compiler = PlenaCompiler(mlen=64, blen=4)

    with pytest.raises(RuntimeError, match="no per-stage precision switch"):
        compiler.qwen3_router_logits_bf16_v0(
            None,
            None,
            rows=1,
            hidden=2048,
            router_vector_format="FP_E3M2",
        )


def test_router_gemv_rejects_unsealed_geometry_and_layout():
    compiler = PlenaCompiler(mlen=64, blen=4)
    x = compiler.alloc("x", 1, 128, strict=False, physical_shape=(4, 128))
    router = compiler.alloc(
        "router", 128, 128, strict=False, physical_shape=(128, 128)
    )
    with pytest.raises(ValueError, match="hidden=64 validation or hidden=2048"):
        compiler.qwen3_router_logits_bf16_v0(
            x,
            router,
            rows=1,
            hidden=128,
            router_vector_format="BF16",
        )

    x64 = compiler.alloc("x64", 1, 64, strict=False, physical_shape=(4, 64))
    padded_router = compiler.alloc(
        "padded_router", 128, 64, strict=False, physical_shape=(132, 64)
    )
    with pytest.raises(ValueError, match="physical router layout"):
        compiler.qwen3_router_logits_bf16_v0(
            x64,
            padded_router,
            rows=1,
            hidden=64,
            router_vector_format="BF16",
        )


def test_target_router_gemv_emits_hidden2048_policy_across_wide_mlen():
    compiler = PlenaCompiler(mlen=256, blen=4)
    x = compiler.alloc(
        "x", 1, 2048, strict=False, physical_shape=(4, 2048)
    )
    router = compiler.alloc(
        "router", 128, 2048, strict=False, physical_shape=(128, 2048)
    )
    compiler.qwen3_router_logits_bf16_v0(
        x,
        router,
        rows=1,
        hidden=2048,
        router_vector_format="BF16",
    )
    instruction = next(
        line
        for line in compiler.compile().splitlines()
        if line.startswith("V_ROUTER_LINEAR_BF16")
    )

    assert instruction.endswith(", 1")


def test_topk8_emits_runtime_device_selection_and_legalizes_large_bases():
    compiler = PlenaCompiler(mlen=64, blen=4)
    logits = compiler.alloc(
        "router_logits",
        rows=2,
        cols=64,
        strict=False,
        physical_shape=(4, 64),
    )
    compiler.qwen3_moe_router_topk8_v0(
        logits,
        token_idx=0,
        route_f32_base=0,
        indices_int_base=(1 << 28) + 64,
    )
    assembly = compiler.compile()

    assert "V_TOPK" in assembly
    assert ", 1" in next(line for line in assembly.splitlines() if "V_TOPK" in line)
    assert "S_LUI_INT" in assembly


def test_complete_reduced_shape_layer_substrate_emits_conserved_top8_path():
    compiler = PlenaCompiler(mlen=64, blen=4)
    x = compiler.alloc("x", 1, 64, strict=False, physical_shape=(4, 64))
    router = compiler.alloc(
        "router", 128, 64, strict=False, physical_shape=(128, 64)
    )
    gate = compiler.input("expert_gate", (64, 64))
    up = compiler.input("expert_up", (64, 64))
    down = compiler.input("expert_down", (64, 64))
    zero = compiler.fp_var("zero", 64)
    one = compiler.fp_var("one", 1)
    neg_one = compiler.fp_var("neg_one", 1)

    output, receipt = compiler.qwen3_moe_decode_layer_v0(
        x,
        router,
        (gate, up, down),
        layer_index=0,
        batch_size=1,
        weight_table_bases=(gate.hbm_addr, up.hbm_addr, down.hbm_addr),
        weight_table_strides=(gate.hbm_size, up.hbm_size, down.hbm_size),
        expert_table_counts=(128, 128, 128),
        route_f32_base=0,
        indices_int_base=0,
        constants=(zero, one, neg_one),
        zero_row=zero,
        router_vector_format="BF16",
        hidden=64,
        intermediate=64,
    )
    assembly = compiler.compile()

    assert output.shape == (1, 64)
    assert receipt == {
        "layer_index": 0,
        "expected_assignments": 8,
        "emitted_assignments": 8,
        "router_linear_transformers_fixture_parity": True,
        "complete_runtime_moe_layer_lowering": True,
        "layer_emulator_transaction_parity": False,
    }
    assert sum(
        line.startswith("V_ROUTER_LINEAR_BF16")
        for line in assembly.splitlines()
    ) == 1
    assert assembly.count("V_TOPK") == 1
    assert assembly.index("V_ROUTER_LINEAR_BF16") < assembly.index("V_TOPK")
    assert sum(
        line.startswith("V_MUL_ROUTE_F32") for line in assembly.splitlines()
    ) == 8
    assert "scores=FP32 storage=route_sram" in assembly
    assert "score=FP32 output_cast=BF16" in assembly
    assert "@moe_router_linear opcode=V_ROUTER_LINEAR_BF16" in assembly
    assert assembly.count("@stage=dispatch_runtime_expert_id pair=") >= 8
    assert assembly.count("@stage=scatter_combine pair=") == 8


def test_layer_substrate_requires_all_128_expert_tables():
    compiler = PlenaCompiler(mlen=64, blen=4)
    x = compiler.alloc("x", 1, 64, strict=False, physical_shape=(4, 64))
    router = compiler.alloc(
        "router", 128, 64, strict=False, physical_shape=(128, 64)
    )
    weights = tuple(
        compiler.input(name, (64, 64))
        for name in ("expert_gate", "expert_up", "expert_down")
    )

    with pytest.raises(ValueError, match="128 addressable"):
        compiler.qwen3_moe_decode_layer_v0(
            x,
            router,
            weights,
            layer_index=0,
            batch_size=1,
            weight_table_bases=tuple(weight.hbm_addr for weight in weights),
            weight_table_strides=tuple(weight.hbm_size for weight in weights),
            expert_table_counts=(8, 8, 8),
            route_f32_base=0,
            indices_int_base=0,
            constants=(None, None, None),
            zero_row=None,
            router_vector_format="BF16",
            hidden=64,
            intermediate=64,
        )


def test_layer_substrate_rejects_whole_batch_route_sram_overflow():
    compiler = PlenaCompiler(mlen=64, blen=4)
    x = compiler.alloc("x", 129, 64, strict=False, physical_shape=(132, 64))
    router = compiler.alloc(
        "router", 128, 64, strict=False, physical_shape=(128, 64)
    )
    weights = tuple(
        compiler.input(name, (64, 64))
        for name in ("expert_gate", "expert_up", "expert_down")
    )

    with pytest.raises(ValueError, match="exceed FP32 route SRAM"):
        compiler.qwen3_moe_decode_layer_v0(
            x,
            router,
            weights,
            layer_index=0,
            batch_size=129,
            weight_table_bases=tuple(weight.hbm_addr for weight in weights),
            weight_table_strides=tuple(weight.hbm_size for weight in weights),
            expert_table_counts=(128, 128, 128),
            route_f32_base=0,
            indices_int_base=0,
            constants=(None, None, None),
            zero_row=None,
            router_vector_format="BF16",
            hidden=64,
            intermediate=64,
        )


class _Norm(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = 1e-6


class _FusedExperts(nn.Module):
    def __init__(self, experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.randn(experts, 2 * intermediate, hidden)
        )
        self.down_proj = nn.Parameter(torch.randn(experts, hidden, intermediate))


def _tiny_fused_layer(hidden: int = 8, intermediate: int = 4, experts: int = 3):
    attention = nn.Module()
    attention.q_proj = nn.Linear(hidden, hidden, bias=False)
    attention.k_proj = nn.Linear(hidden, 4, bias=False)
    attention.v_proj = nn.Linear(hidden, 4, bias=False)
    attention.o_proj = nn.Linear(hidden, hidden, bias=False)
    attention.q_norm = _Norm(4)
    attention.k_norm = _Norm(4)
    mlp = nn.Module()
    mlp.gate = nn.Linear(hidden, experts, bias=False)
    mlp.experts = _FusedExperts(experts, hidden, intermediate)
    layer = nn.Module()
    layer.self_attn = attention
    layer.mlp = mlp
    layer.input_layernorm = _Norm(hidden)
    layer.post_attention_layernorm = _Norm(hidden)
    return layer


def test_fused_extractor_accounts_for_all_runtime_addressable_experts():
    config = ModelConfig(
        hidden_size=8,
        inter_dim=4,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        eps=1e-6,
        rope_theta=10_000.0,
        vocab_size=32,
        model_type="qwen3_moe",
        qk_norm=True,
        num_hidden_layers=1,
        dense_inter_dim=16,
        moe_inter_dim=4,
        num_experts=3,
        experts_per_token=2,
        norm_topk_prob=True,
    )
    weights = extract_layer_weights(_tiny_fused_layer(), config)

    assert isinstance(weights, MoeLayerWeights)
    assert weights.w_router_rows.shape == (3, 8)
    names = [name for name, _ in weights.tensor_entries(0)]
    for expert_id in range(3):
        assert f"W_expert_gate_0_e{expert_id}" in names
        assert f"W_expert_up_0_e{expert_id}" in names
        assert f"W_expert_down_0_e{expert_id}" in names


def test_old_modulelist_expert_abi_is_rejected_instead_of_lowered_dense():
    config = replace(_target_config(), hidden_size=8, num_heads=2, num_kv_heads=1, head_dim=4)
    layer = _tiny_fused_layer()
    del layer.mlp.experts.gate_up_proj
    del layer.mlp.experts.down_proj
    layer.mlp.experts = nn.ModuleList([nn.Linear(8, 8, bias=False)])

    with pytest.raises(ValueError, match=r"Transformers 5\.5\.0 fused"):
        extract_layer_weights(layer, config)


def test_official_shape_config_extracts_qk_norm_and_no_shared_expert_assumption():
    config = extract_model_config(
        SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=2048,
                intermediate_size=6144,
                moe_intermediate_size=768,
                num_attention_heads=32,
                num_key_value_heads=4,
                head_dim=128,
                rms_norm_eps=1e-6,
                rope_theta=10_000_000.0,
                vocab_size=151936,
                model_type="qwen3_moe",
                num_hidden_layers=48,
                num_experts=128,
                num_experts_per_tok=8,
                norm_topk_prob=True,
                decoder_sparse_step=1,
                mlp_only_layers=[],
            )
        )
    )

    assert config == _target_config()
    plan = build_qwen3_30b_a3b_decode_plan(
        config,
        layer_count=48,
        batch_size=1,
        key_precision="MXINT4",
        value_precision="MXINT4",
        shared_expert_count=0,
    )
    assert all(layer.as_dict()["shared_experts"] == 0 for layer in plan.layers)


def test_router_precision_contract_is_independent_but_not_plena_executable():
    fp6 = QWEN3_MOE_ROUTER_PRECISION.bind_expert_vector_precision("FP_E3M2")
    bf16 = QWEN3_MOE_ROUTER_PRECISION.bind_expert_vector_precision("BF16")

    assert fp6["router_precision_contract_hash"] == bf16[
        "router_precision_contract_hash"
    ]
    assert fp6["semantically_independent"] is True
    assert fp6["plena_per_stage_switch_valid"] is False
    assert fp6["publication_rankable"] is False

    with pytest.raises(ValueError, match="sealed Qwen3-MoE router"):
        _build_plan(
            router_precision_contract=replace(
                RouterPrecisionContract(), softmax_dtype="BF16"
            )
        )


def test_cpu_router_oracle_matches_bf16_linear_fp32_softmax_and_renorm():
    generator = torch.Generator().manual_seed(20260725)
    hidden = torch.randn(2, 3, 16, generator=generator)
    weight = torch.randn(128, 16, generator=generator)

    result = qwen3_moe_router_cpu_reference(hidden, weight)
    raw = torch.nn.functional.linear(hidden.reshape(-1, 16).bfloat16(), weight.bfloat16())
    probabilities = torch.softmax(raw, dtype=torch.float32, dim=-1)
    scores, indices = torch.topk(probabilities, 8, dim=-1)
    scores = scores / scores.sum(dim=-1, keepdim=True)

    assert result.raw_logits.dtype == torch.bfloat16
    assert result.probabilities.dtype == torch.float32
    assert result.route_scores.dtype == torch.float32
    assert result.expert_indices.dtype == torch.int64
    assert torch.equal(result.raw_logits, raw)
    assert torch.equal(result.probabilities, probabilities)
    assert torch.equal(result.route_scores, scores)
    assert torch.equal(result.expert_indices, indices)
    assert torch.allclose(
        result.route_scores.sum(dim=-1), torch.ones(6), atol=1e-6, rtol=0
    )
    receipt = result.transaction_receipt()
    assert receipt["expected_assignments"] == 6 * 8
    assert receipt["actual_assignments"] == 6 * 8
    assert receipt["assignment_conserved"] is True
    assert receipt["cpu_reference_valid"] is True
    assert receipt["emulator_equivalence_valid"] is False
    assert receipt["publication_rankable"] is False


def test_cpu_router_oracle_matches_transformers_5_5_abi_when_available():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("sealed comparison requires transformers==5.5.0")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig(
        hidden_size=16,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )
    router = Qwen3MoeTopKRouter(config).to(dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(20260726)
    hidden = torch.randn(7, 16, generator=generator).bfloat16()
    weight = torch.randn(128, 16, generator=generator).bfloat16()
    with torch.no_grad():
        router.weight.copy_(weight)
        probabilities, scores, indices = router(hidden)
    result = qwen3_moe_router_cpu_reference(hidden, weight)

    assert torch.equal(result.probabilities, probabilities)
    assert torch.equal(result.route_scores, scores)
    assert torch.equal(result.expert_indices, indices)


@pytest.mark.parametrize(
    ("hidden_size", "expected_hash"),
    [(64, 0xB678_A0D7_FE63_550A), (2048, 0x81E3_82FF_292F_FD8E)],
)
def test_router_linear_fixtures_pin_all_transformers_bf16_logits(
    hidden_size, expected_hash
):
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("sealed comparison requires transformers==5.5.0")
    indices = torch.arange(hidden_size, dtype=torch.int64)
    experts = torch.arange(128, dtype=torch.int64)[:, None]
    hidden = (
        (((indices * 37 + 11) % 2001) - 1000).to(torch.float32) / 257.0
    ).to(torch.bfloat16)
    weights = (
        (((experts * 97 + indices[None, :] * 53 + 19) % 2001) - 1000).to(
            torch.float32
        )
        / 263.0
    ).to(torch.bfloat16)

    reference = torch.nn.functional.linear(hidden, weights)
    oracle = qwen3_moe_router_cpu_reference(hidden[None, :], weights)
    assert torch.equal(oracle.raw_logits[0], reference)
    hash_value = 0xCBF2_9CE4_8422_2325
    for value in reference.view(torch.int16).tolist():
        bits = value & 0xFFFF
        for byte in (bits & 0xFF, bits >> 8):
            hash_value ^= byte
            hash_value = (hash_value * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    assert hash_value == expected_hash
    if hidden_size == 64:
        assert oracle.expert_indices.tolist() == [
            [53, 115, 12, 33, 95, 74, 32, 94]
        ]
        assert [
            value & 0xFFFF_FFFF
            for value in oracle.route_scores.contiguous()
            .view(torch.int32)
            .flatten()
            .tolist()
        ] == [
            0x3F5D_E77B,
            0x3DF0_4073,
            0x3C82_0EE4,
            0x2E0F_EB3D,
            0x2D00_735A,
            0x2730_7F89,
            0x24AA_8D95,
            0x23FA_F8CC,
        ]


def test_transformers_5_5_pins_emulator_topk_fixture_bits():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("sealed comparison requires transformers==5.5.0")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeTopKRouter,
    )

    config = Qwen3MoeConfig(
        hidden_size=128,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
    )
    router = Qwen3MoeTopKRouter(config).to(dtype=torch.bfloat16)
    hidden = torch.full((1, 128), -100.0, dtype=torch.bfloat16)
    for expert_id, logit in {
        2: 7.0,
        5: 7.0,
        63: 3.0,
        64: 4.0,
        71: 5.0,
        84: 2.5,
        85: 2.25,
        127: 6.0,
    }.items():
        hidden[0, expert_id] = logit
    with torch.no_grad():
        router.weight.copy_(torch.eye(128, dtype=torch.bfloat16))
        _, scores, indices = router(hidden)

    assert indices.tolist() == [[2, 5, 127, 71, 64, 63, 84, 85]]
    assert [
        value & 0xFFFF_FFFF
        for value in scores.contiguous().view(torch.int32).flatten().tolist()
    ] == [
        0x3EC5_99E4,
        0x3EC5_99E4,
        0x3E11_6305,
        0x3D55_F072,
        0x3C9D_685F,
        0x3BE7_A0D4,
        0x3B8C_7D58,
        0x3B5A_D3AD,
    ]


def test_cpu_routed_moe_oracle_matches_transformers_5_5_fused_experts():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("sealed comparison requires transformers==5.5.0")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeSparseMoeBlock,
    )

    config = Qwen3MoeConfig(
        hidden_size=16,
        moe_intermediate_size=12,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    block = Qwen3MoeSparseMoeBlock(config).to(dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(20260727)
    hidden = torch.randn(2, 3, 16, generator=generator).bfloat16()
    router_weight = torch.randn(128, 16, generator=generator).bfloat16()
    gate_up = torch.randn(128, 24, 16, generator=generator).bfloat16()
    down = torch.randn(128, 16, 12, generator=generator).bfloat16()
    with torch.no_grad():
        block.gate.weight.copy_(router_weight)
        block.experts.gate_up_proj.copy_(gate_up)
        block.experts.down_proj.copy_(down)
        expected = block(hidden)
    result = qwen3_moe_cpu_reference(hidden, router_weight, gate_up, down)

    assert torch.equal(result.output, expected)
    receipt = result.transaction_receipt()
    assert receipt["expected_assignments"] == 2 * 3 * 8
    assert receipt["actual_assignments"] == 2 * 3 * 8
    assert receipt["assignment_conserved"] is True
    assert receipt["cpu_routed_moe_reference_valid"] is True
    assert receipt["emulator_equivalence_valid"] is False
    assert receipt["publication_rankable"] is False


def _exact_moe_tail_fixture():
    hidden = intermediate = 64
    experts = 128
    hidden_index = torch.arange(hidden, dtype=torch.int64)
    expert_index = torch.arange(experts, dtype=torch.int64)[:, None, None]
    output_index = torch.arange(intermediate, dtype=torch.int64)[None, :, None]
    input_index = hidden_index[None, None, :]
    attention = (
        ((((hidden_index * 17 + 3) % 127) - 63).float() / 41)
        .to(torch.bfloat16)
        .reshape(1, hidden)
    )
    residual = (
        ((((hidden_index * 29 + 7) % 131) - 65).float() / 43)
        .to(torch.bfloat16)
        .reshape(1, hidden)
    )
    norm_weight = (
        0.75 + ((hidden_index * 11) % 17).float() / 64
    ).to(torch.bfloat16)
    router_weight = (
        (
            (
                torch.arange(experts, dtype=torch.int64)[:, None] * 97
                + hidden_index[None, :] * 53
                + 19
            )
            % 2001
            - 1000
        ).float()
        / 263
    ).to(torch.bfloat16)
    gate = (
        (
            (
                expert_index * 13
                + output_index * 17
                + input_index * 19
                + 5
            )
            % 257
            - 128
        ).float()
        / 509
    ).to(torch.bfloat16)
    up = (
        (
            (
                expert_index * 23
                + output_index * 29
                + input_index * 31
                + 7
            )
            % 257
            - 128
        ).float()
        / 521
    ).to(torch.bfloat16)
    down = (
        (
            (
                expert_index * 37
                + hidden_index[None, :, None] * 41
                + torch.arange(intermediate, dtype=torch.int64)[None, None, :] * 43
                + 11
            )
            % 257
            - 128
        ).float()
        / 523
    ).to(torch.bfloat16)
    return {
        "attention": attention,
        "residual": residual,
        "norm_weight": norm_weight,
        "router_weight": router_weight,
        "fused_gate_up": torch.cat((gate, up), dim=1),
        "down": down,
    }


def _fnv1a_tensor_bytes(tensor: torch.Tensor) -> int:
    value = 0xCBF2_9CE4_8422_2325
    for byte in tensor.contiguous().view(torch.uint8).flatten().tolist():
        value ^= byte
        value = (value * 0x0000_0100_0000_01B3) & 0xFFFF_FFFF_FFFF_FFFF
    return value


def test_post_attention_moe_tail_matches_transformers_5_5_exact_bits():
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "5.5.0":
        pytest.skip("sealed comparison requires transformers==5.5.0")
    from transformers.models.qwen3_moe.configuration_qwen3_moe import (
        Qwen3MoeConfig,
    )
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeRMSNorm,
        Qwen3MoeSparseMoeBlock,
    )

    fixture = _exact_moe_tail_fixture()
    config = Qwen3MoeConfig(
        hidden_size=64,
        moe_intermediate_size=64,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    norm = Qwen3MoeRMSNorm(64, eps=1.0e-6).to(torch.bfloat16)
    block = Qwen3MoeSparseMoeBlock(config).to(torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(fixture["norm_weight"])
        block.gate.weight.copy_(fixture["router_weight"])
        block.experts.gate_up_proj.copy_(fixture["fused_gate_up"])
        block.experts.down_proj.copy_(fixture["down"])
        attention_residual = fixture["attention"] + fixture["residual"]
        expected_normalized = norm(attention_residual)
        expected = attention_residual + block(
            expected_normalized.unsqueeze(1)
        ).squeeze(1)

    result = qwen3_post_attention_moe_cpu_reference(
        fixture["attention"],
        fixture["residual"],
        fixture["norm_weight"],
        fixture["router_weight"],
        fixture["fused_gate_up"],
        fixture["down"],
    )
    assert torch.equal(result.normalized, expected_normalized)
    assert torch.equal(result.output, expected)
    assert _fnv1a_tensor_bytes(result.normalized) == 0x07EB_BAB0_6EDC_1369
    assert _fnv1a_tensor_bytes(result.output) == 0x910C_569B_727D_0EFC
    assert result.moe.router.expert_indices.tolist() == [
        [29, 99, 33, 95, 8, 70, 91, 50]
    ]
    receipt = result.transaction_receipt()
    assert receipt["packedkv_attention_prefix_executed"] is False
    assert receipt["emulator_transaction_parity"] is False


def test_raw_bf16_expert_layout_and_image_are_byte_exact_and_fail_closed():
    fixture = _exact_moe_tail_fixture()
    layout = Qwen3RawBf16ExpertBankLayout.canonical(
        0, hidden=64, intermediate=64
    )
    assert layout.gate_up_base == 64
    assert layout.gate_up_stride_bytes == 16_384
    assert layout.down_base == 2_097_216
    assert layout.down_stride_bytes == 8_192
    assert layout.end_address == 3_145_792
    image = pack_qwen3_raw_bf16_expert_hbm(
        layout, fixture["fused_gate_up"], fixture["down"]
    )
    assert len(image) == layout.total_bytes
    assert image[:8] == b"Q3MOEBF1"
    assert image[:64] == layout.descriptor_bytes()
    assert image[64:66] == fixture["fused_gate_up"].view(torch.uint8).flatten()[:2].numpy().tobytes()

    with pytest.raises(ValueError, match="64-byte aligned"):
        Qwen3RawBf16ExpertBankLayout.canonical(
            1, hidden=64, intermediate=64
        )
    target = Qwen3RawBf16ExpertBankLayout.canonical(
        0, hidden=2048, intermediate=768
    )
    assert target.total_bytes == 1_207_959_616
    with pytest.raises(ValueError, match="materialization limit"):
        pack_qwen3_raw_bf16_expert_hbm(
            target,
            torch.empty(128, 1536, 2048, device="meta"),
            torch.empty(128, 2048, 768, device="meta"),
        )


def test_moe_tail_provenance_pins_executable_dispatch_boundary():
    receipt = qwen3_moe_tail_isa_provenance_receipt()

    assert receipt["abi"] == (
        "V_QWEN3_RMSNORM_BF16@0x3a+V_ROUTER_LINEAR_BF16@0x36+"
        "V_TOPK@0x37+V_MUL_ROUTE_F32@0x38+"
        "V_QWEN3_EXPERT_COMBINE_BF16@0x39"
    )
    assert receipt["tiny_hbm_bytes_read"] == 196_672
    assert receipt["transformers_5_5_cpu_fixture_parity_valid"] is True
    assert receipt["emulator_dispatch_transaction_parity_valid"] is True
    assert receipt["compiler_generated_binary_emulator_parity_valid"] is False
    assert receipt["packedkv_attention_prefix_executed"] is False
    assert receipt["timing_calibrated"] is False
    assert receipt["rtl_valid"] is False
    assert receipt["publication_rankable"] is False


def test_tiny_single_layer_provenance_is_true_only_for_validation_geometry():
    receipt = qwen3_tiny_single_layer_transaction_provenance_receipt()

    assert receipt["scope"] == "hidden64_q_len1_validation_only"
    assert receipt["geometry"] == {
        "batch_size": 1,
        "q_len": 1,
        "hidden": 64,
        "head_dim": 64,
        "kv_heads": 1,
        "intermediate": 64,
        "experts": 128,
        "top_k": 8,
        "cache_position": 3,
    }
    assert receipt["packedkv_attention_prefix_executed"] is True
    assert receipt["compiler_generated_binary_emulator_parity_valid"] is True
    assert receipt["all_bf16_boundaries_byte_exact"] is True
    assert receipt["route_scores_fp32_byte_exact"] is True
    assert receipt["kv_append_and_untouched_byte_conservation_verified"] is True
    assert receipt["target_geometry_valid"] is False
    assert receipt["full_model_compiler_valid"] is False
    assert receipt["emulator_target_valid"] is False
    assert receipt["rtl_valid"] is False
    assert receipt["timing_calibrated"] is False
    assert receipt["publication_rankable"] is False
    assert len(receipt["contract_sha256"]) == 64


def test_compiler_lowers_one_exact_post_attention_moe_tail_with_layout_receipt():
    compiler = PlenaCompiler(mlen=64, blen=4)
    attention = compiler.alloc(
        "packedkv_attention_output", 1, 64, strict=False, physical_shape=(4, 64)
    )
    residual = compiler.alloc(
        "layer_residual", 1, 64, strict=False, physical_shape=(4, 64)
    )
    norm_weight = compiler.alloc(
        "post_attention_norm_weight",
        1,
        64,
        strict=False,
        physical_shape=(4, 64),
    )
    router = compiler.alloc(
        "router_weight", 128, 64, strict=False, physical_shape=(128, 64)
    )
    layout = Qwen3RawBf16ExpertBankLayout.canonical(
        0x4000, hidden=64, intermediate=64
    )
    output, receipt = compiler.qwen3_post_attention_moe_tail_transaction_v0(
        attention,
        residual,
        norm_weight,
        router,
        descriptor_layout=layout,
        layer_index=0,
    )
    assembly = compiler.compile()

    assert output.shape == (1, 64)
    instruction_lines = assembly.splitlines()
    assert sum(line.startswith("V_QWEN3_RMSNORM_BF16") for line in instruction_lines) == 1
    assert sum(line.startswith("V_ROUTER_LINEAR_BF16") for line in instruction_lines) == 1
    assert sum(line.startswith("V_TOPK") for line in instruction_lines) == 1
    assert sum(
        line.startswith("V_QWEN3_EXPERT_COMBINE_BF16")
        for line in instruction_lines
    ) == 1
    assert assembly.index("V_QWEN3_RMSNORM_BF16") < assembly.index(
        "V_ROUTER_LINEAR_BF16"
    ) < assembly.index("V_TOPK") < assembly.index(
        "V_QWEN3_EXPERT_COMBINE_BF16"
    )
    assert receipt["packedkv_attention_prefix_executed"] is False
    assert receipt["compiler_lowering_valid"] is True
    assert receipt["emulator_transaction_parity"] is False
    assert receipt["expert_hbm"]["total_bytes"] == 3_145_792
    assert receipt["vram"]["output"]["dtype"] == "BF16"


class _ShapeOnly:
    def __init__(self, *shape: int):
        self.shape = shape


def _shape_linear(*shape: int):
    return SimpleNamespace(weight=_ShapeOnly(*shape))


def _shape_only_target_model():
    attention = SimpleNamespace(
        q_proj=_shape_linear(4096, 2048),
        o_proj=_shape_linear(2048, 4096),
        k_proj=_shape_linear(512, 2048),
        v_proj=_shape_linear(512, 2048),
        q_norm=_shape_linear(128),
        k_norm=_shape_linear(128),
    )
    mlp = SimpleNamespace(
        gate=_shape_linear(128, 2048),
        experts=SimpleNamespace(
            gate_up_proj=_ShapeOnly(128, 1536, 2048),
            down_proj=_ShapeOnly(128, 2048, 768),
        ),
    )
    layer = SimpleNamespace(
        self_attn=attention,
        mlp=mlp,
        input_layernorm=_shape_linear(2048),
        post_attention_layernorm=_shape_linear(2048),
    )
    config = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        moe_intermediate_size=768,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=10_000_000.0,
        vocab_size=151936,
        model_type="qwen3_moe",
        num_hidden_layers=48,
        num_experts=128,
        num_experts_per_tok=8,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        hidden_act="silu",
    )
    return SimpleNamespace(
        config=config, model=SimpleNamespace(layers=[layer] * 48)
    )


def test_full_model_frontend_audit_covers_all_fused_layers_but_stays_blocked():
    receipt = audit_qwen3_30b_a3b_full_model_frontend(
        _shape_only_target_model(),
        model_id=QWEN3_MOE_MODEL_ID,
        revision=QWEN3_MOE_MODEL_REVISION,
    ).as_dict()

    assert receipt["audited_layer_count"] == 48
    assert receipt["fused_expert_layer_count"] == 48
    assert receipt["all_experts_addressable_per_layer"] == 128
    assert receipt["weights_materialized_by_audit"] is False
    assert receipt["dense_fallback_forbidden"] is True
    assert receipt["full_layer_dataflow_contract_sha256"] == (
        _build_plan(batch_size=1).full_frontend_contract_receipt()[
            "contract_sha256"
        ]
    )
    assert receipt["emulator_fp32_route_path_available"] is True
    assert receipt["route_isa"]["route_sram_dtype"] == "FP32"
    assert (
        receipt["router_linear"][
            "transformers_5_5_cpu_fixture_parity_valid"
        ]
        is True
    )
    assert receipt["compiler_pipeline_valid"] is False
    assert receipt["runtime_routing_dataflow_wired"] is False
    assert receipt["publication_rankable"] is False


def test_native_frontend_rejects_qwen_moe_instead_of_using_dense_fallback():
    with pytest.raises(NotImplementedError, match="dense FFN fallback is forbidden"):
        compile_native_hf_decoder(
            _shape_only_target_model(),
            seq_len=1,
            batch_size=1,
            num_layers=48,
            trace_only=True,
        )


def test_full_model_frontend_audit_rejects_wrong_revision_and_fused_shape():
    model = _shape_only_target_model()
    with pytest.raises(ValueError, match="sealed target revision"):
        audit_qwen3_30b_a3b_full_model_frontend(
            model,
            model_id=QWEN3_MOE_MODEL_ID,
            revision="main",
        )

    model.model.layers[0].mlp.experts.down_proj = _ShapeOnly(128, 2048, 769)
    with pytest.raises(ValueError, match="layer 0 down_proj"):
        audit_qwen3_30b_a3b_full_model_frontend(
            model,
            model_id=QWEN3_MOE_MODEL_ID,
            revision=QWEN3_MOE_MODEL_REVISION,
        )
