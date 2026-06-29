"""Unit tests for GPT-OSS MoE Golden A/B helpers.

Run:
    PYTHONPATH=. python3 aten/tests/test_gpt_oss_moe_reference.py
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS_DIR)))
for _path in [
    os.path.join(_REPO_ROOT, "PLENA_Tools"),
    os.path.join(_REPO_ROOT, "PLENA_Simulator", "PLENA_Tools"),
]:
    if os.path.isdir(_path) and _path not in sys.path:
        sys.path.insert(0, _path)

from aten.models.gpt_oss.moe_reference import (
    assert_clamp_inactive,
    clamp_stats,
    gpt_oss_moe_fixed_routing_host_smoke,
    gpt_oss_moe_golden_a,
    gpt_oss_moe_golden_b_plena_mxfp8,
    gpt_oss_swiglu,
    split_packed_gate_up,
)


def _tiny_inputs(scale: float = 0.05):
    torch.manual_seed(7)
    tokens, hidden, experts, inter = 3, 8, 5, 6
    x = torch.randn(tokens, hidden) * scale
    router_w = torch.randn(hidden, experts) * scale
    router_b = torch.randn(experts) * scale
    gate_up_w = torch.randn(experts, hidden, 2 * inter) * scale
    gate_up_b = torch.randn(experts, 2 * inter) * scale
    down_w = torch.randn(experts, inter, hidden) * scale
    down_b = torch.randn(experts, hidden) * scale
    return x, router_w, router_b, gate_up_w, gate_up_b, down_w, down_b


def _load_hf_gpt_oss_mlp():
    # Some local torch/torchvision builds fail while importing transformers
    # because torchvision registers a fake nms kernel before defining its op.
    # Defining only the schema is enough for this CPU-only GPT-OSS MLP test.
    try:
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except Exception:
        pass

    try:
        from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
    except Exception as exc:
        pytest.skip(f"HF GPT-OSS MLP import unavailable in this environment: {exc}")
    return GptOssConfig, GptOssMLP


def test_bf16_represents_gpt_oss_clamp_boundary_exactly():
    seven = torch.tensor([7.0], dtype=torch.bfloat16).float()
    minus_seven = torch.tensor([-7.0], dtype=torch.bfloat16).float()
    assert seven.item() == 7.0
    assert minus_seven.item() == -7.0


def test_gpt_oss_swiglu_matches_manual_formula():
    gate_up = torch.tensor([[[8.0, -8.0, -2.0, 3.0]]])
    out = gpt_oss_swiglu(gate_up, bf16_intermediates=False)

    gate = torch.tensor([[[7.0, -2.0]]])
    up = torch.tensor([[[-7.0, 3.0]]])
    expected = (up + 1.0) * gate * torch.sigmoid(1.702 * gate)
    assert torch.allclose(out, expected, atol=1e-6)


def test_split_packed_gate_up_uses_even_odd_weight_and_bias():
    weight = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    bias = torch.arange(2 * 8, dtype=torch.float32).reshape(2, 8)

    split = split_packed_gate_up(weight, bias)

    assert torch.equal(split.gate_weight, weight[..., 0::2])
    assert torch.equal(split.up_weight, weight[..., 1::2])
    assert torch.equal(split.gate_bias, bias[..., 0::2])
    assert torch.equal(split.up_bias, bias[..., 1::2])
    assert not torch.equal(split.gate_weight, weight[..., :4])
    assert not torch.equal(split.up_weight, weight[..., 4:])


def test_split_packed_gate_up_reconstructs_hf_projection_lanes():
    torch.manual_seed(11)
    tokens, hidden, inter = 4, 5, 7
    # One-hot rows make this an exact lane-selection check rather than a
    # floating-point matmul associativity test.
    x = torch.eye(hidden, dtype=torch.float32)[:tokens]
    packed_weight = torch.randn(hidden, 2 * inter)
    packed_bias = torch.randn(2 * inter)

    packed = x @ packed_weight + packed_bias
    split = split_packed_gate_up(packed_weight, packed_bias)
    gate = x @ split.gate_weight + split.gate_bias
    up = x @ split.up_weight + split.up_bias

    assert torch.allclose(gate, packed[..., 0::2], atol=0.0, rtol=0.0)
    assert torch.allclose(up, packed[..., 1::2], atol=0.0, rtol=0.0)


def test_golden_a_matches_hf_gpt_oss_mlp_tiny():
    GptOssConfig, GptOssMLP = _load_hf_gpt_oss_mlp()
    args = _tiny_inputs(scale=0.04)
    x, router_w, router_b, gate_up_w, gate_up_b, down_w, down_b = args

    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=gate_up_w.shape[0],
        hidden_size=x.shape[1],
        intermediate_size=down_w.shape[1],
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=x.shape[1],
        num_experts_per_tok=2,
        vocab_size=16,
    )
    hf_mlp = GptOssMLP(config).eval()
    with torch.no_grad():
        hf_mlp.router.weight.copy_(router_w.T.contiguous())
        hf_mlp.router.bias.copy_(router_b)
        hf_mlp.experts.gate_up_proj.copy_(gate_up_w)
        hf_mlp.experts.gate_up_proj_bias.copy_(gate_up_b)
        hf_mlp.experts.down_proj.copy_(down_w)
        hf_mlp.experts.down_proj_bias.copy_(down_b)

        hf_router_logits = torch.nn.functional.linear(x, hf_mlp.router.weight, hf_mlp.router.bias)
        hf_scores, hf_indices = hf_mlp.router(x)
        hf_selected_scores = hf_scores.gather(1, hf_indices)
        hf_expert_out = hf_mlp.experts(x.unsqueeze(0), hf_indices, hf_scores).squeeze(0)
        hf_mlp_out, hf_mlp_scores = hf_mlp(x.unsqueeze(0))
        hf_mlp_selected_scores = hf_mlp_scores.gather(1, hf_indices)

    golden = gpt_oss_moe_golden_a(
        *args,
        experts_per_token=2,
        bf16_intermediates=False,
    )

    assert torch.equal(golden.topk_indices, hf_indices)
    assert torch.allclose(golden.router_logits, hf_router_logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(golden.topk_weights, hf_selected_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(golden.topk_weights, hf_mlp_selected_scores, atol=1e-6, rtol=1e-6)
    assert torch.allclose(golden.output, hf_expert_out, atol=1e-6, rtol=1e-6)
    assert torch.allclose(golden.output, hf_mlp_out.squeeze(0), atol=1e-6, rtol=1e-6)


def test_clamp_inactive_smoke_precondition_and_unclamped_match():
    args = _tiny_inputs(scale=0.02)
    result = gpt_oss_moe_golden_a(*args, experts_per_token=2, bf16_intermediates=True)
    stats = assert_clamp_inactive(result.gate_up_preact)

    unclamped = gpt_oss_moe_golden_a(
        *args,
        experts_per_token=2,
        bf16_intermediates=True,
        apply_clamp=False,
    )
    assert stats.inactive
    assert torch.equal(result.topk_indices, unclamped.topk_indices)
    assert torch.allclose(result.output.float(), unclamped.output.float(), atol=0.0, rtol=0.0)


def test_fixed_routing_wiring_smoke_matches_golden_a_without_clamp_or_quantization():
    args = _tiny_inputs(scale=0.02)
    golden = gpt_oss_moe_golden_a(
        *args,
        experts_per_token=2,
        bf16_intermediates=False,
        apply_clamp=True,
    )
    assert_clamp_inactive(golden.gate_up_preact)
    x, _, _, gate_up_w, gate_up_b, down_w, down_b = args

    smoke = gpt_oss_moe_fixed_routing_host_smoke(
        x,
        golden.topk_indices,
        golden.topk_weights,
        gate_up_w,
        gate_up_b,
        down_w,
        down_b,
        bf16_intermediates=False,
        apply_clamp=False,
    )

    assert torch.allclose(smoke.output, golden.output, atol=1e-6, rtol=1e-6)


def test_fixed_routing_clamp_active_smoke_matches_golden_a_before_quantization():
    args = list(_tiny_inputs(scale=0.04))
    # Force an active clamp site without changing routing.  Golden A and the
    # fixed-routing smoke should still agree when both apply exact clamp.
    args[1] = torch.zeros_like(args[1])
    args[2] = torch.tensor([10.0, 9.0, 0.0, 0.0, 0.0])
    args[3] = args[3].clone()
    args[4] = args[4].clone()
    args[3][0, :, 0] = 0.0
    args[4][0, 0] = 8.0
    args[3][0, :, 1] = 0.0
    args[4][0, 1] = -8.0
    args = tuple(args)

    golden = gpt_oss_moe_golden_a(
        *args,
        experts_per_token=2,
        bf16_intermediates=False,
        apply_clamp=True,
    )
    stats = clamp_stats(golden.gate_up_preact)
    assert not stats.inactive
    x, _, _, gate_up_w, gate_up_b, down_w, down_b = args

    smoke = gpt_oss_moe_fixed_routing_host_smoke(
        x,
        golden.topk_indices,
        golden.topk_weights,
        gate_up_w,
        gate_up_b,
        down_w,
        down_b,
        bf16_intermediates=False,
        apply_clamp=True,
    )

    assert torch.allclose(smoke.output, golden.output, atol=1e-6, rtol=1e-6)


def test_fixed_routing_quantized_smoke_matches_golden_b():
    args = _tiny_inputs(scale=0.08)
    golden_a = gpt_oss_moe_golden_a(
        *args,
        experts_per_token=2,
        bf16_intermediates=True,
        apply_clamp=True,
    )
    x, _, _, gate_up_w, gate_up_b, down_w, down_b = args
    golden_b = gpt_oss_moe_golden_b_plena_mxfp8(
        x,
        golden_a.topk_indices,
        golden_a.topk_weights,
        gate_up_w,
        gate_up_b,
        down_w,
        down_b,
        bf16_intermediates=True,
        apply_clamp=True,
    )

    smoke = gpt_oss_moe_fixed_routing_host_smoke(
        x,
        golden_a.topk_indices,
        golden_a.topk_weights,
        gate_up_w,
        gate_up_b,
        down_w,
        down_b,
        quantize_expert_weights_to_plena_mxfp8=True,
        bf16_intermediates=True,
        apply_clamp=True,
    )

    assert torch.allclose(smoke.output.float(), golden_b.output.float(), atol=0.0, rtol=0.0)


def test_clamp_inactive_check_fails_when_gate_or_up_hits_limit():
    preact = torch.tensor([[[8.0, 0.0, 0.0, -8.0]]])
    try:
        assert_clamp_inactive(preact)
    except AssertionError as exc:
        assert "clamp would be active" in str(exc)
    else:
        raise AssertionError("assert_clamp_inactive should reject active clamp inputs")


def test_golden_b_uses_fixed_routing_and_plena_mxfp8_expert_weights():
    args = _tiny_inputs(scale=0.08)
    a = gpt_oss_moe_golden_a(*args, experts_per_token=2, bf16_intermediates=True)
    x, _, _, gate_up_w, gate_up_b, down_w, down_b = args

    b = gpt_oss_moe_golden_b_plena_mxfp8(
        x,
        a.topk_indices,
        a.topk_weights,
        gate_up_w,
        gate_up_b,
        down_w,
        down_b,
        bf16_intermediates=True,
    )

    assert torch.equal(a.topk_indices, b.topk_indices)
    assert b.output.shape == a.output.shape
    assert b.router_logits.numel() == 0
    assert not torch.equal(a.output, b.output), "MXFP8 expert quantization should affect this seeded case"


if __name__ == "__main__":
    test_bf16_represents_gpt_oss_clamp_boundary_exactly()
    test_gpt_oss_swiglu_matches_manual_formula()
    test_golden_a_matches_hf_gpt_oss_mlp_tiny()
    test_clamp_inactive_smoke_precondition_and_unclamped_match()
    test_fixed_routing_wiring_smoke_matches_golden_a_without_clamp_or_quantization()
    test_fixed_routing_clamp_active_smoke_matches_golden_a_before_quantization()
    test_fixed_routing_quantized_smoke_matches_golden_b()
    test_clamp_inactive_check_fails_when_gate_or_up_hits_limit()
    test_golden_b_uses_fixed_routing_and_plena_mxfp8_expert_weights()
    print("PASS test_gpt_oss_moe_reference")
