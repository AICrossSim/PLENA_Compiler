"""GPT-OSS MoE reference helpers for PLENA bring-up.

These helpers intentionally separate the GPT-OSS mathematical contract from
PLENA's current HBM precision.  Golden A uses high-precision weights and fixed
GPT-OSS semantics.  Golden B reuses Golden A's routing decision and applies the
current PLENA MXFP8 weight model to expert weights only.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GptOssMoeResult:
    output: torch.Tensor
    router_logits: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor
    gate_up_preact: torch.Tensor


@dataclass(frozen=True)
class ClampStats:
    gate_min: float
    gate_max: float
    up_min: float
    up_max: float
    inactive: bool


@dataclass(frozen=True)
class SplitGateUp:
    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    gate_bias: torch.Tensor | None
    up_bias: torch.Tensor | None


@dataclass(frozen=True)
class CompareStats:
    rel_rms: float
    atol: float
    rtol: float
    allclose: bool
    pass_rate: float
    max_abs_error: float


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _maybe_bf16(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    return _bf16(x) if enabled else x.float()


def quantize_to_plena_mxfp8(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize/dequantize through PLENA's current HBM MXFP8 format."""
    from plena_quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


def split_packed_gate_up(
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor | None = None,
) -> SplitGateUp:
    """Split HF/OpenAI GPT-OSS packed gate/up tensors into PLENA tensors.

    HF stores both expert projections in one interleaved tensor:

      gate_up_proj[..., 0::2] -> gate
      gate_up_proj[..., 1::2] -> up

    The packed bias uses the same convention.  Do not split this tensor into
    contiguous halves; doing so is shape-compatible but numerically wrong.
    """
    if gate_up_weight.shape[-1] % 2 != 0:
        raise ValueError(f"gate_up_weight last dim must be even, got {gate_up_weight.shape[-1]}")
    if gate_up_bias is not None and gate_up_bias.shape[-1] != gate_up_weight.shape[-1]:
        raise ValueError(
            "gate_up_bias last dim must match packed gate_up_weight last dim "
            f"({gate_up_bias.shape[-1]} != {gate_up_weight.shape[-1]})"
        )

    return SplitGateUp(
        gate_weight=gate_up_weight[..., 0::2].contiguous(),
        up_weight=gate_up_weight[..., 1::2].contiguous(),
        gate_bias=None if gate_up_bias is None else gate_up_bias[..., 0::2].contiguous(),
        up_bias=None if gate_up_bias is None else gate_up_bias[..., 1::2].contiguous(),
    )


def compare_stats(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol_scale: float = 0.01,
) -> CompareStats:
    """Return GPT-OSS real-layer comparison metrics.

    ``rel_rms`` is ||actual-reference||_2 / ||reference||_2.  The elementwise
    check uses ``atol = atol_scale * std(reference)`` plus the caller-provided
    relative tolerance.
    """
    actual_f = actual.float()
    reference_f = reference.float()
    diff = actual_f - reference_f
    denom = torch.linalg.vector_norm(reference_f).clamp_min(1e-12)
    rel_rms = float((torch.linalg.vector_norm(diff) / denom).item())
    atol = float((reference_f.std(unbiased=False) * atol_scale).item())
    allowed = atol + rtol * reference_f.abs()
    passed = diff.abs() <= allowed
    return CompareStats(
        rel_rms=rel_rms,
        atol=atol,
        rtol=rtol,
        allclose=bool(passed.all().item()),
        pass_rate=float(passed.float().mean().item()),
        max_abs_error=float(diff.abs().max().item()),
    )


def assert_compare_within(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    max_rel_rms: float,
    rtol: float,
    atol_scale: float = 0.01,
) -> CompareStats:
    stats = compare_stats(actual, reference, rtol=rtol, atol_scale=atol_scale)
    if stats.rel_rms > max_rel_rms or not stats.allclose:
        raise AssertionError(
            f"{name} failed: rel_rms={stats.rel_rms:.6g} (limit {max_rel_rms:.6g}), "
            f"allclose={stats.allclose}, pass_rate={stats.pass_rate:.2%}, "
            f"atol={stats.atol:.6g}, rtol={stats.rtol:.6g}, max_abs={stats.max_abs_error:.6g}"
        )
    return stats


def assert_gap_in_band(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    name: str,
    min_rel_rms: float,
    max_rel_rms: float,
) -> CompareStats:
    stats = compare_stats(actual, reference, rtol=max_rel_rms)
    if not (min_rel_rms <= stats.rel_rms <= max_rel_rms):
        raise AssertionError(
            f"{name} gap outside expected band: rel_rms={stats.rel_rms:.6g}, "
            f"expected [{min_rel_rms:.6g}, {max_rel_rms:.6g}]"
        )
    return stats


def gpt_oss_swiglu(
    gate_up: torch.Tensor,
    *,
    limit: float = 7.0,
    alpha: float = 1.702,
    bf16_intermediates: bool = False,
    apply_clamp: bool = True,
) -> torch.Tensor:
    """GPT-OSS expert activation.

    The gate/up projection is interleaved in the final dimension:
    even lanes are gate, odd lanes are up.  GPT-OSS clamps gate only on the
    upper side and clamps up on both sides before applying the gated activation:

        gate * sigmoid(alpha * gate) * (up + 1)
    """
    gate = gate_up[..., ::2].float()
    up = gate_up[..., 1::2].float()

    if bf16_intermediates:
        gate = _bf16(gate)
        up = _bf16(up)

    if apply_clamp:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, min=-limit, max=limit)
        if bf16_intermediates:
            gate = _bf16(gate)
            up = _bf16(up)

    scaled_gate = _maybe_bf16(gate * alpha, bf16_intermediates)
    sigmoid = _maybe_bf16(torch.sigmoid(scaled_gate), bf16_intermediates)
    glu = _maybe_bf16(gate * sigmoid, bf16_intermediates)
    up_plus_one = _maybe_bf16(up + 1.0, bf16_intermediates)
    return _maybe_bf16(up_plus_one * glu, bf16_intermediates)


def clamp_stats(gate_up_preact: torch.Tensor, *, limit: float = 7.0) -> ClampStats:
    """Return whether GPT-OSS clamp would be a no-op for these preactivations."""
    gate = gate_up_preact[..., ::2].float()
    up = gate_up_preact[..., 1::2].float()
    gate_min = float(gate.min().item())
    gate_max = float(gate.max().item())
    up_min = float(up.min().item())
    up_max = float(up.max().item())
    return ClampStats(
        gate_min=gate_min,
        gate_max=gate_max,
        up_min=up_min,
        up_max=up_max,
        inactive=gate_max <= limit and up_min >= -limit and up_max <= limit,
    )


def assert_clamp_inactive(gate_up_preact: torch.Tensor, *, limit: float = 7.0) -> ClampStats:
    """Validate the route-A smoke-test precondition.

    Gate only has an upper clamp in GPT-OSS; up has a two-sided clamp.  A smoke
    test may skip clamp only when those exact clamp sites would be no-ops.
    """
    stats = clamp_stats(gate_up_preact, limit=limit)
    if not stats.inactive:
        raise AssertionError(
            "GPT-OSS clamp would be active: "
            f"gate range [{stats.gate_min:.6g}, {stats.gate_max:.6g}], "
            f"up range [{stats.up_min:.6g}, {stats.up_max:.6g}], limit={limit}"
        )
    return stats


def gpt_oss_moe_golden_a(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    router_bias: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None = None,
    *,
    experts_per_token: int = 4,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    bf16_intermediates: bool = True,
    apply_clamp: bool = True,
) -> GptOssMoeResult:
    """Pure GPT-OSS MoE math with high-precision weights.

    Shapes use PLENA-friendly row-major matrices:
      x:              [tokens, hidden]
      router_weight:  [hidden, experts]
      router_bias:    [experts]
      gate_up_weight: [experts, hidden, 2 * intermediate]
      gate_up_bias:   [experts, 2 * intermediate]
      down_weight:    [experts, intermediate, hidden]
      down_bias:      [experts, hidden] or None

    This function does not use HF's default MXFP4 execution path.  If weights
    come from a quantized checkpoint, callers must pass already-dequantized
    tensors when they want a high-precision Golden A.
    """
    x_ref = _maybe_bf16(x, bf16_intermediates)
    router_logits = x_ref.float() @ router_weight.float() + router_bias.float()
    topk_values, topk_indices = torch.topk(router_logits, k=experts_per_token, dim=-1)
    topk_weights = torch.softmax(topk_values.float(), dim=-1)
    if bf16_intermediates:
        topk_weights = _bf16(topk_weights)

    output, gate_up_preact = _run_selected_experts(
        x_ref,
        topk_indices,
        topk_weights,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        bf16_intermediates=bf16_intermediates,
        apply_clamp=apply_clamp,
    )
    return GptOssMoeResult(output, router_logits, topk_indices, topk_weights, gate_up_preact)


def gpt_oss_moe_golden_b_plena_mxfp8(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None = None,
    *,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    bf16_intermediates: bool = True,
    apply_clamp: bool = True,
) -> GptOssMoeResult:
    """PLENA-aware Golden B with fixed Golden-A routing.

    Router logits and top-k are deliberately not recomputed here.  This isolates
    expert compute plus weighted combine from the fragile discrete routing
    decision.  Expert matrix weights are quantized through PLENA's current HBM
    MXFP8 model; router precision is outside this function by construction.
    """
    x_ref = _maybe_bf16(x, bf16_intermediates)
    gate_up_q = quantize_to_plena_mxfp8(gate_up_weight)
    down_q = quantize_to_plena_mxfp8(down_weight)
    weights_ref = _maybe_bf16(topk_weights, bf16_intermediates)
    output, gate_up_preact = _run_selected_experts(
        x_ref,
        topk_indices,
        weights_ref,
        gate_up_q,
        gate_up_bias,
        down_q,
        down_bias,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        bf16_intermediates=bf16_intermediates,
        apply_clamp=apply_clamp,
    )
    router_logits = torch.empty(x.shape[0], 0, dtype=torch.float32, device=x.device)
    return GptOssMoeResult(output, router_logits, topk_indices, weights_ref, gate_up_preact)


def gpt_oss_moe_fixed_routing_host_smoke(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None = None,
    *,
    quantize_expert_weights_to_plena_mxfp8: bool = False,
    swiglu_limit: float = 7.0,
    swiglu_alpha: float = 1.702,
    bf16_intermediates: bool = True,
    apply_clamp: bool = True,
) -> GptOssMoeResult:
    """Host-side fixed-routing MoE smoke model.

    This mirrors the v0 dataflow deliberately: routing decisions are fixed by
    the caller, tokens are grouped by expert on the host, each selected expert
    is evaluated, and weighted outputs are accumulated back to token order.  It
    exists to test wiring separately from the fragile top-k decision.
    """
    x_ref = _maybe_bf16(x, bf16_intermediates)
    weights_ref = _maybe_bf16(topk_weights, bf16_intermediates)
    if quantize_expert_weights_to_plena_mxfp8:
        gate_up_weight = quantize_to_plena_mxfp8(gate_up_weight)
        down_weight = quantize_to_plena_mxfp8(down_weight)

    tokens, hidden = x.shape
    experts_per_token = topk_indices.shape[1]
    gate_up_width = gate_up_weight.shape[-1]
    # Match HF's standalone GptOssExperts CPU fallback for unquantized Golden A:
    # expert-id sorted loop, BF16 matmul outputs, and BF16 index_add_
    # accumulation.  The quantized Golden-B smoke intentionally keeps the older
    # FP32 matmul + explicit BF16-round model so it stays bit-equivalent to
    # gpt_oss_moe_golden_b_plena_mxfp8.
    hf_cpu_equiv = bf16_intermediates and not quantize_expert_weights_to_plena_mxfp8
    accum_dtype = torch.bfloat16 if hf_cpu_equiv else torch.float32
    next_states = torch.zeros(tokens, hidden, dtype=accum_dtype, device=x.device)
    gate_up_preact = torch.zeros(
        tokens,
        experts_per_token,
        gate_up_width,
        dtype=torch.float32,
        device=x.device,
    )

    for expert_idx_tensor in torch.unique(topk_indices):
        expert_idx = int(expert_idx_tensor.item())
        token_idx, top_k_pos = torch.where(topk_indices == expert_idx)
        if token_idx.numel() == 0:
            continue

        if hf_cpu_equiv:
            current_state = x_ref[token_idx].to(torch.bfloat16)
            gate_up = (
                current_state @ gate_up_weight[expert_idx].to(torch.bfloat16)
                + gate_up_bias[expert_idx].to(torch.bfloat16)
            )
        else:
            current_state = x_ref[token_idx]
            gate_up = current_state.float() @ gate_up_weight[expert_idx].float() + gate_up_bias[expert_idx].float()
        gate_up = _maybe_bf16(gate_up, bf16_intermediates)
        gate_up_preact[token_idx, top_k_pos] = gate_up.float()

        activated = gpt_oss_swiglu(
            gate_up,
            limit=swiglu_limit,
            alpha=swiglu_alpha,
            bf16_intermediates=bf16_intermediates,
            apply_clamp=apply_clamp,
        )
        if hf_cpu_equiv:
            expert_out = activated.to(torch.bfloat16) @ down_weight[expert_idx].to(torch.bfloat16)
            if down_bias is not None:
                expert_out = expert_out + down_bias[expert_idx].to(torch.bfloat16)
        else:
            expert_out = activated.float() @ down_weight[expert_idx].float()
            if down_bias is not None:
                expert_out = expert_out + down_bias[expert_idx].float()
        expert_out = _maybe_bf16(expert_out, bf16_intermediates)

        weighted = expert_out.to(accum_dtype) * weights_ref[token_idx, top_k_pos].to(accum_dtype).unsqueeze(-1)
        next_states.index_add_(0, token_idx, weighted.to(accum_dtype))

    next_states = _maybe_bf16(next_states, bf16_intermediates)
    router_logits = torch.empty(tokens, 0, dtype=torch.float32, device=x.device)
    return GptOssMoeResult(next_states, router_logits, topk_indices, weights_ref, gate_up_preact)


def _run_selected_experts(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_weight: torch.Tensor,
    gate_up_bias: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor | None,
    *,
    swiglu_limit: float,
    swiglu_alpha: float,
    bf16_intermediates: bool,
    apply_clamp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, hidden = x.shape
    experts_per_token = topk_indices.shape[1]
    intermediate = down_weight.shape[1]

    selected_gate_up_w = gate_up_weight[topk_indices].float()
    selected_gate_up_b = gate_up_bias[topk_indices].float()
    gate_up = torch.einsum("th,tkeh->tke", x.float(), selected_gate_up_w.transpose(-1, -2))
    gate_up = gate_up + selected_gate_up_b
    gate_up = _maybe_bf16(gate_up, bf16_intermediates)

    activated = gpt_oss_swiglu(
        gate_up,
        limit=swiglu_limit,
        alpha=swiglu_alpha,
        bf16_intermediates=bf16_intermediates,
        apply_clamp=apply_clamp,
    )
    assert activated.shape == (tokens, experts_per_token, intermediate)

    selected_down_w = down_weight[topk_indices].float()
    expert_out = torch.einsum("tki,tkih->tkh", activated.float(), selected_down_w)
    if down_bias is not None:
        expert_out = expert_out + down_bias[topk_indices].float()
    expert_out = _maybe_bf16(expert_out, bf16_intermediates)

    combined = (expert_out.float() * topk_weights.float().unsqueeze(-1)).sum(dim=1)
    combined = _maybe_bf16(combined, bf16_intermediates)
    assert combined.shape == (tokens, hidden)
    return combined, gate_up
