"""Correctness tests for the Nemotron 3 Mamba-2 executable contract."""

from __future__ import annotations

import pytest
import torch

from aten.models.nemotron3_mamba2.reference import (
    Mamba2Config,
    Mamba2Weights,
    PrecisionPolicy,
    allocate_state,
    mamba2_prefill,
    mamba2_step,
)


def _tiny_config() -> Mamba2Config:
    return Mamba2Config(
        d_model=16,
        d_inner=16,
        num_heads=2,
        head_dim=8,
        state_dim=5,
        groups=2,
        chunk_size=4,
        conv_kernel=4,
    )


def _weights(config: Mamba2Config, seed: int = 7) -> Mamba2Weights:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int, scale: float = 0.08) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * scale

    return Mamba2Weights(
        in_proj_weight=randn(config.projection_size, config.d_model),
        in_proj_bias=randn(config.projection_size),
        conv_weight=randn(config.conv_channels, config.conv_kernel),
        conv_bias=randn(config.conv_channels),
        a_log=randn(config.num_heads, scale=0.2),
        dt_bias=randn(config.num_heads, scale=0.2) - 1.0,
        d_skip=randn(config.num_heads, scale=0.2),
        norm_weight=torch.ones(config.d_inner) + randn(config.d_inner, scale=0.02),
        out_proj_weight=randn(config.d_model, config.output_projection_size),
        out_proj_bias=randn(config.d_model),
    )


@pytest.mark.parametrize(
    "policy,atol,rtol",
    [
        (PrecisionPolicy.FP32_REFERENCE, 2e-5, 2e-5),
        (PrecisionPolicy.BF16_ACTIVATION_FP32_STATE, 2e-3, 2e-3),
    ],
)
def test_prefill_matches_repeated_step(
    policy: PrecisionPolicy, atol: float, rtol: float
):
    config = _tiny_config()
    weights = _weights(config)
    value = torch.randn(
        2, 11, config.d_model, generator=torch.Generator().manual_seed(11)
    )
    initial = allocate_state(config, value.shape[0])

    prefill = mamba2_prefill(value, weights, config, initial.clone(), policy=policy)
    state = initial.clone()
    outputs = []
    for token in range(value.shape[1]):
        result = mamba2_step(value[:, token], weights, config, state, policy=policy)
        outputs.append(result.output)
        state = result.state
    stepped_output = torch.stack(outputs, dim=1)

    assert torch.allclose(
        prefill.output.float(), stepped_output.float(), atol=atol, rtol=rtol
    )
    assert torch.allclose(prefill.state.ssm, state.ssm, atol=atol, rtol=rtol)
    assert torch.allclose(prefill.state.conv, state.conv, atol=atol, rtol=rtol)


def test_chunked_scan_matches_sequential_recurrence_across_partial_chunk():
    config = _tiny_config()
    weights = _weights(config)
    value = torch.randn(
        2, 11, config.d_model, generator=torch.Generator().manual_seed(19)
    )
    initial = allocate_state(config, value.shape[0])
    initial.ssm.uniform_(-0.1, 0.1)
    initial.conv.uniform_(-0.1, 0.1)

    sequential = mamba2_prefill(
        value, weights, config, initial.clone(), scan="sequential"
    )
    chunked = mamba2_prefill(value, weights, config, initial.clone(), scan="chunked")

    assert torch.allclose(chunked.output, sequential.output, atol=2e-5, rtol=2e-5)
    assert torch.allclose(chunked.state.ssm, sequential.state.ssm, atol=2e-5, rtol=2e-5)
    assert torch.equal(chunked.state.conv, sequential.state.conv)


def test_bf16_policy_keeps_recurrent_and_conv_state_fp32():
    config = _tiny_config()
    result = mamba2_prefill(
        torch.randn(1, 9, config.d_model),
        _weights(config),
        config,
        policy=PrecisionPolicy.BF16_ACTIVATION_FP32_STATE,
    )
    assert result.output.dtype == torch.float32
    assert result.state.ssm.dtype == torch.float32
    assert result.state.conv.dtype == torch.float32


def test_nemotron3_official_dimensions_and_state_footprint():
    config = Mamba2Config.nemotron3_nano()
    assert config.projection_size == 10304
    assert config.conv_channels == 6144
    assert config.d_inner * config.state_dim * 4 == 2 * 1024 * 1024
    assert config.conv_channels * config.conv_kernel * 4 == 96 * 1024


def test_invalid_bf16_state_is_rejected():
    config = _tiny_config()
    state = allocate_state(config, 1)
    bad_state = type(state)(state.ssm.bfloat16(), state.conv)
    with pytest.raises(ValueError, match="must be FP32"):
        mamba2_step(torch.zeros(1, config.d_model), _weights(config), config, bad_state)
