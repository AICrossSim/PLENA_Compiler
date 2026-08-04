"""PyTorch ground truth for the Nemotron 3 Nano Mamba-2 mixer.

The implementation follows NVIDIA's ``NemotronHMamba2Mixer`` tensor order and
the Mamba-2 recurrence directly.  It deliberately avoids the quadratic
semiseparable ``L`` matrix: prefill is either a scalar-time recurrence or an
associative affine prefix scan.  Persistent convolution and SSM state are FP32
for both supported precision policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as F


class PrecisionPolicy(IntEnum):
    """Values are part of the shared PLENA Mamba descriptor ABI."""

    FP32_REFERENCE = 0
    BF16_ACTIVATION_FP32_STATE = 1


@dataclass(frozen=True)
class Mamba2Config:
    d_model: int
    d_inner: int
    num_heads: int
    head_dim: int
    state_dim: int
    groups: int
    chunk_size: int
    conv_kernel: int = 4
    rms_norm_eps: float = 1e-5
    dt_min: float = 0.0
    dt_max: float = float("inf")
    d_mlp: int = 0

    def __post_init__(self) -> None:
        positive = {
            "d_model": self.d_model,
            "d_inner": self.d_inner,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "state_dim": self.state_dim,
            "groups": self.groups,
            "chunk_size": self.chunk_size,
            "conv_kernel": self.conv_kernel,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.d_inner != self.num_heads * self.head_dim:
            raise ValueError("d_inner must equal num_heads * head_dim")
        if self.num_heads % self.groups:
            raise ValueError("num_heads must be divisible by groups")
        if self.d_inner % self.groups:
            raise ValueError("d_inner must be divisible by groups for grouped RMSNorm")
        if self.d_mlp < 0:
            raise ValueError("d_mlp must be non-negative")
        if self.dt_min < 0.0 or self.dt_max < self.dt_min:
            raise ValueError("invalid dt clamp interval")

    @property
    def conv_channels(self) -> int:
        return self.d_inner + 2 * self.groups * self.state_dim

    @property
    def projection_size(self) -> int:
        # NVIDIA order: [z0, x0, gate, xBC, dt].
        return 2 * self.d_mlp + self.d_inner + self.conv_channels + self.num_heads

    @property
    def output_projection_size(self) -> int:
        return self.d_mlp + self.d_inner

    @classmethod
    def nemotron3_nano(cls) -> "Mamba2Config":
        return cls(
            d_model=2688,
            d_inner=4096,
            num_heads=64,
            head_dim=64,
            state_dim=128,
            groups=8,
            chunk_size=128,
            conv_kernel=4,
        )


@dataclass(frozen=True)
class Mamba2Weights:
    """Weights in PyTorch ``Linear`` orientation: ``[out_features, in_features]``."""

    in_proj_weight: torch.Tensor
    in_proj_bias: torch.Tensor | None
    conv_weight: torch.Tensor
    conv_bias: torch.Tensor | None
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    d_skip: torch.Tensor
    norm_weight: torch.Tensor
    out_proj_weight: torch.Tensor
    out_proj_bias: torch.Tensor | None

    def validate(self, config: Mamba2Config) -> None:
        expected = {
            "in_proj_weight": (config.projection_size, config.d_model),
            "conv_weight": (config.conv_channels, config.conv_kernel),
            "a_log": (config.num_heads,),
            "dt_bias": (config.num_heads,),
            "d_skip": (config.num_heads,),
            "norm_weight": (config.d_inner,),
            "out_proj_weight": (config.d_model, config.output_projection_size),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} shape must be {shape}, got {actual}")
        optional = {
            "in_proj_bias": (config.projection_size,),
            "conv_bias": (config.conv_channels,),
            "out_proj_bias": (config.d_model,),
        }
        for name, shape in optional.items():
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != shape:
                raise ValueError(
                    f"{name} shape must be {shape}, got {tuple(value.shape)}"
                )


@dataclass(frozen=True)
class Mamba2State:
    """Persistent per-layer state.

    ``ssm`` is ``[batch, head, head_dim, state_dim]``. ``conv`` is
    ``[batch, conv_channel, conv_kernel]`` with oldest sample first.  Both are
    always FP32, matching the shared ABI and avoiding recurrent BF16 drift.
    """

    ssm: torch.Tensor
    conv: torch.Tensor

    def clone(self) -> "Mamba2State":
        return Mamba2State(self.ssm.clone(), self.conv.clone())


@dataclass(frozen=True)
class Mamba2Result:
    output: torch.Tensor
    state: Mamba2State


def allocate_state(
    config: Mamba2Config,
    batch_size: int,
    *,
    device: torch.device | str | None = None,
) -> Mamba2State:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return Mamba2State(
        ssm=torch.zeros(
            batch_size,
            config.num_heads,
            config.head_dim,
            config.state_dim,
            dtype=torch.float32,
            device=device,
        ),
        conv=torch.zeros(
            batch_size,
            config.conv_channels,
            config.conv_kernel,
            dtype=torch.float32,
            device=device,
        ),
    )


def _validate_state(state: Mamba2State, config: Mamba2Config, batch_size: int) -> None:
    expected_ssm = (batch_size, config.num_heads, config.head_dim, config.state_dim)
    expected_conv = (batch_size, config.conv_channels, config.conv_kernel)
    if tuple(state.ssm.shape) != expected_ssm:
        raise ValueError(
            f"ssm state shape must be {expected_ssm}, got {tuple(state.ssm.shape)}"
        )
    if tuple(state.conv.shape) != expected_conv:
        raise ValueError(
            f"conv state shape must be {expected_conv}, got {tuple(state.conv.shape)}"
        )
    if state.ssm.dtype != torch.float32 or state.conv.dtype != torch.float32:
        raise ValueError("Mamba SSM and convolution state must be FP32")


def _bf16_boundary(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.bfloat16).float()


def _activation_boundary(tensor: torch.Tensor, policy: PrecisionPolicy) -> torch.Tensor:
    tensor = tensor.float()
    if policy == PrecisionPolicy.BF16_ACTIVATION_FP32_STATE:
        return _bf16_boundary(tensor)
    if policy != PrecisionPolicy.FP32_REFERENCE:
        raise ValueError(f"unsupported precision policy {policy!r}")
    return tensor


def _linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    policy: PrecisionPolicy,
) -> torch.Tensor:
    value = _activation_boundary(value, policy)
    weight = _activation_boundary(weight, policy)
    bias = None if bias is None else _activation_boundary(bias, policy)
    return _activation_boundary(F.linear(value, weight, bias), policy)


def depthwise_conv_step(
    value: torch.Tensor,
    state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    policy: PrecisionPolicy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one causal depthwise convolution update and SiLU activation."""
    next_state = torch.roll(state.float(), shifts=-1, dims=-1)
    next_state[..., -1] = _activation_boundary(value, policy)
    conv = (next_state * _activation_boundary(weight, policy).unsqueeze(0)).sum(dim=-1)
    if bias is not None:
        conv = conv + _activation_boundary(bias, policy)
    output = _activation_boundary(F.silu(conv), policy)
    return output, next_state


def _repeat_groups(value: torch.Tensor, config: Mamba2Config) -> torch.Tensor:
    return value.repeat_interleave(config.num_heads // config.groups, dim=-2)


def _dt_and_a(
    dt: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    config: Mamba2Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt_fp32 = F.softplus(dt.float() + dt_bias.float())
    dt_fp32 = torch.clamp(dt_fp32, min=config.dt_min, max=config.dt_max)
    a_fp32 = -torch.exp(a_log.float())
    return dt_fp32, a_fp32


def selective_state_step(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_skip: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One exact Mamba-2 recurrence update, with all recurrence math in FP32."""
    x = x.float()
    dt = dt.float()
    b = b.float()
    c = c.float()
    state = state.float()
    decay = torch.exp(dt * a.float()).unsqueeze(-1).unsqueeze(-1)
    drive = (dt.unsqueeze(-1) * b).unsqueeze(-2) * x.unsqueeze(-1)
    next_state = state * decay + drive
    output = (next_state * c.unsqueeze(-2)).sum(dim=-1)
    output = output + d_skip.float().unsqueeze(0).unsqueeze(-1) * x
    return output, next_state


def selective_scan_sequential(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference recurrence over ``[batch, sequence, head, ...]`` tensors."""
    outputs = []
    state = initial_state.float()
    for token in range(x.shape[1]):
        output, state = selective_state_step(
            x[:, token],
            dt[:, token],
            a,
            b[:, token],
            c[:, token],
            d_skip,
            state,
        )
        outputs.append(output)
    if not outputs:
        empty = x.new_empty((*x.shape[:-1], x.shape[-1]), dtype=torch.float32)
        return empty, state
    return torch.stack(outputs, dim=1), state


def selective_scan_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d_skip: torch.Tensor,
    initial_state: torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Associative affine prefix scan without materializing an ``L`` matrix.

    Each token is the affine map ``state -> decay * state + drive``.  A
    Hillis-Steele scan composes maps in ``ceil(log2(chunk_size))`` rounds.  The
    implementation materializes only linear-in-sequence affine coefficients;
    RTL is expected to tile the ``head_dim x state_dim`` drive tensor.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    outputs = []
    state = initial_state.float()
    for start in range(0, x.shape[1], chunk_size):
        stop = min(start + chunk_size, x.shape[1])
        x_chunk = x[:, start:stop].float()
        dt_chunk = dt[:, start:stop].float()
        b_chunk = b[:, start:stop].float()
        c_chunk = c[:, start:stop].float()

        decay = torch.exp(dt_chunk * a.float().view(1, 1, -1))
        decay = decay.unsqueeze(-1).unsqueeze(-1)
        drive = (dt_chunk.unsqueeze(-1) * b_chunk).unsqueeze(-2) * x_chunk.unsqueeze(-1)

        offset = 1
        while offset < stop - start:
            previous_decay = decay.clone()
            previous_drive = drive.clone()
            decay[:, offset:] = previous_decay[:, offset:] * previous_decay[:, :-offset]
            drive[:, offset:] = (
                previous_drive[:, offset:]
                + previous_decay[:, offset:] * previous_drive[:, :-offset]
            )
            offset <<= 1

        states = decay * state.unsqueeze(1) + drive
        chunk_output = (states * c_chunk.unsqueeze(-2)).sum(dim=-1)
        chunk_output = chunk_output + d_skip.float().view(1, 1, -1, 1) * x_chunk
        outputs.append(chunk_output)
        state = states[:, -1]

    if not outputs:
        empty = x.new_empty((*x.shape[:-1], x.shape[-1]), dtype=torch.float32)
        return empty, state
    return torch.cat(outputs, dim=1), state


def gated_group_rms_norm(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    config: Mamba2Config,
    policy: PrecisionPolicy,
) -> torch.Tensor:
    """Nemotron order: group-RMSNorm(value * SiLU(gate))."""
    gated = value.float() * F.silu(_activation_boundary(gate, policy)).float()
    group_size = config.d_inner // config.groups
    grouped = gated.reshape(*gated.shape[:-1], config.groups, group_size)
    inverse_rms = torch.rsqrt(
        grouped.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
    )
    normalized = (grouped * inverse_rms).reshape_as(gated)
    return _activation_boundary(
        normalized * _activation_boundary(weight, policy), policy
    )


def _project_and_split(
    value: torch.Tensor,
    weights: Mamba2Weights,
    config: Mamba2Config,
    policy: PrecisionPolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = _linear(value, weights.in_proj_weight, weights.in_proj_bias, policy)
    return torch.split(
        projected,
        [
            config.d_mlp,
            config.d_mlp,
            config.d_inner,
            config.conv_channels,
            config.num_heads,
        ],
        dim=-1,
    )


def _finish_output(
    scan_output: torch.Tensor,
    gate: torch.Tensor,
    z0: torch.Tensor,
    x0: torch.Tensor,
    weights: Mamba2Weights,
    config: Mamba2Config,
    policy: PrecisionPolicy,
) -> torch.Tensor:
    normalized = gated_group_rms_norm(
        scan_output.reshape(*scan_output.shape[:-2], config.d_inner),
        gate,
        weights.norm_weight,
        config,
        policy,
    )
    if config.d_mlp:
        mlp = _activation_boundary(F.silu(z0.float()) * x0.float(), policy)
        normalized = torch.cat((mlp, normalized), dim=-1)
    return _linear(normalized, weights.out_proj_weight, weights.out_proj_bias, policy)


def mamba2_step(
    value: torch.Tensor,
    weights: Mamba2Weights,
    config: Mamba2Config,
    state: Mamba2State,
    *,
    policy: PrecisionPolicy = PrecisionPolicy.FP32_REFERENCE,
) -> Mamba2Result:
    """Execute one decode token and return updated persistent state."""
    squeeze_sequence = value.ndim == 3
    if squeeze_sequence:
        if value.shape[1] != 1:
            raise ValueError("mamba2_step accepts one token")
        value = value[:, 0]
    if value.ndim != 2 or value.shape[-1] != config.d_model:
        raise ValueError(f"step input must be [batch, {config.d_model}]")
    weights.validate(config)
    _validate_state(state, config, value.shape[0])

    z0, x0, gate, xbc, dt_raw = _project_and_split(value, weights, config, policy)
    xbc, conv_state = depthwise_conv_step(
        xbc, state.conv, weights.conv_weight, weights.conv_bias, policy
    )
    x, b_grouped, c_grouped = torch.split(
        xbc,
        [
            config.d_inner,
            config.groups * config.state_dim,
            config.groups * config.state_dim,
        ],
        dim=-1,
    )
    x = x.reshape(value.shape[0], config.num_heads, config.head_dim)
    b_grouped = b_grouped.reshape(value.shape[0], config.groups, config.state_dim)
    c_grouped = c_grouped.reshape(value.shape[0], config.groups, config.state_dim)
    b = _repeat_groups(b_grouped, config)
    c = _repeat_groups(c_grouped, config)
    dt, a = _dt_and_a(dt_raw, weights.a_log, weights.dt_bias, config)
    scan_output, ssm_state = selective_state_step(
        x, dt, a, b, c, weights.d_skip, state.ssm
    )
    output = _finish_output(scan_output, gate, z0, x0, weights, config, policy)
    if squeeze_sequence:
        output = output.unsqueeze(1)
    return Mamba2Result(output, Mamba2State(ssm_state.float(), conv_state.float()))


def mamba2_prefill(
    value: torch.Tensor,
    weights: Mamba2Weights,
    config: Mamba2Config,
    state: Mamba2State | None = None,
    *,
    policy: PrecisionPolicy = PrecisionPolicy.FP32_REFERENCE,
    scan: str = "chunked",
) -> Mamba2Result:
    """Execute a sequence while preserving state for a following decode step."""
    if value.ndim != 3 or value.shape[-1] != config.d_model:
        raise ValueError(f"prefill input must be [batch, sequence, {config.d_model}]")
    weights.validate(config)
    if state is None:
        state = allocate_state(config, value.shape[0], device=value.device)
    _validate_state(state, config, value.shape[0])

    z0, x0, gate, xbc, dt_raw = _project_and_split(value, weights, config, policy)
    conv_outputs = []
    conv_state = state.conv
    for token in range(value.shape[1]):
        conv_output, conv_state = depthwise_conv_step(
            xbc[:, token], conv_state, weights.conv_weight, weights.conv_bias, policy
        )
        conv_outputs.append(conv_output)
    if conv_outputs:
        xbc = torch.stack(conv_outputs, dim=1)
    else:
        xbc = xbc.new_empty((value.shape[0], 0, config.conv_channels))

    x, b_grouped, c_grouped = torch.split(
        xbc,
        [
            config.d_inner,
            config.groups * config.state_dim,
            config.groups * config.state_dim,
        ],
        dim=-1,
    )
    x = x.reshape(value.shape[0], value.shape[1], config.num_heads, config.head_dim)
    b_grouped = b_grouped.reshape(
        value.shape[0], value.shape[1], config.groups, config.state_dim
    )
    c_grouped = c_grouped.reshape(
        value.shape[0], value.shape[1], config.groups, config.state_dim
    )
    b = _repeat_groups(b_grouped, config)
    c = _repeat_groups(c_grouped, config)
    dt, a = _dt_and_a(dt_raw, weights.a_log, weights.dt_bias, config)

    if scan == "sequential":
        scan_output, ssm_state = selective_scan_sequential(
            x, dt, a, b, c, weights.d_skip, state.ssm
        )
    elif scan == "chunked":
        scan_output, ssm_state = selective_scan_chunked(
            x, dt, a, b, c, weights.d_skip, state.ssm, config.chunk_size
        )
    else:
        raise ValueError(f"unknown scan implementation {scan!r}")

    output = _finish_output(scan_output, gate, z0, x0, weights, config, policy)
    return Mamba2Result(output, Mamba2State(ssm_state.float(), conv_state.float()))
