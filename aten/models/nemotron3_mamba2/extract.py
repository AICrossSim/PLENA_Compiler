"""Strict extraction of NVIDIA Nemotron 3 Mamba-2 mixer modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .reference import Mamba2Config, Mamba2Weights


_MISSING = object()


def _attribute(source: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    if default is not _MISSING:
        return default
    raise ValueError(
        f"{type(source).__name__} is missing required attribute "
        f"{' or '.join(repr(name) for name in names)}"
    )


def _unwrap_mixer(layer_or_mixer: Any) -> Any:
    block_type = getattr(layer_or_mixer, "block_type", None)
    if block_type is not None and block_type != "mamba":
        raise ValueError(f"decoder block_type must be 'mamba', got {block_type!r}")
    mixer = getattr(layer_or_mixer, "mixer", layer_or_mixer)
    required = ("in_proj", "conv1d", "A_log", "dt_bias", "D", "norm", "out_proj")
    missing = [name for name in required if not hasattr(mixer, name)]
    if missing:
        raise ValueError(
            f"{type(mixer).__name__} is not a Nemotron Mamba-2 mixer; "
            f"missing {', '.join(missing)}"
        )
    return mixer


def _runtime_dt_limit(source: Any) -> tuple[float, float]:
    value = getattr(source, "time_step_limit", None)
    if value is None:
        return 0.0, float("inf")
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("time_step_limit must contain exactly (minimum, maximum)")
    return float(value[0]), float(value[1])


def _projection_d_mlp(source: Any, d_inner: int, conv_channels: int, heads: int) -> int:
    in_proj = getattr(source, "in_proj", None)
    if in_proj is None:
        return int(getattr(source, "d_mlp", 0))
    weight = getattr(in_proj, "weight", None)
    if weight is None or weight.ndim != 2:
        raise ValueError("Nemotron in_proj.weight must be a rank-2 tensor")
    residual = int(weight.shape[0]) - d_inner - conv_channels - heads
    if residual < 0 or residual % 2:
        raise ValueError(
            "Nemotron input projection cannot be split as [z0, x0, gate, xBC, dt]"
        )
    return residual // 2


def extract_nemotron3_mamba2_config(source: Any) -> Mamba2Config:
    """Extract the execution dimensions from a mixer or HF configuration.

    ``time_step_min`` and ``time_step_max`` are intentionally ignored: NVIDIA
    uses them to initialize ``dt_bias``. Runtime clamp semantics come from
    ``time_step_limit``.
    """
    if hasattr(source, "mixer"):
        source = _unwrap_mixer(source)
    elif hasattr(source, "config") and not hasattr(source, "hidden_size"):
        source = source.config

    d_model = int(_attribute(source, "hidden_size"))
    num_heads = int(_attribute(source, "num_heads", "mamba_num_heads"))
    head_dim = int(_attribute(source, "head_dim", "mamba_head_dim"))
    state_dim = int(_attribute(source, "ssm_state_size"))
    groups = int(_attribute(source, "n_groups", "mamba_n_groups"))
    chunk_size = int(_attribute(source, "chunk_size", "mamba_chunk_size"))
    conv_kernel = int(
        _attribute(source, "conv_kernel_size", "conv_kernel", "mamba_d_conv")
    )
    d_inner = num_heads * head_dim
    conv_channels = d_inner + 2 * groups * state_dim
    d_mlp = _projection_d_mlp(source, d_inner, conv_channels, num_heads)
    dt_min, dt_max = _runtime_dt_limit(source)
    norm = getattr(source, "norm", None)
    rms_norm_eps = float(
        getattr(
            norm,
            "variance_epsilon",
            _attribute(
                source,
                "layer_norm_epsilon",
                "rms_norm_eps",
                default=1e-5,
            ),
        )
    )
    activation = _attribute(
        source, "activation", "mamba_hidden_act", default="silu"
    ).lower()
    if activation not in ("silu", "swish"):
        raise ValueError(f"Nemotron Mamba activation must be SiLU, got {activation!r}")

    config = Mamba2Config(
        d_model=d_model,
        d_inner=d_inner,
        num_heads=num_heads,
        head_dim=head_dim,
        state_dim=state_dim,
        groups=groups,
        chunk_size=chunk_size,
        conv_kernel=conv_kernel,
        rms_norm_eps=rms_norm_eps,
        dt_min=dt_min,
        dt_max=dt_max,
        d_mlp=d_mlp,
    )
    declared_inner = getattr(source, "intermediate_size", d_inner)
    declared_conv = getattr(source, "conv_dim", conv_channels)
    if int(declared_inner) != d_inner:
        raise ValueError("mixer intermediate_size does not equal num_heads * head_dim")
    if int(declared_conv) != config.conv_channels:
        raise ValueError(
            "mixer conv_dim does not match d_inner + 2 * groups * state_dim"
        )
    return config


def _tensor(parameter: Any, name: str, *, dimensions: int) -> torch.Tensor:
    if not isinstance(parameter, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    value = parameter.detach().float().contiguous()
    if value.ndim != dimensions:
        raise ValueError(
            f"{name} must have rank {dimensions}, got shape {tuple(value.shape)}"
        )
    return value


def _linear_bias(module: Any, name: str) -> torch.Tensor | None:
    bias = getattr(module, "bias", None)
    return None if bias is None else _tensor(bias, name, dimensions=1)


@dataclass(frozen=True)
class ExtractedNemotronMamba2Layer:
    layer_id: int
    config: Mamba2Config
    weights: Mamba2Weights


def extract_nemotron3_mamba2_layer(layer_or_mixer: Any) -> ExtractedNemotronMamba2Layer:
    """Extract one exact Nemotron mixer in the reference model's tensor order."""
    mixer = _unwrap_mixer(layer_or_mixer)
    config = extract_nemotron3_mamba2_config(mixer)

    conv_module = mixer.conv1d
    conv_weight_raw = _tensor(conv_module.weight, "conv1d.weight", dimensions=3)
    if conv_weight_raw.shape[1] != 1:
        raise ValueError(
            "Nemotron conv1d must be depthwise with weight shape [channel, 1, kernel]"
        )
    if (
        int(getattr(conv_module, "groups", config.conv_channels))
        != config.conv_channels
    ):
        raise ValueError("Nemotron conv1d groups must equal conv_channels")

    weights = Mamba2Weights(
        in_proj_weight=_tensor(mixer.in_proj.weight, "in_proj.weight", dimensions=2),
        in_proj_bias=_linear_bias(mixer.in_proj, "in_proj.bias"),
        conv_weight=conv_weight_raw[:, 0, :].contiguous(),
        conv_bias=_linear_bias(conv_module, "conv1d.bias"),
        a_log=_tensor(mixer.A_log, "A_log", dimensions=1),
        dt_bias=_tensor(mixer.dt_bias, "dt_bias", dimensions=1),
        d_skip=_tensor(mixer.D, "D", dimensions=1),
        norm_weight=_tensor(mixer.norm.weight, "norm.weight", dimensions=1),
        out_proj_weight=_tensor(mixer.out_proj.weight, "out_proj.weight", dimensions=2),
        out_proj_bias=_linear_bias(mixer.out_proj, "out_proj.bias"),
    )
    weights.validate(config)
    norm_group_size = getattr(mixer.norm, "group_size", config.d_inner // config.groups)
    if int(norm_group_size) != config.d_inner // config.groups:
        raise ValueError("gated RMSNorm group_size does not match d_inner / groups")
    layer_id = int(getattr(mixer, "layer_idx", getattr(layer_or_mixer, "layer_idx", 0)))
    if layer_id < 0:
        raise ValueError("layer_idx must be non-negative")
    return ExtractedNemotronMamba2Layer(layer_id, config, weights)
