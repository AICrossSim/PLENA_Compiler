"""CPU reference implementations of the Mamba-2 ops registered in native_ops.yaml.

Thin wrappers over ``aten/models/mamba2/reference.py`` so the registry's CPU
backend and the numerical golden are the same code -- if they were separate, a
divergence between them would be invisible until it showed up as an emulator
mismatch nobody could localise.
"""

from __future__ import annotations

import torch

from compiler.aten.models.mamba2.reference import (
    _causal_depthwise_conv1d,
    mamba2_recurrent_reference,
    ssd_chunk_reference,
)


def causal_conv1d_cpu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None):
    """`x` [batch, seq, channels], `weight` [channels, kernel]."""
    return _causal_depthwise_conv1d(x, weight, bias)


def dt_activation_cpu(
    dt_raw: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    dt_min: float = 0.0,
    dt_max: float = float("inf"),
):
    dt = dt_raw if dt_bias is None else dt_raw + dt_bias
    return torch.clamp(torch.nn.functional.softplus(dt), min=dt_min, max=dt_max)


def gated_rmsnorm_cpu(
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor | None = None,
    n_groups: int = 1,
    eps: float = 1e-5,
):
    """``RMSNorm(y * silu(z)) * norm_weight``, gate applied before the variance.

    This exists because pointing the registry at ``norm_ops.rms_norm_cpu`` did not
    work: dispatch forwards positionally and that function's signature is
    ``(input, eps, eps_offset, reci_hid_offset)``, so `z` bound to `eps` and
    `sqrt(mean(y^2) + z)` went negative wherever `z` did -- NaN, silently, with the
    gate and the weight both discarded.
    """
    y = y * torch.nn.functional.silu(z)
    *lead, d_inner = y.shape
    group_width = d_inner // n_groups
    yg = y.reshape(*lead, n_groups, group_width)
    yg = yg * torch.rsqrt(yg.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = yg.reshape(*lead, d_inner)
    return y if norm_weight is None else y * norm_weight


def ssd_scan_cpu(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    initial_state: torch.Tensor | None = None,
):
    """Chunked SSD scan -- the form the PLENA prefill lowering implements.

    `chunk_size` is required: the PLENA lowering pins it to MLEN, so a default
    here would silently encode one build's tile size, and on this CPU path it
    would produce a numerically valid but differently-chunked result instead of
    an error.
    """
    return ssd_chunk_reference(x, dt, A, B, C, D, chunk_size, initial_state=initial_state)


def ssm_recurrent_step_cpu(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    state: torch.Tensor,
):
    """One decode step. `x`/`dt`/`B`/`C` carry a length-1 sequence axis."""
    return mamba2_recurrent_reference(x, dt, A, B, C, D, initial_state=state)


__all__ = [
    "causal_conv1d_cpu",
    "gated_rmsnorm_cpu",
    "dt_activation_cpu",
    "ssd_scan_cpu",
    "ssm_recurrent_step_cpu",
]
