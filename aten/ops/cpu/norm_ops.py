"""CPU (PyTorch) reference implementations for normalization."""

import torch


def rms_norm_cpu(
    input: torch.Tensor,
    eps: float = 1e-6,
    eps_offset: int = 1,
    reci_hid_offset: int = 2,
) -> torch.Tensor:
    """CPU reference: RMS normalization."""
    del eps_offset, reci_hid_offset
    x = input.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


def layer_norm_cpu(
    input: torch.Tensor,
    eps: float = 1e-6,
    eps_offset: int = 1,
    reci_hid_offset: int = 2,
) -> torch.Tensor:
    """CPU reference: Layer normalization (zero-mean, unit-variance per row)."""
    del eps_offset, reci_hid_offset
    x = input.float()
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)
