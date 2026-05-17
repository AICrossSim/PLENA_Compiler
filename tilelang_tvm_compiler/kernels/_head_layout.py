"""HBM head-layout view helpers — BSHD <-> B,S,1,H*D.

Both layouts share the *same* row-major fp16 bytes in HBM. The only
difference is how the next kernel's ``T.Tensor((...), ...)`` signature
declares the logical shape. Use these helpers when a producer kernel
emits BSHD but the consumer wants B,S,1,H*D (or vice versa).

Pure ``torch.Tensor.view`` — zero copy, contiguous-preserving.
"""

from __future__ import annotations

import torch


def pack_heads(x_bshd: torch.Tensor) -> torch.Tensor:
    """[B, S, H, D] -> [B, S, 1, H*D].

    Same memory, just a different logical view. The producing kernel
    wrote H*D contiguous fp16 elements per (B,S) row; the consuming
    kernel declares them as a single "head" of width H*D.
    """
    if x_bshd.dim() != 4:
        raise ValueError(f"pack_heads expects 4D BSHD; got shape {tuple(x_bshd.shape)}")
    if not x_bshd.is_contiguous():
        raise ValueError(
            "pack_heads requires a contiguous tensor (view-only op). "
            "Call .contiguous() upstream if the producer's output was permuted."
        )
    B, S, H, D = x_bshd.shape
    return x_bshd.view(B, S, 1, H * D)


def unpack_heads(x_packed: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """[B, S, 1, H*D] -> [B, S, H, D].

    Inverse of ``pack_heads``. ``num_heads * head_dim`` must equal the
    last-dim of the packed tensor.
    """
    if x_packed.dim() != 4:
        raise ValueError(
            f"unpack_heads expects 4D B,S,1,H*D; got shape {tuple(x_packed.shape)}"
        )
    if x_packed.shape[2] != 1:
        raise ValueError(
            f"unpack_heads expects head-axis == 1 (packed); got shape "
            f"{tuple(x_packed.shape)}"
        )
    if not x_packed.is_contiguous():
        raise ValueError(
            "unpack_heads requires a contiguous tensor (view-only op)."
        )
    B, S, _, HD = x_packed.shape
    if HD != num_heads * head_dim:
        raise ValueError(
            f"unpack_heads: packed last-dim {HD} != num_heads*head_dim "
            f"{num_heads}*{head_dim} = {num_heads * head_dim}"
        )
    return x_packed.view(B, S, num_heads, head_dim)


__all__ = ["pack_heads", "unpack_heads"]
