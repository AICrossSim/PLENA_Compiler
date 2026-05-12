"""CPU (PyTorch) reference implementations for linear projection."""

import torch


def linear_cpu(
    input: torch.Tensor,
    weight: torch.Tensor,
    name: str = "linear_out",
) -> torch.Tensor:
    """CPU reference: input @ weight (float32 accumulation)."""
    del name
    return torch.matmul(input.float(), weight.float())
