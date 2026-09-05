"""KDA head geometry.

Deliberately torch-free and in its own module. ``aten/plena/compiler.py``
imports the KDA lowering at module scope, so anything the lowering needs at
import time is pulled into every ``compiler.aten.plena.*`` import -- including
the ones the ``moe-stage-guard`` CI job runs with only pytest and pyyaml
installed. Leaving ``KdaShape`` in ``reference.py`` next to ``import torch``
made that job fail to collect.

Same object for the reference and the lowering, so ``isinstance`` and dataclass
equality hold across the two.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["KdaShape"]


@dataclass(frozen=True)
class KdaShape:
    hidden_size: int
    num_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    chunk_size: int = 16
    gate_lower_bound: float = -5.0

    def __post_init__(self) -> None:
        for name in ("hidden_size", "num_heads", "key_dim", "value_dim", "conv_kernel", "chunk_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.gate_lower_bound >= 0:
            raise ValueError("gate_lower_bound must be negative")

    @classmethod
    def kimi_k3(cls) -> KdaShape:
        return cls(7168, 96, 128, 128, 4)

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.key_dim

    @property
    def state_elements(self) -> int:
        return self.num_heads * self.value_dim * self.key_dim

    @property
    def conv_state_elements(self) -> int:
        channels = self.num_heads * (2 * self.key_dim + self.value_dim)
        return channels * self.conv_kernel
