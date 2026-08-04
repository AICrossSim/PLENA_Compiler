"""Exact validation-only lowering for Qwen rotate-half RoPE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


T = TypeVar("T", int, float)

QWEN_ROPE_VALIDATION_SCHEMA = "plena-qwen-rope-validation"


def rotate_half(values: Sequence[T]) -> list[T]:
    """Return ``[-x[d/2:], x[:d/2]]`` for one even-width head."""

    if not values or len(values) % 2:
        raise ValueError("rotate_half requires a non-empty even-width head")
    half = len(values) // 2
    return [-value for value in values[half:]] + list(values[:half])


def permute_norm_weight(values: Sequence[T]) -> list[T]:
    """Move learned RMSNorm weights with the rotate-half lane permutation."""

    if not values or len(values) % 2:
        raise ValueError("norm weight requires a non-empty even-width head")
    half = len(values) // 2
    return list(values[half:]) + list(values[:half])


def rotate_projection_columns(
    weight: Sequence[Sequence[T]],
    *,
    head_dim: int,
) -> list[list[T]]:
    """Derive columns for a projection that produces rotate-half outputs."""

    if head_dim <= 0 or head_dim % 2:
        raise ValueError("head_dim must be a positive even integer")
    if not weight:
        raise ValueError("projection weight must contain at least one row")
    width = len(weight[0])
    if width == 0 or width % head_dim:
        raise ValueError("projection width must contain complete heads")
    if any(len(row) != width for row in weight):
        raise ValueError("projection rows must have equal width")

    result: list[list[T]] = []
    half = head_dim // 2
    for row in weight:
        rotated: list[T] = []
        for start in range(0, width, head_dim):
            head = row[start : start + head_dim]
            rotated.extend(-value for value in head[half:])
            rotated.extend(head[:half])
        result.append(rotated)
    return result


@dataclass(frozen=True)
class QwenRopeValidationCost:
    """Additional work needed by the duplicated validation path."""

    hidden_size: int
    query_heads: int
    kv_heads: int
    head_dim: int
    block_size: int = 8

    def __post_init__(self) -> None:
        for name in ("hidden_size", "query_heads", "kv_heads", "head_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        half = self.head_dim // 2
        if self.head_dim != self.block_size and half % self.block_size:
            raise ValueError(
                "rotate-half must preserve complete microscaling blocks"
            )

    @property
    def projected_width(self) -> int:
        return (self.query_heads + self.kv_heads) * self.head_dim

    @property
    def extra_weight_elements(self) -> int:
        return self.hidden_size * self.projected_width

    @property
    def extra_projection_macs_per_token(self) -> int:
        return self.extra_weight_elements

    @property
    def extra_norm_elements_per_token(self) -> int:
        return self.projected_width

    def to_dict(self) -> dict[str, object]:
        """Return immutable cost and legality metadata for an artifact."""

        return {
            "schema_version": QWEN_ROPE_VALIDATION_SCHEMA,
            "method": "duplicated_projection_and_segmented_norm",
            "runtime_rotated_tensor_input": False,
            "headline_datapath": False,
            "projection_multiplier_q_plus_k": 2,
            "segmented_norm_multiplier_q_plus_k": 2,
            "extra_weight_elements": self.extra_weight_elements,
            "extra_projection_macs_per_token": (
                self.extra_projection_macs_per_token
            ),
            "extra_norm_elements_per_token": self.extra_norm_elements_per_token,
            "required_existing_operations": [
                "matrix_projection",
                "segmented_affine_rms_norm",
                "rope_elementwise",
            ],
        }
