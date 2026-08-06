"""Shared Matrix-SRAM residency planning for packed GQA K/V tiles.

The planner is intentionally independent of assembler emission.  Native
compilation, CostEmitter, and DSE policy selection use the same deterministic
mapping from physical Matrix SRAM capacity to resident and streamed K/V tiles.
Addresses are expressed in Matrix SRAM elements, matching the PLENA ISA.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


MATRIX_SRAM_POLICIES = (
    "streaming",
    "projection-full",
    "kv-25",
    "kv-50",
    "kv-75",
    "kv-100",
)

_POLICY_FRACTIONS = {
    "kv-25": 0.25,
    "kv-50": 0.50,
    "kv-75": 0.75,
    "kv-100": 1.00,
}


@dataclass(frozen=True)
class KVResidencyPlan:
    """Physical prefix-cache allocation for one local attention sequence."""

    k_blocks: int
    resident_prefix_blocks: int
    streaming_blocks: int
    stream_k_address: int | None
    stream_v_address: int | None
    requested_residency_fraction: float
    realized_residency_fraction: float
    matrix_sram_tiles: int
    tile_elements: int
    policy: str

    @property
    def full_resident(self) -> bool:
        return self.resident_prefix_blocks == self.k_blocks

    @property
    def resident_k_addresses(self) -> tuple[int, ...]:
        return tuple(
            block * self.tile_elements
            for block in range(self.resident_prefix_blocks)
        )

    @property
    def resident_v_addresses(self) -> tuple[int, ...]:
        base = self.resident_prefix_blocks * self.tile_elements
        return tuple(
            base + block * self.tile_elements
            for block in range(self.resident_prefix_blocks)
        )

    @property
    def peak_live_tiles(self) -> int:
        stream_tiles = (
            1
            if self.streaming_blocks
            and self.stream_k_address == self.stream_v_address
            else 2
            if self.streaming_blocks
            else 0
        )
        return 2 * self.resident_prefix_blocks + stream_tiles

    @property
    def tile_utilization(self) -> float:
        return self.peak_live_tiles / self.matrix_sram_tiles

    def is_resident(self, block: int) -> bool:
        if not 0 <= block < self.k_blocks:
            raise IndexError(f"K/V block {block} outside [0, {self.k_blocks})")
        return block < self.resident_prefix_blocks

    def k_address(self, block: int) -> int:
        if self.is_resident(block):
            # Do not materialize the complete resident-address tuple for a
            # single lookup. Long-context causal attention queries this path
            # once per logical Q/K pair, so tuple construction here turns an
            # otherwise linear address calculation into a major compiler
            # allocation hotspot.
            return block * self.tile_elements
        if self.stream_k_address is None:
            raise ValueError("streaming K address requested for a full-resident plan")
        return self.stream_k_address

    def v_address(self, block: int) -> int:
        if self.is_resident(block):
            return (
                self.resident_prefix_blocks + block
            ) * self.tile_elements
        if self.stream_v_address is None:
            raise ValueError("streaming V address requested for a full-resident plan")
        return self.stream_v_address

    def expected_tile_loads(self, *, q_blocks: int, causal: bool) -> int:
        """Return physical K+V tile loads for one batch and one KV head."""

        if q_blocks <= 0:
            raise ValueError(f"q_blocks must be positive, got {q_blocks}")
        resident = 2 * self.resident_prefix_blocks
        if causal:
            streamed = 2 * sum(
                max(
                    0,
                    min(q_block + 1, self.k_blocks)
                    - self.resident_prefix_blocks,
                )
                for q_block in range(q_blocks)
            )
        else:
            streamed = (
                2
                * q_blocks
                * (self.k_blocks - self.resident_prefix_blocks)
            )
        return resident + streamed

    def expected_cache_hits(self, *, q_blocks: int, causal: bool) -> int:
        """Return K/V demands served from the preloaded resident prefix."""

        if q_blocks <= 0:
            raise ValueError(f"q_blocks must be positive, got {q_blocks}")
        if causal:
            return 2 * sum(
                min(
                    self.resident_prefix_blocks,
                    min(q_block + 1, self.k_blocks),
                )
                for q_block in range(q_blocks)
            )
        return 2 * q_blocks * self.resident_prefix_blocks

    def average_live_tile_count(
        self,
        *,
        q_blocks: int,
        causal: bool,
    ) -> float:
        """Average live allocation over logical Q/K block interactions."""

        if q_blocks <= 0:
            raise ValueError(f"q_blocks must be positive, got {q_blocks}")
        if causal:
            total_block_demands = sum(
                min(q_block + 1, self.k_blocks)
                for q_block in range(q_blocks)
            )
            streamed_block_demands = sum(
                max(
                    0,
                    min(q_block + 1, self.k_blocks)
                    - self.resident_prefix_blocks,
                )
                for q_block in range(q_blocks)
            )
        else:
            total_block_demands = q_blocks * self.k_blocks
            streamed_block_demands = (
                q_blocks * (self.k_blocks - self.resident_prefix_blocks)
            )
        stream_slot_tiles = self.peak_live_tiles - (
            2 * self.resident_prefix_blocks
        )
        return (
            2.0 * self.resident_prefix_blocks
            + stream_slot_tiles
            * streamed_block_demands
            / total_block_demands
        )

    def metadata(self, *, q_blocks: int, causal: bool) -> dict[str, object]:
        loads = self.expected_tile_loads(q_blocks=q_blocks, causal=causal)
        average_live = self.average_live_tile_count(
            q_blocks=q_blocks,
            causal=causal,
        )
        return {
            **asdict(self),
            "resident_k_addresses": self.resident_k_addresses,
            "resident_v_addresses": self.resident_v_addresses,
            "peak_live_tiles": self.peak_live_tiles,
            "average_live_tiles": average_live,
            "tile_utilization": self.tile_utilization,
            "kv_tile_load_count": loads,
            "kv_cache_hits": self.expected_cache_hits(
                q_blocks=q_blocks,
                causal=causal,
            ),
            "kv_cache_misses": loads,
            "kv_reload_factor": loads / (2 * self.k_blocks),
        }


def plan_kv_residency(
    *,
    k_blocks: int,
    mlen: int,
    matrix_sram_tiles: int,
    requested_residency_fraction: float | None = None,
    policy: str = "raw-tiles",
    force_streaming: bool = False,
) -> KVResidencyPlan:
    """Use all safely available capacity as a deterministic prefix cache."""

    if k_blocks <= 0 or mlen <= 0:
        raise ValueError("k_blocks and mlen must be positive")
    if matrix_sram_tiles < 1:
        raise ValueError("Matrix SRAM requires at least one tile")
    if requested_residency_fraction is not None and not (
        0.0 <= requested_residency_fraction <= 1.0
    ):
        raise ValueError("requested_residency_fraction must be in [0, 1]")

    if force_streaming:
        resident = 0
    elif matrix_sram_tiles >= 2 * k_blocks:
        resident = k_blocks
    else:
        resident = min(k_blocks - 1, max(0, (matrix_sram_tiles - 2) // 2))

    streaming = k_blocks - resident
    tile_elements = mlen * mlen
    stream_k = 2 * resident * tile_elements if streaming else None
    stream_v = (
        stream_k
        if streaming and matrix_sram_tiles == 1
        else (2 * resident + 1) * tile_elements
        if streaming
        else None
    )
    requested = (
        resident / k_blocks
        if requested_residency_fraction is None
        else requested_residency_fraction
    )
    return KVResidencyPlan(
        k_blocks=k_blocks,
        resident_prefix_blocks=resident,
        streaming_blocks=streaming,
        stream_k_address=stream_k,
        stream_v_address=stream_v,
        requested_residency_fraction=requested,
        realized_residency_fraction=resident / k_blocks,
        matrix_sram_tiles=matrix_sram_tiles,
        tile_elements=tile_elements,
        policy=policy,
    )


def derive_matrix_sram_policy(
    *,
    policy: str,
    k_blocks: int,
    mlen: int,
    projection_tiles: int,
) -> KVResidencyPlan:
    """Resolve one public DSE policy into physical capacity and residency."""

    if policy not in MATRIX_SRAM_POLICIES:
        raise ValueError(
            f"unsupported Matrix SRAM policy {policy!r}; "
            f"expected one of {MATRIX_SRAM_POLICIES}"
        )
    if projection_tiles <= 0:
        raise ValueError("projection_tiles must be positive")

    if policy == "streaming":
        tiles = 2
        fraction = 0.0
        force_streaming = True
    elif policy == "projection-full":
        tiles = max(2, projection_tiles)
        fraction = None
        force_streaming = False
    else:
        fraction = _POLICY_FRACTIONS[policy]
        requested_prefix = math.ceil(fraction * k_blocks)
        cache_tiles = (
            2 * k_blocks
            if fraction == 1.0
            else 2 + 2 * requested_prefix
        )
        tiles = max(projection_tiles, cache_tiles)
        force_streaming = False

    return plan_kv_residency(
        k_blocks=k_blocks,
        mlen=mlen,
        matrix_sram_tiles=tiles,
        requested_residency_fraction=fraction,
        policy=policy,
        force_streaming=force_streaming,
    )


__all__ = [
    "KVResidencyPlan",
    "MATRIX_SRAM_POLICIES",
    "derive_matrix_sram_policy",
    "plan_kv_residency",
]
