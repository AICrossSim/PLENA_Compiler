"""Shared structural plan for dense FFN projection lowering.

The plan describes the loop topology and K-chunk partition independently of
either the assembly renderer or CostEmitter.  Keeping this information in one
place prevents Matrix SRAM capacity from selecting a different implementation
with different address-generation costs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._k_split import k_chunks


FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2 = "affine-loop-v2"
FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1 = "legacy-auto-v1"
FFN_PROJECTION_SCHEDULES = (
    FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2,
    FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1,
)
FFN_LOOP_PLAN_VERSION = "ffn-affine-loop-ir-v2"


@dataclass(frozen=True)
class FfnProjectionChunkPlan:
    """One K chunk executed without changing the partial-sum order."""

    chunk_index: int
    k_start_tile: int
    k_tile_count: int
    output_block_count: int
    output_tiles_per_block: int
    activation_column_count: int


@dataclass(frozen=True)
class FfnProjectionPlan:
    """Loop census shared by assembly and analytical cost lowering."""

    schedule: str
    version: str
    mlen: int
    blen: int
    batch_rows: int
    k_size: int
    out_size: int
    max_k_tiles: int
    chunks: tuple[FfnProjectionChunkPlan, ...]
    explicit_loop_depth: int
    agu_streams_by_axis: tuple[tuple[str, int], ...]

    @property
    def matrix_compute_count(self) -> int:
        return sum(
            chunk.output_block_count
            * chunk.output_tiles_per_block
            * chunk.activation_column_count
            * chunk.k_tile_count
            for chunk in self.chunks
        )

    @property
    def matrix_writeout_count(self) -> int:
        return sum(
            chunk.output_block_count
            * chunk.output_tiles_per_block
            * chunk.activation_column_count
            for chunk in self.chunks
        )

    @property
    def weight_prefetch_count(self) -> int:
        return sum(
            chunk.output_block_count * chunk.k_tile_count
            for chunk in self.chunks
        )

    @property
    def partial_sum_accumulate_count(self) -> int:
        if len(self.chunks) <= 1:
            return 0
        return (len(self.chunks) - 1) * (
            self.out_size * self.batch_rows // self.mlen
        )

    def semantic_census(self) -> dict[str, int]:
        return {
            "k_chunk_count": len(self.chunks),
            "M_MM": self.matrix_compute_count,
            "M_MM_WO": self.matrix_writeout_count,
            "H_PREFETCH_M": self.weight_prefetch_count,
            "V_ADD_VV": self.partial_sum_accumulate_count,
        }


def build_ffn_projection_plan(
    *,
    schedule: str,
    mlen: int,
    blen: int,
    batch_rows: int,
    k_size: int,
    out_size: int,
    max_k_tiles: int,
) -> FfnProjectionPlan:
    if schedule not in FFN_PROJECTION_SCHEDULES:
        raise ValueError(
            f"unsupported FFN projection schedule {schedule!r}; "
            f"expected one of {FFN_PROJECTION_SCHEDULES}"
        )
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError(f"invalid Matrix shape MLEN={mlen}, BLEN={blen}")
    if batch_rows <= 0 or batch_rows % blen:
        raise ValueError(
            f"FFN batch_rows={batch_rows} must be a positive multiple of BLEN={blen}"
        )
    if k_size <= 0 or k_size % mlen:
        raise ValueError(f"FFN K={k_size} must be a positive multiple of MLEN={mlen}")
    if out_size <= 0 or out_size % mlen:
        raise ValueError(
            f"FFN output size={out_size} must be a positive multiple of MLEN={mlen}"
        )
    if max_k_tiles <= 0:
        raise ValueError("max_k_tiles must be positive")

    chunks = tuple(
        FfnProjectionChunkPlan(
            chunk_index=index,
            k_start_tile=k_start,
            k_tile_count=k_count,
            output_block_count=out_size // mlen,
            output_tiles_per_block=mlen // blen,
            activation_column_count=batch_rows // blen,
        )
        for index, (k_start, k_count) in enumerate(
            k_chunks(k_size // mlen, max_k_tiles)
        )
    )
    return FfnProjectionPlan(
        schedule=schedule,
        version=FFN_LOOP_PLAN_VERSION,
        mlen=mlen,
        blen=blen,
        batch_rows=batch_rows,
        k_size=k_size,
        out_size=out_size,
        max_k_tiles=max_k_tiles,
        chunks=chunks,
        explicit_loop_depth=4,
        agu_streams_by_axis=(
            ("output_block", 2),
            ("output_tile", 1),
            ("activation_column", 2),
            ("k_accumulation", 2),
            ("k_prefetch", 2),
        ),
    )


__all__ = [
    "FFN_LOOP_PLAN_VERSION",
    "FFN_PROJECTION_SCHEDULE_AFFINE_LOOP_V2",
    "FFN_PROJECTION_SCHEDULE_LEGACY_AUTO_V1",
    "FFN_PROJECTION_SCHEDULES",
    "FfnProjectionChunkPlan",
    "FfnProjectionPlan",
    "build_ffn_projection_plan",
]
