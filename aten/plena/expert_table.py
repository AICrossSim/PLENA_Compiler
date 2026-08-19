"""Physical HBM layout shared by routed-MoE model frontends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compiler.aten.plena.vars import InputVar

if TYPE_CHECKING:
    from compiler.aten.plena.compiler import PlenaCompiler


@dataclass(frozen=True)
class ExpertWeightTable:
    """A tile-major expert table addressable by a runtime expert id."""

    template: InputVar
    base: int
    stride: int
    num_experts: int
    tile_group_stride: int | None = None


def reserve_expert_weight_table(
    prog: PlenaCompiler,
    *,
    name: str,
    num_experts: int,
    rows: int,
    cols: int,
) -> ExpertWeightTable:
    """Reserve a tile-major HBM expert table and its logical template.

    A 32-bit GP cannot form ``expert_id * full_expert_size`` for large expert
    tables. Tile-major storage keeps the dynamic term to one MLEN square tile;
    each power-of-two tile group stays within one 4-GiB low-address window,
    while the compiler writes the group's static high bits to the HBM address
    register.
    """
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    stride = prog.hbm_tensor_size(rows * cols)
    block_size = prog.mlen * prog.mlen
    row_blocks = (rows + prog.mlen - 1) // prog.mlen
    col_blocks = (cols + prog.mlen - 1) // prog.mlen
    # One MXFP8 tile also owns the scale bytes that immediately follow its
    # element stream. Keeping each tile self-contained gives every expert one
    # constant runtime byte stride while C_SET_SCALE_REG remains the logical
    # element count.
    tile_hbm_stride = prog.hbm_tensor_size(block_size)
    raw_tile_group = num_experts * tile_hbm_stride
    tile_group_stride = 1 << (raw_tile_group - 1).bit_length()
    if tile_group_stride > 1 << 32:
        raise ValueError("one expert tile group must fit a 32-bit dynamic offset")
    aligned = (
        (prog._next_hbm_addr + tile_group_stride - 1) // tile_group_stride
    ) * tile_group_stride
    prog._next_hbm_addr = aligned
    base = prog._allocate_hbm(row_blocks * col_blocks * tile_group_stride)
    template = prog.input(
        name,
        shape=(rows, cols),
        physical_shape=(rows, cols),
        hbm_addr=base,
    )
    return ExpertWeightTable(
        template=template,
        base=base,
        stride=stride,
        num_experts=num_experts,
        tile_group_stride=tile_group_stride,
    )


__all__ = ["ExpertWeightTable", "reserve_expert_weight_table"]
