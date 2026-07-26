"""Shared FFN pointer-liveness and invariant-stride planning."""

from __future__ import annotations

from dataclasses import dataclass

from ._imm import IMM2_BOUND
from ._k_split import k_chunks


FFN_ADDRESS_SCHEDULE_LEGACY = "legacy"
FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1 = "live-stride-v1"
FFN_ADDRESS_SCHEDULES = (
    FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1,
    FFN_ADDRESS_SCHEDULE_LEGACY,
)


@dataclass(frozen=True)
class FfnAddressPlan:
    mode: str
    k_tile_count: int
    num_activation_columns: int
    prefetch_pointer_updates: int
    matrix_pointer_updates: int
    activation_pointer_updates: int
    output_pointer_updates: int
    dead_k_pointer_updates_elided: int
    dead_prefetch_updates_elided: int
    dead_output_updates_elided: int


def build_ffn_address_plan(
    *,
    mode: str,
    k_tile_count: int,
    num_activation_columns: int,
) -> FfnAddressPlan:
    if mode not in FFN_ADDRESS_SCHEDULES:
        raise ValueError(
            f"unsupported FFN address schedule {mode!r}; "
            f"expected one of {FFN_ADDRESS_SCHEDULES}"
        )
    if k_tile_count <= 0 or num_activation_columns <= 0:
        raise ValueError("FFN tile and activation-column counts must be positive")
    if mode == FFN_ADDRESS_SCHEDULE_LEGACY:
        return FfnAddressPlan(
            mode=mode,
            k_tile_count=k_tile_count,
            num_activation_columns=num_activation_columns,
            prefetch_pointer_updates=k_tile_count,
            matrix_pointer_updates=k_tile_count,
            activation_pointer_updates=k_tile_count,
            output_pointer_updates=num_activation_columns,
            dead_k_pointer_updates_elided=0,
            dead_prefetch_updates_elided=0,
            dead_output_updates_elided=0,
        )

    live_k_updates = k_tile_count if k_tile_count > 1 else 0
    return FfnAddressPlan(
        mode=mode,
        k_tile_count=k_tile_count,
        num_activation_columns=num_activation_columns,
        prefetch_pointer_updates=max(0, k_tile_count - 1),
        matrix_pointer_updates=live_k_updates,
        activation_pointer_updates=live_k_updates,
        output_pointer_updates=max(0, num_activation_columns - 1),
        dead_k_pointer_updates_elided=2 if k_tile_count == 1 else 0,
        dead_prefetch_updates_elided=2,
        dead_output_updates_elided=1,
    )


def uses_invariant_stride(value: int, *, update_count: int, mode: str) -> bool:
    return (
        mode == FFN_ADDRESS_SCHEDULE_LIVE_STRIDE_V1
        and update_count > 0
        and value >= IMM2_BOUND
    )


def summarize_ffn_address_optimization(
    *,
    mode: str,
    mlen: int,
    blen: int,
    batch_rows: int,
    hidden_size: int,
    intermediate_size: int,
    max_k_tiles: int,
) -> dict[str, int | str]:
    """Count eliminated raw pointer updates across the three FFN projections."""

    totals = {
        "ffn_dead_k_pointer_updates_elided": 0,
        "ffn_dead_prefetch_updates_elided": 0,
        "ffn_dead_output_updates_elided": 0,
        "ffn_invariant_stride_loads": 0,
    }
    if mode == FFN_ADDRESS_SCHEDULE_LEGACY:
        return {"ffn_address_schedule": mode, **totals}

    activation_columns = batch_rows // blen
    projections = (
        (hidden_size, intermediate_size),
        (hidden_size, intermediate_size),
        (intermediate_size, hidden_size),
    )
    for k_size, out_size in projections:
        output_blocks = out_size // mlen
        output_rows = out_size // blen
        for _, tile_count in k_chunks(k_size // mlen, max_k_tiles):
            plan = build_ffn_address_plan(
                mode=mode,
                k_tile_count=tile_count,
                num_activation_columns=activation_columns,
            )
            cells = output_rows * activation_columns
            totals["ffn_dead_k_pointer_updates_elided"] += (
                plan.dead_k_pointer_updates_elided * cells
            )
            totals["ffn_dead_prefetch_updates_elided"] += (
                plan.dead_prefetch_updates_elided * output_blocks
            )
            totals["ffn_dead_output_updates_elided"] += (
                plan.dead_output_updates_elided * output_rows
            )

    live_stride_values = (
        mlen * mlen,
        mlen * batch_rows,
        blen * mlen,
    )
    totals["ffn_invariant_stride_loads"] = sum(
        value >= IMM2_BOUND for value in live_stride_values
    )
    return {"ffn_address_schedule": mode, **totals}
