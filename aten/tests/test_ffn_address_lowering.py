from __future__ import annotations

import math
from collections import Counter

import pytest

from compiler.asm_templates.ffn_asm import ffn_asm
from compiler.asm_templates.ffn_address_plan import (
    build_ffn_address_plan,
    summarize_ffn_address_optimization,
)
from compiler.asm_templates.ffn_projection_plan import (
    build_ffn_projection_plan,
)
from compiler.aten.agu import optimize_agu_assembly
from compiler.aten.plena.cost_kernels import ffn_unrolled_cost_counts


def _summary(mode: str):
    return ffn_unrolled_cost_counts(
        mlen=8192,
        vlen=8192,
        blen=32,
        batch_rows=32768,
        hidden_size=8192,
        intermediate_size=32768,
        activation_base_address=0,
        workspace_base_address=1 << 32,
        matrix_sram_size=2 * 8192,
        gate_weight_hbm_base=0,
        up_weight_hbm_base=1 << 30,
        down_weight_hbm_base=2 << 30,
        hbm_prefetch_amount=8192,
        ffn_address_schedule=mode,
    )


def test_live_stride_preserves_matrix_and_dma_work() -> None:
    legacy = _summary("legacy")
    live = _summary("live-stride-v1")

    for opcode in ("M_MM", "M_MM_WO", "H_PREFETCH_M"):
        assert live.dynamic[opcode] == legacy.dynamic[opcode]
    assert sum(stream.multiplicity for stream in live.memory_streams) == sum(
        stream.multiplicity for stream in legacy.memory_streams
    )


def test_live_stride_removes_large_immediate_expansion() -> None:
    legacy = _summary("legacy")
    live = _summary("live-stride-v1")

    assert live.dynamic["S_ADDI_INT"] < legacy.dynamic["S_ADDI_INT"] * 0.35
    assert live.dynamic["S_ADD_INT"] > legacy.dynamic["S_ADD_INT"]


@pytest.mark.parametrize("k_tile_count", [1, 2, 4])
def test_live_stride_pointer_liveness(k_tile_count: int) -> None:
    plan = build_ffn_address_plan(
        mode="live-stride-v1",
        k_tile_count=k_tile_count,
        num_activation_columns=7,
    )

    assert plan.prefetch_pointer_updates == k_tile_count - 1
    assert plan.matrix_pointer_updates == (0 if k_tile_count == 1 else k_tile_count)
    assert plan.activation_pointer_updates == plan.matrix_pointer_updates
    assert plan.output_pointer_updates == 6
    assert plan.dead_k_pointer_updates_elided == (2 if k_tile_count == 1 else 0)


def test_address_metadata_counts_physical_output_rows() -> None:
    summary = summarize_ffn_address_optimization(
        mode="live-stride-v1",
        mlen=8192,
        blen=32,
        batch_rows=32768,
        hidden_size=8192,
        intermediate_size=32768,
        max_k_tiles=2,
    )

    # Up and gate each have 1024 output rows and one K tile. Down has four
    # K tiles split into two two-tile chunks, so only the first two projections
    # eliminate dead post-M_MM K-pointer updates.
    assert summary["ffn_dead_k_pointer_updates_elided"] == 2 * 2 * 1024 * 1024
    # One dead output increment is removed per physical output row and K chunk.
    assert summary["ffn_dead_output_updates_elided"] == 2 * 1024 + 2 * 256


def _dynamic_histogram(assembly: str) -> Counter[str]:
    histogram: Counter[str] = Counter()
    loops: list[tuple[int, bool]] = []
    for raw_line in assembly.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        opcode = line.split(maxsplit=1)[0]
        multiplier = math.prod(count for count, _ in loops)
        if opcode == "C_LOOP_END" and loops[-1][1]:
            loops.pop()
            continue
        histogram[opcode] += multiplier
        if opcode == "C_LOOP_START":
            loops.append((int(line.rsplit(",", 1)[1]), False))
        elif opcode == "C_LOOP_START_AGU":
            loops.append((int(line.rsplit(",", 1)[1]), True))
        elif opcode == "C_LOOP_END":
            loops.pop()
    assert not loops
    return histogram


def test_affine_projection_plan_preserves_semantic_census() -> None:
    args = dict(
        mlen=64,
        blen=8,
        batch_rows=128,
        k_size=256,
        out_size=512,
        max_k_tiles=3,
    )
    affine = build_ffn_projection_plan(schedule="affine-loop-v2", **args)
    compatibility = build_ffn_projection_plan(schedule="legacy-auto-v1", **args)

    assert affine.semantic_census() == compatibility.semantic_census()
    assert tuple((item.k_start_tile, item.k_tile_count) for item in affine.chunks) == (
        (0, 3),
        (3, 1),
    )
    assert affine.explicit_loop_depth == 4
    assert max(dict(affine.agu_streams_by_axis).values()) == 2


def test_affine_loop_matches_legacy_matrix_and_dma_work() -> None:
    common = dict(
        mlen=64,
        vlen=64,
        blen=8,
        batch=128,
        seq_len=1,
        hidden_size=128,
        intermediate_size=256,
        alive_registers=list(range(1, 16)),
        gate_weight_hbm_offset_reg=1,
        up_weight_hbm_offset_reg=2,
        down_weight_hbm_offset_reg=3,
        const_one_fp_address=5,
        activation_base_address=1 << 20,
        use_loop_instructions=True,
        matrix_sram_size=4 * 64,
        workspace_base_address=1 << 24,
        ffn_address_schedule="live-stride-v1",
    )
    legacy, _ = optimize_agu_assembly(
        ffn_asm(
            **common,
            ffn_projection_schedule="legacy-auto-v1",
        ),
        mode="loop-agu-v1",
    )
    affine, metadata = optimize_agu_assembly(
        ffn_asm(
            **common,
            ffn_projection_schedule="affine-loop-v2",
        ),
        mode="loop-agu-v1",
    )
    legacy_counts = _dynamic_histogram(legacy)
    affine_counts = _dynamic_histogram(affine)

    for opcode in ("M_MM", "M_MM_WO", "H_PREFETCH_M", "V_ADD_VV"):
        assert affine_counts[opcode] == legacy_counts[opcode]
    assert affine_counts["S_ADDI_INT"] <= legacy_counts["S_ADDI_INT"]
    assert max(int(key) for key in metadata["agu_stream_count_histogram"]) <= 6


@pytest.mark.parametrize(
    ("mlen", "blen"),
    [
        (256, 32),
        (512, 64),
        (1024, 128),
        (2048, 1024),
        (4096, 64),
        (8192, 32),
    ],
)
@pytest.mark.parametrize(
    ("hidden_tiles", "intermediate_tiles"),
    [(1, 1), (2, 4), (4, 7)],
)
def test_affine_loop_post_agu_domain_has_no_address_regression(
    mlen: int,
    blen: int,
    hidden_tiles: int,
    intermediate_tiles: int,
) -> None:
    """Exercise the full supported width range against the real AGU rewrite."""

    common = dict(
        mlen=mlen,
        vlen=mlen,
        blen=blen,
        batch=max(2 * blen, mlen),
        seq_len=1,
        hidden_size=hidden_tiles * mlen,
        intermediate_size=intermediate_tiles * mlen,
        alive_registers=list(range(1, 16)),
        gate_weight_hbm_offset_reg=1,
        up_weight_hbm_offset_reg=2,
        down_weight_hbm_offset_reg=3,
        const_one_fp_address=5,
        activation_base_address=1 << 28,
        use_loop_instructions=True,
        matrix_sram_size=max(hidden_tiles, intermediate_tiles) * mlen,
        workspace_base_address=1 << 36,
        ffn_address_schedule="live-stride-v1",
    )
    legacy, _ = optimize_agu_assembly(
        ffn_asm(**common, ffn_projection_schedule="legacy-auto-v1"),
        mode="loop-agu-v1",
    )
    affine, metadata = optimize_agu_assembly(
        ffn_asm(**common, ffn_projection_schedule="affine-loop-v2"),
        mode="loop-agu-v1",
    )
    legacy_counts = _dynamic_histogram(legacy)
    affine_counts = _dynamic_histogram(affine)

    for opcode in ("M_MM", "M_MM_WO", "H_PREFETCH_M", "V_ADD_VV"):
        assert affine_counts[opcode] == legacy_counts[opcode]

    address_control = {
        "S_ADDI_INT",
        "S_ADD_INT",
        "S_LUI_INT",
        "C_LOOP_START",
        "C_LOOP_START_AGU",
        "C_AGU_CONFIG",
        "C_LOOP_END",
    }
    assert sum(affine_counts[opcode] for opcode in address_control) <= sum(
        legacy_counts[opcode] for opcode in address_control
    )
    assert max(int(key) for key in metadata["agu_stream_count_histogram"]) <= 6
