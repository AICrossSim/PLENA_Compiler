"""Parity tests for the shared native normalization instruction plans."""

from __future__ import annotations

from collections import Counter
import math

from compiler.aten.plena.normalization_plan import (
    build_active_row_rms_norm,
    build_grouped_segmented_rms_norm,
    build_split_head_segmented_rms_norm,
)


def _dynamic_histogram(rendered: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    loop_stack: list[int] = []
    for raw_line in rendered.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        opcode = line.split(maxsplit=1)[0]
        counts[opcode] += math.prod(loop_stack)
        if opcode == "C_LOOP_START":
            loop_stack.append(int(line.rsplit(",", 1)[1]))
        elif opcode == "C_LOOP_END":
            loop_stack.pop()
    assert not loop_stack
    return counts


def test_grouped_segmented_norm_asm_and_cost_counts_match() -> None:
    lowering = build_grouped_segmented_rms_norm(
        name="q_norm",
        tensor_base_address=1024,
        scratch_base_address=8192,
        physical_rows=16,
        physical_cols=16,
        mlen=16,
        hlen=4,
        segments=((0, 0), (0, 1), (0, 2), (0, 3)),
        active_row_ranges=((0, 7), (8, 15)),
        gp_src=1,
        gp_scratch=2,
        gp_mask=3,
        gp_loop=4,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    # Four direct square evaluations per active row preserve legacy reduction
    # semantics, while source copy is shared once by the four lanes.
    assert lowering.dynamic_opcodes["V_MUL_VV"] == 14 * 4
    assert lowering.dynamic_opcodes["V_ADD_VV"] == 14
    assert lowering.metadata == {
        "segmented_norm_square_ops_elided": 8,
        "segmented_norm_copy_ops_elided": 50,
        "segmented_norm_constant_loads_elided": 126,
        "inactive_norm_rows_elided": 8,
    }


def test_active_row_rms_norm_asm_and_cost_counts_match() -> None:
    lowering = build_active_row_rms_norm(
        name="decoder_norm",
        activation_base_address=2048,
        scratch_base_address=16384,
        physical_rows=16,
        hidden_dim=32,
        vlen=16,
        active_row_ranges=((0, 7), (8, 15)),
        gp_row=1,
        gp_scratch=2,
        gp_stats=3,
        gp_act=4,
        gp_loop=5,
        gp_stride=6,
        epsilon_slot=3,
        reciprocal_hidden_slot=4,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM"] == 14 * 2
    assert lowering.dynamic_opcodes["V_MUL_VF"] == 14 * 2
    assert lowering.metadata["inactive_norm_rows_elided"] == 2
    assert lowering.metadata["rms_norm_address_loads_elided"] > 0
    assert lowering.metadata["rms_norm_nops_elided"] == 42


def test_rtl_v2_grouped_norm_uses_segment_reduction_and_scalar_extensions() -> None:
    lowering = build_grouped_segmented_rms_norm(
        name="q_norm_rtl_v2",
        tensor_base_address=1024,
        scratch_base_address=8192,
        physical_rows=16,
        physical_cols=16,
        mlen=16,
        hlen=4,
        segments=((0, 0), (0, 1), (0, 2), (0, 3)),
        active_row_ranges=((0, 7), (8, 15)),
        gp_src=1,
        gp_scratch=2,
        gp_mask=3,
        gp_loop=4,
        rtl_v2=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM_SEG"] == 14 * 4
    assert lowering.dynamic_opcodes["V_RED_SUM"] == 0
    assert lowering.dynamic_opcodes["S_RSQRT_FP"] == 14 * 4
    assert lowering.dynamic_opcodes["S_SQRT_FP"] == 0
    assert lowering.dynamic_opcodes["S_RECI_FP"] == 0
    assert lowering.metadata["segment_reductions_emitted"] == 14 * 4


def test_rtl_v2_active_row_norm_uses_move_and_rsqrt() -> None:
    lowering = build_active_row_rms_norm(
        name="decoder_norm_rtl_v2",
        activation_base_address=2048,
        scratch_base_address=16384,
        physical_rows=16,
        hidden_dim=32,
        vlen=16,
        active_row_ranges=((0, 7), (8, 15)),
        gp_row=1,
        gp_scratch=2,
        gp_stats=3,
        gp_act=4,
        gp_loop=5,
        gp_stride=6,
        epsilon_slot=3,
        reciprocal_hidden_slot=4,
        rtl_v2=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["S_RSQRT_FP"] == 14
    assert lowering.dynamic_opcodes["S_SQRT_FP"] == 0
    assert lowering.dynamic_opcodes["S_RECI_FP"] == 0
    assert lowering.dynamic_opcodes["S_MV_FP"] == 15


def test_rtl_v3_grouped_norm_emits_one_multi_reduction_per_block_row() -> None:
    lowering = build_grouped_segmented_rms_norm(
        name="q_norm_rtl_v3",
        tensor_base_address=1024,
        scratch_base_address=8192,
        physical_rows=16,
        physical_cols=16,
        mlen=16,
        hlen=4,
        segments=((0, 0), (0, 1), (0, 2), (0, 3)),
        active_row_ranges=((0, 7), (8, 15)),
        gp_src=1,
        gp_scratch=2,
        gp_mask=3,
        gp_loop=4,
        gp_stats=5,
        rtl_v3=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM_SEGS"] == 14
    assert lowering.dynamic_opcodes["S_LD_VLANE_FP"] == 14 * 4
    assert lowering.dynamic_opcodes["S_ST_VLANE_FP"] == 14 * 4
    assert lowering.dynamic_opcodes["V_MUL_VSEG"] == 14
    assert lowering.metadata["single_segment_reductions_elided"] == 14 * 3


def test_rtl_v4_grouped_norm_replaces_scalar_lane_chains() -> None:
    lowering = build_grouped_segmented_rms_norm(
        name="q_norm_rtl_v4",
        tensor_base_address=1024,
        scratch_base_address=8192,
        physical_rows=16,
        physical_cols=16,
        mlen=16,
        hlen=4,
        segments=((0, 0), (0, 1), (0, 2), (0, 3)),
        active_row_ranges=((0, 7), (8, 15)),
        gp_src=1,
        gp_scratch=2,
        gp_mask=3,
        gp_loop=4,
        gp_stats=5,
        rtl_v3=True,
        rtl_v4=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM_SEGS"] == 14
    assert lowering.dynamic_opcodes["V_STAT_MUL_F"] == 14
    assert lowering.dynamic_opcodes["V_STAT_ADD_F"] == 14
    assert lowering.dynamic_opcodes["V_STAT_RSQRT"] == 14
    assert lowering.dynamic_opcodes["S_LD_VLANE_FP"] == 0
    assert lowering.dynamic_opcodes["S_ST_VLANE_FP"] == 0
    assert lowering.metadata["compact_scalar_chain_ops_elided"] == 14 * 4 * 5
    assert lowering.metadata["compact_lane_selectors_elided"] == 14 * 4 * 2


def test_rtl_v4_grouped_norm_falls_back_for_more_than_16_segments() -> None:
    lowering = build_grouped_segmented_rms_norm(
        name="q_norm_rtl_v4_wide",
        tensor_base_address=1024,
        scratch_base_address=32768,
        physical_rows=4,
        physical_cols=4096,
        mlen=4096,
        hlen=128,
        segments=tuple((0, lane) for lane in range(32)),
        active_row_ranges=((0, 4),),
        gp_src=1,
        gp_scratch=2,
        gp_mask=3,
        gp_loop=4,
        gp_stats=5,
        rtl_v3=True,
        rtl_v4=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM_SEGS"] == 0
    assert lowering.dynamic_opcodes["V_STAT_RSQRT"] == 0
    assert lowering.dynamic_opcodes["V_RED_SUM_SEG"] == 4 * 32
    assert lowering.dynamic_opcodes["S_RSQRT_FP"] == 4 * 32
    assert lowering.metadata["segment_parallel_fallback_blocks"] == 1
    assert lowering.metadata["segment_parallel_fallback_segments"] == 32
    assert lowering.metadata["segment_parallel_requested_rtl_v4"] == 1


def test_rtl_v3_split_k_norm_emits_one_multi_reduction_for_all_heads() -> None:
    lowering = build_split_head_segmented_rms_norm(
        name="k_norm_rtl_v3",
        tensor_base_addresses=(1024, 2048, 3072, 4096),
        scratch_base_address=8192,
        physical_rows=16,
        mlen=16,
        hlen=4,
        active_row_ranges=((0, 7), (8, 15)),
        gp_heads=(1, 2, 3, 4),
        gp_packed=5,
        gp_shifted=6,
        gp_stats=7,
        gp_index=8,
        gp_loop=9,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_RED_SUM_SEGS"] == 14
    assert lowering.dynamic_opcodes["V_SHIFT_V"] == 14 * 4
    assert lowering.dynamic_opcodes["V_MUL_VF"] == 14 * 5
    assert lowering.metadata["single_segment_reductions_elided"] == 14 * 3


def test_rtl_v4_split_k_norm_keeps_only_required_lane_reads() -> None:
    lowering = build_split_head_segmented_rms_norm(
        name="k_norm_rtl_v4",
        tensor_base_addresses=(1024, 2048, 3072, 4096),
        scratch_base_address=8192,
        physical_rows=16,
        mlen=16,
        hlen=4,
        active_row_ranges=((0, 7), (8, 15)),
        gp_heads=(1, 2, 3, 4),
        gp_packed=5,
        gp_shifted=6,
        gp_stats=7,
        gp_index=8,
        gp_loop=9,
        rtl_v4=True,
    )

    assert lowering.dynamic_opcodes == _dynamic_histogram(lowering.rendered_asm)
    assert lowering.dynamic_opcodes["V_STAT_MUL_F"] == 14
    assert lowering.dynamic_opcodes["V_STAT_ADD_F"] == 14
    assert lowering.dynamic_opcodes["V_STAT_RSQRT"] == 14
    assert lowering.dynamic_opcodes["S_LD_VLANE_FP"] == 14 * 4
    assert lowering.dynamic_opcodes["S_ST_VLANE_FP"] == 0
    assert lowering.dynamic_opcodes["S_MUL_FP"] == 0
    assert lowering.dynamic_opcodes["S_ADD_FP"] == 0
    assert lowering.dynamic_opcodes["S_RSQRT_FP"] == 0
    assert lowering.metadata["compact_scalar_chain_ops_elided"] == 14 * 4 * 4
    assert lowering.metadata["compact_lane_selectors_elided"] == 14 * 4
    assert lowering.metadata["compact_lane_selectors_remaining"] == 14 * 4
