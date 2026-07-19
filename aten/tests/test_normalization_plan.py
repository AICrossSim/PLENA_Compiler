"""Parity tests for the shared native normalization instruction plans."""

from __future__ import annotations

from collections import Counter
import math

from compiler.aten.plena.normalization_plan import (
    build_active_row_rms_norm,
    build_grouped_segmented_rms_norm,
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
