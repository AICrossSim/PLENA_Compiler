from __future__ import annotations

import pytest

from compiler.aten.plena.matrix_prefill_handoff import (
    KdaStateOrientation,
    MatrixViewAxis,
    build_prefill_handoff_report,
    required_handoff_axis,
    validate_handoff_axis,
)


def test_prefill_and_decode_use_opposite_axes_without_moving_values() -> None:
    assert required_handoff_axis(
        KdaStateOrientation.PREFILL_VALUE_KEY,
        KdaStateOrientation.DECODE_KEY_VALUE,
    ) is MatrixViewAxis.COLUMN
    assert required_handoff_axis(
        KdaStateOrientation.DECODE_KEY_VALUE,
        KdaStateOrientation.DECODE_KEY_VALUE,
    ) is MatrixViewAxis.ROW
    with pytest.raises(ValueError, match="state view mismatch"):
        validate_handoff_axis(
            stored=KdaStateOrientation.PREFILL_VALUE_KEY,
            requested=KdaStateOrientation.DECODE_KEY_VALUE,
            selected=MatrixViewAxis.ROW,
        )


def test_real_kimi_shape_measures_the_emitted_identity_gemm() -> None:
    report = build_prefill_handoff_report()
    legacy = report["legacy_identity_gemm"]
    assert legacy["static_instructions"] == 49
    assert legacy["dynamic_issued_instructions_per_head"] == 42_584
    assert legacy["dynamic_opcode_census_per_head"]["M_TMM"] == 4_096
    assert legacy["dynamic_opcode_census_per_head"]["M_MM_WO"] == 4_096
    assert legacy["logical_macs_all_kda_layers"] == 13_891_534_848
    assert legacy["emitted_padded_macs_all_kda_layers"] == 56_899_726_737_408
    assert legacy["padding_over_logical_macs"] == 4_096
    assert legacy["emitted_matrix_cycles_all_kda_layers"] == 868_220_928


def test_matrix_view_handoff_checks_values_and_keeps_fp32_out_of_matrix_sram() -> None:
    report = build_prefill_handoff_report()
    view = report["matrix_view_handoff"]
    assert view["handoff_arithmetic_instructions"] == 0
    assert view["handoff_macs"] == 0
    assert view["same_physical_cells"] is True
    assert view["decode_axis"] == "column"
    assert view["descriptor"]["shape"] == {
        "rows": 128,
        "cols": 128,
        "tile_count": 1,
    }
    assert view["descriptor"]["mapping"]["fixed_wiring_alpha"] == 1
    assert view["direct_cross_head_residence_claimed"] is False
    assert view["value_evidence"]["values_checked"] == 16_384
    assert view["value_evidence"]["wrong_axis_rejected_before_execution"] is True

    boundary = report["precision_and_capacity_boundary"]
    assert boundary["official_fp32_state_bytes_per_layer"] == 6 * 1024 * 1024
    assert boundary["matrix_sram_bytes"] == 1024 * 1024
    assert boundary["official_fp32_state_matrix_resident"] is False
    assert boundary["bf16_heads_per_resident_window"] == 32
    assert boundary["bf16_windows_per_layer"] == 3
