from pathlib import Path

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.plena.hybrid_l_tile_schedule import (
    build_official_hybrid_l_tile_report,
    build_official_hybrid_l_tile_schedules,
)
from compiler.aten.plena.matrix_recurrence_lowering import (
    KIMI_KDA,
    NEMOTRON_MAMBA,
    RecurrenceLayout,
)
from compiler.aten.plena.mview import validate_matrix_view_dominance


ROOT = Path(__file__).resolve().parents[2]
MODEL_LIB = ROOT / "doc/Model_Lib"


@pytest.mark.parametrize(
    ("model", "layers", "recurrent_layers", "state_bytes"),
    [
        ("nemotron3", 52, 23, 23 * NEMOTRON_MAMBA.state_bytes_per_layer),
        ("kimi_k3", 93, 69, 69 * KIMI_KDA.state_bytes_per_layer),
    ],
)
@pytest.mark.parametrize("layout", list(RecurrenceLayout))
def test_every_official_recurrent_layer_emits_l_tile(
    model: str,
    layers: int,
    recurrent_layers: int,
    state_bytes: int,
    layout: RecurrenceLayout,
) -> None:
    schedule = build_official_hybrid_l_tile_schedules(
        MODEL_LIB,
        layout=layout,
    )[model]
    report = schedule.to_report()

    assert report["layer_count"] == layers
    assert report["recurrent_layer_count"] == recurrent_layers
    assert report["all_recurrent_layers_emit_l_tile"] is True
    assert len(schedule.records) == recurrent_layers
    assert report["state_hbm_arena"]["bytes"] == state_bytes
    assert report["l_tile_exec_count"] == sum(
        record.l_tile_exec_count for record in schedule.records
    )
    # The frozen Matrix-SRAM path must not depend on the historical Vector
    # stream experiment that shares opcode 0x3f at funct1=0.
    assert all(
        not line.strip().startswith("L_CFG ")
        for line in schedule.assembly.splitlines()
    )
    for layer in range(1, layers + 1):
        assert f"@hybrid_layer_begin layer={layer} " in schedule.assembly
        assert f"@hybrid_layer_end layer={layer}" in schedule.assembly


@pytest.mark.parametrize("layout", list(RecurrenceLayout))
def test_hybrid_schedule_owns_disjoint_state_and_field_regions(
    layout: RecurrenceLayout,
) -> None:
    for schedule in build_official_hybrid_l_tile_schedules(
        MODEL_LIB,
        layout=layout,
    ).values():
        ranges = sorted(
            [
                (record.state_hbm_begin, record.state_hbm_end)
                for record in schedule.records
            ]
            + [
                (record.field_hbm_begin, record.field_hbm_end)
                for record in schedule.records
            ]
        )
        assert all(begin % 64 == end % 64 == 0 for begin, end in ranges)
        assert all(left[1] <= right[0] for left, right in zip(ranges, ranges[1:]))
        assert schedule.field_hbm_begin >= schedule.state_hbm_end


@pytest.mark.parametrize("model", ["nemotron3", "kimi_k3"])
def test_complete_affine_recurrence_schedule_is_legal_machine_code(
    model: str,
    tmp_path: Path,
) -> None:
    schedule = build_official_hybrid_l_tile_schedules(
        MODEL_LIB,
        layout=RecurrenceLayout.AFFINE,
    )[model]
    validate_matrix_view_dominance(schedule.assembly)
    asm_path = tmp_path / f"{model}.asm"
    asm_path.write_text(schedule.assembly)
    assembler = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    )
    words = [
        assembler._convert_to_binary(instruction)
        for instruction in parse_asm_file(str(asm_path))
    ]
    assert len(words) == schedule.to_report()["static_instructions"]
    assert all(0 <= word < 2**32 for word in words)
    assert any((word & 0x3F) == 0x3F for word in words)


def test_report_states_the_full_model_execution_boundary() -> None:
    report = build_official_hybrid_l_tile_report(MODEL_LIB)
    for variants in report["variants"].values():
        for model in ("nemotron3", "kimi_k3"):
            boundary = variants[model]["architectural_boundary"]
            assert boundary["recurrent_layer_lowering"] == (
                "executable PLENA instructions"
            )
            assert boundary["full_model_numerical_rust_execution"] is False
            assert boundary["cache"] is False
            assert boundary["private_state_sram"] is False
            assert boundary["runtime_scheduler"] is False
            assert boundary["new_mac_array"] is False
            assert variants[model]["all_recurrent_layers_emit_l_tile"] is True
