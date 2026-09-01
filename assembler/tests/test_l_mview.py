from pathlib import Path

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction


ROOT = Path(__file__).resolve().parents[2]


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )


def test_full_and_field_share_0x3f_and_use_distinct_functs() -> None:
    assembler = _assembler()
    full = assembler._convert_to_binary(
        Instruction("L_MVIEW_FULL", 2, 7, 9, None, None, None)
    )
    field = assembler._convert_to_binary(
        Instruction("L_MVIEW_FIELD", 2, None, 9, None, None, None, imm=2)
    )
    assert full & 0x3F == field & 0x3F == 0x3F
    assert (full >> 22) & 0xF == 1
    assert (field >> 22) & 0xF == 2


def test_matrix_fourth_operand_is_an_explicit_view_slot() -> None:
    assembler = _assembler()
    # M_BMV architecturally consumes rd as its matrix offset, so a non-zero rd
    # locks every register field without violating M_MV's rd=0 convention.
    plain = assembler._convert_to_binary(Instruction("M_BMV", 5, 1, 2, None, None, None))
    viewed = assembler._convert_to_binary(Instruction("M_BMV", 5, 1, 2, 3, None, None))
    assert plain == 0x09 | (5 << 6) | (1 << 10) | (2 << 14)
    assert viewed & ((1 << 22) - 1) == plain
    assert (viewed >> 22) & 0xF == 4


def test_view_slot_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="view slot"):
        _assembler()._convert_to_binary(Instruction("M_TMM", 0, 1, 2, 4, None, None))


def test_matrix_view_vector_alias_keeps_arithmetic_opcode_and_marks_operands() -> None:
    assembler = _assembler()
    plain = assembler._convert_to_binary(
        Instruction("V_ADD_VV", 4, 5, 6, 0, 0, None)
    )
    viewed = assembler._convert_to_binary(
        Instruction("V_ADD_VV.MV", 4, 5, 6, 0, 0b110, None)
    )
    assert plain & 0x3F == viewed & 0x3F == 0x0D
    assert plain >> 22 == 0
    assert (viewed >> 22) & 0xF == 0x8 | 0b110


def test_matrix_view_vector_alias_rejects_an_empty_operand_mask() -> None:
    with pytest.raises(ValueError, match="cannot be zero"):
        _assembler()._convert_to_binary(
            Instruction("V_MUL_VV.MV", 1, 1, 2, 0, 0, None)
        )


def test_accumulator_writeback_uses_existing_m_mm_wo_with_explicit_view() -> None:
    assembler = _assembler()
    legacy = assembler._convert_to_binary(
        Instruction("M_MM_WO", 4, 0, None, None, None, None, imm=5)
    )
    viewed = assembler._convert_to_binary(
        Instruction("M_MM_WO", 4, 0, None, 2, None, None, imm=5)
    )
    assert legacy & 0x3F == viewed & 0x3F == 0x06
    legacy_imm = legacy >> 14
    viewed_imm = viewed >> 14
    assert legacy_imm == 5
    assert viewed_imm == (1 << 17) | (2 << 15) | 5


def test_legacy_m_mm_wo_cannot_alias_the_view_marker() -> None:
    with pytest.raises(ValueError, match="bit 17"):
        _assembler()._convert_to_binary(
            Instruction("M_MM_WO", 4, 0, None, None, None, None, imm=1 << 17)
        )


def test_generate_binary_enforces_matrix_view_dominance(tmp_path: Path) -> None:
    source = tmp_path / "bad.asm"
    output = tmp_path / "bad.mem"
    source.write_text("M_TMM gp0, gp1, gp2, 2\n")

    with pytest.raises(ValueError, match="before a dominating configuration"):
        _assembler().generate_binary(str(source), str(output))

    assert not output.exists()
