from pathlib import Path

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction
from compiler.aten.plena.mview import LTilePrimitive


ROOT = Path(__file__).resolve().parents[2]


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )


def test_config_uses_0x3f_and_the_config_funct() -> None:
    word = _assembler()._convert_to_binary(
        Instruction("L_TILE_CFG", 2, 7, 9, None, None, None)
    )
    assert word & 0x3F == 0x3F
    assert (word >> 22) & 0xF == 1


@pytest.mark.parametrize("primitive", list(LTilePrimitive))
def test_exec_uses_0x3f_and_a_distinct_funct(
    primitive: LTilePrimitive,
) -> None:
    word = _assembler()._convert_to_binary(
        Instruction("L_TILE_EXEC", 4, 5, 6, int(primitive), None, None)
    )
    assert word & 0x3F == 0x3F
    assert (word >> 22) & 0xF == 3
    assert (word >> 18) & 0xF == int(primitive)


def test_exec_rejects_a_reserved_primitive() -> None:
    with pytest.raises(ValueError, match="reserved L_TILE primitive"):
        _assembler()._convert_to_binary(
            Instruction("L_TILE_EXEC", 4, 5, 6, 3, None, None)
        )


def test_exec_encodes_optional_operand_axes_without_spending_an_opcode() -> None:
    word = _assembler()._convert_to_binary(
        Instruction("L_TILE_EXEC", 4, 5, 6, 1, 0b10, None)
    )
    assert word & 0x3F == 0x3F
    assert (word >> 22) & 0xF == 3
    assert (word >> 26) & 0b11 == 0b10


def test_exec_rejects_reserved_operand_axes() -> None:
    with pytest.raises(ValueError, match="axis mask"):
        _assembler()._convert_to_binary(
            Instruction("L_TILE_EXEC", 4, 5, 6, 1, 4, None)
        )


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


@pytest.mark.parametrize(
    ("mnemonic", "opcode"),
    [("H_PREFETCH_V.MV", 0x29), ("H_STORE_V.MV", 0x2A)],
)
def test_vector_dma_matrix_view_form_reuses_opcode_and_names_slot(
    mnemonic: str, opcode: int
) -> None:
    word = _assembler()._convert_to_binary(
        Instruction(mnemonic, 1, 2, 3, 1, 2, 3)
    )
    assert word & 0x3F == opcode
    assert word >> 31 == 1
    assert (word >> 29) & 0b11 == 3
    assert (word >> 26) & 0b111 == 0
    assert (word >> 22) & 0xF == 2
    assert word & ((1 << 26) - 1) == (
        opcode | (1 << 6) | (2 << 10) | (3 << 14) | (1 << 18) | (2 << 22)
    )


@pytest.mark.parametrize("precision", [0, 1, 2])
def test_vector_dma_matrix_view_accepts_all_canonical_precisions(
    precision: int,
) -> None:
    word = _assembler()._convert_to_binary(
        Instruction("H_PREFETCH_V.MV", 1, 2, 3, 1, precision, 0)
    )
    assert (word >> 22) & 0xF == precision


def test_vector_dma_matrix_view_form_rejects_reserved_slot() -> None:
    with pytest.raises(ValueError, match="view slot"):
        _assembler()._convert_to_binary(
            Instruction("H_PREFETCH_V.MV", 1, 2, 3, 1, 2, 4)
        )


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
