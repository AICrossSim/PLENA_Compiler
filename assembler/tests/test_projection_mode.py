"""The projection mode is explicit; legacy Matrix machine words stay unchanged."""

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import parse_asm_file
from compiler.aten.plena.mview import validate_matrix_view_dominance


def test_projection_register_fields_and_legacy_words(tmp_path):
    source = tmp_path / "projection.asm"
    source.write_text(
        "M_MM.P gp1, gp2, gp3, gp4, 3\nM_MM 0, gp2, gp3\nM_MM 0, gp2, gp3, 3\n"
    )
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    words = [assembler._convert_to_binary(i) for i in parse_asm_file(str(source))]
    legacy = 0x01 | (2 << 10) | (3 << 14)
    assert words == [
        legacy | (1 << 6) | (4 << 18) | (11 << 22),
        legacy,
        legacy | (4 << 22),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "M_MM.P gp1, gp2, gp3, 4, 0",
        "M_MM.P gp1, gp2, gp3, gp4",
        "M_MM.P gp1, gp2, gp3, gp4, 4",
        "M_MM.P gp16, gp2, gp3, gp4, 0",
        "M_MM.P gp1, gp2, gp3, gp4, x",
        "M_MM.P gp1, gp2, gp3, gp4, 0, 0",
    ],
)
def test_projection_rejects_ambiguous_or_overflowing_encoding(tmp_path, line):
    source = tmp_path / "invalid.asm"
    source.write_text(line + "\n")
    assembler = AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")
    with pytest.raises(ValueError):
        for instruction in parse_asm_file(str(source)):
            assembler._convert_to_binary(instruction)


def test_projection_requires_dominating_view():
    use = "M_MM.P gp1, gp2, gp3, gp4, 2\n"
    with pytest.raises(ValueError, match="dominating"):
        validate_matrix_view_dominance(use)
    validate_matrix_view_dominance("L_TILE_CFG 2, gp12, gp13\n" + use)
