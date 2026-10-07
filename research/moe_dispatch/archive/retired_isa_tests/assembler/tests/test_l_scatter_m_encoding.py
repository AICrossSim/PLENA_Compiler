from __future__ import annotations

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction
from aten.state import (
    LayoutMode,
    decode_layout_instruction,
    encode_layout_instruction,
)


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")


def test_assembler_matches_typed_l_scatter_m_codec() -> None:
    instruction = Instruction(
        "L_SCATTER_M",
        1,
        2,
        3,
        1,
        int(LayoutMode.KDA_SKEW),
        None,
        None,
    )
    word = _assembler()._convert_to_binary(instruction)
    assert word == encode_layout_instruction(1, 2, 3, 1, LayoutMode.KDA_SKEW)
    assert decode_layout_instruction(word)["mode"] == LayoutMode.KDA_SKEW


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rd", 16, "context_gp"),
        ("rs1", 16, "descriptor_offset_gp"),
        ("rs2", 8, "descriptor_hbm_reg"),
        ("rstride", 16, "buffer_id"),
        ("funct1", 5, "layout_mode"),
    ],
)
def test_assembler_rejects_noncanonical_layout_fields(
    field: str, value: int, message: str
) -> None:
    values = {
        "rd": 1,
        "rs1": 2,
        "rs2": 3,
        "rstride": 1,
        "funct1": int(LayoutMode.MAMBA_SKEW),
    }
    values[field] = value
    instruction = Instruction(
        "L_SCATTER_M",
        values["rd"],
        values["rs1"],
        values["rs2"],
        values["rstride"],
        values["funct1"],
        None,
        None,
    )
    with pytest.raises(ValueError, match=message):
        _assembler()._convert_to_binary(instruction)
