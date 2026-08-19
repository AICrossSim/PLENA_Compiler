from __future__ import annotations

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction
from aten.state import StateSubop, decode_instruction


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")


def test_assembler_matches_typed_x_state_codec() -> None:
    instruction = Instruction(
        "X_STATE",
        1,
        2,
        3,
        4,
        int(StateSubop.STEP),
        None,
        None,
    )
    word = _assembler()._convert_to_binary(instruction)
    assert decode_instruction(word)["subop"] == StateSubop.STEP
    assert (word >> 18) & 0xF == 4


def test_assembler_rejects_noncanonical_fence() -> None:
    instruction = Instruction(
        "X_STATE",
        1,
        0,
        0,
        2,
        int(StateSubop.FENCE),
        None,
        None,
    )
    with pytest.raises(ValueError, match="FENCE"):
        _assembler()._convert_to_binary(instruction)


@pytest.mark.parametrize("field_index", [1, 2])
def test_assembler_rejects_operands_wider_than_their_field(field_index: int) -> None:
    # rd=16 used to encode 0x00c0043d, which decodes as a different
    # (context_gp, descriptor_offset_gp) pair instead of raising: 16 << 6 carries
    # straight into the neighbouring four-bit field, well under the 32-bit
    # overflow guard.
    operands = [None, 1, 2, 3, 4, int(StateSubop.STEP), None, None]
    operands[field_index] = 16
    with pytest.raises(ValueError, match="four-bit instruction field"):
        _assembler()._convert_to_binary(Instruction("X_STATE", *operands[1:]))
