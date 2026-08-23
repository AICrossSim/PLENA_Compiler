from __future__ import annotations

import json
from pathlib import Path

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from assembler.parser import Instruction, load_isa_definitions, parse_asm_file


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary("doc/operation.svh", "doc/configuration.svh")


def _encode_text(tmp_path, text: str) -> list[int]:
    asm_path = tmp_path / "topk.asm"
    asm_path.write_text(text, encoding="utf-8")
    assembler = _assembler()
    return [assembler._convert_to_binary(op) for op in parse_asm_file(asm_path)]


def test_legacy_policy_spelling_is_byte_identical_to_explicit_target_zero(
    tmp_path,
) -> None:
    words = _encode_text(
        tmp_path,
        "C_SET_TOPK_REG gp7\nC_SET_TOPK_REG gp7, 0\n",
    )
    legacy_word = 0x38 | (7 << 6)
    assert words == [legacy_word, legacy_word]


def test_bias_target_uses_rs1_bits_without_changing_opcode(tmp_path) -> None:
    [word] = _encode_text(tmp_path, "C_SET_TOPK_REG gp13, 1\n")
    assert word == 0x38 | (13 << 6) | (1 << 10)
    assert word & 0x3F == 0x38
    assert (word >> 10) & 0xF == 1
    assert word >> 14 == 0


def test_encoder_matches_cross_repo_control_instruction_golden(tmp_path) -> None:
    golden = json.loads(
        Path("spec/x_state_v2_golden.json").read_text(encoding="utf-8")
    )["control_instruction_words"]
    words = _encode_text(
        tmp_path,
        "C_SET_TOPK_REG gp7\nC_SET_TOPK_REG gp13, 1\n",
    )
    assert [f"{word:08x}" for word in words] == [
        golden["c_set_topk_policy_gp7"],
        golden["c_set_topk_bias_gp13"],
    ]


@pytest.mark.parametrize("target", [-1, 2, 15, 16])
def test_rejects_reserved_or_out_of_range_targets(target: int) -> None:
    instruction = Instruction(
        "C_SET_TOPK_REG",
        4,
        None,
        None,
        None,
        None,
        None,
        target,
    )
    with pytest.raises(ValueError, match="target must be 0 .* or 1"):
        _assembler()._convert_to_binary(instruction)


def test_rejects_register_second_operand_instead_of_silently_defaulting_target() -> (
    None
):
    instruction = Instruction(
        "C_SET_TOPK_REG",
        4,
        1,
        None,
        None,
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="optional numeric target"):
        _assembler()._convert_to_binary(instruction)


def test_rejects_rd_that_would_carry_into_target_field() -> None:
    instruction = Instruction(
        "C_SET_TOPK_REG",
        16,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    with pytest.raises(ValueError, match="rd must fit"):
        _assembler()._convert_to_binary(instruction)


def test_legacy_bias_mnemonic_does_not_occupy_the_route_opcode() -> None:
    opcodes = load_isa_definitions("doc/operation.svh")
    assert opcodes["C_SET_TOPK_REG"] == 0x38
    assert "C_SET_TOPK_BIAS_REG" not in opcodes
    route_reservations = {
        0x39: "C_ROUTE_BEGIN",
        0x3A: "C_ROUTE_LOOP_START",
        0x3B: "C_ROUTE_LOOP_END",
        0x3C: "V_ROUTE_MUL",
    }
    for opcode, reserved_name in route_reservations.items():
        occupant = next(
            (name for name, value in opcodes.items() if value == opcode), None
        )
        assert occupant in {None, reserved_name}
    assert opcodes["X_STATE"] == 0x3D
    assert 0x3E not in opcodes.values()
    assert opcodes["L_SCATTER_M"] == 0x3F
