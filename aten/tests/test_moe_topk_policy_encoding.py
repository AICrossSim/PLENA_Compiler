"""Encoder guards on the packed ``C_SET_TOPK_REG`` policy immediate.

The packing ``(num_experts << 8) | top_k`` must fit ``S_ADDI_INT``'s 18-bit
``IMM_2_WIDTH`` field (bits 14..32), not the 22-bit ``IMM_WIDTH`` field that
belongs to ``S_LUI_INT``/``C_LOOP_START``.

These tests pin both tiers, in both directions, and assemble the result -- an
unlegalized wide ``S_ADDI_INT`` is rejected by ``AssemblyToBinary``, so the
round-trip is what makes the claim falsifiable rather than decorative.
"""

from __future__ import annotations

import pathlib

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.isa_builder import IsaBuilder, gp
from compiler.aten.plena.program_routed_moe import (
    _TOPK_POLICY_EXPERT_MASK,
    _TOPK_POLICY_CORRECTION_BIAS,
    _TOPK_POLICY_MAX_PACKED,
    _TOPK_POLICY_SIGMOID_NORMALIZED,
    _TOPK_POLICY_SINGLE_ADDI_MAX_PACKED,
    _pack_topk_policy,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: ``S_ADDI_INT`` immediate width, from ``doc/configuration.svh``: the assembler
#: places it at ``OPCODE_WIDTH + 2 * OPERAND_WIDTH`` = bit 14 of a 32-bit word.
IMM_2_WIDTH = 18


def _assembler() -> AssemblyToBinary:
    return AssemblyToBinary(
        str(REPO_ROOT / "doc" / "operation.svh"),
        str(REPO_ROOT / "doc" / "configuration.svh"),
    )


def test_single_addi_ceiling_matches_the_imm2_field() -> None:
    assert _TOPK_POLICY_SINGLE_ADDI_MAX_PACKED == (1 << IMM_2_WIDTH) - 1 == 262143
    # 1023 experts, not 16383, is what one instruction buys.
    assert _TOPK_POLICY_SINGLE_ADDI_MAX_PACKED >> 8 == 1023


def _render_policy(
    num_experts: int,
    top_k: int,
    route_weight_mode: str = "softmax",
) -> list[str]:
    packed = _pack_topk_policy(num_experts, top_k, route_weight_mode)
    asm = IsaBuilder()
    asm.instr("S_ADDI_INT", gp(4), gp(0), packed)
    asm.instr("C_SET_TOPK_REG", gp(4))
    return asm.render().splitlines()


@pytest.mark.parametrize(
    ("num_experts", "top_k", "packed"),
    [
        (32, 4, 8196),  # GPT-OSS
        (128, 8, 32776),  # Qwen3-30B-A3B
        (256, 8, 65544),  # DeepSeek-V3 / Kimi K2
        (1023, 255, 262143),  # last shape one S_ADDI_INT admits
    ],
)
def test_narrow_policies_emit_one_addi(num_experts: int, top_k: int, packed: int) -> None:
    assert _pack_topk_policy(num_experts, top_k) == packed
    assert packed <= _TOPK_POLICY_SINGLE_ADDI_MAX_PACKED
    assert _render_policy(num_experts, top_k) == [
        f"S_ADDI_INT gp4, gp0, {packed}",
        "C_SET_TOPK_REG gp4",
    ]


@pytest.mark.parametrize(("num_experts", "top_k"), [(1024, 1), (16383, 255)])
def test_wide_policies_are_legalized_into_a_lui_pair(num_experts: int, top_k: int) -> None:
    packed = _pack_topk_policy(num_experts, top_k)
    assert packed > _TOPK_POLICY_SINGLE_ADDI_MAX_PACKED
    assert _render_policy(num_experts, top_k) == [
        f"S_LUI_INT gp4, {packed >> 12}",
        f"S_ADDI_INT gp4, gp4, {packed & 0xFFF}",
        "C_SET_TOPK_REG gp4",
    ]


def test_packing_ceiling_is_enforced() -> None:
    assert _TOPK_POLICY_MAX_PACKED == (1 << 24) - 1
    _pack_topk_policy(_TOPK_POLICY_EXPERT_MASK, 255)
    with pytest.raises(ValueError, match="does not fit 14 bits"):
        _pack_topk_policy(_TOPK_POLICY_EXPERT_MASK + 1, 1)


def test_kimi_sigmoid_normalized_mode_is_packed_and_legalized() -> None:
    packed = _pack_topk_policy(
        896,
        16,
        "sigmoid_normalized",
        correction_bias=True,
    )
    assert packed == (
        _TOPK_POLICY_CORRECTION_BIAS
        | _TOPK_POLICY_SIGMOID_NORMALIZED
        | (896 << 8)
        | 16
    )
    asm = IsaBuilder()
    asm.instr("S_ADDI_INT", gp(4), gp(0), packed)
    asm.instr("C_SET_TOPK_REG", gp(4))
    assert asm.render().splitlines() == [
        f"S_LUI_INT gp4, {packed >> 12}",
        f"S_ADDI_INT gp4, gp4, {packed & 0xFFF}",
        "C_SET_TOPK_REG gp4",
    ]


@pytest.mark.parametrize(("num_experts", "top_k"), [(32, 4), (1023, 255), (1024, 1), (16383, 255)])
def test_emitted_policy_assembles(tmp_path: pathlib.Path, num_experts: int, top_k: int) -> None:
    """The whole point: a wide raw S_ADDI_INT does not survive the assembler."""
    asm_path = tmp_path / "policy.asm"
    asm_path.write_text("\n".join(_render_policy(num_experts, top_k)) + "\n")
    _assembler().generate_binary(str(asm_path), str(tmp_path / "policy.bin"))
    words = [int(line, 16) for line in (tmp_path / "policy.bin").read_text().split()]
    assert words, "the assembler produced no words"
    assert all(0 <= word <= 0xFFFFFFFF for word in words)


def test_unlegalized_wide_immediate_is_rejected(tmp_path: pathlib.Path) -> None:
    """Guards the claim the other way: 2**18 really is the S_ADDI_INT ceiling."""
    asm_path = tmp_path / "raw.asm"
    asm_path.write_text(f"S_ADDI_INT gp4, gp0, {1 << IMM_2_WIDTH}\n")
    with pytest.raises(ValueError, match="Instruction encoding overflow"):
        _assembler().generate_binary(str(asm_path), str(tmp_path / "raw.bin"))
    # One below the ceiling is the widest value that does survive.
    ok_path = tmp_path / "ok.asm"
    ok_path.write_text(f"S_ADDI_INT gp4, gp0, {(1 << IMM_2_WIDTH) - 1}\n")
    _assembler().generate_binary(str(ok_path), str(tmp_path / "ok.bin"))
