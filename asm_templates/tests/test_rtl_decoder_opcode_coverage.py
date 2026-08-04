"""Pin the RTL matrix-opcode execution contract used by attention lowering.

`definitions/operation.svh` assigns an encoding to the whole matrix opcode
family, but an encoding is executable only when the decoder both classifies it
and maps it to a non-stall ``m_op``. These tests parse the semantic assignment,
not unrelated opcode comparisons elsewhere in the decoder.

This matters for the two attention lowerings. The batch-packed path
(`q_len > 1`) issues `M_BTMM` / `M_BMM_WO` for QK^T and `M_MM` / `M_MM_WO` for
PV. The single-token path (`q_len == 1`) issues `M_BTMV` / `M_BMV_WO` instead,
and those are among the encodings the decoder does not match, so the emitted
program stalls rather than computing.

The single-vector broadcast instructions remain explicit ISA reservations and
must not be treated as executable until their datapaths are implemented.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

def _rtl_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == "PLENA_RTL" and (parent / "src").is_dir():
            return parent
    return Path(__file__).resolve().parents[4] / "PLENA_RTL"


RTL_ROOT = _rtl_root()
DECODER = RTL_ROOT / "src" / "frontend" / "rtl" / "decoder.sv"
OPERATIONS = RTL_ROOT / "src" / "definitions" / "operation.svh"

# Opcodes each attention lowering emits, from asm_templates/flashattn.
BATCH_PACKED_OPCODES = {"M_BTMM", "M_BMM_WO", "M_MM", "M_MM_WO"}
SINGLE_TOKEN_OPCODES = {"M_BTMV", "M_BMV_WO", "M_MV", "M_MV_WO"}
ISSUED_MATRIX_OPCODES = {
    "M_MM", "M_TMM", "M_BMM", "M_BTMM", "M_MM_WO", "M_BMM_WO",
    "M_MV", "M_TMV", "M_MV_WO",
}
UNSUPPORTED_MATRIX_OPCODES = {"M_BMV", "M_BTMV", "M_BMV_WO"}
TRANSPOSED_MATRIX_OPCODES = {"M_TMM", "M_BTMM", "M_TMV"}

requires_rtl = pytest.mark.skipif(
    not DECODER.is_file() or not OPERATIONS.is_file(),
    reason="RTL checkout not present",
)


def declared_matrix_opcodes() -> set[str]:
    """Matrix opcodes the ISA header assigns an encoding to."""
    return set(
        re.findall(r"^\s*(M_[A-Z_]+)\s*=\s*6'h[0-9a-fA-F]+", OPERATIONS.read_text(), re.M)
    )


def classified_matrix_opcodes() -> set[str]:
    """Matrix opcodes classified as matrix instructions."""
    text = DECODER.read_text()
    loaded_case = re.search(r"case\s*\(loaded_opcode\)(.*?)endcase", text, re.S)
    if loaded_case is None:
        raise AssertionError("could not find loaded_opcode case statement")
    arms = re.findall(
        r"^\s*((?:M_[A-Z_]+\s*,\s*)*M_[A-Z_]+)\s*:\s*begin",
        loaded_case.group(1),
        re.M,
    )
    matched: set[str] = set()
    for arm in arms:
        matched.update(name.strip() for name in arm.split(","))
    return matched


def _semantic_matrix_opcodes(text: str) -> set[str]:
    matrix_arm = re.search(r"^\s*M:\s*begin(.*?)^\s*V:\s*begin", text, re.M | re.S)
    if matrix_arm is None:
        raise AssertionError("could not find semantic matrix decode arm")
    assignment = re.search(
        r"decode_stage_op\.m_op\s*<=\s*(.*?);",
        matrix_arm.group(1),
        re.S,
    )
    if assignment is None:
        raise AssertionError("could not find semantic m_op assignment")
    return set(re.findall(r"decode_instr_info\.opcode\s*==\s*(M_[A-Z_]+)", assignment.group(1)))


def issued_matrix_opcodes() -> set[str]:
    """Matrix opcodes mapped to a non-stall matrix-machine operation."""
    return _semantic_matrix_opcodes(DECODER.read_text())


def transposed_matrix_opcodes() -> set[str]:
    text = DECODER.read_text()
    assignment = re.search(
        r"decode_stage_op\.m_transposed_read\s*<=\s*(.*?)\?\s*1'b1\s*:\s*1'b0",
        text,
        re.S,
    )
    if assignment is None:
        raise AssertionError("could not find semantic transposed-read assignment")
    return set(re.findall(r"decode_instr_info\.opcode\s*==\s*(M_[A-Z_]+)", assignment.group(1)))


@requires_rtl
def test_batch_packed_attention_uses_only_decoded_opcodes() -> None:
    """The validated decode program must be executable on the RTL."""
    missing = BATCH_PACKED_OPCODES - issued_matrix_opcodes()
    assert not missing, (
        f"batch-packed attention emits {sorted(missing)}, which "
        f"{DECODER.name} does not decode"
    )


@requires_rtl
def test_single_token_attention_opcodes_are_not_all_decoded() -> None:
    """The single-token path is not executable, and that is the current gap.

    `M_BTMV` and `M_BMV_WO` carry ISA encodings but no decoder arm, so a
    `q_len == 1` program stalls. Pinning it keeps the gap visible; implementing
    those opcodes in the RTL is what makes this test change.
    """
    undecoded = SINGLE_TOKEN_OPCODES - issued_matrix_opcodes()
    assert undecoded == {"M_BTMV", "M_BMV_WO"}, (
        f"expected exactly M_BTMV and M_BMV_WO to be undecoded, got {sorted(undecoded)}"
    )


@requires_rtl
def test_undecoded_opcodes_still_carry_isa_encodings() -> None:
    """The gap is in the decoder, not the ISA: the encodings do exist."""
    declared = declared_matrix_opcodes()
    for opcode in UNSUPPORTED_MATRIX_OPCODES:
        assert opcode in declared, f"{opcode} missing from {OPERATIONS.name}"
        assert opcode not in classified_matrix_opcodes()
        assert opcode not in issued_matrix_opcodes()


@requires_rtl
def test_matrix_opcode_support_is_explicit() -> None:
    assert classified_matrix_opcodes() == ISSUED_MATRIX_OPCODES
    assert issued_matrix_opcodes() == ISSUED_MATRIX_OPCODES
    assert declared_matrix_opcodes() == ISSUED_MATRIX_OPCODES | UNSUPPORTED_MATRIX_OPCODES


def test_transposed_matrix_operations_select_transposed_reads() -> None:
    assert transposed_matrix_opcodes() == TRANSPOSED_MATRIX_OPCODES


def test_semantic_parser_ignores_unrelated_opcode_comparisons() -> None:
    """A parser miss or a broad whole-file regex must fail this self-check."""
    synthetic = """
        M: begin
            debug_match = decode_instr_info.opcode == M_DISTRACTOR;
            decode_stage_op.m_op <=
                (decode_instr_info.opcode == M_ALPHA ||
                 decode_instr_info.opcode == M_BETA) ? MM_IC :
                (decode_instr_info.opcode == M_GAMMA) ? MV_IC : STALL_M;
        end
        V: begin
        end
    """
    assert _semantic_matrix_opcodes(synthetic) == {"M_ALPHA", "M_BETA", "M_GAMMA"}
