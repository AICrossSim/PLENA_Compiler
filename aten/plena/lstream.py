"""ISA contract for PLENA's model-independent affine stream configuration.

``L_STREAM_CFG`` configures how an existing Matrix/Vector/scalar operand is
addressed.  Existing opcodes still define all arithmetic and existing
``C_LOOP_START/END`` still define repetition.  This avoids both an
algorithm-specific step instruction and a second loop ISA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, IntFlag

from compiler.aten.isa_builder import IsaBuilder, fp, gp
from compiler.aten.plena.affine_layout import AffineLayout


L_STREAM_CFG_OPCODE = 0x3C
L_STREAM_CONTRACT_VERSION = 1
L_STREAM_MAX_SLOTS = 4


class StreamConfigField(IntEnum):
    RESET = 0
    FLAGS = 1
    BASE = 2
    EXTENT_MINOR = 3
    EXTENT_MAJOR = 4
    EXTENT_FIELD = 5
    EXTENT_GROUP = 6
    BANK_ROW_PITCH = 7
    ALPHA = 8
    BETA = 9
    GAMMA = 10
    ADVANCE = 11
    PACKET_ELEMENTS = 12
    STORAGE_ATOM = 13
    PHYSICAL_BASE_ROW = 14


class StreamFlags(IntFlag):
    ENABLE = 1 << 0
    AUTO_ADVANCE = 1 << 1
    AFFINE = 1 << 2
    TARGET_FP = 1 << 3
    WRITE = 1 << 4
    LANE_RESTORE = 1 << 5
    STRICT_BOUNDS = 1 << 6


@dataclass(frozen=True)
class StreamBinding:
    slot: int
    target_register: int
    target_is_fp: bool
    base: int
    advance: int
    packet_elements: int
    storage_atom: int
    auto_advance: bool = True
    write: bool = False
    lane_restore: bool = True

    def validate(self) -> None:
        if not 0 <= self.slot < L_STREAM_MAX_SLOTS:
            raise ValueError(f"stream slot must be in [0, {L_STREAM_MAX_SLOTS}), got {self.slot}")
        target_limit = 8 if self.target_is_fp else 16
        if not 0 <= self.target_register < target_limit:
            raise ValueError(
                f"stream target register must be in [0, {target_limit}), got {self.target_register}"
            )
        if self.base < 0 or self.advance < 0:
            raise ValueError("stream base and advance must be non-negative")
        if self.packet_elements <= 0 or self.storage_atom <= 0:
            raise ValueError("stream packet and storage atom must be positive")


def encode_l_stream_cfg_word(
    *, value_register: int, target_register: int, slot: int, field: StreamConfigField | int
) -> int:
    """Encode the canonical register-register-immediate L_STREAM_CFG word."""

    field_value = int(field)
    try:
        StreamConfigField(field_value)
    except ValueError as error:
        raise ValueError(f"reserved L_STREAM_CFG field {field_value}") from error
    for name, value in (
        ("value_register", value_register),
        ("target_register", target_register),
        ("slot", slot),
        ("field", field_value),
    ):
        if not 0 <= value < 16:
            raise ValueError(f"{name} must fit four bits, got {value}")
    if slot >= L_STREAM_MAX_SLOTS:
        raise ValueError(f"slot {slot} exceeds the implemented {L_STREAM_MAX_SLOTS} slots")
    return (
        L_STREAM_CFG_OPCODE
        | value_register << 6
        | target_register << 10
        | slot << 14
        | field_value << 18
    )


def decode_l_stream_cfg_word(word: int) -> tuple[int, int, int, StreamConfigField]:
    if word < 0 or word > 0xFFFF_FFFF:
        raise ValueError("instruction word must be unsigned 32-bit")
    if word & 0x3F != L_STREAM_CFG_OPCODE or word >> 22:
        raise ValueError("word is not a canonical L_STREAM_CFG encoding")
    value_register = word >> 6 & 0xF
    target_register = word >> 10 & 0xF
    slot = word >> 14 & 0xF
    if slot >= L_STREAM_MAX_SLOTS:
        raise ValueError(f"slot {slot} exceeds the implemented {L_STREAM_MAX_SLOTS} slots")
    try:
        field = StreamConfigField(word >> 18 & 0xF)
    except ValueError as error:
        raise ValueError("reserved L_STREAM_CFG field") from error
    return value_register, target_register, slot, field


def emit_stream_configuration(
    *,
    value_gp: int,
    binding: StreamBinding,
    layout: AffineLayout,
) -> IsaBuilder:
    """Emit deterministic slot setup; FLAGS is written last to validate/enable."""

    binding.validate()
    target = fp(binding.target_register) if binding.target_is_fp else gp(binding.target_register)
    asm = IsaBuilder().comment(f"L-stream slot {binding.slot} affine configuration")

    def write(field: StreamConfigField, value: int) -> None:
        asm.instr("S_ADDI_INT", gp(value_gp), gp(0), value)
        asm.instr("L_STREAM_CFG", gp(value_gp), target, binding.slot, int(field))

    # RESET has no payload, so gp0 is canonical and costs one instruction.
    asm.instr(
        "L_STREAM_CFG",
        gp(0),
        target,
        binding.slot,
        int(StreamConfigField.RESET),
    )
    for field, value in (
        (StreamConfigField.BASE, binding.base),
        (StreamConfigField.EXTENT_MINOR, layout.minors),
        (StreamConfigField.EXTENT_MAJOR, layout.majors),
        (StreamConfigField.EXTENT_FIELD, layout.fields),
        (StreamConfigField.EXTENT_GROUP, layout.groups),
        (StreamConfigField.BANK_ROW_PITCH, layout.bank_row_pitch),
        (StreamConfigField.ALPHA, layout.alpha),
        (StreamConfigField.BETA, layout.beta),
        (StreamConfigField.GAMMA, layout.gamma),
        (StreamConfigField.ADVANCE, binding.advance),
        (StreamConfigField.PACKET_ELEMENTS, binding.packet_elements),
        (StreamConfigField.STORAGE_ATOM, binding.storage_atom),
        (StreamConfigField.PHYSICAL_BASE_ROW, layout.bank_row_base),
    ):
        defaults = {
            StreamConfigField.EXTENT_MINOR: 1,
            StreamConfigField.EXTENT_MAJOR: 1,
            StreamConfigField.EXTENT_FIELD: 1,
            StreamConfigField.EXTENT_GROUP: 1,
            StreamConfigField.BANK_ROW_PITCH: 0,
            StreamConfigField.ALPHA: 0,
            StreamConfigField.BETA: 0,
            StreamConfigField.GAMMA: 0,
            StreamConfigField.ADVANCE: 0,
            StreamConfigField.PACKET_ELEMENTS: 1,
            StreamConfigField.STORAGE_ATOM: 1,
            StreamConfigField.PHYSICAL_BASE_ROW: 0,
        }
        if field == StreamConfigField.BASE or value != defaults[field]:
            write(field, value)

    flags = StreamFlags.ENABLE | StreamFlags.STRICT_BOUNDS
    if binding.auto_advance:
        flags |= StreamFlags.AUTO_ADVANCE
    if layout.alpha or layout.beta or layout.gamma:
        flags |= StreamFlags.AFFINE
    if binding.target_is_fp:
        flags |= StreamFlags.TARGET_FP
    if binding.write:
        flags |= StreamFlags.WRITE
    if binding.lane_restore:
        flags |= StreamFlags.LANE_RESTORE
    write(StreamConfigField.FLAGS, int(flags))
    return asm


__all__ = [
    "L_STREAM_CFG_OPCODE",
    "L_STREAM_CONTRACT_VERSION",
    "L_STREAM_MAX_SLOTS",
    "StreamBinding",
    "StreamConfigField",
    "StreamFlags",
    "decode_l_stream_cfg_word",
    "emit_stream_configuration",
    "encode_l_stream_cfg_word",
]
