"""Binary codec for the shared PLENA Mamba command ABI.

This module only encodes and validates the frozen wire format.  It does not
claim that the compiler, simulator, or RTL executes ``X_MAMBA`` yet.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping

from .generated_contract import (
    MAMBA_COMPLETION_FIELDS,
    MAMBA_COMPLETION_SIZE,
    MAMBA_DESCRIPTOR_FIELDS,
    MAMBA_DESCRIPTOR_MAGIC,
    MAMBA_DESCRIPTOR_SIZE,
    MAMBA_DESCRIPTOR_VERSION,
    MAMBA_SUBOPS,
    OPCODES,
)


_STRUCT_FORMAT = {"u16": "H", "u32": "I", "u64": "Q"}
_DESCRIPTOR_DEFAULTS = {
    "magic": MAMBA_DESCRIPTOR_MAGIC,
    "version": MAMBA_DESCRIPTOR_VERSION,
    "size_bytes": MAMBA_DESCRIPTOR_SIZE,
}


def _pack_record(
    fields: Mapping[str, tuple[int, str]],
    size: int,
    values: Mapping[str, int],
) -> bytes:
    unknown = set(values) - set(fields)
    if unknown:
        raise ValueError(f"unknown ABI fields: {sorted(unknown)}")
    result = bytearray(size)
    for name, (offset, field_type) in fields.items():
        value = values.get(name, 0)
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        try:
            struct.pack_into(f"<{_STRUCT_FORMAT[field_type]}", result, offset, value)
        except struct.error as error:
            raise ValueError(f"{name}={value} does not fit {field_type}") from error
    return bytes(result)


def _unpack_record(
    fields: Mapping[str, tuple[int, str]], size: int, data: bytes
) -> dict[str, int]:
    if len(data) != size:
        raise ValueError(f"ABI record must be exactly {size} bytes, got {len(data)}")
    return {
        name: struct.unpack_from(f"<{_STRUCT_FORMAT[field_type]}", data, offset)[0]
        for name, (offset, field_type) in fields.items()
    }


def pack_mamba_descriptor(values: Mapping[str, int]) -> bytes:
    """Pack a 256-byte little-endian descriptor with canonical header values."""
    merged = dict(_DESCRIPTOR_DEFAULTS)
    merged.update(values)
    data = _pack_record(MAMBA_DESCRIPTOR_FIELDS, MAMBA_DESCRIPTOR_SIZE, merged)
    unpack_mamba_descriptor(data)
    return data


def unpack_mamba_descriptor(data: bytes) -> dict[str, int]:
    """Unpack a descriptor and reject an incompatible header."""
    values = _unpack_record(MAMBA_DESCRIPTOR_FIELDS, MAMBA_DESCRIPTOR_SIZE, data)
    expected = _DESCRIPTOR_DEFAULTS
    for name, value in expected.items():
        if values[name] != value:
            raise ValueError(
                f"invalid Mamba descriptor {name}: expected {value}, got {values[name]}"
            )
    return values


def pack_mamba_completion(values: Mapping[str, int]) -> bytes:
    return _pack_record(MAMBA_COMPLETION_FIELDS, MAMBA_COMPLETION_SIZE, values)


def unpack_mamba_completion(data: bytes) -> dict[str, int]:
    return _unpack_record(MAMBA_COMPLETION_FIELDS, MAMBA_COMPLETION_SIZE, data)


def _check_u4(name: str, value: int) -> None:
    if not isinstance(value, int) or not 0 <= value < 16:
        raise ValueError(f"{name} must fit the 4-bit operand field, got {value!r}")


def encode_mamba_instruction(
    context_gp: int,
    descriptor_offset_gp: int,
    descriptor_hbm_reg: int,
    queue_id: int,
    subop: int,
) -> int:
    """Encode `X_MAMBA rd, rs1, rs2, queue, subop` in `R_FUNCT` format."""
    _check_u4("context_gp", context_gp)
    _check_u4("descriptor_offset_gp", descriptor_offset_gp)
    _check_u4("descriptor_hbm_reg", descriptor_hbm_reg)
    _check_u4("queue_id", queue_id)
    _check_u4("subop", subop)
    if descriptor_hbm_reg >= 8:
        raise ValueError(
            "portable Mamba descriptors can use only HBM address registers 0..7"
        )
    if subop not in MAMBA_SUBOPS.values():
        raise ValueError(f"unknown Mamba sub-operation {subop}")
    return (
        OPCODES["X_MAMBA"]
        | (context_gp << 6)
        | (descriptor_offset_gp << 10)
        | (descriptor_hbm_reg << 14)
        | (queue_id << 18)
        | (subop << 22)
    )


def decode_mamba_instruction(word: int) -> dict[str, int]:
    """Decode a canonical Mamba instruction and reject reserved high bits."""
    if not isinstance(word, int) or not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("instruction word must be an unsigned 32-bit integer")
    if word & 0xFC000000:
        raise ValueError("reserved instruction bits 26..31 must be zero")
    if word & 0x3F != OPCODES["X_MAMBA"]:
        raise ValueError("instruction is not X_MAMBA")
    values = {
        "context_gp": (word >> 6) & 0xF,
        "descriptor_offset_gp": (word >> 10) & 0xF,
        "descriptor_hbm_reg": (word >> 14) & 0xF,
        "queue_id": (word >> 18) & 0xF,
        "subop": (word >> 22) & 0xF,
    }
    if values["descriptor_hbm_reg"] >= 8:
        raise ValueError("non-portable Mamba HBM address-register index")
    if values["subop"] not in MAMBA_SUBOPS.values():
        raise ValueError(f"unknown Mamba sub-operation {values['subop']}")
    return values
