"""Compatibility names for the Nemotron scheduler on the common X_STATE ABI."""

from aten.state import (
    FLAG_LAST_CHUNK,
    STREAMING_SRAM_OFFSET,
    Mamba2Payload,
    PrecisionCode,
    StateCommand,
    StateDescriptor,
    StateIdentity,
    StateLifecycle,
    StateSubop,
    X_STATE_OPCODE,
    decode_instruction,
    encode_instruction,
)

MAMBA_OPCODE = X_STATE_OPCODE
MambaCommand = StateCommand
MambaDescriptor = StateDescriptor
MambaSubop = StateSubop

__all__ = [
    "FLAG_LAST_CHUNK",
    "MAMBA_OPCODE",
    "STREAMING_SRAM_OFFSET",
    "Mamba2Payload",
    "MambaCommand",
    "MambaDescriptor",
    "MambaSubop",
    "PrecisionCode",
    "StateIdentity",
    "StateLifecycle",
    "decode_instruction",
    "encode_instruction",
]
