from __future__ import annotations

import pytest

from aten.mamba.contract import (
    DESCRIPTOR_SIZE,
    MAMBA_OPCODE,
    MambaDescriptor,
    MambaSubop,
    StateLifecycle,
    decode_instruction,
    encode_instruction,
)


def test_instruction_codec_uses_single_free_opcode() -> None:
    word = encode_instruction(1, 2, 3, 4, MambaSubop.STEP)
    assert word & 0x3F == MAMBA_OPCODE == 0x39
    assert decode_instruction(word) == {
        "context_gp": 1,
        "descriptor_offset_gp": 2,
        "descriptor_hbm_reg": 3,
        "queue_id": 4,
        "subop": MambaSubop.STEP,
    }


def test_descriptor_round_trip_is_fixed_size() -> None:
    descriptor = MambaDescriptor(
        sequence_length=257,
        token_offset=256,
        valid_tokens=1,
        projection_addr=0x1234_5678_9ABC,
        state_addr=0x4567_89AB_CDEF,
    )
    packed = descriptor.pack()
    assert len(packed) == DESCRIPTOR_SIZE
    assert MambaDescriptor.unpack(packed) == descriptor


def test_state_lifecycle_rejects_step_without_resident_state() -> None:
    lifecycle = StateLifecycle()
    key = (0, 0)
    with pytest.raises(ValueError, match="nonresident"):
        lifecycle.apply(key, MambaSubop.STEP)
    lifecycle.apply(key, MambaSubop.STATE_PREFETCH)
    lifecycle.apply(key, MambaSubop.STEP)
    with pytest.raises(ValueError, match="non-clean"):
        lifecycle.apply(key, MambaSubop.STATE_EVICT)
    lifecycle.apply(key, MambaSubop.STATE_COMMIT)
    lifecycle.apply(key, MambaSubop.STATE_EVICT)
