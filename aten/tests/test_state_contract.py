from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from aten.state import (
    STREAMING_SRAM_OFFSET,
    KdaPayload,
    Mamba2Payload,
    PrecisionCode,
    StateCommand,
    StateCompletion,
    StateDescriptor,
    StateLifecycle,
    StateSubop,
    X_STATE_OPCODE,
    decode_instruction,
    encode_instruction,
)
from aten.state.generated_contract import STATE_DESCRIPTOR_SIZE


GOLDEN = json.loads(
    (Path(__file__).parents[2] / "spec" / "x_state_v2_golden.json").read_text()
)
RTL_CANDIDATE = json.loads(
    (Path(__file__).parents[2] / "spec" / "x_state_v2_rtl_candidate.json").read_text()
)


def test_generated_contract_is_current() -> None:
    """Covers this repo's generated Python only.

    ``--check`` compares the Simulator's ``generated_contract.rs`` and descriptor
    golden only when ``--simulator-root`` is passed, and this repo's CI has no
    Simulator checkout to point it at. The cross-repo half is enforced from the
    other side, where PLENA_Compiler is a submodule, by
    ``transactional_emulator/testbench/test_x_state_contract_sync.py``. A green
    run here is therefore not evidence that the Rust contract is in sync.
    """
    subprocess.run(
        [sys.executable, "tools/state_contract.py", "--check"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "tools/generate_state_golden.py", "--check"],
        check=True,
    )


def test_provisional_rtl_candidate_matches_frozen_contract_and_sram_budget() -> None:
    isa = RTL_CANDIDATE["isa"]
    datapath = RTL_CANDIDATE["recurrent_datapath"]
    conv = RTL_CANDIDATE["conv_datapath"]
    memory = RTL_CANDIDATE["memory"]

    assert isa["opcode"] == X_STATE_OPCODE
    assert isa["descriptor_bytes"] == STATE_DESCRIPTOR_SIZE
    assert datapath["row_lanes"] * datapath["column_lanes"] == datapath["banks_per_head"]
    assert datapath["fma_lanes_per_head"] == datapath["banks_per_head"]
    assert memory["minimum_transient_state_sram_bytes"] == (
        datapath["recurrent_head_tile_sram_bytes"]
        + conv["max_transient_conv_state_bytes"]
    )
    assert (
        datapath["mamba_bf16_recurrent_head_tile_sram_bytes"]
        <= datapath["recurrent_head_tile_sram_bytes"]
    )
    assert 32 * 1024 * 1024 in memory["future_cache_options_bytes"]
    assert "mx8_b128_activation" not in RTL_CANDIDATE["precision_scope"][
        "activation_first_iteration"
    ]


def test_instruction_codec_uses_post_route_opcode() -> None:
    word = encode_instruction(1, 2, 3, 4, StateSubop.STEP)
    assert word == 0x00D0C87D
    assert word & 0x3F == X_STATE_OPCODE == 0x3D
    assert decode_instruction(word) == {
        "context_gp": 1,
        "descriptor_offset_gp": 2,
        "descriptor_hbm_reg": 3,
        "queue_id": 4,
        "subop": StateSubop.STEP,
    }


def test_fence_has_no_descriptor_fetch_operands() -> None:
    command = StateCommand(StateSubop.FENCE, queue_id=3)
    assert command.instruction_word == 0x018C003D
    assert decode_instruction(command.instruction_word) == {
        "context_gp": 0,
        "descriptor_offset_gp": 0,
        "descriptor_hbm_reg": 0,
        "queue_id": 3,
        "subop": StateSubop.FENCE,
    }
    with pytest.raises(ValueError, match="canonical zero"):
        encode_instruction(1, 0, 0, 3, StateSubop.FENCE)


def test_mamba_descriptor_round_trip_and_real_state_bytes() -> None:
    descriptor = StateDescriptor(
        payload=Mamba2Payload(
            conv_weight_addr=0x1_0000,
            a_log_addr=0x2_0000,
            dt_bias_addr=0x3_0000,
            d_skip_addr=0x4_0000,
        ),
        sequence_length=257,
        token_offset=256,
        valid_tokens=1,
        state_hbm_addr=0x10_0000,
        conv_state_hbm_addr=0x40_0000,
    )
    packed = descriptor.pack()
    assert len(packed) == STATE_DESCRIPTOR_SIZE
    assert StateDescriptor.unpack(packed) == descriptor
    assert descriptor.state_bytes == 2 * 1024 * 1024
    assert descriptor.conv_state_bytes == 96 * 1024
    assert descriptor.streaming
    assert int.from_bytes(packed[48:52], "little") == STREAMING_SRAM_OFFSET
    assert (
        hashlib.sha256(packed).hexdigest()
        == "8c5dc5ab2fdde4b3a3c5b88bd510e5cdc4e855ffa27c5fcd1c3acc656ad59d2c"
    )
    assert packed.hex() == GOLDEN["descriptors"]["mamba2_real"]["hex"]


def test_kda_descriptor_uses_the_same_common_header() -> None:
    descriptor = StateDescriptor(
        payload=KdaPayload(
            q_conv_weight_addr=0x1_0000,
            k_conv_weight_addr=0x2_0000,
            v_conv_weight_addr=0x3_0000,
            a_log_addr=0x4_0000,
            dt_bias_addr=0x5_0000,
        ),
        num_heads=96,
        chunk_size=16,
        state_precision=PrecisionCode.FP32,
        conv_state_precision=PrecisionCode.BF16,
        state_hbm_addr=0x80_0000,
        conv_state_hbm_addr=0xC0_0000,
    )
    packed = descriptor.pack()
    assert StateDescriptor.unpack(packed) == descriptor
    assert descriptor.state_bytes == 6 * 1024 * 1024
    assert descriptor.conv_state_bytes == 288 * 1024
    assert descriptor.algorithm.name == "KDA"
    assert "layout" not in descriptor.to_dict()
    assert (
        hashlib.sha256(packed).hexdigest()
        == "62252044cebd325837993e452549a66e6a860e4d6d6d3568c2a6d57c37f0d84b"
    )
    assert packed.hex() == GOLDEN["descriptors"]["kda_real"]["hex"]


def test_unpack_rejects_nonzero_reserved_bytes() -> None:
    packed = bytearray(StateDescriptor(payload=Mamba2Payload()).pack())
    packed[255] = 1
    with pytest.raises(ValueError, match="reserved"):
        StateDescriptor.unpack(bytes(packed))


def test_resident_state_lifecycle_rejects_dirty_eviction() -> None:
    descriptor = StateDescriptor(payload=Mamba2Payload(), state_sram_offset=0)
    lifecycle = StateLifecycle()
    lifecycle.seed_hbm(descriptor.identity)
    lifecycle.apply(descriptor, StateSubop.PRELOAD)
    lifecycle.apply(descriptor, StateSubop.STEP)
    with pytest.raises(ValueError, match="non-clean"):
        lifecycle.apply(descriptor, StateSubop.EVICT)
    lifecycle.apply(descriptor, StateSubop.COMMIT)
    lifecycle.apply(descriptor, StateSubop.EVICT)


def test_streaming_step_commits_without_fake_cache_commands() -> None:
    descriptor = StateDescriptor(payload=Mamba2Payload())
    lifecycle = StateLifecycle()
    lifecycle.seed_hbm(descriptor.identity)
    lifecycle.apply(descriptor, StateSubop.STEP)
    with pytest.raises(ValueError, match="invalid for streaming"):
        lifecycle.apply(descriptor, StateSubop.PRELOAD)


def test_completion_record_is_byte_exact() -> None:
    completion = StateCompletion(status=1, completion_event=7, elapsed_cycles=99)
    assert StateCompletion.unpack(completion.pack()) == completion


def test_write_completion_requires_an_address() -> None:
    with pytest.raises(ValueError, match="nonzero completion_addr"):
        StateDescriptor(payload=Mamba2Payload(), flags=2)
