from __future__ import annotations

from aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import MambaScheduleConfig, Nemotron3MambaScheduler, SchedulePhase
from aten.state import (
    LoweredStateTrace,
    StateDescriptor,
    StateSubop,
    decode_instruction,
    lower_state_trace,
)
from aten.state.generated_contract import STATE_DESCRIPTOR_SIZE


def _assert_lowered(trace) -> LoweredStateTrace:
    lowered = lower_state_trace(trace, descriptor_base=0x9000_0000)
    state_events = [event for event in trace.events if event.instruction_word is not None]
    assert len(lowered.commands) == len(state_events)
    assert lowered.commands[-1].operation == "FENCE"
    assert lowered.commands[-1].descriptor_address is None
    descriptor_commands = [
        command for command in lowered.commands if command.descriptor_address is not None
    ]
    assert lowered.descriptor_count == len(descriptor_commands)
    assert len(lowered.descriptor_image) == lowered.descriptor_count * STATE_DESCRIPTOR_SIZE
    assert lowered.layout_descriptor_count == trace.count("PROJECTION_SCATTER")
    assert lowered.layout_descriptor_base % 64 == 0
    assert (
        lowered.layout_descriptor_base
        >= lowered.descriptor_base + len(lowered.descriptor_image)
    )
    assert len(lowered.layout_commands) == lowered.layout_descriptor_count
    for index, command in enumerate(descriptor_commands):
        start = index * STATE_DESCRIPTOR_SIZE
        packed = lowered.descriptor_image[start : start + STATE_DESCRIPTOR_SIZE]
        descriptor = StateDescriptor.unpack(packed)
        assert command.descriptor_offset == start
        assert command.descriptor_address == 0x9000_0000 + start
        assert descriptor.context_id == command.register_writes[0].value
    for index, command in enumerate(lowered.layout_commands):
        assert command.descriptor_offset == index * 256
        assert command.descriptor_address == lowered.layout_descriptor_base + index * 256
    document = lowered.to_dict()
    assert document["layout_contract"]["contract"] == "plena-l-scatter-m-v1"
    assert document["layout_contract"]["instruction_opcode"] == 0x3F
    assert len(document["layout_contract"]["sha256"]) == 64
    return lowered


def test_lowers_real_nemotron_decode_trace() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    lowered = _assert_lowered(trace)
    assert lowered.layout_descriptor_count == 23
    assert len(lowered.layout_commands) == 23


def test_lowers_real_kimi_kda_decode_trace() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    lowered = _assert_lowered(trace)
    assert lowered.layout_descriptor_count == 69
    assert len(lowered.layout_commands) == 69


def test_lowering_preserves_async_queue_ids_and_fences() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            batch_size=2,
            decode_tokens=1,
            async_pipeline=True,
        )
    ).build()
    lowered = lower_state_trace(trace)
    first_steps = [
        decode_instruction(command.instruction_word)
        for command in lowered.commands
        if command.operation == "STEP"
    ][:2]
    assert [fields["queue_id"] for fields in first_steps] == [0, 1]
    assert all(fields["subop"] == StateSubop.STEP for fields in first_steps)
