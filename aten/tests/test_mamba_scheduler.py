from __future__ import annotations

from aten.mamba.contract import FLAG_LAST_CHUNK
from aten.state import PrecisionCode
from aten.mamba.scheduler import (
    CachePolicy,
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)


def test_decode_without_cache_uses_direct_state_streaming() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=2)
    ).build()
    assert trace.count("STEP") == 46
    assert trace.count("PRELOAD") == 0
    assert trace.count("COMMIT") == 0
    assert trace.count("EVICT") == 0
    assert trace.count("IN_PROJECTION") == 46
    assert trace.count("PROJECTION_SCATTER") == 46
    assert trace.count("GATED_GROUP_RMSNORM") == 46
    assert trace.count("OUT_PROJECTION") == 46
    assert trace.count("FENCE") == 47
    assert trace.cache_hits == 0
    assert trace.cache_misses == 46
    assert all(
        event.descriptor.streaming
        for event in trace.events
        if event.operation == "STEP" and event.descriptor is not None
    )
    for event in trace.events:
        descriptor = event.descriptor
        if event.operation != "STEP" or descriptor is None:
            continue
        assert descriptor.input_vram_addr % 64 == 0
        assert descriptor.output_vram_addr % 64 == 0
        assert descriptor.input_token_stride % 64 == 0
        assert descriptor.output_token_stride % 64 == 0
        assert descriptor.chunk_size == 4
        assert (
            descriptor.input_vram_addr
            + descriptor.input_token_stride * trace.config.chunk_size
            <= trace.config.vector_sram_elements
        )
        assert (
            descriptor.output_vram_addr
            + descriptor.output_token_stride * trace.config.chunk_size
            <= trace.config.vector_sram_elements
        )
    plans = [
        event.projection_scatter
        for event in trace.events
        if event.operation == "PROJECTION_SCATTER"
    ]
    assert all(plan.fifo_capacity_values == 64 for plan in plans)
    for index, event in enumerate(trace.events):
        if event.operation == "GATED_GROUP_RMSNORM":
            assert trace.events[index - 1].operation == "FENCE"


def test_full_cache_reuses_state_on_second_decode_token() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=2,
            state_cache_entries=23,
            cache_policy=CachePolicy.LRU,
        )
    ).build()
    assert trace.count("STEP") == 46
    assert trace.count("PRELOAD") == 23
    assert trace.count("COMMIT") == 23
    assert trace.count("EVICT") == 0
    assert trace.cache_hits == 23
    assert trace.cache_misses == 23


def test_partial_lru_cache_thrashes_for_layer_ordered_decode() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=2,
            state_cache_entries=4,
            cache_policy=CachePolicy.LRU,
        )
    ).build()
    assert trace.cache_hits == 0
    assert trace.cache_misses == 46
    assert trace.cache_evictions == 42


def test_pinned_cache_keeps_a_useful_subset() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=2,
            state_cache_entries=4,
            cache_policy=CachePolicy.PINNED,
        )
    ).build()
    assert trace.cache_hits == 4
    assert trace.cache_misses == 42


def test_prefill_is_chunked_without_continue_state_flag() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.PREFILL, sequence_length=257)
    ).build()
    assert trace.count("RESET") == 23
    assert trace.count("PREFILL") == 69
    assert trace.count("IN_PROJECTION") == 69
    assert trace.count("GATED_GROUP_RMSNORM") == 69
    assert trace.count("COMMIT") == 0
    chunks = [event for event in trace.events if event.operation == "PREFILL"]
    assert [event.valid_tokens for event in chunks[:3]] == [128, 128, 1]
    assert chunks[0].descriptor is not None
    assert chunks[1].descriptor is not None
    assert chunks[2].descriptor is not None
    assert chunks[0].descriptor.flags & FLAG_LAST_CHUNK == 0
    assert chunks[1].descriptor.flags & FLAG_LAST_CHUNK == 0
    assert chunks[2].descriptor.flags & FLAG_LAST_CHUNK


def test_prefill_resident_state_is_committed_once() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.PREFILL,
            sequence_length=257,
            state_cache_entries=23,
            cache_policy=CachePolicy.LRU,
        )
    ).build()
    assert trace.count("RESET") == 23
    assert trace.count("PREFILL") == 69
    assert trace.count("COMMIT") == 23


def test_mx8_flush_uses_value_and_scale_bytes_for_slot_offsets() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=1,
            state_cache_entries=23,
            cache_policy=CachePolicy.LRU,
            state_precision=PrecisionCode.MX8_B128,
        )
    ).build()
    commits = [event.descriptor for event in trace.events if event.operation == "COMMIT"]
    assert all(descriptor is not None for descriptor in commits)
    resident_bytes = commits[0].resident_bytes
    assert [descriptor.state_sram_offset for descriptor in commits] == [
        index * resident_bytes for index in range(23)
    ]


def test_async_decode_pipelines_two_requests_with_distinct_queues() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            batch_size=2,
            decode_tokens=1,
            async_pipeline=True,
        )
    ).build()
    assert trace.count("STEP") == 46
    assert trace.count("FENCE") == 48
    first_layer = [event for event in trace.events if event.layer_id == 0]
    steps = [event for event in first_layer if event.operation == "STEP"]
    assert [event.queue_id for event in steps] == [0, 1]
    first_gate = next(
        index
        for index, event in enumerate(first_layer)
        if event.operation == "GATED_GROUP_RMSNORM"
    )
    assert sum(event.operation == "STEP" for event in first_layer[:first_gate]) == 2


def test_prefill_alternates_the_projection_double_buffer_across_chunks() -> None:
    # buffer_index used to be parity of the raw token offset, which advances by
    # chunk_size in prefill and therefore never flipped for an even chunk size:
    # every chunk landed in buffer 0 and half the allocated Vector SRAM was dead.
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.PREFILL, sequence_length=512, chunk_size=128
        )
    ).build()
    chunks = [
        event.descriptor
        for event in trace.events
        if event.operation == "PREFILL" and event.layer_id == 0
    ]
    assert [chunk.token_offset for chunk in chunks] == [0, 128, 256, 384]
    inputs = [chunk.input_vram_addr for chunk in chunks]
    outputs = [chunk.output_vram_addr for chunk in chunks]
    assert inputs[0] == inputs[2] and inputs[1] == inputs[3]
    assert inputs[0] != inputs[1]
    assert outputs[0] == outputs[2] and outputs[1] == outputs[3]
    assert outputs[0] != outputs[1]
    # Each chunk also needs its own completion record.
    assert len({chunk.completion_addr for chunk in chunks}) == len(chunks)


def test_decode_still_alternates_the_double_buffer_per_token() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=4)
    ).build()
    steps = [
        event.descriptor
        for event in trace.events
        if event.operation == "STEP" and event.layer_id == 0
    ]
    inputs = [step.input_vram_addr for step in steps]
    assert inputs[0] == inputs[2] and inputs[1] == inputs[3]
    assert inputs[0] != inputs[1]
