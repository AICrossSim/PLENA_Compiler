from __future__ import annotations

from aten.mamba.contract import FLAG_CONTINUE_STATE
from aten.mamba.scheduler import (
    CachePolicy,
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)


def test_decode_without_cache_streams_every_layer_state() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=2)
    ).build()
    assert trace.count("STEP") == 46
    assert trace.count("STATE_PREFETCH") == 46
    assert trace.count("STATE_COMMIT") == 46
    assert trace.count("STATE_EVICT") == 46
    assert trace.count("IN_PROJECTION") == 46
    assert trace.count("PROJECTION_SCATTER") == 46
    assert trace.count("GATED_GROUP_RMSNORM") == 46
    assert trace.count("OUT_PROJECTION") == 46
    assert trace.cache_hits == 0
    assert trace.cache_misses == 46


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
    assert trace.count("STATE_PREFETCH") == 23
    assert trace.count("STATE_COMMIT") == 23
    assert trace.count("STATE_EVICT") == 0
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


def test_prefill_is_chunked_and_keeps_state_across_chunks() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.PREFILL, sequence_length=257)
    ).build()
    assert trace.count("STATE_RESET") == 23
    assert trace.count("PREFILL") == 69
    assert trace.count("IN_PROJECTION") == 69
    assert trace.count("GATED_GROUP_RMSNORM") == 69
    assert trace.count("STATE_COMMIT") == 23
    chunks = [event for event in trace.events if event.operation == "PREFILL"]
    assert [event.valid_tokens for event in chunks[:3]] == [128, 128, 1]
    assert chunks[0].descriptor is not None
    assert chunks[1].descriptor is not None
    assert chunks[0].descriptor.flags & FLAG_CONTINUE_STATE == 0
    assert chunks[1].descriptor.flags & FLAG_CONTINUE_STATE
