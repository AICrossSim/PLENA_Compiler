from __future__ import annotations

from aten.kda.scheduler import (
    KIMI_K3_KDA_LAYERS,
    KdaScheduleConfig,
    KimiK3KdaScheduler,
)
from aten.mamba.scheduler import CachePolicy, SchedulePhase
from aten.state import PrecisionCode
from aten.state.projection import ProjectionLayout


def test_kimi_layer_contract_has_69_kda_mixers() -> None:
    assert len(KIMI_K3_KDA_LAYERS) == 69
    assert all((layer + 1) % 4 for layer in KIMI_K3_KDA_LAYERS)


def test_decode_without_cache_streams_all_kda_states() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=2)
    ).build()
    assert trace.model_name == "kimi_k3"
    assert trace.count("STEP") == 138
    assert trace.count("PRELOAD") == 0
    assert trace.count("COMMIT") == 0
    assert trace.count("KDA_QKV_PROJECTION") == 138
    assert trace.count("KDA_DECAY_BETA_PROJECTION") == 138
    assert trace.count("KDA_OUTPUT_GATE_RMSNORM") == 138
    assert trace.count("KDA_OUT_PROJECTION") == 138
    assert trace.count("FENCE") == 139
    for index, event in enumerate(trace.events):
        if event.operation == "KDA_OUTPUT_GATE_RMSNORM":
            assert trace.events[index - 1].operation == "FENCE"
            assert event.aux_vram_addr is not None
            assert event.aux_vram_addr % 64 == 0
            assert (
                event.aux_vram_addr
                + event.descriptor.output_token_stride * trace.config.chunk_size
                <= trace.config.vector_sram_elements
            )
            gate_projection = next(
                candidate
                for candidate in reversed(trace.events[:index])
                if candidate.operation == "KDA_OUTPUT_GATE_PROJECTION"
                and candidate.request_id == event.request_id
                and candidate.layer_id == event.layer_id
            )
            assert gate_projection.aux_vram_addr == event.aux_vram_addr
    step = next(event for event in trace.events if event.operation == "STEP")
    assert step.descriptor is not None
    assert step.descriptor.state_bytes == 6 * 1024 * 1024
    assert step.descriptor.conv_state_bytes == 288 * 1024
    assert step.descriptor.state_precision == PrecisionCode.FP32
    assert step.descriptor.conv_state_precision == PrecisionCode.BF16
    assert step.descriptor.input_token_stride == 49_280
    assert step.descriptor.chunk_size == 4
    for event in trace.events:
        descriptor = event.descriptor
        if event.operation != "STEP" or descriptor is None:
            continue
        assert descriptor.input_vram_addr % 64 == 0
        assert descriptor.output_vram_addr % 64 == 0
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


def test_full_kda_cache_reuses_second_decode_token() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=2,
            state_cache_entries=69,
            cache_policy=CachePolicy.LRU,
        )
    ).build()
    assert trace.cache_misses == 69
    assert trace.cache_hits == 69
    assert trace.count("PRELOAD") == 69
    assert trace.count("COMMIT") == 69


def test_kda_prefill_chunks_and_commits_resident_state_once() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(
            phase=SchedulePhase.PREFILL,
            sequence_length=33,
            state_cache_entries=69,
            cache_policy=CachePolicy.LRU,
        )
    ).build()
    assert trace.count("RESET") == 69
    assert trace.count("PREFILL") == 207
    assert trace.count("COMMIT") == 69
    chunks = [
        event.valid_tokens for event in trace.events if event.operation == "PREFILL"
    ]
    assert chunks[:3] == [16, 16, 1]
    prefill = next(event for event in trace.events if event.operation == "PREFILL")
    assert prefill.descriptor is not None
    assert prefill.descriptor.chunk_size == 16


def test_kda_async_decode_uses_the_common_two_request_pipeline() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(
            phase=SchedulePhase.DECODE,
            batch_size=2,
            decode_tokens=1,
            async_pipeline=True,
        )
    ).build()
    assert trace.count("STEP") == 138
    assert trace.count("FENCE") == 140
    first_layer = [event for event in trace.events if event.layer_id == 0]
    steps = [event for event in first_layer if event.operation == "STEP"]
    assert [event.queue_id for event in steps] == [0, 1]
    first_gate = next(
        index
        for index, event in enumerate(first_layer)
        if event.operation == "KDA_OUTPUT_GATE_RMSNORM"
    )
    assert sum(event.operation == "STEP" for event in first_layer[:first_gate]) == 2


def test_kda_builds_under_both_projection_layouts() -> None:
    # The field placement and the layout recorded in the plan must agree.
    # Placing row-major but recording group-major made ProjectionScatterPlan
    # divide bank-unaligned offsets by the bank count, which aliased group 0's
    # beta onto group 1's q at (row 32, bank 0) and aborted the whole trace.
    for layout in ProjectionLayout:
        trace = KimiK3KdaScheduler(
            KdaScheduleConfig(
                phase=SchedulePhase.DECODE,
                decode_tokens=1,
                projection_layout=layout,
            )
        ).build()
        plans = [
            event.projection_scatter
            for event in trace.events
            if event.operation == "PROJECTION_SCATTER"
        ]
        assert len(plans) == 69
        assert all(plan.layout == layout for plan in plans)
        if layout == ProjectionLayout.ROW_MAJOR:
            assert all(
                field.skew_kind == "none" for plan in plans for field in plan.fields
            )
        else:
            for plan in plans:
                fields = {field.name: field for field in plan.fields}
                assert fields["k"].skew_kind == "field_constant"
                assert fields["k"].skew_stride == 8
                assert fields["beta"].skew_kind == "group_stride"
                assert fields["beta"].skew_stride == 1
                assert all(
                    field.skew_kind == "none"
                    for name, field in fields.items()
                    if name not in {"k", "beta"}
                )


def test_kda_defaults_to_buffering_independent_projection_fields() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    plans = [
        event.projection_scatter
        for event in trace.events
        if event.operation == "PROJECTION_SCATTER"
    ]
    assert all(plan.flow.value == "buffered" for plan in plans)
    assert all(plan.fifo_capacity_values == 64 for plan in plans)
