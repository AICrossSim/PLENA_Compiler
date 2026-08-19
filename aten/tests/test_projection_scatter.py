from __future__ import annotations

from aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from aten.state import lower_state_trace
from aten.state.projection import ProjectionFlow, ProjectionLayout


def _scatter_events(trace):
    return [event for event in trace.events if event.operation == "PROJECTION_SCATTER"]


def test_nemotron_scatter_is_a_bijective_physical_plan() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    plan = _scatter_events(trace)[0].projection_scatter
    assert plan is not None
    assert plan.algorithm == "mamba2"
    assert plan.layout == ProjectionLayout.GROUP_MAJOR_SKEWED
    assert plan.flow == ProjectionFlow.FIFO_WITH_SPILL
    assert plan.source_values_per_token == 10304
    assert plan.physical_values_per_token == 10368
    assert plan.padding_values_per_token == 64
    assert plan.groups == 8
    assert [field.name for field in plan.fields] == ["x", "gate", "b", "c", "dt"]
    assert {field.name: field.source_offset for field in plan.fields} == {
        "x": 4096,
        "gate": 0,
        "b": 8192,
        "c": 9216,
        "dt": 10240,
    }

    physical = set()
    sources = set()
    for field in plan.fields:
        for group in range(plan.groups):
            for local_row in range(field.local_rows):
                for lane in range(field.local_lanes):
                    source, row, bank = plan.address(field.name, group, local_row, lane)
                    sources.add(source)
                    physical.add((row, bank))
    assert sources == set(range(10304))
    assert len(physical) == 10304
    assert len(plan.mapping_sha256) == 64


def test_decode_scatter_uses_the_same_double_buffer_as_x_state() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=2)
    ).build()
    scatters = _scatter_events(trace)
    for event in scatters:
        plan = event.projection_scatter
        descriptor = event.descriptor
        assert plan is not None and descriptor is not None
        assert plan.fallback_vram_addr == descriptor.input_vram_addr
        assert plan.fallback_token_stride == descriptor.input_token_stride
        expected_index = descriptor.input_vram_addr // (
            descriptor.input_token_stride * descriptor.chunk_size
        )
        assert plan.physical_buffer_index == expected_index
    assert {event.projection_scatter.physical_buffer_index for event in scatters} == {
        0,
        1,
    }


def test_disabling_bypass_materializes_every_projection_value() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(
            phase=SchedulePhase.DECODE,
            projection_direct_bypass=False,
        )
    ).build()
    plan = _scatter_events(trace)[0].projection_scatter
    assert plan is not None
    assert plan.flow == ProjectionFlow.BUFFERED
    assert plan.spill_policy == "always"


def test_kda_scatter_groups_heads_and_rotates_k_by_one_packet() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    plan = _scatter_events(trace)[0].projection_scatter
    assert plan is not None
    assert plan.algorithm == "kda"
    assert plan.layout == ProjectionLayout.GROUP_MAJOR_SKEWED
    assert plan.flow == ProjectionFlow.BUFFERED
    assert plan.fifo_capacity_values == 64
    assert plan.groups == 96
    assert plan.source_values_per_token == 49248
    assert [field.name for field in plan.fields] == ["q", "k", "v", "decay", "beta"]
    assert plan.physical_values_per_token == 50688
    assert plan.padding_values_per_token == 1440
    assert {field.name: field.skew_stride for field in plan.fields} == {
        "q": 0,
        "k": 8,
        "v": 0,
        "decay": 0,
        "beta": 1,
    }
    beta = plan.fields[-1]
    assert beta.skew_kind == "group_stride"
    assert {plan.address("beta", group, 0, 0)[2] for group in range(16)} == set(
        range(16)
    )


def test_kda_row_major_ablation_disables_field_rotations() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(
            phase=SchedulePhase.DECODE,
            decode_tokens=1,
            projection_layout=ProjectionLayout.ROW_MAJOR,
        )
    ).build()
    plan = _scatter_events(trace)[0].projection_scatter
    assert plan is not None
    assert plan.layout == ProjectionLayout.ROW_MAJOR
    assert all(field.skew_stride == 0 for field in plan.fields)


def test_lowered_trace_contains_one_executable_layout_per_scatter() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.PREFILL, sequence_length=129)
    ).build()
    lowered = lower_state_trace(trace)
    assert len(lowered.projection_scatters) == trace.count("PROJECTION_SCATTER") == 46
    document = lowered.to_dict()
    assert (
        document["projection_scatter_contract"]["contract"]
        == "plena.projection_scatter"
    )
    assert document["projection_scatter_contract"]["version"] == 1
    assert document["projection_scatter_contract"]["isa_opcode"] == 0x3F
    assert document["layout_descriptor_count"] == 46
    assert len(document["layout_commands"]) == 46
    assert len(document["projection_scatter_contract"]["sha256"]) == 64
    assert document["projection_scatters"][0]["plan"]["mapping_sha256"]
