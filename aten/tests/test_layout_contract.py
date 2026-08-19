from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aten.kda.scheduler import KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import (
    MambaScheduleConfig,
    Nemotron3MambaScheduler,
    SchedulePhase,
)
from aten.state import (
    LAYOUT_DESCRIPTOR_SIZE,
    LayoutMode,
    LayoutScatterDescriptor,
    StateDescriptor,
    decode_layout_instruction,
    encode_layout_instruction,
)


GOLDEN = Path(__file__).parents[2] / "spec" / "l_scatter_m_v1_golden.json"


def _first_scatter(trace):
    return next(event for event in trace.events if event.operation == "PROJECTION_SCATTER")


def test_tiny_cross_repo_golden_parses_with_both_executable_contracts() -> None:
    document = json.loads(GOLDEN.read_text())
    assert document["contract"] == "plena-l-scatter-m-v1"
    for name, expected_mode in (
        ("mamba2_tiny", LayoutMode.MAMBA_SKEW),
        ("kda_tiny", LayoutMode.KDA_SKEW),
    ):
        entry = document[name]
        state = StateDescriptor.unpack(bytes.fromhex(entry["state_hex"]))
        layout = LayoutScatterDescriptor.unpack(bytes.fromhex(entry["layout_hex"]))
        instruction = decode_layout_instruction(entry["instruction_word"])
        assert layout.context_id == state.context_id == 7
        assert layout.request_id == state.request_id == 11
        assert layout.layer_id == state.layer_id == 13
        assert layout.source_values_per_token == len(entry["projected"])
        assert layout.mode == instruction["mode"] == expected_mode


@pytest.mark.parametrize("algorithm", ["mamba2", "kda"])
def test_real_skew_descriptor_roundtrip_matches_projection_plan(algorithm: str) -> None:
    if algorithm == "mamba2":
        trace = Nemotron3MambaScheduler(
            MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
        ).build()
        expected_mode = LayoutMode.MAMBA_SKEW
    else:
        trace = KimiK3KdaScheduler(
            KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
        ).build()
        expected_mode = LayoutMode.KDA_SKEW
    event = _first_scatter(trace)
    assert event.projection_scatter is not None and event.descriptor is not None

    descriptor = LayoutScatterDescriptor.from_projection_plan(
        event.projection_scatter, event.descriptor
    )
    packed = descriptor.pack()

    assert len(packed) == LAYOUT_DESCRIPTOR_SIZE
    assert LayoutScatterDescriptor.unpack(packed) == descriptor
    assert descriptor.mode == expected_mode
    assert sorted(source for source, _, _ in descriptor.mapping()) == list(
        range(descriptor.source_values_per_token)
    )
    assert len({(row, bank) for _, row, bank in descriptor.mapping()}) == len(
        descriptor.mapping()
    )


def test_layout_instruction_roundtrip_and_reserved_bits() -> None:
    word = encode_layout_instruction(1, 2, 3, 1, LayoutMode.KDA_SKEW)
    assert decode_layout_instruction(word) == {
        "context_gp": 1,
        "descriptor_offset_gp": 2,
        "descriptor_hbm_reg": 3,
        "buffer_id": 1,
        "mode": LayoutMode.KDA_SKEW,
    }
    with pytest.raises(ValueError, match="canonical"):
        decode_layout_instruction(word | (1 << 31))


def test_transpose_descriptor_is_a_real_permutation() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    event = _first_scatter(trace)
    assert event.projection_scatter is not None and event.descriptor is not None
    baseline = LayoutScatterDescriptor.from_projection_plan(
        event.projection_scatter, event.descriptor
    )
    descriptor = replace(
        baseline,
        mode=LayoutMode.TRANSPOSE,
        fields=(),
        logical_rows=8,
        logical_cols=baseline.source_values_per_token // 8,
    )
    mapping = descriptor.mapping()
    source0 = mapping[0]
    source1 = mapping[1]
    assert source0[0] == 0
    assert source1[0] == 1
    assert source1[1:] != (
        baseline.physical_buffer_base_row,
        1,
    )
    assert LayoutScatterDescriptor.unpack(descriptor.pack()) == descriptor


def test_descriptor_crc_detects_mapping_corruption() -> None:
    trace = KimiK3KdaScheduler(
        KdaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    event = _first_scatter(trace)
    assert event.projection_scatter is not None and event.descriptor is not None
    descriptor = LayoutScatterDescriptor.from_projection_plan(
        event.projection_scatter, event.descriptor
    )
    packed = bytearray(descriptor.pack())
    packed[82] = 2  # FIELD skew
    packed[83] = 1  # rotate the q field by one bank
    with pytest.raises(ValueError, match="CRC"):
        LayoutScatterDescriptor.unpack(bytes(packed))


def test_descriptor_rejects_two_sources_aliasing_one_bank_cell() -> None:
    trace = Nemotron3MambaScheduler(
        MambaScheduleConfig(phase=SchedulePhase.DECODE, decode_tokens=1)
    ).build()
    event = _first_scatter(trace)
    assert event.projection_scatter is not None and event.descriptor is not None
    descriptor = LayoutScatterDescriptor.from_projection_plan(
        event.projection_scatter, event.descriptor
    )
    first, second, *rest = descriptor.fields
    aliased_second = replace(
        second,
        physical_offset=first.physical_offset,
        skew_kind=first.skew_kind,
        skew_stride=first.skew_stride,
    )
    with pytest.raises(ValueError, match="aliases"):
        replace(descriptor, mode=LayoutMode.CUSTOM, fields=(first, aliased_second, *rest))
