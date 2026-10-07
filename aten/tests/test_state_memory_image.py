"""Guards for the emulator preload images and the compact HBM arena.

Every assertion here corresponds to a way the physical KDA program failed to run
before: a 24.5 GiB address span the emulator cannot allocate, an oversized scalar
SRAM preload that panics instead of truncating, and assembly handed to an
``--opcode`` flag that only parses hex.
"""

from __future__ import annotations

import hashlib

import pytest

from aten.kda.scheduler import KdaHbmLayout, KdaScheduleConfig, KimiK3KdaScheduler
from aten.mamba.scheduler import CachePolicy, SchedulePhase
from aten.state import lower_state_trace
from aten.state.hbm_image import (
    SCALAR_SRAM_ENTRIES,
    assemble_words,
    build_memory_images,
    render_mem_image,
)
from aten.state.isa_lowering import lower_kda_trace_to_existing_isa


GIB = 1024**3
MIB = 1024**2


def _config(**overrides: object) -> KdaScheduleConfig:
    settings: dict[str, object] = {
        "phase": SchedulePhase.DECODE,
        "decode_tokens": 1,
        "state_cache_entries": 5,
        "cache_policy": CachePolicy.PINNED,
    }
    settings.update(overrides)
    return KdaScheduleConfig(**settings)  # type: ignore[arg-type]


def _layout(config: KdaScheduleConfig) -> KdaHbmLayout:
    # Via the scheduler on purpose: rebuilding the layout here would omit the
    # precision-derived state stride and test a layout nothing actually uses.
    return KimiK3KdaScheduler(config).hbm_layout()


def test_sparse_bases_do_not_fit_a_flat_allocation() -> None:
    # This is why the program was never executed, so it is worth pinning: the
    # default layout is unrunnable by construction, not by accident.
    assert _layout(_config()).arena_bytes > 20 * GIB


def test_compact_arena_fits_the_state_it_has_to_hold() -> None:
    config = _config(hbm_arena_base=0x10000)
    layout = _layout(config)
    # 69 layers x 6 MiB of FP32 recurrent state is the floor, plus the conv
    # state, scales and per-layer conv weights; it must stay far below the
    # sparse span and inside a size the emulator can allocate.
    assert 414 * MIB <= layout.arena_bytes < 768 * MIB


def test_state_regions_do_not_overlap() -> None:
    # The literal stride was 4 MiB, sized when KDA state was BF16. FP32 state is
    # 6 MiB, so consecutive layers overlapped by 2 MiB and each layer's writeback
    # corrupted the next layer's state. Byte counts were unchanged, so no timing
    # model could have caught it.
    for overrides in ({}, {"hbm_arena_base": 0x10000}):
        config = _config(**overrides)
        trace = KimiK3KdaScheduler(config).build()
        first_by_layer: dict[int, object] = {}
        for event in trace.events:
            if event.descriptor is not None:
                first_by_layer.setdefault(event.descriptor.layer_id, event.descriptor)
        previous_end = None
        for layer_id in sorted(first_by_layer):
            descriptor = first_by_layer[layer_id]
            start = descriptor.state_hbm_addr
            if previous_end is not None:
                assert start >= previous_end, (
                    f"layer {layer_id} state starts at {start} inside the previous "
                    f"layer's region ending at {previous_end}"
                )
            previous_end = start + descriptor.state_bytes


def test_kda_parameter_regions_do_not_overlap_between_layers() -> None:
    config = _config(hbm_arena_base=0x10000)
    scheduler = KimiK3KdaScheduler(config)
    trace = scheduler.build()
    layout = scheduler.hbm_layout()
    first_by_layer: dict[int, object] = {}
    for event in trace.events:
        if event.descriptor is not None:
            first_by_layer.setdefault(event.descriptor.layer_id, event.descriptor)

    key_elements = config.kda_num_heads * config.kda_key_dim
    value_elements = config.kda_num_heads * config.kda_value_dim
    parameter_bytes = config.parameter_precision.element_bytes
    live_sizes = {
        "q_conv_weight": key_elements * config.kda_conv_kernel * parameter_bytes,
        "k_conv_weight": key_elements * config.kda_conv_kernel * parameter_bytes,
        "v_conv_weight": value_elements * config.kda_conv_kernel * parameter_bytes,
        "q_conv_bias": key_elements * parameter_bytes,
        "k_conv_bias": key_elements * parameter_bytes,
        "v_conv_bias": value_elements * parameter_bytes,
        "a_log": config.kda_num_heads * parameter_bytes,
        "dt_bias": key_elements * parameter_bytes,
    }
    for region, live_size in live_sizes.items():
        assert layout.strides[region] >= live_size
        addresses = [
            getattr(descriptor.payload, f"{region}_addr")
            for _, descriptor in sorted(first_by_layer.items())
        ]
        for start, following in zip(addresses, addresses[1:]):
            assert start + live_size <= following


def test_packed_regions_do_not_run_into_each_other() -> None:
    """Region-vs-region, not just entry-vs-entry within one region.

    The first version of the packing loop advanced the cursor by the table's
    literal stride while addressing used the widened one, so the state region ran
    414 MiB past a base only 276 MiB apart and overwrote the convolution state and
    the conv weights. Adjacent-entry checks pass under that bug.
    """
    layout = _layout(_config(hbm_arena_base=0x10000))
    spans = sorted(
        (
            layout.bases[name],
            layout.bases[name] + layout.strides[name] * layout.counts[name],
            name,
        )
        for name in layout.bases
    )
    for (_, end, name), (start, _, following) in zip(spans, spans[1:]):
        assert end <= start, f"region {name} overruns {following}"
    assert spans[-1][1] <= layout.arena_bytes


def test_descriptor_image_is_pinned() -> None:
    # Pinned so an address-layout change has to be deliberate. This differs from
    # the sha in the committed cache32 artifact: that artifact predates the state
    # stride fix above and therefore encodes overlapping state regions.
    trace = KimiK3KdaScheduler(_config()).build()
    image = lower_state_trace(trace).descriptor_image
    assert len(image) == 79 * 256
    assert hashlib.sha256(image).hexdigest() == (
        "1a6af5d77ce0b999287bcb657a18aab5767463dc0ed42afa12272834186a9a6f"
    )


def test_compact_mode_moves_every_descriptor_address_below_the_arena_end() -> None:
    config = _config(hbm_arena_base=0x10000)
    trace = KimiK3KdaScheduler(config).build()
    end = _layout(config).realized_arena_bytes(len(trace.events))
    for event in trace.events:
        descriptor = event.descriptor
        if descriptor is None:
            continue
        addresses = [
            descriptor.state_hbm_addr,
            descriptor.conv_state_hbm_addr,
            descriptor.state_scale_addr,
            descriptor.completion_addr,
        ] + [
            value
            for name in dir(descriptor.payload)
            if name.endswith("_addr")
            and isinstance(value := getattr(descriptor.payload, name), int)
        ]
        assert max(addresses) < end


def test_scalar_preloads_match_the_emulator_file_length() -> None:
    layout = _layout(_config(hbm_arena_base=0x10000))
    images = build_memory_images(
        descriptor_image=b"\x00" * 256,
        descriptor_base=0x1000,
        arena_base=layout.arena_base,
        arena_bytes=layout.arena_bytes,
    )
    # `ScalarSram::new` is a fixed 1024 entries and `load_*` copies into a slice
    # of that length, so anything longer panics rather than truncating.
    assert len(images.fpsram) == SCALAR_SRAM_ENTRIES * 2
    assert len(images.intsram) == SCALAR_SRAM_ENTRIES * 4
    with pytest.raises(ValueError, match="fpsram_entries"):
        build_memory_images(
            descriptor_image=b"\x00" * 256,
            descriptor_base=0x1000,
            arena_base=layout.arena_base,
            arena_bytes=layout.arena_bytes,
            fpsram_entries=SCALAR_SRAM_ENTRIES + 1,
        )


def test_descriptor_image_may_not_overlap_the_arena() -> None:
    with pytest.raises(ValueError, match="overlaps the HBM arena"):
        build_memory_images(
            descriptor_image=b"\x00" * 4096,
            descriptor_base=0x1000,
            arena_base=0x1400,
            arena_bytes=64 * MIB,
        )


def test_hbm_file_only_carries_the_descriptor_image() -> None:
    descriptor_image = b"\xab" * (79 * 256)
    images = build_memory_images(
        descriptor_image=descriptor_image,
        descriptor_base=0x1000,
        arena_base=0x10000,
        arena_bytes=417 * MIB,
    )
    # Weights and initial state are read as zero, so the file stays tiny while
    # the allocation still has to span the arena.
    assert len(images.hbm) == 0x1000 + len(descriptor_image)
    assert images.hbm[0x1000:] == descriptor_image
    assert images.hbm_size_bytes >= 417 * MIB


def test_hbm_file_places_state_and_layout_descriptors_without_overlap() -> None:
    state = bytes(range(256))
    layout = bytes(reversed(range(256)))
    images = build_memory_images(
        descriptor_image=state,
        descriptor_base=0x1000,
        layout_descriptor_image=layout,
        layout_descriptor_base=0x1100,
        arena_base=0x2000,
        arena_bytes=64 * MIB,
    )
    assert images.hbm[0x1000:0x1100] == state
    assert images.hbm[0x1100:0x1200] == layout
    assert images.layout_descriptor_bytes == 256
    with pytest.raises(ValueError, match="overlap"):
        build_memory_images(
            descriptor_image=state,
            descriptor_base=0x1000,
            layout_descriptor_image=layout,
            layout_descriptor_base=0x1080,
            arena_base=0x2000,
            arena_bytes=64 * MIB,
        )


def test_mem_image_is_hex_words_the_emulator_can_parse() -> None:
    assert render_mem_image([0x0, 0xFFFFFFFF]) == "0x00000000\n0xFFFFFFFF\n"
    with pytest.raises(ValueError, match="does not fit a 32-bit word"):
        render_mem_image([0x1_0000_0000])


def test_physical_program_assembles_to_legal_machine_words() -> None:
    trace = KimiK3KdaScheduler(_config(hbm_arena_base=0x10000)).build()
    program = lower_kda_trace_to_existing_isa(trace, descriptor_base=0x1000)
    words = assemble_words(program.assembly)
    assert len(words) == program.instruction_count
    assert all(0 <= word <= 0xFFFFFFFF for word in words)
    assert all(word != 0 for word in words)
    # The rendered image is what `--opcode` reads; feeding it assembly fails on
    # the first comment token.
    assert render_mem_image(words).count("\n") == len(words)
