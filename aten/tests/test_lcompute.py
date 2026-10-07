from pathlib import Path
from copy import deepcopy

from compiler.aten.plena.isa_matrix_view import (
    MatrixViewAllocation, MatrixViewDescriptor, MatrixViewMap, MatrixViewShape,
    validate_disjoint_matrix_views,
)

import pytest

from assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_lcompute import (
    BF16_BYTES,
    KIMI_KDA,
    NEMOTRON_MAMBA,
    ONE_MIB,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceKind,
    RecurrenceLayout,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    validate_recurrence_field_loads,
    validate_recurrence_output_stores,
)
from compiler.aten.plena.isa_matrix_view import MatrixViewFlags, validate_matrix_view_dominance


ROOT = Path(__file__).resolve().parents[2]


def test_recurrence_point_enforces_the_simulator_bank_limit() -> None:
    MatrixSramPoint(mlen=64, banks=64, bank_width=1).validate()
    with pytest.raises(ValueError, match="no larger than 64"):
        MatrixSramPoint(mlen=128, banks=128, bank_width=1).validate()


def test_official_shapes_and_plena_bf16_state_sizes_are_explicit() -> None:
    assert (NEMOTRON_MAMBA.heads, NEMOTRON_MAMBA.row_elements) == (64, 64)
    assert (KIMI_KDA.heads, KIMI_KDA.row_elements) == (96, 128)
    assert NEMOTRON_MAMBA.recurrence_rows == KIMI_KDA.recurrence_rows == 128
    assert NEMOTRON_MAMBA.state_bytes_per_layer == ONE_MIB
    assert KIMI_KDA.state_bytes_per_layer == 3 * ONE_MIB


def test_mamba_field_contract_is_reusable_by_a_real_checkpoint_shape() -> None:
    checkpoint = MatrixRecurrenceSpec(
        name="mamba2_130m",
        kind=RecurrenceKind.MAMBA,
        heads=24,
        row_elements=64,
        recurrence_rows=128,
        primitives=NEMOTRON_MAMBA.primitives,
    )
    point = MatrixSramPoint()
    working_set = build_recurrence_working_set(
        checkpoint,
        layout=RecurrenceLayout.AFFINE,
        point=point,
    )
    assembly = lower_matrix_recurrence(
        checkpoint,
        layout=RecurrenceLayout.AFFINE,
        point=point,
    )

    assert working_set.group_heads == 24
    assert working_set.groups == 1
    assert checkpoint.state_bytes_per_layer == 24 * 128 * 64 * BF16_BYTES
    assert "@stage=mamba2_130m_matrix_recurrence" in assembly
    assert assembly.count("L_TILE_EXEC") == 4
    validate_matrix_view_dominance(assembly)


@pytest.mark.parametrize(
    ("capacity_bytes", "spec", "fixed_heads", "affine_heads"),
    [
        (ONE_MIB, NEMOTRON_MAMBA, 32, 32),
        (ONE_MIB, KIMI_KDA, 16, 16),
        (2 * ONE_MIB, NEMOTRON_MAMBA, 32, 32),
        (2 * ONE_MIB, KIMI_KDA, 16, 16),
    ],
)
def test_capacity_point_sets_an_honest_head_group(
    capacity_bytes: int,
    spec,
    fixed_heads: int,
    affine_heads: int,
) -> None:
    point = MatrixSramPoint(capacity_bytes=capacity_bytes)
    for layout, group_heads in (
        (RecurrenceLayout.FIXED, fixed_heads),
        (RecurrenceLayout.AFFINE, affine_heads),
    ):
        working_set = build_recurrence_working_set(spec, layout=layout, point=point)
        assert working_set.group_heads == group_heads
        assert working_set.packet_values == group_heads * spec.row_elements
        assert working_set.capacity_facts["bank_words"] <= point.capacity_bank_words
        assert working_set.capacity_facts["max_bank_row"] <= point.depth_rows


@pytest.mark.parametrize(
    ("capacity_bytes", "spec", "expected_scalar_cols"),
    [
        (ONE_MIB, NEMOTRON_MAMBA, 64),
        (ONE_MIB, KIMI_KDA, 32),
        (2 * ONE_MIB, NEMOTRON_MAMBA, 64),
        (2 * ONE_MIB, KIMI_KDA, 32),
    ],
)
def test_recurrence_scalar_and_head_major_fields_match_their_consumers(
    capacity_bytes: int,
    spec,
    expected_scalar_cols: int,
) -> None:
    point = MatrixSramPoint(capacity_bytes=capacity_bytes)
    for layout in RecurrenceLayout:
        working_set = build_recurrence_working_set(spec, layout=layout, point=point)
        compact = {"beta"} if spec is KIMI_KDA else {"dt", "update", "c", "d"}
        for name in compact:
            descriptor = working_set.allocation(name).descriptor
            assert descriptor.shape.tile_count == 1
            assert descriptor.shape.cols == expected_scalar_cols
            assert descriptor.mapping.flags & MatrixViewFlags.BROADCAST_MINOR
            # One row stores one [a,b] pair per head at most; it must not
            # replicate each scalar across a complete recurrent row.
            assert descriptor.shape.cols < (working_set.group_heads * spec.row_elements)
        if spec is KIMI_KDA:
            for name, rows in (("decay", 2), ("key", 1), ("query", 1)):
                descriptor = working_set.allocation(name).descriptor
                assert descriptor.shape.tile_count == working_set.group_heads
                assert descriptor.shape.rows == rows
                assert descriptor.shape.cols >= working_set.state_rows_per_chunk
                assert descriptor.mapping.flags & MatrixViewFlags.BROADCAST_MINOR


def test_every_active_recurrence_view_uses_uniform_bf16() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        assert spec.state_bytes_per_layer == (
            spec.heads
            * spec.recurrence_rows
            * spec.row_elements
            * BF16_BYTES
        )
        for layout in RecurrenceLayout:
            working_set = build_recurrence_working_set(spec, layout=layout)
            assert working_set.point.element_bytes == BF16_BYTES
            assert all(
                not allocation.descriptor.mapping.flags
                & ~MatrixViewFlags.BROADCAST_MINOR
                for allocation in working_set.allocations
            )


def test_state_transfers_are_real_viewed_dma_not_timing_comments() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        assembly = lower_matrix_recurrence(
            spec,
            layout="affine",
            state_hbm_base=0x20000,
            hbm_address_register=2,
        )
        lines = assembly.splitlines()
        markers = [
            index
            for index, line in enumerate(lines)
            if line.startswith("; @matrix_state_")
        ]
        assert markers
        assert assembly.count("H_PREFETCH_V.MV") > 0
        assert assembly.count("H_STORE_V.MV") > 0
        for index in markers:
            marker = lines[index]
            assert "precision=bf16" in marker
            assert "hbm_byte_offset=" in marker
            next_marker = next(
                (
                    candidate
                    for candidate in range(index + 1, len(lines))
                    if lines[candidate].startswith("; @matrix_state_")
                    or lines[candidate].startswith("; @matrix_field_load")
                    or lines[candidate].startswith("; @l_tile_step=")
                ),
                len(lines),
            )
            transfer = "\n".join(lines[index:next_marker])
            assert "L_TILE_CFG 3" in transfer
            assert ("H_PREFETCH_V.MV" in transfer) ^ ("H_STORE_V.MV" in transfer)
            assert ", a2, 0, 2, 3" in transfer


def test_every_recurrence_field_is_loaded_into_a_consumer_view() -> None:
    for spec in (NEMOTRON_MAMBA, KIMI_KDA):
        for layout in RecurrenceLayout:
            working_set = build_recurrence_working_set(spec, layout=layout)
            manifest = build_recurrence_field_manifest(
                working_set,
                field_hbm_base=spec.state_bytes_per_layer,
            )
            assembly = lower_matrix_recurrence(spec, layout=layout)
            assert validate_recurrence_field_loads(
                assembly,
                expected={
                    packet.key
                    for packet in manifest.packets
                    if packet.field != "output_result"
                },
            )
            assert assembly.count("H_PREFETCH_V.MV") > working_set.groups

    mamba = lower_matrix_recurrence(NEMOTRON_MAMBA, layout="affine")
    for field in ("x", "scratch_zero", "dt", "update", "c", "output_zero", "d"):
        assert f"@matrix_field_load field={field} " in mamba
    for step in (
        "mamba_dt_times_x",
        "mamba_state_decay_rank1_update",
        "mamba_c_readout",
        "mamba_skip",
    ):
        assert f"@l_tile_step={step}" in mamba

    kimi = lower_matrix_recurrence(KIMI_KDA, layout="affine")
    for field in (
        "prediction_zero",
        "decay",
        "key",
        "value",
        "beta",
        "output_zero",
        "query",
    ):
        assert f"@matrix_field_load field={field} " in kimi
    for step in (
        "kda_decay",
        "kda_prediction",
        "kda_beta_error",
        "kda_rank1_update",
        "kda_readout",
    ):
        assert f"@l_tile_step={step}" in kimi


def test_comment_only_field_claim_is_rejected() -> None:
    with pytest.raises(ValueError, match="comment-only"):
        validate_recurrence_field_loads(
            "; @matrix_field_write=x,dt layout=consumer_view\nL_TILE_EXEC gp1, gp2, gp3, 0\n"
        )


@pytest.mark.parametrize("spec", [NEMOTRON_MAMBA, KIMI_KDA])
@pytest.mark.parametrize("layout", list(RecurrenceLayout))
def test_every_head_group_output_is_stored_once_without_aliasing(spec, layout) -> None:
    working_set = build_recurrence_working_set(spec, layout=layout)
    assembly = lower_matrix_recurrence(spec, layout=layout)
    stores = validate_recurrence_output_stores(
        assembly,
        expected_groups=working_set.groups,
    )
    assert len(stores) == working_set.groups
    assert len(set(stores.values())) == working_set.groups


def test_missing_head_group_output_store_is_rejected() -> None:
    working_set = build_recurrence_working_set(
        NEMOTRON_MAMBA,
        layout=RecurrenceLayout.AFFINE,
    )
    assembly = lower_matrix_recurrence(
        NEMOTRON_MAMBA,
        layout=RecurrenceLayout.AFFINE,
    )
    marker = "; @matrix_field_store field=output_result"
    first = assembly.index(marker)
    next_group = assembly.index("; @head_group=1/", first)
    broken = assembly[:first] + assembly[next_group:]
    with pytest.raises(ValueError, match="output coverage differs"):
        validate_recurrence_output_stores(
            broken,
            expected_groups=working_set.groups,
        )


@pytest.mark.parametrize("spec", [NEMOTRON_MAMBA, KIMI_KDA])
def test_fixed_chunks_have_distinct_field_packets(spec) -> None:
    working_set = build_recurrence_working_set(spec, layout="fixed")
    assert working_set.chunks > 1
    manifest = build_recurrence_field_manifest(
        working_set,
        field_hbm_base=spec.state_bytes_per_layer,
    )
    chunk_fields = (
        ("update", "c") if spec is NEMOTRON_MAMBA else ("decay", "key", "query")
    )
    for group in range(working_set.groups):
        for field in chunk_fields:
            packets = [
                manifest.packet(field, group=group, chunk=chunk)
                for chunk in range(working_set.chunks)
            ]
            assert (
                len({packet.hbm_byte_offset for packet in packets})
                == working_set.chunks
            )
            assert all(packet.logical_values > 0 for packet in packets)


def test_complete_lowerings_assemble_and_views_dominate_every_exec(
    tmp_path: Path,
) -> None:
    assembler = AssemblyToBinary(
        str(ROOT / "doc" / "operation.svh"),
        str(ROOT / "doc" / "configuration.svh"),
    )
    for capacity_bytes in (ONE_MIB, 2 * ONE_MIB):
        point = MatrixSramPoint(capacity_bytes=capacity_bytes)
        for spec in (NEMOTRON_MAMBA, KIMI_KDA):
            for layout in RecurrenceLayout:
                assembly = lower_matrix_recurrence(spec, layout=layout, point=point)
                validate_matrix_view_dominance(assembly)
                stem = f"{spec.name}_{capacity_bytes}_{layout}"
                asm_path = tmp_path / f"{stem}.asm"
                mem_path = tmp_path / f"{stem}.mem"
                asm_path.write_text(assembly)
                assembler.generate_binary(str(asm_path), str(mem_path))
                words = [line for line in mem_path.read_text().splitlines() if line]
                assert words
                assert all(0 <= int(word, 16) < 2**32 for word in words)


@pytest.mark.parametrize("spec", [NEMOTRON_MAMBA, KIMI_KDA])
@pytest.mark.parametrize("layout", [RecurrenceLayout.FIXED, RecurrenceLayout.AFFINE])
def test_diagnostic_snapshots_are_explicit_and_do_not_replace_persistent_state(spec, layout):
    working = build_recurrence_working_set(spec, layout=layout)
    program = lower_matrix_recurrence(spec, layout=layout, snapshot_hbm_base=64 * 1024 * 1024)
    assert program.count("@state_snapshot ") == working.groups * working.chunks
    assert program.count("@matrix_state_store ") == working.groups * working.chunks
    for bad_base in (0, 3, spec.state_bytes_per_layer):
        with pytest.raises(ValueError, match="snapshot"):
            lower_matrix_recurrence(spec, layout=layout, snapshot_hbm_base=bad_base)


@pytest.mark.parametrize("field_base", [0, 64, KIMI_KDA.state_bytes_per_layer - 64])
def test_explicit_prepared_fields_cannot_overlap_live_state(field_base):
    with pytest.raises(ValueError, match="fields overlap live recurrent state"):
        lower_matrix_recurrence(
            KIMI_KDA, layout="affine", state_hbm_base=0, field_hbm_base=field_base
        )


def test_adjacent_state_and_field_arenas_are_legal_in_both_orders():
    working = build_recurrence_working_set(KIMI_KDA, layout="affine")
    fields = build_recurrence_field_manifest(working, field_hbm_base=0)
    for state_base, field_base in ((0, KIMI_KDA.state_bytes_per_layer), (fields.end, 0)):
        program = lower_matrix_recurrence(
            KIMI_KDA, layout="affine",
            state_hbm_base=state_base, field_hbm_base=field_base,
        )
        assert "L_TILE_EXEC" in program


@pytest.mark.parametrize("base", [2**32, 2**32 - 64])
@pytest.mark.parametrize("arena", ["state", "field", "snapshot"])
def test_dma_arenas_cannot_wrap_the_gp_address_window(arena, base):
    with pytest.raises(ValueError, match="32-bit GP DMA offset window"):
        lower_matrix_recurrence(
            KIMI_KDA, layout="affine", **{f"{arena}_hbm_base": base}
        )


def test_exact_top_of_gp_window_is_legal_with_nonzero_hbm_base_register(tmp_path):
    working = build_recurrence_working_set(KIMI_KDA, layout="affine")
    fields = build_recurrence_field_manifest(working, field_hbm_base=0)
    field_base = 2**32 - fields.end
    state_base = field_base - KIMI_KDA.state_bytes_per_layer
    program = lower_matrix_recurrence(
        KIMI_KDA, layout="affine", state_hbm_base=state_base,
        field_hbm_base=field_base, hbm_address_register=7,
    )
    assert f"@field_hbm_end={2**32}" in program
    assert ", a7, 0, 2, 3" in program
    asm_path, mem_path = tmp_path / "top.asm", tmp_path / "top.mem"
    asm_path.write_text(program)
    assembler = AssemblyToBinary(ROOT / "doc/operation.svh", ROOT / "doc/configuration.svh")
    assembler.generate_binary(str(asm_path), str(mem_path))
    assert all(0 <= int(line, 16) < 2**32 for line in mem_path.read_text().splitlines())


def test_projection_accumulator_directly_writes_a_configured_matrix_view(tmp_path):
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    x_hbm = program.input(
        "view_x",
        shape=(1, 64),
        physical_shape=(4, 64),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="view_x_vram")
    weight = program.input(
        "view_weight",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=1.0,
    )
    output = program.alloc(
        "view_output",
        1,
        64,
        strict=False,
        physical_shape=(4, 64),
    )
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=1, cols=8, tile_count=8),
        mapping=MatrixViewMap(tile_pitch_rows=2),
    )

    program.vram_sub_projection_stream_k_accum_to(
        x,
        0,
        weight,
        0,
        output,
        0,
        0,
        max_k_tiles=1,
        matrix_view_descriptor=descriptor,
        matrix_view_slot=2,
    )
    assembly = program.get_code()

    validate_matrix_view_dominance(assembly)
    assert assembly.count("L_TILE_CFG") == 1
    assert assembly.count("M_MM_WO") == 1
    assert "C_LOOP_START" in assembly
    assert "S_ADDI_INT gp" in assembly
    assert ", 4" in assembly
    # All sixteen output micro-columns reuse the one resident weight tile.
    assert assembly.count("H_PREFETCH_M") == 1
    assert "L_MVIEW_LOAD" not in assembly
    assert "L_MVIEW_WO" not in assembly

    asm_path = tmp_path / "matrix_view_projection.asm"
    mem_path = tmp_path / "matrix_view_projection.mem"
    asm_path.write_text(assembly)
    assembler = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    )
    words = assembler.generate_binary(str(asm_path), str(mem_path))
    assert sum(word & 0x3F == 0x3F for word in words) == 1
    viewed_writebacks = [word for word in words if word & 0x3F == 0x06]
    assert len(viewed_writebacks) == 1
    assert all((word >> 31) & 1 for word in viewed_writebacks)


def test_wide_projection_reserves_existing_matrix_scratch_and_streams_weights():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=2)
    x_hbm = program.input(
        "wide_view_x",
        shape=(1, 128),
        physical_shape=(4, 128),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="wide_view_x_vram")
    weight = program.input(
        "wide_view_weight",
        shape=(128, 64),
        physical_shape=(128, 64),
        real_data_ratio=1.0,
    )
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=1, cols=8, tile_count=8),
        mapping=MatrixViewMap(tile_pitch_rows=2),
    )

    program.linear_projection_bf16(
        x,
        weight,
        name="wide_view_output",
        matrix_view_descriptor=descriptor,
        matrix_view_slot=1,
    )
    assembly = program.get_code()

    validate_matrix_view_dominance(assembly)
    assert assembly.count("L_TILE_CFG") == 1
    assert assembly.count("M_MM_WO") == 16
    assert "L_MVIEW_LOAD" not in assembly
    # One Matrix tile is statically reserved for the view. The remaining tile
    # streams the two K chunks without aliasing that scratch address.
    assert assembly.count("H_PREFETCH_M") == 32


def test_direct_view_projection_reuses_a_fitting_wide_k_weight_set():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    x_hbm = program.input(
        "fit_view_x",
        shape=(1, 128),
        physical_shape=(4, 128),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="fit_view_x_vram")
    weight = program.input(
        "fit_view_weight",
        shape=(128, 64),
        physical_shape=(128, 64),
        real_data_ratio=1.0,
    )
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=1, cols=8, tile_count=8),
        mapping=MatrixViewMap(tile_pitch_rows=2),
    )

    program.linear_projection_bf16(
        x,
        weight,
        name="fit_view_output",
        matrix_view_descriptor=descriptor,
    )
    assembly = program.get_code()

    assert assembly.count("M_MM_WO") == 1
    assert "C_LOOP_START" in assembly
    assert assembly.count("H_PREFETCH_M") == 2


def test_direct_view_projection_rejects_a_partial_consumer_packet():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    x_hbm = program.input(
        "bad_view_x",
        shape=(1, 64),
        physical_shape=(4, 64),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="bad_view_x_vram")
    weight = program.input(
        "bad_view_weight",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=1.0,
    )
    incomplete = MatrixViewDescriptor(
        shape=MatrixViewShape(rows=1, cols=8, tile_count=4),
        mapping=MatrixViewMap(tile_pitch_rows=2),
    )

    try:
        program.linear_projection_bf16(
            x,
            weight,
            name="bad_view_output",
            matrix_view_descriptor=incomplete,
        )
    except ValueError as error:
        assert "complete MLEN-wide consumer packet" in str(error)
    else:
        raise AssertionError("partial Matrix consumer packet was accepted")


def _view_projection_inputs(program, *, k=64, cols=64):
    x_hbm = program.input("guard_x", shape=(1, k), physical_shape=(4, k), real_data_ratio=1.0)
    x = program.load_batch(x_hbm, name="guard_x_vram")
    weight = program.input("guard_weight", shape=(k, cols), physical_shape=(k, cols), real_data_ratio=1.0)
    return x, weight


def _projection_state(program):
    return deepcopy((
        program.get_code(),
        vars(program.mram_allocator._vmm),
        program.mram_allocator.reserved_blocks,
        vars(program.vram_allocator._vmm),
        vars(program.register_allocator),
    ))


@pytest.mark.parametrize("entry", ["linear", "tile"])
def test_direct_view_rejects_live_weight_alias_before_emitting_or_allocating(entry):
    # tile 4 of this descriptor would overwrite the weight at 4096.
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=2)
    x, weight = _view_projection_inputs(program)
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 8, 8), MatrixViewMap(16))
    output = program.alloc("guard_output", 1, 64, strict=False, physical_shape=(4, 64))
    before = _projection_state(program)
    with pytest.raises(ValueError, match="one reserved scratch tile"):
        if entry == "linear":
            program.linear_projection_bf16(x, weight, matrix_view_descriptor=descriptor)
        else:
            program.vram_sub_projection_stream_k_accum_to(
                x, 0, weight, 0, output, 0, 0, max_k_tiles=1,
                matrix_view_descriptor=descriptor,
            )
    assert _projection_state(program) == before


@pytest.mark.parametrize("cols, physical_shape", [(128, None), (64, (4, 128)), (64, (128, 64))])
def test_direct_view_rejects_multiple_output_blocks_without_losing_first_block(cols, physical_shape):
    # independent output blocks previously reset the same view offset.
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=2)
    x, weight = _view_projection_inputs(program, cols=cols)
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 8, 8), MatrixViewMap(2))
    before = _projection_state(program)
    with pytest.raises(ValueError, match="exactly one output tile"):
        program.linear_projection_bf16(
            x, weight, matrix_view_descriptor=descriptor, physical_shape=physical_shape
        )
    assert _projection_state(program) == before


@pytest.mark.parametrize("base, error", [
    (0, "persistent MRAM reservation"),
    (4, "tile-aligned reservation base"),
    (-4096, "tile-aligned reservation base"),
    (3 * 4096, "footprint exceeds Matrix SRAM"),
])
def test_direct_view_requires_a_reserved_in_bounds_owner(base, error):
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=3)
    x, weight = _view_projection_inputs(program)
    output = program.alloc("guard_output", 1, 64, strict=False, physical_shape=(4, 64))
    # A transient allocation is not sufficient: _prepare_projection resets it.
    assert program.mram_allocator.allocate("old_weight", 4096) == 0
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 8, 8), MatrixViewMap(2))
    before = _projection_state(program)
    with pytest.raises(ValueError, match=error):
        program.vram_sub_projection_stream_k_accum_to(
            x, 0, weight, 0, output, 0, 0, max_k_tiles=1,
            matrix_view_descriptor=descriptor, matrix_view_base=base,
        )
    assert _projection_state(program) == before


def test_direct_view_rejects_missing_weight_capacity_before_reserving_scratch():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=1)
    x, weight = _view_projection_inputs(program)
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 8, 8), MatrixViewMap(2))
    before = _projection_state(program)
    with pytest.raises(ValueError, match="insufficient room for weight tiles"):
        program.linear_projection_bf16(x, weight, matrix_view_descriptor=descriptor)
    assert _projection_state(program) == before


@pytest.mark.parametrize("entry", ["linear", "tile"])
def test_nonzero_reserved_view_keeps_weights_disjoint_when_streaming_k(entry, tmp_path):
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=3)
    # A pre-existing allocation puts scratch at a nonzero base. Reset does not
    # reuse the hole below scratch, leaving only ONE tile for two K chunks.
    program.mram_allocator.allocate("old_weight", 4096)
    base = program.reserve_matrix_view_scratch_v0()
    assert base == 4096
    x, weight = _view_projection_inputs(program, k=128)
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 8, 8), MatrixViewMap(2))
    if entry == "linear":
        program.linear_projection_bf16(x, weight, matrix_view_descriptor=descriptor)
    else:
        output = program.alloc("guard_output", 1, 64, strict=False, physical_shape=(4, 64))
        program.vram_sub_projection_stream_k_accum_to(
            x, 0, weight, 0, output, 0, 0, max_k_tiles=2,
            matrix_view_descriptor=descriptor, matrix_view_base=base,
        )
    assembly = program.get_code()
    assert assembly.count("H_PREFETCH_M") == 32
    assert assembly.count("M_MM_WO") == 16
    resident = [block for block in program.mram_allocator._vmm.used_stack
                if block.name != "__matrix_view_scratch"]
    assert [block.addr for block in resident] == [8192]
    facts = validate_disjoint_matrix_views(
        [MatrixViewAllocation("scratch", base, descriptor),
         MatrixViewAllocation("weight", resident[0].addr,
                              MatrixViewDescriptor(MatrixViewShape(64, 64), MatrixViewMap(64)))],
        mlen=64, banks=16, bank_width=4, depth_rows=192,
    )
    assert facts["max_bank_row"] == 192
    asm_path, mem_path = tmp_path / "nonzero.asm", tmp_path / "nonzero.mem"
    asm_path.write_text(assembly)
    words = AssemblyToBinary(str(ROOT / "doc/operation.svh"), str(ROOT / "doc/configuration.svh")).generate_binary(
        str(asm_path), str(mem_path)
    )
    assert words


def test_reserved_view_cannot_be_freed_or_confused_with_a_transient_weight():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=3)
    program.reserve_matrix_view_scratch_v0()
    with pytest.raises(ValueError, match="cannot be freed"):
        program.mram_allocator.free("__matrix_view_scratch")
    program.mram_allocator.allocate("live_weight", 4096)
    before = _projection_state(program)
    with pytest.raises(ValueError, match="conflicts with a live allocation"):
        program.reserve_matrix_view_scratch_v0("live_weight")
    assert _projection_state(program) == before
