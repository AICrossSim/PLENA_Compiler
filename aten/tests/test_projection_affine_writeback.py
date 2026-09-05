from pathlib import Path
from copy import deepcopy

import pytest

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind
from compiler.aten.plena.lstream import StreamConfigField
from compiler.aten.plena.mview import (
    MatrixViewAllocation,
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
    validate_matrix_view_dominance,
    validate_disjoint_matrix_views,
)


ROOT = Path(__file__).resolve().parents[2]


def _projection(*, affine: bool) -> str:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    x_hbm = program.input(
        "x",
        shape=(4, 64),
        physical_shape=(4, 64),
        real_data_ratio=1.0,
    )
    x = program.load_batch(x_hbm, name="x_vram")
    weight = program.input(
        "weight",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=1.0,
    )
    output = program.alloc(
        "output",
        4,
        64,
        strict=False,
        physical_shape=(4, 64),
    )
    layout = None
    if affine:
        layout = AffineLayout(
            kind=LayoutKind.AFFINE_SKEW,
            groups=1,
            fields=1,
            majors=4,
            minors=64,
            alpha=1,
        )
    program.vram_sub_projection_to(
        x,
        0,
        weight,
        0,
        output,
        0,
        0,
        output_layout=layout,
    )
    return program.get_code()


def test_projection_affine_writeback_is_opt_in_and_brackets_matrix_writeout(tmp_path):
    baseline = _projection(affine=False)
    affine = _projection(affine=True)

    assert "L_CFG" not in baseline
    assert "L_CFG" in affine
    writeout = affine.index("M_MM_WO")
    setup = affine.index("L_CFG")
    reset = affine.rindex(
        f", 3, {int(StreamConfigField.RESET)}"
    )
    assert setup < writeout < reset

    asm_path = tmp_path / "affine_projection.asm"
    mem_path = tmp_path / "affine_projection.mem"
    asm_path.write_text(affine)
    assembler = AssemblyToBinary(
        str(ROOT / "doc/operation.svh"),
        str(ROOT / "doc/configuration.svh"),
    )
    words = assembler.generate_binary(str(asm_path), str(mem_path))
    assert words
    assert sum(word & 0x3F == 0x3F for word in words) >= 2


def test_projection_write_layout_does_not_replace_the_static_result_pointer():
    affine = _projection(affine=True)
    # Matrix writeback owns slot 3 by convention. Existing scalar address
    # arithmetic remains authoritative; the view changes only physical bank
    # placement at M_MM_WO.
    flags_cfg = [
        line
        for line in affine.splitlines()
        if line.startswith("L_CFG")
        and line.endswith(f", {int(StreamConfigField.FLAGS)}")
    ]
    assert len(flags_cfg) == 1
    assert "S_ADDI_INT" in affine
    assert "C_LOOP_START" in affine


def test_wide_k_projection_applies_affine_layout_only_at_final_writeback():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=1)
    x_hbm = program.input("wide_x", shape=(4, 128), physical_shape=(4, 128))
    x = program.load_batch(x_hbm, name="wide_x_vram")
    weight = program.input("wide_w", shape=(128, 64), physical_shape=(128, 64))
    layout = AffineLayout(LayoutKind.AFFINE_SKEW, 1, 1, 4, 64, alpha=1)

    program.linear_projection(x, weight, name="wide_y", output_layout=layout)
    code = program.get_code()

    assert "VRAM Sub Projection microtile accumulate" in code
    assert "VRAM Matrix Add" not in code
    assert code.count("M_MM_WO") == 16
    # One setup/reset pair covers all sixteen 4x4 microtiles in the output tile.
    cfg_fields = [
        int(line.rsplit(",", 1)[1])
        for line in code.splitlines()
        if line.startswith("L_CFG")
    ]
    assert cfg_fields.count(int(StreamConfigField.RESET)) == 2
    assert cfg_fields.count(int(StreamConfigField.FLAGS)) == 1


def test_identity_relayout_records_the_same_major_packed_layout_for_consumers():
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=64)
    source_hbm = program.input("source", shape=(16, 64), physical_shape=(16, 64))
    source = program.load_batch(source_hbm, name="source_vram")
    identity = program.input("identity", shape=(64, 64), physical_shape=(64, 64))
    output = program.alloc(
        "packed_output",
        16,
        64,
        strict=False,
        physical_shape=(16, 64),
    )
    layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        1,
        1,
        16,
        64,
        alpha=1,
        major_packed=True,
    )

    program.vram_identity_relayout_to(
        source=source,
        identity=identity,
        out=output,
        output_layout=layout,
    )

    assert output.physical_layout == layout
    flags_lines = [
        line
        for line in program.get_code().splitlines()
        if line.startswith("L_CFG")
        and line.endswith(f", {int(StreamConfigField.FLAGS)}")
    ]
    assert len(flags_lines) == 1


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
    # Review F1: tile 4 of this descriptor would overwrite the weight at 4096.
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
    # Review F2: independent output blocks previously reset the same view offset.
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
