from __future__ import annotations

import pytest

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.mview import (
    LTilePrimitive,
    L_MVIEW_OPCODE,
    MatrixViewAllocation,
    MatrixViewAxis,
    MatrixViewDescriptor,
    MatrixViewFlags,
    MatrixViewForm,
    MatrixViewMap,
    MatrixViewShape,
    decode_l_tile_exec_word,
    decode_matrix_view_dma_slot,
    decode_l_mview_word,
    encode_l_tile_exec,
    encode_l_tile_cfg,
    encode_matrix_view_dma_word,
    validate_matrix_view_dominance,
    validate_disjoint_matrix_views,
)


def test_real_shapes_and_fixed_wiring_mapping_roundtrip() -> None:
    shape = MatrixViewShape(rows=128, cols=128, tile_count=96)
    mapping = MatrixViewMap(tile_pitch_rows=128)

    assert MatrixViewShape.unpack(shape.pack()) == shape
    assert MatrixViewMap.unpack(mapping.pack()) == mapping


def test_affine_mapping_roundtrips_without_model_specific_fields() -> None:
    mapping = MatrixViewMap(tile_pitch_rows=128, row_skew=3, tile_skew=11)
    restored = MatrixViewMap.unpack(mapping.pack())
    assert restored.tile_pitch_rows == 128
    assert restored.row_skew == 3
    assert restored.tile_skew == 11
    assert restored.flags & MatrixViewFlags.AFFINE


def test_zero_pitch_compacts_tiles_into_distinct_banks() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=128, cols=128, tile_count=8),
        MatrixViewMap(tile_pitch_rows=0, row_skew=1, tile_skew=4),
    )
    descriptor.validate_for_machine(banks=64, bank_width=32)
    assert MatrixViewMap.unpack(descriptor.mapping.pack()).tile_pitch_rows == 0


def test_zero_pitch_without_a_tile_phase_is_rejected_as_aliasing() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=128, cols=128, tile_count=8),
        MatrixViewMap(tile_pitch_rows=0),
    )
    with pytest.raises(ValueError, match="aliases logical bank words"):
        descriptor.validate_for_machine(banks=64, bank_width=32)


def test_nonzero_skew_without_affine_flag_is_rejected() -> None:
    word = 4 | (3 << 16) | (1 << 28)
    with pytest.raises(ValueError, match="requires the AFFINE flag"):
        MatrixViewMap.unpack(word)


def test_config_word_is_canonical_and_uses_the_shared_opcode() -> None:
    word = encode_l_tile_cfg(slot=2, shape_register=7, map_register=9)

    assert word & 0x3F == L_MVIEW_OPCODE
    assert decode_l_mview_word(word) == (MatrixViewForm.CONFIG, 2, 7, 9)


@pytest.mark.parametrize("primitive", list(LTilePrimitive))
def test_l_tile_exec_is_canonical_and_uses_the_same_opcode(
    primitive: LTilePrimitive,
) -> None:
    word = encode_l_tile_exec(
        dst_register=4,
        src1_register=5,
        src2_register=6,
        primitive=primitive,
    )
    assert word & 0x3F == L_MVIEW_OPCODE
    assert (word >> 22) & 0xF == MatrixViewForm.EXEC
    assert decode_l_tile_exec_word(word) == (
        4,
        5,
        6,
        primitive,
        MatrixViewAxis.ROW,
        MatrixViewAxis.ROW,
    )


def test_l_tile_exec_carries_explicit_source_and_scale_axes() -> None:
    word = encode_l_tile_exec(
        dst_register=4,
        src1_register=5,
        src2_register=6,
        primitive=LTilePrimitive.DOT_REDUCE,
        source_axis=MatrixViewAxis.ROW,
        scale_axis=MatrixViewAxis.COLUMN,
    )
    assert decode_l_tile_exec_word(word) == (
        4,
        5,
        6,
        LTilePrimitive.DOT_REDUCE,
        MatrixViewAxis.ROW,
        MatrixViewAxis.COLUMN,
    )


def test_l_tile_exec_rejects_reserved_primitives_and_high_bits() -> None:
    with pytest.raises(ValueError, match="reserved L_TILE primitive"):
        encode_l_tile_exec(
            dst_register=1, src1_register=2, src2_register=3, primitive=3
        )
    valid = encode_l_tile_exec(
        dst_register=1,
        src1_register=2,
        src2_register=3,
        primitive=LTilePrimitive.DOT_REDUCE,
    )
    with pytest.raises(ValueError, match="reserved L_TILE_EXEC bits"):
        decode_l_tile_exec_word(valid | (1 << 31))


def test_matrix_view_dma_qualifier_is_explicit_and_legacy_compatible() -> None:
    legacy = 0x29 | (1 << 6) | (2 << 10) | (3 << 14) | (1 << 18) | (2 << 22)
    assert decode_matrix_view_dma_slot(legacy) is None
    viewed = encode_matrix_view_dma_word(legacy, slot=3)
    assert viewed & ((1 << 26) - 1) == legacy
    assert decode_matrix_view_dma_slot(viewed) == 3
    with pytest.raises(ValueError, match="non-canonical"):
        decode_matrix_view_dma_slot(viewed | (1 << 28))


def test_descriptor_rejects_a_pitch_that_aliases_tiles() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=128, cols=128, tile_count=16),
        MatrixViewMap(tile_pitch_rows=127),
    )
    with pytest.raises(ValueError, match="aliases logical bank words"):
        descriptor.validate_for_machine(banks=64, bank_width=2)


@pytest.mark.parametrize(
    ("banks", "bank_width"),
    [
        (8, 4),
        (16, 4),
        (32, 4),
        (64, 32),
    ],
)
def test_descriptor_does_not_encode_machine_bank_geometry(
    banks: int, bank_width: int
) -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=64, cols=2048, tile_count=32),
        MatrixViewMap(
            tile_pitch_rows=4096,
            flags=MatrixViewFlags.STRICT_BOUNDS,
        ),
    )
    descriptor.validate_for_machine(banks=banks, bank_width=bank_width)


@pytest.mark.parametrize(
    "word",
    [
        L_MVIEW_OPCODE,
        L_MVIEW_OPCODE | (2 << 22),
        L_MVIEW_OPCODE | (6 << 22),
        encode_l_tile_cfg(slot=0, shape_register=1, map_register=2) | (1 << 18),
        encode_l_tile_cfg(slot=0, shape_register=1, map_register=2) | (1 << 31),
    ],
)
def test_noncanonical_or_reserved_words_fail(word: int) -> None:
    with pytest.raises(ValueError):
        decode_l_mview_word(word)


def test_matrix_consumer_must_be_dominated_by_an_explicit_view() -> None:
    validate_matrix_view_dominance(
        """
L_TILE_CFG 2, gp7, gp9
M_MM 0, gp1, gp2, 2
"""
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("M_TMM 0, gp1, gp2, 2")


def test_matrix_view_dma_requires_a_dominating_explicit_view() -> None:
    validate_matrix_view_dominance(
        "L_TILE_CFG 2, gp7, gp9\n"
        "H_PREFETCH_V.MV gp1, gp2, a0, 0, 2, 2"
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance(
            "H_STORE_V.MV gp1, gp2, a0, 0, 2, 2"
        )


def test_l_tile_exec_requires_all_three_explicit_views() -> None:
    configured = "\n".join(
        f"L_TILE_CFG {slot}, gp7, gp9" for slot in range(3)
    )
    validate_matrix_view_dominance(
        configured + "\nL_TILE_EXEC gp1, gp2, gp3, 0, 2"
    )
    with pytest.raises(ValueError, match="Matrix view 2"):
        validate_matrix_view_dominance(
            "L_TILE_CFG 0, gp7, gp9\n"
            "L_TILE_CFG 1, gp7, gp9\n"
            "L_TILE_EXEC gp1, gp2, gp3, 0"
        )

    with pytest.raises(ValueError, match="axis mask"):
        validate_matrix_view_dominance(
            configured + "\nL_TILE_EXEC gp1, gp2, gp3, 0, 4"
        )


def test_loop_backedge_requires_the_view_on_every_iteration() -> None:
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance(
            """
C_LOOP_START gp10, 2
M_MV 0, gp1, gp2, 1
C_LOOP_END gp10
"""
        )

    validate_matrix_view_dominance(
        """
C_LOOP_START gp10, 2
L_TILE_CFG 1, gp7, gp9
M_MV 0, gp1, gp2, 1
C_LOOP_END gp10
"""
    )


def test_break_cannot_hide_an_unconfigured_reachable_consumer() -> None:
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance(
            """
C_BREAK
M_MV 0, gp1, gp2, 1
"""
        )

    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance(
            """
C_LOOP_START gp10, 2
C_BREAK
L_TILE_CFG 1, gp7, gp9
C_LOOP_END gp10
M_MV 0, gp1, gp2, 1
"""
        )


def test_matrix_view_vector_operands_require_dominating_views() -> None:
    validate_matrix_view_dominance(
        "L_TILE_CFG 1, gp2, gp3\n"
        "L_TILE_CFG 2, gp2, gp3\n"
        "V_MUL_VV.MV gp4, gp5, gp6, 0, 6"
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("V_MUL_VV.MV gp4, gp5, gp6, 0, 6")


def test_direct_accumulator_writeback_is_canonical_and_dominated() -> None:
    validate_matrix_view_dominance(
        "L_TILE_CFG 1, gp2, gp3\nM_MM_WO gp4, gp0, 5, 1"
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("M_MM_WO gp4, gp0, 5, 1")


def test_compiler_emits_one_generic_matrix_view_data_path() -> None:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=4)
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=4, tile_count=16),
        MatrixViewMap(tile_pitch_rows=1),
    )
    matrix_base = program.reserve_matrix_view_scratch_v0()

    program.configure_matrix_view_v0(descriptor, slot=2)
    program.matrix_view_accumulator_store_v0(
        matrix_base=matrix_base,
        logical_offset=12,
        slot=2,
    )
    assembly = program.compile()

    validate_matrix_view_dominance(assembly)
    assert assembly.count("L_TILE_CFG") == 1
    assert assembly.count("M_MM_WO") == 1
    assert "L_MVIEW_LOAD" not in assembly
    assert "L_MVIEW_STORE" not in assembly


def test_matrix_view_scratch_is_static_not_a_runtime_cache() -> None:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=4)
    scratch = program.reserve_matrix_view_scratch_v0()
    first_weight = program.mram_allocator.allocate("weight_before_reset", 64 * 64)
    program.reset_mram()
    same_scratch = program.reserve_matrix_view_scratch_v0()
    second_weight = program.mram_allocator.allocate("weight_after_reset", 64 * 64)

    assert scratch == same_scratch == 0
    assert first_weight == second_weight == 64 * 64


def _paper_addr(row: int, bank_phase: int) -> int:
    return row * 2048 + bank_phase * 32


def test_compiler_proves_official_kda_chunk_and_fields_are_disjoint() -> None:
    state = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=128, tile_count=16),
        MatrixViewMap(tile_pitch_rows=8, row_skew=1, tile_skew=4),
    )
    scalar = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=32, tile_count=16),
        MatrixViewMap(tile_pitch_rows=1, row_skew=1, tile_skew=3),
    )
    vector = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=128, tile_count=16),
        MatrixViewMap(tile_pitch_rows=1, row_skew=1, tile_skew=3),
    )
    allocations = [
        MatrixViewAllocation("state", _paper_addr(0, 0), state),
        MatrixViewAllocation("decay", _paper_addr(136, 0), scalar),
        MatrixViewAllocation("key", _paper_addr(136, 1), scalar),
        MatrixViewAllocation("query", _paper_addr(136, 2), scalar),
        MatrixViewAllocation("value_or_error", _paper_addr(168, 0), vector),
        MatrixViewAllocation("prediction_or_output", _paper_addr(168, 4), vector),
    ]
    facts = validate_disjoint_matrix_views(
        allocations, mlen=2048, banks=64, bank_width=32, depth_rows=256
    )
    assert facts["max_bank_row"] == 184
    assert facts["bank_words"] < facts["capacity_bank_words"]


def test_compiler_proves_official_mamba_chunk_and_fields_are_disjoint() -> None:
    state = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=64, tile_count=32),
        MatrixViewMap(tile_pitch_rows=4, row_skew=1, tile_skew=2),
    )
    scalar = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=32, tile_count=32),
        MatrixViewMap(tile_pitch_rows=1, row_skew=1, tile_skew=1),
    )
    vector = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=64, tile_count=32),
        MatrixViewMap(tile_pitch_rows=1, row_skew=1, tile_skew=1),
    )
    allocations = [
        MatrixViewAllocation("state", _paper_addr(0, 0), state),
        MatrixViewAllocation("decay_and_b", _paper_addr(140, 0), scalar),
        MatrixViewAllocation("c", _paper_addr(140, 16), scalar),
        MatrixViewAllocation("dt_and_skip", _paper_addr(140, 32), scalar),
        MatrixViewAllocation("x", _paper_addr(188, 0), vector),
        MatrixViewAllocation("scratch", _paper_addr(188, 2), vector),
        MatrixViewAllocation("output", _paper_addr(188, 4), vector),
    ]
    facts = validate_disjoint_matrix_views(
        allocations, mlen=2048, banks=64, bank_width=32, depth_rows=256
    )
    assert facts["max_bank_row"] == 220


def test_compiler_rejects_cross_view_aliasing() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=128, tile_count=16),
        MatrixViewMap(tile_pitch_rows=16),
    )
    with pytest.raises(ValueError, match="first.*second"):
        validate_disjoint_matrix_views(
            [
                MatrixViewAllocation("first", 0, descriptor),
                MatrixViewAllocation("second", 0, descriptor),
            ],
            mlen=2048,
            banks=64,
            bank_width=32,
            depth_rows=256,
        )
