from __future__ import annotations

import pytest

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.mview import (
    L_MVIEW_OPCODE,
    MatrixViewDescriptor,
    MatrixViewField,
    MatrixViewFlags,
    MatrixViewForm,
    MatrixViewMap,
    MatrixViewShape,
    decode_l_mview_word,
    encode_l_mview_field,
    encode_l_mview_full,
    validate_matrix_view_dominance,
)


def test_real_shapes_and_64_bank_skew_roundtrip() -> None:
    shape = MatrixViewShape(rows=128, cols=128, tile_count=96)
    mapping = MatrixViewMap(tile_pitch_rows=128, alpha=4)

    assert MatrixViewShape.unpack(shape.pack()) == shape
    assert MatrixViewMap.unpack(mapping.pack()) == mapping
    assert mapping.alpha == 4


def test_full_and_field_words_are_canonical_and_share_one_opcode() -> None:
    full = encode_l_mview_full(slot=2, shape_register=7, map_register=9)
    field = encode_l_mview_field(
        slot=2, field=MatrixViewField.MAPPING, value_register=9
    )

    assert full & 0x3F == field & 0x3F == L_MVIEW_OPCODE
    assert decode_l_mview_word(full) == (MatrixViewForm.FULL, 2, 7, 9)
    assert decode_l_mview_word(field) == (
        MatrixViewForm.FIELD,
        2,
        9,
        int(MatrixViewField.MAPPING),
    )


def test_descriptor_rejects_a_pitch_that_aliases_tiles() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=128, cols=128, tile_count=16),
        MatrixViewMap(tile_pitch_rows=127, alpha=1),
    )
    with pytest.raises(ValueError, match="aliases consecutive tiles"):
        descriptor.validate_for_machine(banks=64, bank_width=2)


def test_descriptor_does_not_encode_machine_bank_geometry() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=64, cols=2048, tile_count=32),
        MatrixViewMap(
            tile_pitch_rows=64,
            alpha=1,
            flags=MatrixViewFlags.STRICT_BOUNDS,
        ),
    )
    descriptor.validate_for_machine(banks=64, bank_width=32)
    descriptor.validate_for_machine(banks=32, bank_width=64)


@pytest.mark.parametrize(
    "word",
    [
        L_MVIEW_OPCODE,
        L_MVIEW_OPCODE | (6 << 22),
        encode_l_mview_full(slot=0, shape_register=1, map_register=2) | (1 << 18),
        encode_l_mview_full(slot=0, shape_register=1, map_register=2) | (1 << 31),
    ],
)
def test_noncanonical_or_reserved_words_fail(word: int) -> None:
    with pytest.raises(ValueError):
        decode_l_mview_word(word)


def test_matrix_consumer_must_be_dominated_by_an_explicit_view() -> None:
    validate_matrix_view_dominance(
        """
L_MVIEW_FULL 2, gp7, gp9
M_MM 0, gp1, gp2, 2
"""
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("M_TMM 0, gp1, gp2, 2")


def test_field_pair_is_cold_equivalent_of_full_and_reset_kills_dominance() -> None:
    validate_matrix_view_dominance(
        """
L_MVIEW_FIELD 1, 1, gp7
L_MVIEW_FIELD 1, 2, gp9
M_MV 0, gp1, gp2, 1
"""
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance(
            """
L_MVIEW_FULL 1, gp7, gp9
L_MVIEW_FIELD 1, 0, gp0
M_MV 0, gp1, gp2, 1
"""
        )


def test_matrix_view_vector_operands_require_dominating_views() -> None:
    validate_matrix_view_dominance(
        "L_MVIEW_FULL 1, gp2, gp3\n"
        "L_MVIEW_FULL 2, gp2, gp3\n"
        "V_MUL_VV.MV gp4, gp5, gp6, 0, 6"
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("V_MUL_VV.MV gp4, gp5, gp6, 0, 6")


def test_direct_accumulator_writeback_is_canonical_and_dominated() -> None:
    validate_matrix_view_dominance(
        "L_MVIEW_FULL 1, gp2, gp3\nM_MM_WO gp4, gp0, 5, 1"
    )
    with pytest.raises(ValueError, match="before a dominating configuration"):
        validate_matrix_view_dominance("M_MM_WO gp4, gp0, 5, 1")


def test_compiler_emits_one_generic_matrix_view_data_path() -> None:
    program = PlenaCompiler(mlen=64, blen=4, mram_tile_capacity=4)
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=4, tile_count=16),
        MatrixViewMap(tile_pitch_rows=1, alpha=1),
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
    assert assembly.count("L_MVIEW_FULL") == 1
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
