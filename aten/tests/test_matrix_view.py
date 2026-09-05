from __future__ import annotations

import pytest

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.isa_matrix_view import (
    L_MVIEW_CONTRACT_VERSION,
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


def test_matrix_view_enforces_the_simulator_bank_limit() -> None:
    descriptor = MatrixViewDescriptor(MatrixViewShape(1, 1), MatrixViewMap(1))
    descriptor.validate_for_machine(banks=64, bank_width=1)
    with pytest.raises(ValueError, match="no larger than 64"):
        descriptor.validate_for_machine(banks=128, bank_width=1)


def test_matrix_view_v3_golden_mapping_words() -> None:
    assert L_MVIEW_CONTRACT_VERSION == 3
    assert MatrixViewMap(tile_pitch_rows=64).pack() == 0x00000040
    assert (
        MatrixViewMap(tile_pitch_rows=0, tile_phase_stride=4).pack()
        == 0x01000000
    )
    assert (
        MatrixViewMap(
            tile_pitch_rows=0,
            tile_phase_stride=4,
            flags=MatrixViewFlags.BROADCAST_MINOR,
        ).pack()
        == 0x81000000
    )


def test_real_shapes_and_fixed_wiring_mapping_roundtrip() -> None:
    shape = MatrixViewShape(rows=128, cols=128, tile_count=96)
    mapping = MatrixViewMap(tile_pitch_rows=128)

    assert MatrixViewShape.unpack(shape.pack()) == shape
    assert MatrixViewMap.unpack(mapping.pack()) == mapping


def test_phased_mapping_roundtrips_without_model_specific_fields() -> None:
    mapping = MatrixViewMap(tile_pitch_rows=128, tile_phase_stride=11)
    restored = MatrixViewMap.unpack(mapping.pack())
    assert restored.tile_pitch_rows == 128
    assert restored.tile_phase_stride == 11


def test_programmable_row_coefficient_is_rejected_before_rtl() -> None:
    word = 128 | (3 << 16) | (11 << 22)
    with pytest.raises(ValueError, match=r"bits \[21:16\] are reserved"):
        MatrixViewMap.unpack(word)


def test_zero_pitch_compacts_tiles_into_distinct_banks() -> None:
    descriptor = MatrixViewDescriptor(
        MatrixViewShape(rows=128, cols=128, tile_count=8),
        MatrixViewMap(
            tile_pitch_rows=0,
            tile_phase_stride=4,
        ),
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


@pytest.mark.parametrize("reserved_flag", [1 << 0, 1 << 1, 1 << 2])
def test_unused_mapping_flag_bits_are_reserved(reserved_flag: int) -> None:
    word = 4 | (reserved_flag << 28)
    with pytest.raises(ValueError, match="unknown Matrix-view flags"):
        MatrixViewMap.unpack(word)








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
            flags=MatrixViewFlags(0),
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


def test_matrix_view_reservation_survives_weight_reset() -> None:
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
        MatrixViewMap(
            tile_pitch_rows=8,
            tile_phase_stride=4,
        ),
    )
    scalar = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=32, tile_count=16),
        MatrixViewMap(
            tile_pitch_rows=1,
            tile_phase_stride=3,
        ),
    )
    vector = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=128, tile_count=16),
        MatrixViewMap(
            tile_pitch_rows=1,
            tile_phase_stride=3,
        ),
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
        MatrixViewMap(
            tile_pitch_rows=4,
            tile_phase_stride=2,
        ),
    )
    scalar = MatrixViewDescriptor(
        MatrixViewShape(rows=16, cols=32, tile_count=32),
        MatrixViewMap(
            tile_pitch_rows=1,
            tile_phase_stride=1,
        ),
    )
    vector = MatrixViewDescriptor(
        MatrixViewShape(rows=1, cols=64, tile_count=32),
        MatrixViewMap(
            tile_pitch_rows=1,
            tile_phase_stride=1,
        ),
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
