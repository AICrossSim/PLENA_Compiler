from __future__ import annotations

from compiler.aten.plena.matrix_access_packets import (
    PacketGeometry,
    coissued_packet_histogram,
    extract_matrix_access_packets,
    matrix_access_instruction_count,
    packet_histogram,
)
from compiler.aten.plena.mview import MatrixViewMap, MatrixViewShape

import pytest


GEOMETRY = PacketGeometry(mlen=64, blen=4, hlen=16)


def test_public_instruction_count_matches_multi_operand_extraction_boundary() -> None:
    shape = MatrixViewShape(rows=1, cols=4, tile_count=16).pack()
    mapping = MatrixViewMap(tile_pitch_rows=1).pack()
    assembly = f"""
S_ADDI_INT gp1, gp0, {shape}
S_ADDI_INT gp2, gp0, {mapping}
L_MVIEW_FULL 0, gp1, gp2
L_MVIEW_FULL 1, gp1, gp2
L_MVIEW_FULL 2, gp1, gp2
V_ADD_VV.MV gp3, gp4, gp5, 0, 7
M_MM_WO gp3, gp0, 0, 0
"""
    packets = extract_matrix_access_packets(assembly, GEOMETRY)
    assert matrix_access_instruction_count(assembly) == 2
    assert len({packet.instruction_index for packet in packets}) == 2
    assert len(packets) == 4


def test_extracts_every_matrix_access_and_resolves_known_tile_addresses() -> None:
    assembly = """
; @stage=kda_qkv_proj
S_ADDI_INT gp1, gp0, 4096
S_ADDI_INT gp2, gp0, 128
M_MM 0, gp1, gp2
M_TMM 0, gp1, gp2
; @stage=attention
M_BTMM 0, gp1, gp2
H_PREFETCH_M gp1, gp0, a0, gp0, 0
"""
    packets = extract_matrix_access_packets(assembly, GEOMETRY)

    assert len(packets) == 4
    assert packets[0].matrix_address == 4096
    assert packets[0].sample_cells[0].tile == 1
    assert packets[0].axis == "row"
    assert packets[1].axis == "column"
    assert packets[2].elements_per_tile == 16
    assert packets[3].direction == "write"


def test_hardware_loops_change_multiplicity_not_packet_shape() -> None:
    assembly = """
; @stage=moe_expert_projection
C_LOOP_START gp7, 8
M_MM 0, gp1, gp2
C_LOOP_END gp7
"""
    packet = extract_matrix_access_packets(assembly, GEOMETRY)[0]
    assert packet.tile_count == 1
    assert packet.elements_per_tile == 4
    assert packet.repeats == 8 * 64


def test_existing_matrix_ops_expose_only_single_tile_accesses() -> None:
    packets = extract_matrix_access_packets(
        """
; @stage=mamba_projection
M_MM 0, gp1, gp2
; @stage=kda_projection
M_MM 0, gp3, gp4
; @stage=attention
M_TMM 0, gp5, gp6
; @stage=moe
M_MM 0, gp7, gp8
""",
        GEOMETRY,
    )
    histogram = packet_histogram(packets)

    assert all(entry["tiles"] == 1 for entry in histogram)
    assert not any(entry["per_tile_skew_can_help"] for entry in histogram)


def test_optional_view_is_explicit_on_the_consumer() -> None:
    shape = MatrixViewShape(rows=1, cols=4, tile_count=16).pack()
    mapping = MatrixViewMap(tile_pitch_rows=1).pack()
    packet = extract_matrix_access_packets(
        f"""
S_ADDI_INT gp7, gp0, {shape}
S_ADDI_INT gp8, gp0, {mapping}
L_MVIEW_FULL 2, gp7, gp8
M_MM 0, gp1, gp2, 2
""",
        GEOMETRY,
    )[0]
    assert packet.view_slot == 2
    assert packet.tile_count == 16
    assert packet.elements_per_tile == 4


def test_extracts_real_multi_tile_view_packets_and_direct_writeback() -> None:
    shape = MatrixViewShape(rows=1, cols=4, tile_count=16).pack()
    mapping = MatrixViewMap(tile_pitch_rows=1).pack()
    packets = extract_matrix_access_packets(
        f"""
; @stage=kda_projection @axis=cross_head
S_ADDI_INT gp1, gp0, {shape}
S_ADDI_INT gp2, gp0, {mapping}
L_MVIEW_FULL 0, gp1, gp2
L_MVIEW_FULL 1, gp1, gp2
L_MVIEW_FULL 2, gp1, gp2
S_ADDI_INT gp3, gp0, 4096
S_ADDI_INT gp4, gp0, 8192
S_ADDI_INT gp5, gp0, 12288
V_MUL_VV.MV gp3, gp4, gp5, 0, 7
M_MM_WO gp3, gp0, 0, 0
""",
        GEOMETRY,
    )

    destination, source1, source2, writeback = packets
    assert {packet.instruction_index for packet in packets[:3]} == {
        destination.instruction_index
    }
    assert destination.direction == "write"
    assert (source1.direction, source2.direction) == ("read", "read")
    assert [packet.operand for packet in packets[:3]] == [
        "destination",
        "source1",
        "source2",
    ]
    assert source1.axis == "cross_head"
    assert source1.tile_count == 16
    assert source1.elements_per_tile == 4
    assert len(source1.sample_cells) == 64
    assert source1.sample_cells[-1].tile == 15
    assert writeback.direction == "write"
    assert writeback.tile_count == 1
    assert writeback.elements_per_tile == 4
    assert packet_histogram((source1,))[0]["per_tile_skew_can_help"]

    coissued = coissued_packet_histogram(packets)
    read_group = next(entry for entry in coissued if entry["direction"] == "read")
    assert read_group["same_cycle_operands"] == 2
    assert read_group["tiles_total"] == 32
    assert read_group["values_per_service"] == 128
    assert [operand["name"] for operand in read_group["operands"]] == [
        "source1",
        "source2",
    ]
    destination_group = next(
        entry
        for entry in coissued
        if entry["direction"] == "write" and entry["opcode"] == "V_MUL_VV.MV"
    )
    assert destination_group["same_cycle_operands"] == 1


def test_view_packet_extraction_rejects_an_unconfigured_slot() -> None:
    with pytest.raises(ValueError, match="unconfigured Matrix-view slot"):
        extract_matrix_access_packets("V_MUL_VV.MV gp1, gp2, gp3, 0, 2", GEOMETRY)
