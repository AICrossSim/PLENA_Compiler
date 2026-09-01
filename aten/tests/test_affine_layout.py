from compiler.aten.plena.affine_layout import (
    AffineLayout,
    BankGeometry,
    LayoutKind,
    LogicalCoord,
)


def test_affine_layout_is_bijective_and_round_row_is_one_cycle():
    geometry = BankGeometry(banks=16, bank_width=4)
    layout = AffineLayout(
        kind=LayoutKind.AFFINE_SKEW,
        groups=2,
        fields=3,
        majors=8,
        minors=64,
        alpha=1,
        beta=5,
        gamma=7,
    )
    layout.assert_bijective(geometry)
    packet = [LogicalCoord(0, 0, 0, minor) for minor in range(64)]
    stats = layout.packet_service(packet, geometry)
    assert stats.service_cycles == 1
    assert stats.bandwidth_floor_cycles == 1
    assert stats.conflict_stall_cycles == 0


def test_affine_skew_removes_a_real_multirow_bank_conflict():
    geometry = BankGeometry(banks=16, bank_width=4)
    row = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 8, 64)
    skew = AffineLayout(LayoutKind.AFFINE_SKEW, 1, 1, 8, 64, alpha=1)
    # One four-element word from each of eight rows. Row-major sends all eight
    # words to bank 0; alpha=1 rotates them across eight banks.
    packet = [LogicalCoord(0, 0, major, minor) for major in range(8) for minor in range(4)]
    row_stats = row.packet_service(packet, geometry)
    skew_stats = skew.packet_service(packet, geometry)
    assert row_stats.service_cycles == 8
    assert skew_stats.service_cycles == 1
    assert skew_stats.conflict_stall_cycles == 0


def test_bandwidth_floor_does_not_mislabel_two_cycles_as_a_conflict():
    geometry = BankGeometry(banks=16, bank_width=1)
    layout = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 2, 16)
    packet = list(layout.iter_coords())
    stats = layout.packet_service(packet, geometry)
    assert stats.bank_words == 32
    assert stats.bandwidth_floor_cycles == 2
    assert stats.service_cycles == 2
    assert stats.conflict_stall_cycles == 0


def test_an_aliasing_pitch_is_rejected():
    geometry = BankGeometry(banks=4, bank_width=1)
    layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=1,
        fields=1,
        majors=2,
        minors=8,
        bank_row_pitch=1,
    )
    try:
        layout.assert_bijective(geometry)
    except ValueError as error:
        assert "smaller than" in str(error)
    else:
        raise AssertionError("an undersized physical pitch must fail")


def test_transpose_exchanges_row_and_column_service_costs():
    geometry = BankGeometry(banks=4, bank_width=2)
    row = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 4, 8)
    transpose = AffineLayout(LayoutKind.TRANSPOSE, 1, 1, 4, 8)
    logical_row = [LogicalCoord(0, 0, 0, minor) for minor in range(8)]
    logical_column = [LogicalCoord(0, 0, major, 0) for major in range(4)]

    assert row.packet_service(logical_row, geometry).service_cycles == 1
    assert row.packet_service(logical_column, geometry).service_cycles == 4
    assert transpose.packet_service(logical_row, geometry).service_cycles == 8
    assert transpose.packet_service(logical_column, geometry).service_cycles == 1
    transpose.assert_bijective(geometry)


def test_major_packed_layout_coalesces_one_short_row_per_bank():
    geometry = BankGeometry(banks=32, bank_width=64)
    row = AffineLayout(LayoutKind.ROW_MAJOR, 1, 1, 32, 64)
    packed = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=1,
        fields=1,
        majors=32,
        minors=64,
        alpha=1,
        major_packed=True,
    )
    packet = list(packed.iter_coords())

    packed.assert_bijective(geometry)
    assert len({row.place(coord, geometry).bank_row for coord in packet}) == 32
    assert len({packed.place(coord, geometry).bank_row for coord in packet}) == 1
    assert row.packet_service(packet, geometry).service_cycles == 32
    assert packed.packet_service(packet, geometry).service_cycles == 1
    assert packed.packet_service(packet, geometry).conflict_stall_cycles == 0


def test_major_packed_kda_subviews_advance_by_complete_bank_groups():
    geometry = BankGeometry(banks=32, bank_width=64)
    layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=1,
        fields=1,
        majors=96,
        minors=128,
        alpha=1,
        major_packed=True,
    )

    layout.assert_bijective(geometry)
    assert layout.major_start_row_offset(0, geometry) == 0
    assert layout.major_start_row_offset(32, geometry) == 2
    assert layout.major_start_row_offset(64, geometry) == 4
    try:
        layout.major_start_row_offset(1, geometry)
    except ValueError as error:
        assert "complete bank group" in str(error)
    else:
        raise AssertionError("an unaligned major-packed subview must fail")


def test_major_packed_layout_rejects_a_non_permuting_bank_rotation():
    geometry = BankGeometry(banks=32, bank_width=64)
    layout = AffineLayout(
        LayoutKind.AFFINE_SKEW,
        groups=1,
        fields=1,
        majors=32,
        minors=64,
        alpha=2,
        major_packed=True,
    )

    try:
        layout.assert_bijective(geometry)
    except ValueError as error:
        assert "must permute every physical bank" in str(error)
    else:
        raise AssertionError("a major-packed mapping that skips banks must fail")
