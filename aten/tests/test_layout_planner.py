from compiler.aten.plena.affine_layout import BankGeometry, LayoutKind, LogicalCoord
from compiler.aten.plena.layout_planner import (
    AccessPacket,
    AffineLayoutPlanner,
    LayoutRequest,
    full_row_packets,
)


def test_planner_selects_affine_only_when_total_cost_is_lower():
    geometry = BankGeometry(banks=4, bank_width=1)
    request = LayoutRequest(
        name="four_parallel_rows",
        groups=1,
        fields=1,
        majors=4,
        minors=4,
        producer_packets=full_row_packets(groups=1, fields=1, majors=4, minors=4),
        consumer_packets=(
            AccessPacket(
                "column_word",
                tuple(LogicalCoord(0, 0, major, 0) for major in range(4)),
                repeats=16,
            ),
        ),
        transpose_supported=False,
        lane_restore_cycles_per_packet=1,
    )
    plan = AffineLayoutPlanner(geometry).plan(request)

    assert plan.baseline.read_cycles == 64
    assert plan.selected.layout.kind == LayoutKind.AFFINE_SKEW
    assert plan.selected.read_cycles == 16
    assert plan.selected.conflict_stall_cycles == 0
    assert plan.selected.total_cycles < plan.baseline.total_cycles


def test_planner_keeps_row_major_when_rotation_cannot_repay_restore():
    geometry = BankGeometry(banks=4, bank_width=1)
    request = LayoutRequest(
        name="ordinary_rows",
        groups=1,
        fields=1,
        majors=2,
        minors=4,
        producer_packets=full_row_packets(groups=1, fields=1, majors=2, minors=4),
        consumer_packets=full_row_packets(
            groups=1, fields=1, majors=2, minors=4, name="consumer_row"
        ),
        lane_restore_cycles_per_packet=1,
    )
    plan = AffineLayoutPlanner(geometry).plan(request)

    assert plan.selected.name == "row_major"
    assert plan.speedup == 1.0


def test_consumer_major_is_a_zero_gather_software_candidate():
    geometry = BankGeometry(banks=4, bank_width=1)
    request = LayoutRequest(
        name="direct_write",
        groups=1,
        fields=1,
        majors=2,
        minors=4,
        producer_packets=full_row_packets(groups=1, fields=1, majors=2, minors=4),
        consumer_packets=full_row_packets(
            groups=1, fields=1, majors=2, minors=4, name="consumer_row"
        ),
        baseline_reorder_cycles=20,
        consumer_major_supported=True,
        lane_restore_cycles_per_packet=5,
    )
    plan = AffineLayoutPlanner(geometry).plan(request)

    assert plan.selected.name == "consumer_major"
    assert plan.selected.layout.kind == LayoutKind.CONSUMER_MAJOR
    assert plan.selected.reorder_cycles == 0
    assert plan.baseline.reorder_cycles == 20


def test_request_rejects_out_of_bounds_or_duplicate_packets():
    try:
        AccessPacket(
            "duplicate",
            (LogicalCoord(0, 0, 0, 0), LogicalCoord(0, 0, 0, 0)),
        )
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate logical values must fail")

    try:
        LayoutRequest(
            name="bad",
            groups=1,
            fields=1,
            majors=1,
            minors=1,
            producer_packets=(AccessPacket("bad", (LogicalCoord(0, 0, 1, 0),)),),
            consumer_packets=(AccessPacket("ok", (LogicalCoord(0, 0, 0, 0),)),),
        )
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("out-of-range packet coordinates must fail")


def test_planner_prices_transpose_write_scatter_before_selecting_it():
    geometry = BankGeometry(banks=4, bank_width=2)
    request = LayoutRequest(
        name="transpose_candidate",
        groups=1,
        fields=1,
        majors=4,
        minors=8,
        producer_packets=full_row_packets(groups=1, fields=1, majors=4, minors=8),
        consumer_packets=(
            AccessPacket(
                "column",
                tuple(LogicalCoord(0, 0, major, 0) for major in range(4)),
            ),
        ),
        transpose_supported=True,
    )
    plan = AffineLayoutPlanner(geometry).plan(request)
    transpose = next(candidate for candidate in plan.candidates if candidate.name == "transpose")
    row = plan.baseline

    assert transpose.read_cycles < row.read_cycles
    assert transpose.write_cycles > row.write_cycles
    assert transpose.total_cycles > row.total_cycles
