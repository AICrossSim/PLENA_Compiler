"""Unit tests for mid_ir.passes.split (pass_3).

Coverage:
  * BLOCK_IDX axis with extent == cluster_count → number=1, phase=cluster
  * BLOCK_IDX axis with extent == 2*cluster_count → number=2, phase=cluster
  * non-lane BLOCK_IDX (q_block) preserved untouched
  * For (T.serial) preserved untouched (never split)
  * Lane-aware buffers (scope != "global") get an outer LANE dim
  * Global buffers (HBM params) stay unchanged
  * BufferRef.indices NOT touched
  * ParallelAxis nested INSIDE a For (conv2d-style) gets handled too
  * Multi-axis lane fusion: two axes both split
  * Extent not divisible → SplitError

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_split
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.mark import run as mark_run
from tilelang_tvm_compiler.frontend.mid_ir.passes.split import (
    SplitError,
    run as split_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _slice_ref(buf):
    return ir.BufferRef(buffer=buf, indices=[ir.Slice() for _ in buf.shape])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _block_idx(name, extent, body, tag="blockIdx.y"):
    return ir.ParallelAxis(
        axis_name=name, extent=extent, body=body,
        kind=ir.ParallelKind.BLOCK_IDX, thread_tag=tag,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_split_extent_eq_cluster() -> int:
    print("test_split_extent_eq_cluster — head_count == LANE")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh  = _mk_buf("Q_sh",  [64, 16])
    fn = ir.MidFunc(
        name="t",
        params=[Q_hbm], allocs=[Q_sh],
        body=[_block_idx("by", LANE, [
            ir.Dma(src=_slice_ref(Q_hbm), dst=_slice_ref(Q_sh)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    failures += _check("cluster_counts", out.cluster_counts, [LANE])
    if not (out.body and isinstance(out.body[0], ir.ParallelAxis)):
        return 1
    by_number = out.body[0]
    failures += _check("by_number axis_name", by_number.axis_name, "by_number")
    failures += _check("by_number kind", by_number.kind, ir.ParallelKind.BLOCK_IDX)
    failures += _check("by_number extent", by_number.extent, 1)
    failures += _check("by_number thread_tag", by_number.thread_tag, "blockIdx.y")
    by_phase = by_number.body[0]
    failures += _check("by_phase axis_name", by_phase.axis_name, "by_phase")
    failures += _check("by_phase kind", by_phase.kind, ir.ParallelKind.CLUSTER)
    failures += _check("by_phase extent", by_phase.extent, LANE)
    failures += _check("by_phase thread_tag", by_phase.thread_tag, None)
    # cluster → grid back-link
    failures += _check("by_phase parent_grid_axis_name",
                       by_phase.parent_grid_axis_name, "by_number")
    failures += _check("by_number parent_grid_axis_name",
                       by_number.parent_grid_axis_name, None)
    return failures


def test_split_extent_multiple() -> int:
    print("test_split_extent_multiple — head_count == 2*LANE")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[_block_idx("by", 2 * LANE, [
            ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    by_number = out.body[0]
    by_phase = by_number.body[0]
    return (_check("by_number extent", by_number.extent, 2)
            + _check("by_phase extent", by_phase.extent, LANE))


def test_split_buffer_growth() -> int:
    print("test_split_buffer_growth")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh  = _mk_buf("Q_sh",  [64, 16], scope="shared")
    S_loc = _mk_buf("S_loc", [64, 64], scope="fragment")
    fn = ir.MidFunc(
        name="t", params=[Q_hbm], allocs=[Q_sh, S_loc],
        body=[_block_idx("by", LANE, [
            ir.Dma(src=_slice_ref(Q_hbm), dst=_slice_ref(Q_sh)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    failures += _check("Q_hbm shape (global)", out.params[0].shape,
                       [1, 64, 4, 16])
    Q_sh_grown = next(b for b in out.allocs if b.name == "Q_sh")
    S_loc_grown = next(b for b in out.allocs if b.name == "S_loc")
    failures += _check("Q_sh shape", Q_sh_grown.shape, [LANE, 64, 16])
    failures += _check("S_loc shape", S_loc_grown.shape, [LANE, 64, 64])
    return failures


def test_split_indices_unchanged() -> int:
    """BufferRef.indices stay rank-2 even though the underlying buffer
    is now rank-3. pass_4 will fix the mismatch."""
    print("test_split_indices_unchanged")
    Q_sh = _mk_buf("Q_sh", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q_sh],
        body=[_block_idx("by", LANE, [
            ir.Dma(src=_slice_ref(Q_sh), dst=_slice_ref(Q_sh)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    by_number = out.body[0]
    by_phase = by_number.body[0]
    dma = by_phase.body[0]
    failures = 0
    failures += _check("dma.src buffer rank", len(dma.src.buffer.shape), 3)
    failures += _check("dma.src.indices rank", len(dma.src.indices), 2)
    return failures


def test_split_non_lane_blockidx_preserved() -> int:
    print("test_split_non_lane_blockidx_preserved — q_block stays")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[_block_idx("q_block", 2, [
            _block_idx("by", LANE, [
                ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
            ]),
        ], tag="blockIdx.x")],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    qb = out.body[0]
    failures += _check("q_block axis_name", qb.axis_name, "q_block")
    failures += _check("q_block kind", qb.kind, ir.ParallelKind.BLOCK_IDX)
    failures += _check("q_block extent", qb.extent, 2)
    failures += _check("q_block thread_tag", qb.thread_tag, "blockIdx.x")
    by_number = qb.body[0]
    failures += _check("by_number axis_name", by_number.axis_name, "by_number")
    return failures


def test_split_for_serial_preserved() -> int:
    """A real T.serial For (e.g. conv2d's `for oc`) is NEVER split.
    split only touches BLOCK_IDX ParallelAxis nodes."""
    print("test_split_for_serial_preserved")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[ir.For(loop_var="oc", extent=4, body=[
            ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    f = out.body[0]
    return (_check("type", type(f).__name__, "For")
            + _check("loop_var", f.loop_var, "oc")
            + _check("kind", f.kind, "serial"))


def test_split_parallel_axis_inside_for() -> int:
    """conv2d-style structure: outer For(serial) wraps a ParallelAxis
    that needs splitting. Walker recurses into For body and splits the
    inner ParallelAxis."""
    print("test_split_parallel_axis_inside_for")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[ir.For(loop_var="oc", extent=4, body=[
            _block_idx("by", LANE, [
                ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
            ]),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    f = out.body[0]
    failures += _check("outer For preserved", type(f).__name__, "For")
    failures += _check("outer For loop_var", f.loop_var, "oc")
    inner_number = f.body[0]
    failures += _check("inner is ParallelAxis", type(inner_number).__name__,
                       "ParallelAxis")
    failures += _check("inner axis_name", inner_number.axis_name, "by_number")
    inner_phase = inner_number.body[0]
    failures += _check("phase axis_name", inner_phase.axis_name, "by_phase")
    failures += _check("phase kind", inner_phase.kind, ir.ParallelKind.CLUSTER)
    return failures


def test_split_logical_grid_axis() -> int:
    """A LOGICAL_GRID axis (unfolded T.Parallel) is split the same way
    as a BLOCK_IDX axis. The number axis stays LOGICAL_GRID (no
    thread_tag); the phase axis is CLUSTER and back-references it."""
    print("test_split_logical_grid_axis — LOGICAL_GRID can also be split")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[ir.ParallelAxis(
            axis_name="m", extent=LANE, kind=ir.ParallelKind.LOGICAL_GRID,
            thread_tag=None,
            body=[ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q))],
        )],
        lane_axes=["m"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    if not (out.body and isinstance(out.body[0], ir.ParallelAxis)):
        return 1
    m_number = out.body[0]
    failures += _check("m_number axis_name", m_number.axis_name, "m_number")
    failures += _check("m_number kind preserved",
                       m_number.kind, ir.ParallelKind.LOGICAL_GRID)
    failures += _check("m_number thread_tag", m_number.thread_tag, None)
    m_phase = m_number.body[0]
    failures += _check("m_phase axis_name", m_phase.axis_name, "m_phase")
    failures += _check("m_phase kind", m_phase.kind, ir.ParallelKind.CLUSTER)
    failures += _check("m_phase parent_grid_axis_name",
                       m_phase.parent_grid_axis_name, "m_number")
    return failures


def test_split_extent_not_divisible_raises() -> int:
    print("test_split_extent_not_divisible_raises")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[_block_idx("by", LANE + 1, [
            ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    try:
        split_run(fn)
    except SplitError as e:
        print(f"  [OK]   raised SplitError: {e}")
        return 0
    return 1


def test_split_no_lane_axes_no_op() -> int:
    """Kernel without lane_axes: split is a no-op (returns input
    unchanged), no error. This is the cluster_guard skip path."""
    print("test_split_no_lane_axes_no_op")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q))],
        lane_axes=[],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    failures += _check("body unchanged length", len(out.body), 1)
    failures += _check("body[0] still Dma", type(out.body[0]).__name__, "Dma")
    failures += _check("Q shape unchanged", out.allocs[0].shape, [64, 16])
    failures += _check("cluster_counts empty", out.cluster_counts, [])
    return failures


def test_split_skipped_when_d_ge_mlen() -> int:
    """Every non-global buffer's last dim >= MLEN (=64): split is
    a no-op even with lane_axes declared. One lane already fills a
    whole HW vector."""
    print("test_split_skipped_when_d_ge_mlen — D=64 buffers don't need cluster")
    A = _mk_buf("A", [4, 64], scope="shared")    # last dim = 64 = MLEN
    B = _mk_buf("B", [4, 128], scope="fragment") # last dim = 128 > MLEN
    fn = ir.MidFunc(
        name="t", params=[], allocs=[A, B],
        body=[_block_idx("by", LANE, [
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(B)),
        ])],
        lane_axes=["by"],   # declared but unneeded
    )
    fn = mark_run(fn)
    out = split_run(fn)
    failures = 0
    # Body should be unchanged: still one ParallelAxis(BLOCK_IDX, "by", extent=4)
    failures += _check("body[0] is ParallelAxis",
                       type(out.body[0]).__name__, "ParallelAxis")
    failures += _check("axis_name unchanged", out.body[0].axis_name, "by")
    failures += _check("extent unchanged", out.body[0].extent, 4)
    failures += _check("A shape unchanged", out.allocs[0].shape, [4, 64])
    failures += _check("B shape unchanged", out.allocs[1].shape, [4, 128])
    return failures


def test_split_runs_when_one_buffer_d_lt_mlen() -> int:
    """Even one buffer with D<MLEN forces full cluster pipeline."""
    print("test_split_runs_when_one_buffer_d_lt_mlen — D=16 forces cluster")
    A = _mk_buf("A", [4, 64], scope="shared")    # D=64=MLEN → would skip alone
    B = _mk_buf("B", [4, 16], scope="fragment")  # D=16<MLEN → forces cluster
    fn = ir.MidFunc(
        name="t", params=[], allocs=[A, B],
        body=[_block_idx("by", LANE, [
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(B)),
        ])],
        lane_axes=["by"],
    )
    fn = mark_run(fn)
    out = split_run(fn)
    # Should split into number → cluster phase
    by_number = out.body[0]
    return (_check("split happened", by_number.axis_name, "by_number")
            + _check("cluster_counts set", out.cluster_counts, [LANE])
            + _check("A grown", out.allocs[0].shape, [LANE, 4, 64]))


def test_split_multi_axis() -> int:
    print("test_split_multi_axis — lane_axes=['q_block', 'by']")
    Q = _mk_buf("Q", [64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[_block_idx("q_block", LANE, [
            _block_idx("by", LANE, [
                ir.Dma(src=_slice_ref(Q), dst=_slice_ref(Q)),
            ]),
        ], tag="blockIdx.x")],
        lane_axes=["q_block", "by"],
    )
    fn = mark_run(fn)
    out = split_run(fn, cluster_counts=[LANE, LANE])
    failures = 0
    failures += _check("cluster_counts", out.cluster_counts, [LANE, LANE])
    qb_num = out.body[0]
    failures += _check("q_block_number axis_name", qb_num.axis_name, "q_block_number")
    failures += _check("q_block_number kind", qb_num.kind, ir.ParallelKind.BLOCK_IDX)
    qb_phase = qb_num.body[0]
    failures += _check("q_block_phase axis_name", qb_phase.axis_name, "q_block_phase")
    failures += _check("q_block_phase kind", qb_phase.kind, ir.ParallelKind.CLUSTER)
    by_num = qb_phase.body[0]
    failures += _check("by_number axis_name", by_num.axis_name, "by_number")
    by_phase = by_num.body[0]
    failures += _check("by_phase axis_name", by_phase.axis_name, "by_phase")
    failures += _check("Q shape outer", out.allocs[0].shape[0], LANE * LANE)
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_split_extent_eq_cluster()
    failures += test_split_extent_multiple()
    failures += test_split_buffer_growth()
    failures += test_split_indices_unchanged()
    failures += test_split_non_lane_blockidx_preserved()
    failures += test_split_for_serial_preserved()
    failures += test_split_parallel_axis_inside_for()
    failures += test_split_logical_grid_axis()
    failures += test_split_extent_not_divisible_raises()
    failures += test_split_no_lane_axes_no_op()
    failures += test_split_skipped_when_d_ge_mlen()
    failures += test_split_runs_when_one_buffer_d_lt_mlen()
    failures += test_split_multi_axis()
    print()
    if failures == 0:
        print("PASS — all mid_ir.split tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
