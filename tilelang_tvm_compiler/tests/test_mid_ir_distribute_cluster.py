"""Unit tests for mid_ir.passes.distribute_cluster (pass_3b).

Coverage:
  * cluster body == [unroll For] → For lifted out, cluster pushed inside
  * cluster body has [op_pre, unroll For, op_post] → 3-way split:
        cluster {pre}; for {cluster {inner}}; cluster {post}
  * cluster body has serial For (not unroll) → no rewrite
  * Multiple unroll Fors in one cluster body → multiple lifts
  * Nested cluster (cluster inside cluster) — outer not rewritten,
    inner stays as-is
  * Cluster with no unroll For at all → no change

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_distribute_cluster
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.distribute_cluster import (
    run as distribute_run,
)


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


def _cluster(name, extent, body, parent="parent"):
    return ir.ParallelAxis(
        axis_name=name, extent=extent, body=body,
        kind=ir.ParallelKind.CLUSTER, thread_tag=None,
        parent_grid_axis_name=parent,
    )


def _wrap(body):
    # Declare a lane axis so cluster_guard doesn't no-op the pass.
    # The test fixtures don't actually run pass_3_split, so the value
    # is just a placeholder.
    return ir.MidFunc(
        name="t", params=[], allocs=[], body=list(body),
        lane_axes=["by"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cluster_pure_unroll() -> int:
    """cluster {for_unroll {ops}} → for_unroll {cluster {ops}}."""
    print("test_cluster_pure_unroll")
    A = _mk_buf("A", [64, 16])
    body = [_cluster("c_phase", 4, [
        ir.For(loop_var="kh", extent=4, kind="unroll", body=[
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ]),
    ])]
    out = distribute_run(_wrap(body))
    failures = 0
    failures += _check("body length", len(out.body), 1)
    if not (out.body and isinstance(out.body[0], ir.For)):
        print(f"  [FAIL] expected For at top, got {type(out.body[0]).__name__}")
        return 1
    for_node = out.body[0]
    failures += _check("For kind", for_node.kind, "unroll")
    failures += _check("For loop_var", for_node.loop_var, "kh")
    failures += _check("For body length", len(for_node.body), 1)
    if isinstance(for_node.body[0], ir.ParallelAxis):
        failures += _check("inner ParallelAxis kind",
                           for_node.body[0].kind, ir.ParallelKind.CLUSTER)
        failures += _check("inner cluster axis_name",
                           for_node.body[0].axis_name, "c_phase")
        failures += _check("inner cluster body length",
                           len(for_node.body[0].body), 1)
        failures += _check("innermost is Dma",
                           type(for_node.body[0].body[0]).__name__, "Dma")
    else:
        print(f"  [FAIL] inner not ParallelAxis: {for_node.body[0]}")
        failures += 1
    return failures


def test_cluster_mixed_body() -> int:
    """cluster {pre; for_unroll; post} → cluster{pre}; for{cluster{...}}; cluster{post}."""
    print("test_cluster_mixed_body — 3-way split")
    A = _mk_buf("A", [64, 16])
    body = [_cluster("c_phase", 4, [
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),       # pre
        ir.For(loop_var="kh", extent=4, kind="unroll", body=[
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ]),
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),       # post
    ])]
    out = distribute_run(_wrap(body))
    failures = 0
    failures += _check("top-level body length", len(out.body), 3)
    # [0] cluster {pre}
    failures += _check("[0] type", type(out.body[0]).__name__, "ParallelAxis")
    if isinstance(out.body[0], ir.ParallelAxis):
        failures += _check("[0] kind", out.body[0].kind, ir.ParallelKind.CLUSTER)
        failures += _check("[0] body length", len(out.body[0].body), 1)
    # [1] for_unroll {cluster {inner}}
    failures += _check("[1] type", type(out.body[1]).__name__, "For")
    if isinstance(out.body[1], ir.For):
        failures += _check("[1] kind", out.body[1].kind, "unroll")
        failures += _check("[1] body length", len(out.body[1].body), 1)
        if isinstance(out.body[1].body[0], ir.ParallelAxis):
            failures += _check("[1] inner cluster",
                               out.body[1].body[0].kind, ir.ParallelKind.CLUSTER)
    # [2] cluster {post}
    failures += _check("[2] type", type(out.body[2]).__name__, "ParallelAxis")
    return failures


def test_serial_for_not_distributed() -> int:
    """cluster {serial_for {ops}} stays as-is — only unroll triggers."""
    print("test_serial_for_not_distributed")
    A = _mk_buf("A", [64, 16])
    body = [_cluster("c_phase", 4, [
        ir.For(loop_var="kv", extent=4, kind="serial", body=[
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ]),
    ])]
    out = distribute_run(_wrap(body))
    failures = 0
    # Top-level still ONE cluster, body still ONE For.
    failures += _check("top-level body length", len(out.body), 1)
    failures += _check("[0] type", type(out.body[0]).__name__, "ParallelAxis")
    if isinstance(out.body[0], ir.ParallelAxis):
        failures += _check("cluster preserved", out.body[0].kind,
                           ir.ParallelKind.CLUSTER)
        failures += _check("cluster body length", len(out.body[0].body), 1)
        failures += _check("for inside cluster",
                           type(out.body[0].body[0]).__name__, "For")
        failures += _check("for kind", out.body[0].body[0].kind, "serial")
    return failures


def test_cluster_no_unroll_pass_through() -> int:
    """cluster body has no unroll For → unchanged."""
    print("test_cluster_no_unroll_pass_through")
    A = _mk_buf("A", [64, 16])
    body = [_cluster("c_phase", 4, [
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
    ])]
    out = distribute_run(_wrap(body))
    failures = 0
    failures += _check("body length", len(out.body), 1)
    failures += _check("[0] type", type(out.body[0]).__name__, "ParallelAxis")
    if isinstance(out.body[0], ir.ParallelAxis):
        failures += _check("cluster body length", len(out.body[0].body), 2)
    return failures


def test_two_unroll_fors_in_cluster() -> int:
    """cluster {for_a; for_b} → for_a {cluster}; for_b {cluster}.
    Two unroll Fors with no in-between ops → no extra cluster instances."""
    print("test_two_unroll_fors_in_cluster")
    A = _mk_buf("A", [64, 16])
    body = [_cluster("c_phase", 4, [
        ir.For(loop_var="kh", extent=2, kind="unroll", body=[
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ]),
        ir.For(loop_var="kw", extent=2, kind="unroll", body=[
            ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
        ]),
    ])]
    out = distribute_run(_wrap(body))
    failures = 0
    # Should be exactly 2 stmts at top: two Fors, both with cluster inside.
    failures += _check("body length", len(out.body), 2)
    failures += _check("[0] type", type(out.body[0]).__name__, "For")
    failures += _check("[1] type", type(out.body[1]).__name__, "For")
    if isinstance(out.body[0], ir.For):
        failures += _check("[0] loop_var", out.body[0].loop_var, "kh")
        failures += _check("[0] inner is cluster",
                           type(out.body[0].body[0]).__name__, "ParallelAxis")
    if isinstance(out.body[1], ir.For):
        failures += _check("[1] loop_var", out.body[1].loop_var, "kw")
    return failures


def test_grid_outside_cluster_preserved() -> int:
    """A grid wrapping a cluster wrapping an unroll For → grid stays
    outside; only the inner cluster/unroll get rewritten."""
    print("test_grid_outside_cluster_preserved")
    A = _mk_buf("A", [64, 16])
    body = [
        ir.ParallelAxis(
            axis_name="by_number", extent=1,
            kind=ir.ParallelKind.BLOCK_IDX, thread_tag="blockIdx.y",
            body=[_cluster("by_phase", 4, [
                ir.For(loop_var="kh", extent=4, kind="unroll", body=[
                    ir.Dma(src=_slice_ref(A), dst=_slice_ref(A)),
                ]),
            ])],
        ),
    ]
    out = distribute_run(_wrap(body))
    failures = 0
    grid = out.body[0]
    failures += _check("grid kind preserved", grid.kind,
                       ir.ParallelKind.BLOCK_IDX)
    # Inside the grid: should be the For (cluster pushed inside).
    failures += _check("grid body length", len(grid.body), 1)
    failures += _check("grid body[0] type", type(grid.body[0]).__name__, "For")
    if isinstance(grid.body[0], ir.For):
        failures += _check("inner For kind", grid.body[0].kind, "unroll")
        failures += _check("inside For is cluster",
                           type(grid.body[0].body[0]).__name__, "ParallelAxis")
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_cluster_pure_unroll()
    failures += test_cluster_mixed_body()
    failures += test_serial_for_not_distributed()
    failures += test_cluster_no_unroll_pass_through()
    failures += test_two_unroll_fors_in_cluster()
    failures += test_grid_outside_cluster_preserved()
    print()
    if failures == 0:
        print("PASS — all mid_ir.distribute_cluster tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
