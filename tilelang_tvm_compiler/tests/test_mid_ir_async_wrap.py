"""Unit tests for mid_ir.passes.async_wrap (pass_4).

Coverage:
  * can_async=True ops inside cluster body get wrapped in Async (one
    op per Async region — strict)
  * can_async=False ops (Reduce, broadcast Elementwise) stay unwrapped
  * Ops outside cluster (top-level RawStore, etc.) not touched
  * Ops in non-cluster ParallelAxis (grid / logical_grid) not wrapped
  * Multiple consecutive can_async ops → multiple Async regions
  * BufferRef indices NOT rewritten — that's the next (view) pass

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_async_wrap
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.async_wrap import (
    AsyncWrapError,
    run as async_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _ref(buf, indices):
    return ir.BufferRef(buf, list(indices))


def _slice_ref(buf):
    return _ref(buf, [ir.Slice() for _ in buf.shape])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _cluster(body):
    return ir.ParallelAxis(
        axis_name="by_phase", extent=LANE, body=body,
        kind=ir.ParallelKind.CLUSTER, thread_tag=None,
        parent_grid_axis_name="by_number",
    )


def _grid(body):
    return ir.ParallelAxis(
        axis_name="by_number", extent=1, body=body,
        kind=ir.ParallelKind.BLOCK_IDX, thread_tag="blockIdx.y",
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


def test_can_async_true_gets_wrapped() -> int:
    """A Dma with can_async=True inside a cluster gets wrapped in Async."""
    print("test_can_async_true_gets_wrapped")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")
    fn = _wrap([_grid([_cluster([
        ir.Dma(src=_slice_ref(Q_hbm), dst=_slice_ref(Q_sh),
               marker=ir.Marker.DMA, can_async=True),
    ])])])
    out = async_run(fn)
    cluster = out.body[0].body[0]
    failures = 0
    failures += _check("cluster body length", len(cluster.body), 1)
    failures += _check("body[0] is Async", type(cluster.body[0]).__name__, "Async")
    if isinstance(cluster.body[0], ir.Async):
        async_node = cluster.body[0]
        failures += _check("Async body length", len(async_node.body), 1)
        failures += _check("inner is Dma", type(async_node.body[0]).__name__, "Dma")
    return failures


def test_can_async_false_not_wrapped() -> int:
    """A Reduce (can_async=False) stays bare in the cluster body."""
    print("test_can_async_false_not_wrapped")
    S = _mk_buf("S", [LANE, 64, 64], scope="fragment")
    M = _mk_buf("M", [LANE, 64], scope="fragment")
    fn = _wrap([_grid([_cluster([
        ir.Reduce(dst=_slice_ref(M), src=_slice_ref(S),
                  op=ir.ReduceOp.MAX, axis=1,
                  marker=ir.Marker.LANE_OP, can_async=False),
    ])])])
    out = async_run(fn)
    cluster = out.body[0].body[0]
    return (_check("body length", len(cluster.body), 1)
            + _check("body[0] is Reduce (not Async)",
                     type(cluster.body[0]).__name__, "Reduce"))


def test_strict_one_async_one_op() -> int:
    """Two consecutive can_async ops → two separate Async regions."""
    print("test_strict_one_async_one_op")
    A = _mk_buf("A", [LANE, 64, 16])
    B = _mk_buf("B", [LANE, 64, 16])
    fn = _wrap([_grid([_cluster([
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(B),
               marker=ir.Marker.DMA, can_async=True),
        ir.Dma(src=_slice_ref(B), dst=_slice_ref(A),
               marker=ir.Marker.DMA, can_async=True),
    ])])])
    out = async_run(fn)
    cluster_body = out.body[0].body[0].body
    failures = 0
    failures += _check("two stmts", len(cluster_body), 2)
    failures += _check("[0] type", type(cluster_body[0]).__name__, "Async")
    failures += _check("[1] type", type(cluster_body[1]).__name__, "Async")
    failures += _check("scope_ids unique",
                       cluster_body[0].scope_id != cluster_body[1].scope_id,
                       True)
    return failures


def test_mixed_async_and_non_async() -> int:
    """Cluster body with mixed can_async + can_async=False ops:
    only the True ones get Async-wrapped."""
    print("test_mixed_async_and_non_async")
    Q = _mk_buf("Q", [LANE, 64, 16])
    K = _mk_buf("K", [LANE, 64, 16])
    S = _mk_buf("S", [LANE, 64, 64], scope="fragment")
    M = _mk_buf("M", [LANE, 64], scope="fragment")
    fn = _wrap([_grid([_cluster([
        ir.Dma(src=_slice_ref(Q), dst=_slice_ref(K),
               marker=ir.Marker.DMA, can_async=True),                       # → Async
        ir.Reduce(dst=_slice_ref(M), src=_slice_ref(S),
                  op=ir.ReduceOp.MAX, axis=1,
                  marker=ir.Marker.LANE_OP, can_async=False),                # bare
        ir.Dma(src=_slice_ref(K), dst=_slice_ref(Q),
               marker=ir.Marker.DMA, can_async=True),                       # → Async
    ])])])
    out = async_run(fn)
    body = out.body[0].body[0].body
    failures = 0
    failures += _check("body length", len(body), 3)
    failures += _check("[0] Async", type(body[0]).__name__, "Async")
    failures += _check("[1] Reduce", type(body[1]).__name__, "Reduce")
    failures += _check("[2] Async", type(body[2]).__name__, "Async")
    return failures


def test_outside_cluster_untouched() -> int:
    """RawStore in a top-level For (no cluster around) is NOT wrapped."""
    print("test_outside_cluster_untouched")
    padded = _mk_buf("padded", [67], scope="fragment")
    fn = _wrap([
        ir.For(loop_var="k", extent=3, body=[
            ir.RawStore(
                dst=_ref(padded, [{"op": "add", "args": [64, "k"]}]),
                value="<opaque>",
            ),
        ]),
    ])
    out = async_run(fn)
    f = out.body[0]
    return (_check("For preserved", type(f).__name__, "For")
            + _check("body[0] still RawStore",
                     type(f.body[0]).__name__, "RawStore"))


def test_grid_body_not_wrapped() -> int:
    """Op directly inside a grid (no cluster wrapper) is not wrapped."""
    print("test_grid_body_not_wrapped — only CLUSTER body triggers wrapping")
    A = _mk_buf("A", [LANE, 64, 16])
    fn = _wrap([_grid([
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A),
               marker=ir.Marker.DMA, can_async=True),
    ])])
    out = async_run(fn)
    grid_body = out.body[0].body
    return (_check("body length", len(grid_body), 1)
            + _check("body[0] is Dma (not Async)",
                     type(grid_body[0]).__name__, "Dma"))


def test_buffer_refs_not_rewritten() -> int:
    """pass_4 only wraps async; it must NOT rewrite BufferRef indices.
    Buffer rank-vs-ref-rank mismatch (set up by pass_3 split) must
    persist past pass_4 — the view pass resolves it later."""
    print("test_buffer_refs_not_rewritten")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16], scope="shared")    # already grown
    # ref to Q_sh has the OLD rank (2D), mismatching its grown shape (3D)
    old_ref = _ref(Q_sh, [ir.Slice(), ir.Slice()])
    fn = _wrap([_grid([_cluster([
        ir.Dma(
            src=_ref(Q_hbm, [0, ir.Slice(), "by", ir.Slice()]),
            dst=old_ref,
            marker=ir.Marker.DMA, can_async=True,
        ),
    ])])])
    out = async_run(fn)
    dma = out.body[0].body[0].body[0].body[0]   # grid → cluster → async → dma
    failures = 0
    # HBM ref indices unchanged: still [0, Slice, "by", Slice]
    failures += _check("HBM[2] still 'by'", dma.src.indices[2], "by")
    failures += _check("HBM rank unchanged", len(dma.src.indices), 4)
    # On-chip ref indices unchanged: still 2D (mismatch with 3D buffer)
    failures += _check("Q_sh rank still 2 (mismatch persists)",
                       len(dma.dst.indices), 2)
    return failures


def test_inside_for_inside_cluster() -> int:
    """cluster → unroll For → cluster → ops (the post-distribute_cluster
    shape). Wrapping happens in the inner cluster body, not at the For
    level."""
    print("test_inside_for_inside_cluster")
    A = _mk_buf("A", [LANE, 64, 16])
    inner_cluster = _cluster([
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(A),
               marker=ir.Marker.DMA, can_async=True),
    ])
    fn = _wrap([_grid([
        ir.For(loop_var="kh", extent=4, kind="unroll", body=[inner_cluster]),
    ])])
    out = async_run(fn)
    grid = out.body[0]
    for_node = grid.body[0]
    inner = for_node.body[0]                # the cluster
    return (_check("For preserved", type(for_node).__name__, "For")
            + _check("inner cluster preserved",
                     type(inner).__name__, "ParallelAxis")
            + _check("dma wrapped in Async",
                     type(inner.body[0]).__name__, "Async"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_can_async_true_gets_wrapped()
    failures += test_can_async_false_not_wrapped()
    failures += test_strict_one_async_one_op()
    failures += test_mixed_async_and_non_async()
    failures += test_outside_cluster_untouched()
    failures += test_grid_body_not_wrapped()
    failures += test_buffer_refs_not_rewritten()
    failures += test_inside_for_inside_cluster()
    print()
    if failures == 0:
        print("PASS — all mid_ir.async_wrap tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
