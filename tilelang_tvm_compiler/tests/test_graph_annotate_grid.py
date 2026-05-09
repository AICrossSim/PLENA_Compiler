"""Tests for the graph-layer ``annotate_grid`` pass.

Equivalent semantics to the legacy stmt-walker ``annotate_group``, but
operating on a :class:`graph_ir.Graph` produced by ``lift_from_raw``.
The graph pass sets ``ATTR_GROUP_EXTENT`` on ForRoots (from blockIdx > 1
grid bindings) and on NestedForGroups derived from ``T.Parallel``
loops, and rewrites PARALLEL kind to SERIAL.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes.lift_from_raw import (
    lift_from_raw_primfunc,
)
from tilelang_tvm_compiler.frontend.passes.graph_passes import annotate_grid
from tilelang_tvm_compiler.frontend.passes.graph_ir import (
    Graph, ForRoot, NestedForGroup, LaneGroup, NodeRoot,
    ATTR_GROUP_EXTENT,
)


def _collect_extents(graph: Graph):
    """Walk a graph, collect every ATTR_GROUP_EXTENT seen on ForRoots /
    NestedForGroups."""
    found = []

    def visit_items(items):
        for it in items:
            if isinstance(it, NestedForGroup):
                if ATTR_GROUP_EXTENT in it.attrs:
                    found.append(it.attrs[ATTR_GROUP_EXTENT])
                visit_items(it.items)

    def visit_root(root):
        if isinstance(root, ForRoot):
            if ATTR_GROUP_EXTENT in root.attrs:
                found.append(root.attrs[ATTR_GROUP_EXTENT])
            visit_root(root.body)
            return
        if isinstance(root, (LaneGroup, NodeRoot)):
            visit_items(root.items)

    visit_root(graph.root)
    return found


def _has_parallel(graph: Graph) -> bool:
    """Any NestedForGroup with PARALLEL kind anywhere?"""

    def visit_items(items):
        for it in items:
            if isinstance(it, NestedForGroup):
                if it.kind == tir.ForKind.PARALLEL:
                    return True
                if visit_items(it.items):
                    return True
        return False

    def visit_root(root):
        if isinstance(root, ForRoot):
            return visit_root(root.body)
        if isinstance(root, (LaneGroup, NodeRoot)):
            return visit_items(root.items)
        return False

    return visit_root(graph.root)


# ---------------------------------------------------------------------------
# Test kernels (same shapes as test_frontend_annotate_group)
# ---------------------------------------------------------------------------

def _make_single_block_kernel():
    @T.prim_func
    def k(
        Q: T.Tensor((1, 64, 4, 16), "float16"),
        K: T.Tensor((1, 64, 4, 16), "float16"),
        S: T.Tensor((1, 64, 4, 64), "float16"),
    ):
        with T.Kernel(1, 4, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            K_sh = T.alloc_shared((64, 16), "float16")
            S_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(K[0, 0, by, 0], K_sh)
            T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)
            T.copy(S_loc, S[0, 0, by, 0])
    return k


def _make_extent_one_kernel():
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64, 64), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(A_sh, C[0, 0, 0, 0])
    return k


def _make_two_block_axes_kernel():
    @T.prim_func
    def k(
        Q: T.Tensor((2, 64, 4, 16), "float16"),
        S: T.Tensor((2, 64, 4, 64), "float16"),
    ):
        with T.Kernel(2, 4, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            S_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(Q[bx, 0, by, 0], Q_sh)
            T.copy(S_loc, S[bx, 0, by, 0])
    return k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_head_axis_becomes_group_with_extent_4():
    g = lift_from_raw_primfunc(_make_single_block_kernel())
    g = annotate_grid.run(g)
    # by=4 grid binding → one ForRoot with ATTR_GROUP_EXTENT=4.
    # bx=1 dropped at lift; threadIdx.* dropped at lift.
    assert sorted(_collect_extents(g)) == [4]


def test_extent_one_grid_drops_to_no_group():
    g = lift_from_raw_primfunc(_make_extent_one_kernel())
    g = annotate_grid.run(g)
    assert _collect_extents(g) == []


def test_two_block_axes_two_groups():
    g = lift_from_raw_primfunc(_make_two_block_axes_kernel())
    g = annotate_grid.run(g)
    assert sorted(_collect_extents(g)) == [2, 4]


def test_no_parallel_for_remains():
    g = lift_from_raw_primfunc(_make_single_block_kernel())
    g = annotate_grid.run(g)
    assert not _has_parallel(g)


def test_idempotent():
    g = lift_from_raw_primfunc(_make_single_block_kernel())
    once = annotate_grid.run(g)
    twice = annotate_grid.run(once)
    assert sorted(_collect_extents(once)) == sorted(_collect_extents(twice))


if __name__ == "__main__":
    test_head_axis_becomes_group_with_extent_4()
    test_extent_one_grid_drops_to_no_group()
    test_two_block_axes_two_groups()
    test_no_parallel_for_remains()
    test_idempotent()
    print("graph annotate_grid tests passed")
