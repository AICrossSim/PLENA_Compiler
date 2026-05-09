"""Tests for the graph-layer ``split_lane_groups`` pass.

Equivalent semantics to the legacy stmt-walker
``split_lane_groups``, but operating on a :class:`graph_ir.Graph`
post-``annotate_grid`` + ``annotate_sync``. A grid-binding ForRoot whose
extent > lane_count is split into ``outer × lane_count`` ForRoots.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes.lift_from_raw import (
    lift_from_raw_primfunc,
)
from tilelang_tvm_compiler.frontend.passes.graph_passes import (
    annotate_grid, annotate_sync as g_annotate_sync, split_lane_groups,
)
from tilelang_tvm_compiler.frontend.passes.graph_ir import (
    Graph, ForRoot, NestedForGroup, LaneGroup, NodeRoot,
    ATTR_GROUP_EXTENT, ATTR_IS_LANE_FOR,
)


def _collect_group_extents(graph: Graph):
    """Walk the graph; return all ATTR_GROUP_EXTENT values seen."""
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
    return sorted(found)


def _has_lane_for(graph: Graph) -> bool:
    """Check that some for in the graph carries ATTR_IS_LANE_FOR=True
    (the inner-of-pair after a split)."""
    found = False

    def visit_items(items):
        nonlocal found
        for it in items:
            if isinstance(it, NestedForGroup):
                if it.attrs.get(ATTR_IS_LANE_FOR):
                    found = True
                visit_items(it.items)

    def visit_root(root):
        nonlocal found
        if isinstance(root, ForRoot):
            if root.attrs.get(ATTR_IS_LANE_FOR):
                found = True
            visit_root(root.body)
            return
        if isinstance(root, (LaneGroup, NodeRoot)):
            visit_items(root.items)

    visit_root(graph.root)
    return found


def _kernel_extent_4_no_split():
    @T.prim_func
    def k(
        Q: T.Tensor((1, 64, 4, 16), "float16"),
        S: T.Tensor((1, 64, 4, 64), "float16"),
    ):
        with T.Kernel(1, 4, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            S_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(S_loc, S[0, 0, by, 0])
    return k


def _kernel_extent_8_splits():
    @T.prim_func
    def k(
        Q: T.Tensor((1, 64, 8, 16), "float16"),
        S: T.Tensor((1, 64, 8, 64), "float16"),
    ):
        with T.Kernel(1, 8, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            S_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(S_loc, S[0, 0, by, 0])
    return k


def _kernel_no_sync_no_split():
    @T.prim_func
    def k(C: T.Tensor((1, 64, 1, 64), "float16")):
        with T.Kernel(1, 8, threads=128) as (bx, by):
            C_loc = T.alloc_fragment((64, 64), "float16")
            T.clear(C_loc)
    return k


def _pipeline(kernel_factory, lane_count=4):
    g = lift_from_raw_primfunc(kernel_factory())
    g = annotate_grid.run(g)
    g = g_annotate_sync.run(g)
    return split_lane_groups.run(g, lane_count=lane_count)


def test_extent_matches_lane_count_unchanged():
    g = _pipeline(_kernel_extent_4_no_split)
    extents = _collect_group_extents(g)
    assert extents == [4]
    assert not _has_lane_for(g)


def test_extent_8_splits_into_2_and_4():
    g = _pipeline(_kernel_extent_8_splits)
    extents = _collect_group_extents(g)
    assert 8 not in extents
    assert 2 in extents
    assert 4 in extents
    assert _has_lane_for(g)


def test_no_sync_means_no_split():
    g = _pipeline(_kernel_no_sync_no_split)
    extents = _collect_group_extents(g)
    # No sync op inside means split doesn't fire.
    assert 8 in extents
    assert 2 not in extents


def test_idempotent_repeat_run():
    g = _pipeline(_kernel_extent_8_splits)
    once = _collect_group_extents(g)
    g_twice = split_lane_groups.run(g, lane_count=4)
    twice = _collect_group_extents(g_twice)
    assert once == twice


if __name__ == "__main__":
    test_extent_matches_lane_count_unchanged()
    test_extent_8_splits_into_2_and_4()
    test_no_sync_means_no_split()
    test_idempotent_repeat_run()
    print("graph split_lane_groups tests passed")
