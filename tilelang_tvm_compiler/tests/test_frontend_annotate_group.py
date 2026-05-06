"""Tests for the `annotate_group` pass.

The pass converts tilelang grid bindings (blockIdx.* / threadIdx.*) and
parallel for-loops into PLENA *groups* — serial for-loops wrapped in a
``T.attr(0, "plena.group", extent)`` AttrStmt.
"""

from __future__ import annotations

import pytest
from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import annotate_group
from tilelang_tvm_compiler.frontend.passes.annotate_group import (
    GROUP_KEY, GroupAnnotateError,
)


# ---------------------------------------------------------------------------
# Invariant predicates
# ---------------------------------------------------------------------------

def _walk_collect(func: tir.PrimFunc, predicate):
    """Collect every Stmt for which `predicate(stmt)` returns True."""
    found = []

    def visit(s):
        if predicate(s):
            found.append(s)
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
        elif isinstance(s, tir.BlockRealize):
            visit(s.block)
        elif isinstance(s, tir.Block):
            visit(s.body)
            if s.init is not None:
                visit(s.init)
        elif isinstance(s, (tir.AttrStmt, tir.For, tir.LetStmt)):
            visit(s.body)
        elif isinstance(s, tir.IfThenElse):
            visit(s.then_case)
            if s.else_case is not None:
                visit(s.else_case)

    visit(func.body)
    return found


def _has_thread_extent(func) -> bool:
    return bool(_walk_collect(
        func,
        lambda s: isinstance(s, tir.AttrStmt) and s.attr_key == "thread_extent",
    ))


def _has_parallel_for(func) -> bool:
    return bool(_walk_collect(
        func,
        lambda s: isinstance(s, tir.For) and s.kind == tir.ForKind.PARALLEL,
    ))


def _group_attrs(func):
    return _walk_collect(
        func,
        lambda s: isinstance(s, tir.AttrStmt) and s.attr_key == GROUP_KEY,
    )


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------

def _make_single_block_kernel():
    """T.Kernel(1, 4) — bx is degenerate (extent=1, dropped), by is a group."""
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
    """T.Kernel(1) — single bx with extent 1 must be dropped entirely."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_thread_extent_attr_is_gone():
    func = annotate_group.run(_make_single_block_kernel())
    assert not _has_thread_extent(func), func.script()


def test_parallel_for_kind_is_gone():
    func = annotate_group.run(_make_single_block_kernel())
    assert not _has_parallel_for(func), func.script()


def test_head_axis_becomes_group_with_extent_4():
    func = annotate_group.run(_make_single_block_kernel())
    groups = _group_attrs(func)
    extents = sorted(int(g.value.value) for g in groups)
    # by=4 -> one group. threadIdx.* are unconditionally dropped on PLENA
    # (single-thread HW, no parallel meaning).
    assert extents == [4], extents


def test_each_group_attr_is_wrapped_by_matching_for():
    """Every plena.group AttrStmt is the body of a serial For with the
    same extent — that's how iterations of the group are scheduled."""
    func = annotate_group.run(_make_single_block_kernel())
    pairs = []  # list of (For, group_extent)

    def visit(s):
        if isinstance(s, tir.For) and isinstance(s.body, tir.AttrStmt) \
                and s.body.attr_key == GROUP_KEY:
            pairs.append((s, int(s.body.value.value)))
        if isinstance(s, tir.SeqStmt):
            for c in s.seq:
                visit(c)
        elif isinstance(s, tir.BlockRealize):
            visit(s.block)
        elif isinstance(s, tir.Block):
            visit(s.body)
        elif isinstance(s, (tir.AttrStmt, tir.For, tir.LetStmt)):
            visit(s.body)

    visit(func.body)
    assert pairs, f"no group-wrapping For found:\n{func.script()}"
    for for_stmt, group_extent in pairs:
        assert isinstance(for_stmt.extent, tir.IntImm), for_stmt
        assert int(for_stmt.extent.value) == group_extent
        assert for_stmt.kind == tir.ForKind.SERIAL


def test_extent_one_grid_drops_to_no_group():
    func = annotate_group.run(_make_extent_one_kernel())
    # bx=1 (degenerate) drops; threadIdx.* are unconditionally dropped.
    # No groups should remain.
    extents = sorted(int(g.value.value) for g in _group_attrs(func))
    assert extents == [], extents
    assert not _has_thread_extent(func)


def _make_two_block_axes_kernel():
    """T.Kernel(2, 4) — two block axes both extent>1; expect two nested groups."""
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


def test_nested_groups_for_two_block_axes():
    """Two extent>1 block axes -> two nested plena.group AttrStmts in
    distinct For wrappers."""
    func = annotate_group.run(_make_two_block_axes_kernel())
    extents = sorted(int(g.value.value) for g in _group_attrs(func))
    # Expected: bx=2, by=4 (the two extent>1 block axes). threadIdx.x=128
    # drops on PLENA.
    assert extents == [2, 4], extents
    assert not _has_thread_extent(func)
    assert not _has_parallel_for(func)


def test_repeat_run_is_idempotent():
    """Running annotate_group twice should be a no-op the second time
    (no thread_extent / parallel left to convert)."""
    once = annotate_group.run(_make_single_block_kernel())
    twice = annotate_group.run(once)
    assert _group_attrs(once) and _group_attrs(twice)
    assert not _has_thread_extent(twice)
    assert not _has_parallel_for(twice)


if __name__ == "__main__":
    test_thread_extent_attr_is_gone()
    test_parallel_for_kind_is_gone()
    test_head_axis_becomes_group_with_extent_4()
    test_each_group_attr_is_wrapped_by_matching_for()
    test_nested_groups_for_two_block_axes()
    test_extent_one_grid_drops_to_no_group()
    test_repeat_run_is_idempotent()
    print("annotate_group tests passed")
