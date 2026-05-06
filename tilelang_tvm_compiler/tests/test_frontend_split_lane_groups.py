"""Tests for the `split_lane_groups` pass.

The pass takes a group axis ``for v in range(N): plena.group(N)`` whose
body contains a ``plena.sync`` op referencing ``v``, and (when
``N > lane_count`` and ``N % lane_count == 0``) splits it into nested
``for v_outer × for v_inner`` with ``v -> v_outer * lane_count + v_inner``.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import (
    annotate_gemm_kind, annotate_group, annotate_sync, split_lane_groups,
)
from tilelang_tvm_compiler.frontend.passes.annotate_group import GROUP_KEY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_collect(func: tir.PrimFunc, predicate):
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


def _group_extents(func):
    return sorted(
        int(g.value.value) for g in _walk_collect(
            func, lambda s: isinstance(s, tir.AttrStmt) and s.attr_key == GROUP_KEY,
        )
    )


def _for_extents(func):
    return sorted(
        int(s.extent.value) for s in _walk_collect(
            func,
            lambda s: isinstance(s, tir.For) and isinstance(s.extent, tir.IntImm),
        )
    )


# ---------------------------------------------------------------------------
# Run helper: full pre-stack so the input matches what split_lane_groups
# would actually see in the pipeline.
# ---------------------------------------------------------------------------

def _run(kernel_factory, lane_count=4):
    func = kernel_factory()
    func = annotate_gemm_kind.run(func)
    func = annotate_group.run(func)
    func = annotate_sync.run(func)
    return split_lane_groups.run(func, lane_count=lane_count)


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------

def _kernel_extent_4_no_split():
    """T.Kernel(1, 4) — head axis already matches lane_count=4. No split."""
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
    """T.Kernel(1, 8) with lane_count=4 — head axis splits 8 -> 2*4."""
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
    """No DMA, no btmm — no sync ops -> no split even if extent > lane_count."""
    @T.prim_func
    def k(C: T.Tensor((1, 64, 1, 64), "float16")):
        with T.Kernel(1, 8, threads=128) as (bx, by):
            C_loc = T.alloc_fragment((64, 64), "float16")
            T.clear(C_loc)
    return k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extent_matches_lane_count_unchanged():
    """When the group extent already equals lane_count, no split happens.
    The group attr stays at extent 4."""
    func = _run(_kernel_extent_4_no_split, lane_count=4)
    extents = _group_extents(func)
    # by=4 -> one group of extent 4. threadIdx is dropped on PLENA.
    assert extents == [4], extents


def test_extent_8_splits_into_2_and_4():
    """With lane_count=4, an 8-extent head group splits into 2 (outer)
    and 4 (inner)."""
    func = _run(_kernel_extent_8_splits, lane_count=4)
    extents = _group_extents(func)
    # After split: by_outer=2 group, by_inner=4 group, plus tx=128.
    assert 2 in extents, extents
    assert 4 in extents, extents
    # And the original 8 should be GONE.
    assert 8 not in extents, extents
    # New for-loop pair appears: extents 2 and 4 are added.
    for_extents = _for_extents(func)
    assert 2 in for_extents and 4 in for_extents, for_extents
    # The original 8-extent for is gone.
    assert 8 not in for_extents, for_extents


def test_no_sync_means_no_split():
    """An 8-extent group with no sync op inside is left alone — split is
    sync-driven, not blanket."""
    func = _run(_kernel_no_sync_no_split, lane_count=4)
    extents = _group_extents(func)
    # 8 should still be present; 2 and 4 should NOT have appeared from a split.
    assert extents == [8], extents


def test_idempotent_repeat_run():
    """Running split_lane_groups twice doesn't keep splitting (after one
    pass extents are already lane_count or smaller)."""
    func = _run(_kernel_extent_8_splits, lane_count=4)
    once = _group_extents(func)
    twice_func = split_lane_groups.run(func, lane_count=4)
    twice = _group_extents(twice_func)
    assert once == twice, f"split_lane_groups not idempotent: {once} -> {twice}"


if __name__ == "__main__":
    test_extent_matches_lane_count_unchanged()
    test_extent_8_splits_into_2_and_4()
    test_no_sync_means_no_split()
    test_idempotent_repeat_run()
    print("split_lane_groups tests passed")
