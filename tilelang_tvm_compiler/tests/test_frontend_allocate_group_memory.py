"""Tests for `allocate_group_memory` — role-based two-mode expansion.

Rules under test:
  * BTMM gemm inputs (arg 0/1) get last-dim * lane_count (col-pack).
  * BTMM gemm output (arg 2) gets first-dim * lane_count (row-stack).
  * DMA local-side inside a lane group gets last-dim * lane_count
    (col-pack).
  * Matmul (kind=overwrite) operands are NEUTRAL — they neither trigger
    nor prevent expansion. A matmul-only buffer outside any lane group
    is unchanged; a matmul operand also touched by a DMA in a lane
    group still gets expanded by the DMA rule.
  * Buffers outside any lane group are unchanged.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import (
    annotate_gemm_kind, annotate_group, annotate_sync, split_lane_groups,
    scope_inference, allocate_group_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _walk_collect(stmt, predicate):
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

    visit(stmt)
    return found


def _alloc_buffers(func: tir.PrimFunc):
    blocks = _walk_collect(func.body, lambda s: isinstance(s, tir.Block))
    out = []
    for b in blocks:
        out.extend(b.alloc_buffers)
    return out


def _alloc_by_name(func: tir.PrimFunc, name: str):
    for buf in _alloc_buffers(func):
        if buf.name == name:
            return buf
    return None


def _run(kernel_factory, lane_count=4):
    func = kernel_factory()
    func = annotate_gemm_kind.run(func)
    func = annotate_group.run(func)
    func = annotate_sync.run(func)
    func = split_lane_groups.run(func, lane_count=lane_count)
    scopes = scope_inference.infer(func)
    return allocate_group_memory.run(func, scopes, lane_count=lane_count)


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------

def _btmm_kernel():
    """T.Kernel(1, 4) — by is the lane var. Q_sh, K_sh are btmm inputs;
    S_loc is btmm output."""
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
            with T.attr(0, "plena.gemm_kind", "btmm"):
                T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)
            T.copy(S_loc, S[0, 0, by, 0])
    return k


def _matmul_in_lane_group_kernel():
    """T.Kernel(1, 4) but the gemm is regular matmul (kind=overwrite).
    Despite being inside the by lane group, matmul operands should NOT
    expand."""
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 4, 64), "float16"),
    ):
        with T.Kernel(1, 4, threads=128) as (bx, by):
            A_sh = T.alloc_shared((64, 64), "float16")
            B_sh = T.alloc_shared((64, 64), "float16")
            C_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            T.gemm(A_sh, B_sh, C_loc)  # default kind=overwrite
            T.copy(C_loc, C[0, 0, by, 0])
    return k


def _no_lane_group_kernel():
    """T.Kernel(1) — no head axis at all. Nothing should expand."""
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


def _fpram_lane_kernel():
    """Per-lane FP scratch buffers should gain an implicit lane dim."""
    @T.prim_func
    def k():
        with T.Kernel(1, 4, threads=128) as (bx, by):
            M_INIT = T.alloc_fragment((64,), "float16")
            M_OLD = T.alloc_fragment((64,), "float16")
            for row in T.serial(64):
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_INIT[row], M_OLD[row],
                ))
    return k


def _fpram_split_head_kernel():
    """Logical head_count=8 splits into outer×hardware-lane. FPRAM follows
    the nearest hardware lane group, not the full logical head_count."""
    @T.prim_func
    def k(
        Q: T.Tensor((1, 64, 8, 16), "float16"),
        K: T.Tensor((1, 64, 8, 16), "float16"),
    ):
        with T.Kernel(1, 8, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            K_sh = T.alloc_shared((64, 16), "float16")
            S_loc = T.alloc_fragment((64, 64), "float16")
            M_INIT = T.alloc_fragment((64,), "float16")
            M_OLD = T.alloc_fragment((64,), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(K[0, 0, by, 0], K_sh)
            with T.attr(0, "plena.gemm_kind", "btmm"):
                T.gemm(Q_sh, K_sh, S_loc, transpose_B=True)
            for row in T.serial(64):
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_INIT[row], M_OLD[row],
                ))
    return k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_btmm_inputs_expand_to_4d_BSHD_packed():
    """BTMM inputs (per-lane (rows, hlen)) → 4D (1, rows, lane_count, hlen)
    BSHD-packed-narrow."""
    func = _run(_btmm_kernel, lane_count=4)
    Q_sh = _alloc_by_name(func, "Q_sh")
    K_sh = _alloc_by_name(func, "K_sh")
    assert Q_sh is not None and K_sh is not None
    assert tuple(int(s) for s in Q_sh.shape) == (1, 64, 4, 16), Q_sh.shape
    assert tuple(int(s) for s in K_sh.shape) == (1, 64, 4, 16), K_sh.shape


def test_btmm_output_expands_to_4d_BHSD_stacked():
    """S_loc is the btmm gemm dst → 4D (1, lane_count, rows, mlen)
    BHSD-stacked."""
    func = _run(_btmm_kernel, lane_count=4)
    S_loc = _alloc_by_name(func, "S_loc")
    assert S_loc is not None
    assert tuple(int(s) for s in S_loc.shape) == (1, 4, 64, 64), S_loc.shape


def test_matmul_neutral_dma_still_expands():
    """Matmul operands inside a lane group: matmul itself is neutral, but
    the DMA copies inside the same lane group still expand the buffers
    (col-pack to 4D BSHD-packed)."""
    func = _run(_matmul_in_lane_group_kernel, lane_count=4)
    for name in ("A_sh", "B_sh", "C_loc"):
        buf = _alloc_by_name(func, name)
        assert buf is not None, name
        # The user-declared shape was (64, 64); after col-pack expansion
        # to 4D it becomes (1, 64, 4, 64).
        assert tuple(int(s) for s in buf.shape) == (1, 64, 4, 64), \
            f"{name} expected (1, 64, 4, 64), got {buf.shape}"


def test_no_lane_group_means_no_expansion():
    func = _run(_no_lane_group_kernel, lane_count=4)
    A_sh = _alloc_by_name(func, "A_sh")
    assert A_sh is not None
    assert tuple(int(s) for s in A_sh.shape) == (64, 64), A_sh.shape


def test_fpram_fragments_expand_to_lane_stacked_2d():
    func = _run(_fpram_lane_kernel, lane_count=4)
    M_INIT = _alloc_by_name(func, "M_INIT")
    M_OLD = _alloc_by_name(func, "M_OLD")
    assert M_INIT is not None and M_OLD is not None
    assert tuple(int(s) for s in M_INIT.shape) == (4, 64), M_INIT.shape
    assert tuple(int(s) for s in M_OLD.shape) == (4, 64), M_OLD.shape


def test_fpram_follows_hardware_lane_domain_not_logical_head_count():
    func = _run(_fpram_split_head_kernel, lane_count=4)
    Q_sh = _alloc_by_name(func, "Q_sh")
    M_INIT = _alloc_by_name(func, "M_INIT")
    assert Q_sh is not None and M_INIT is not None
    assert tuple(int(s) for s in Q_sh.shape) == (1, 64, 4, 16), Q_sh.shape
    assert tuple(int(s) for s in M_INIT.shape) == (4, 64), M_INIT.shape


if __name__ == "__main__":
    test_btmm_inputs_expand_to_4d_BSHD_packed()
    test_btmm_output_expands_to_4d_BHSD_stacked()
    test_matmul_neutral_dma_still_expands()
    test_no_lane_group_means_no_expansion()
    test_fpram_fragments_expand_to_lane_stacked_2d()
    test_fpram_follows_hardware_lane_domain_not_logical_head_count()
    print("allocate_group_memory tests passed")
