"""Tests for the `annotate_sync` pass.

The pass wraps DMA copies and `kind=btmm` gemms in
``T.attr(0, "plena.sync", 1)`` AttrStmts. Other ops are left alone.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import (
    annotate_gemm_kind, annotate_sync,
)
from tilelang_tvm_compiler.frontend.passes.annotate_sync import SYNC_KEY


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


def _sync_attrs(func):
    return _walk_collect(
        func,
        lambda s: isinstance(s, tir.AttrStmt) and s.attr_key == SYNC_KEY,
    )


def _sync_wraps_op(func, op_name):
    """True iff there is at least one plena.sync AttrStmt whose body is
    an Evaluate(Call(<op_name>))."""
    for attr in _sync_attrs(func):
        body = attr.body
        if isinstance(body, tir.Evaluate) and isinstance(body.value, tir.Call):
            if body.value.op.name == op_name:
                return True
    return False


def _evaluate_calls(func, op_name):
    return [
        s for s in _walk_collect(
            func,
            lambda s: isinstance(s, tir.Evaluate)
                      and isinstance(s.value, tir.Call)
                      and s.value.op.name == op_name,
        )
    ]


# ---------------------------------------------------------------------------
# Test kernels
# ---------------------------------------------------------------------------

def _make_dma_only_kernel():
    """Two HBM↔shared copies, no gemm. Both copies should get sync."""
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


def _make_btmm_kernel():
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


def _make_overwrite_only_kernel():
    """gemm without kind (defaults to overwrite). Should NOT get sync."""
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64, 64), "float16")
            B_sh = T.alloc_shared((64, 64), "float16")
            C_loc = T.alloc_fragment((64, 64), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            T.gemm(A_sh, B_sh, C_loc)
            T.copy(C_loc, C[0, 0, 0, 0])
    return k


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run(func):
    """Run annotate_gemm_kind first (sync needs the kind annotation)."""
    func = annotate_gemm_kind.run(func)
    return annotate_sync.run(func)


def test_dma_copies_get_sync():
    func = _run(_make_dma_only_kernel())
    syncs = _sync_attrs(func)
    # Two HBM↔shared copies → two syncs.
    assert len(syncs) == 2, f"expected 2 sync wrappers, got {len(syncs)}\n{func.script()}"
    assert _sync_wraps_op(func, "tl.tileop.copy")


def test_btmm_gemm_gets_sync():
    func = _run(_make_btmm_kernel())
    syncs = _sync_attrs(func)
    # 3 syncs: Q DMA, K DMA, BTMM gemm. The S DMA also -> 4 total.
    assert len(syncs) == 4, f"expected 4 syncs (3 DMAs + btmm), got {len(syncs)}\n{func.script()}"
    assert _sync_wraps_op(func, "tl.tileop.gemm_py")


def test_overwrite_gemm_does_not_get_sync():
    func = _run(_make_overwrite_only_kernel())
    syncs = _sync_attrs(func)
    # 3 DMAs (A in, B in, C out) — the gemm (default kind=overwrite)
    # should NOT be wrapped.
    assert len(syncs) == 3, f"expected 3 syncs (DMAs only), got {len(syncs)}\n{func.script()}"
    for attr in syncs:
        body = attr.body
        if isinstance(body, tir.Evaluate) and isinstance(body.value, tir.Call):
            assert body.value.op.name == "tl.tileop.copy", body.value.op.name


def test_no_double_wrap_on_repeat_run():
    """Running annotate_sync twice should be a no-op the second time —
    sync wrappers are idempotent."""
    once = _run(_make_btmm_kernel())
    twice = annotate_sync.run(once)
    n_once = len(_sync_attrs(once))
    n_twice = len(_sync_attrs(twice))
    assert n_once == n_twice, (
        f"sync count changed on repeat run: {n_once} -> {n_twice}\n"
        f"once:\n{once.script()}\ntwice:\n{twice.script()}"
    )


if __name__ == "__main__":
    test_dma_copies_get_sync()
    test_btmm_gemm_gets_sync()
    test_overwrite_gemm_does_not_get_sync()
    test_no_double_wrap_on_repeat_run()
    print("annotate_sync tests passed")
