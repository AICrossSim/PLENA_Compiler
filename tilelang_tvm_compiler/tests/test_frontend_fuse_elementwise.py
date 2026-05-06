"""Tests for `fuse_elementwise`.

Target pattern::

    for i in T.Parallel(N):
        C[i] = A[i] + B[i]

After ``annotate_group`` it becomes a ``for + plena.group(N)`` wrapping
a single elementwise BufferStore. ``fuse_elementwise`` should collapse
the entire for-loop to a single ``plena.v_add`` extern call.
"""

from __future__ import annotations

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import (
    annotate_gemm_kind, annotate_group, annotate_sync, fuse_elementwise,
)


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


def _has_extern_call(func, name: str) -> bool:
    for s in _walk_collect(
        func.body,
        lambda s: isinstance(s, tir.Evaluate) and isinstance(s.value, tir.Call),
    ):
        call = s.value
        if (call.op.name == "tir.call_extern"
                and isinstance(call.args[0], tir.StringImm)
                and call.args[0].value == name):
            return True
    return False


def _count_elementwise_for(func) -> int:
    """Number of `tir.For` statements whose body is a BufferStore (i.e.
    surviving elementwise loops that didn't get fused)."""

    def predicate(s):
        if not isinstance(s, tir.For):
            return False
        body = s.body
        # Strip an optional plena.group wrapper.
        if isinstance(body, tir.AttrStmt) and body.attr_key == "plena.group":
            body = body.body
        return isinstance(body, tir.BufferStore)

    return len(_walk_collect(func.body, predicate))


def _run(kernel_factory):
    func = kernel_factory()
    func = annotate_gemm_kind.run(func)
    func = annotate_group.run(func)
    func = annotate_sync.run(func)
    return fuse_elementwise.run(func)


def _add_kernel():
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64,), "float16")
            B_sh = T.alloc_shared((64,), "float16")
            C_sh = T.alloc_shared((64,), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            for i in T.Parallel(64):
                C_sh[i] = A_sh[i] + B_sh[i]
            T.copy(C_sh, C[0, 0, 0, 0])
    return k


def _no_parallel_kernel():
    """Same kernel without T.Parallel — uses T.serial. Should NOT be fused."""
    @T.prim_func
    def k(
        A: T.Tensor((1, 64, 1, 64), "float16"),
        B: T.Tensor((1, 64, 1, 64), "float16"),
        C: T.Tensor((1, 64, 1, 64), "float16"),
    ):
        with T.Kernel(1, threads=128) as bx:
            A_sh = T.alloc_shared((64,), "float16")
            B_sh = T.alloc_shared((64,), "float16")
            C_sh = T.alloc_shared((64,), "float16")
            T.copy(A[0, 0, 0, 0], A_sh)
            T.copy(B[0, 0, 0, 0], B_sh)
            for i in T.serial(64):
                C_sh[i] = A_sh[i] + B_sh[i]
            T.copy(C_sh, C[0, 0, 0, 0])
    return k


def test_parallel_add_fuses_to_v_add():
    func = _run(_add_kernel)
    assert _has_extern_call(func, "plena.v_add"), func.script()
    # The original for-loop must be gone (replaced by Evaluate(call_extern)).
    assert _count_elementwise_for(func) == 0, func.script()


def test_serial_loop_is_not_fused():
    """Serial for-loop bodies don't get fused (no plena.group wrapper)."""
    func = _run(_no_parallel_kernel)
    assert not _has_extern_call(func, "plena.v_add"), func.script()
    # The serial for-loop with elementwise body should still be present.
    assert _count_elementwise_for(func) >= 1, func.script()


if __name__ == "__main__":
    test_parallel_add_fuses_to_v_add()
    test_serial_loop_is_not_fused()
    print("fuse_elementwise tests passed")
