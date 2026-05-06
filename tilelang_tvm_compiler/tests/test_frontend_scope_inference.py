"""Tests for the slim `scope_inference` pass.

The pass returns a `BufferScopeMap` (name -> scope string). It does not
modify the IR.
"""

from __future__ import annotations

import pytest

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.frontend.passes import scope_inference
from tilelang_tvm_compiler.frontend.passes.scope_inference import (
    ScopeInferenceError,
)


def _basic_kernel():
    """A @ B → C, all 64×64. A is shared.dyn (vram), B is shared.dyn (mram
    because it appears as gemm RHS), C is local.fragment (vram)."""
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


def _no_gemm_kernel():
    """No gemm — all shared buffers default to vram."""
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


def _fpram_kernel():
    """FP scalar scratch written in tilelang style via buffer indexing."""
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


def test_hbm_params_get_hbm_scope():
    func = _basic_kernel()
    scopes = scope_inference.infer(func)
    # Param names come from the @T.prim_func signature: A, B, C.
    assert scopes.get("A") == "hbm", scopes
    assert scopes.get("B") == "hbm", scopes
    assert scopes.get("C") == "hbm", scopes


def test_gemm_rhs_buffer_is_mram():
    func = _basic_kernel()
    scopes = scope_inference.infer(func)
    assert scopes.get("B_sh") == "mram", scopes


def test_gemm_lhs_buffer_is_vram():
    func = _basic_kernel()
    scopes = scope_inference.infer(func)
    assert scopes.get("A_sh") == "vram", scopes


def test_fragment_buffer_is_vram():
    func = _basic_kernel()
    scopes = scope_inference.infer(func)
    assert scopes.get("C_loc") == "vram", scopes


def test_shared_default_is_vram_when_no_gemm():
    func = _no_gemm_kernel()
    scopes = scope_inference.infer(func)
    assert scopes.get("A_sh") == "vram", scopes


def test_fp_scalar_fragment_is_fpram():
    func = _fpram_kernel()
    scopes = scope_inference.infer(func)
    assert scopes.get("M_INIT") == "fpram", scopes
    assert scopes.get("M_OLD") == "fpram", scopes


def test_unknown_scope_raises():
    """An alloc_buffer with a non-shared-non-fragment scope should raise."""
    from tvm import tir
    import tvm

    A_data = tir.Var("A", tvm.ir.PointerType(tvm.ir.PrimType("float16"), "weird.scope"))
    A_buf = tir.decl_buffer(shape=(64, 64), dtype="float16", name="A_weird",
                              data=A_data, scope="weird.scope")
    body = tir.Block(
        iter_vars=[], reads=[], writes=[], name_hint="root",
        body=tir.Evaluate(tir.IntImm("int32", 0)),
        alloc_buffers=[A_buf],
    )
    body = tir.BlockRealize(
        iter_values=[], predicate=tir.IntImm("bool", True), block=body,
    )
    func = tir.PrimFunc(params=[], body=body, ret_type=None, buffer_map={})
    with pytest.raises(ScopeInferenceError, match="unsupported declared scope"):
        scope_inference.infer(func)


if __name__ == "__main__":
    test_hbm_params_get_hbm_scope()
    test_gemm_rhs_buffer_is_mram()
    test_gemm_lhs_buffer_is_vram()
    test_fragment_buffer_is_vram()
    test_shared_default_is_vram_when_no_gemm()
    test_fp_scalar_fragment_is_fpram()
    test_unknown_scope_raises()
    print("scope_inference tests passed")
