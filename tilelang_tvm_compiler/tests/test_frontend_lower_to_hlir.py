"""End-to-end tests for the new frontend pipeline through `lower_to_hlir`.

The pipeline runs every pass and the resulting TIR is fed into
`PlenaCodegen` and the back-end ISA emitter — exercising the whole
tilelang → HLIR → ISA path.
"""

from __future__ import annotations

import re

from tvm import tir

import tilelang_tvm_compiler  # bootstrap TVM 0.23
import tilelang.language as T

from tilelang_tvm_compiler.address_alloc import FPRAM_USER_BASE
from tilelang_tvm_compiler.frontend import compile_func, compile_to_tir_text
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget


# ---------------------------------------------------------------------------
# Reference kernels
# ---------------------------------------------------------------------------

def _mm64_kernel():
    """Single 64×64 matmul (kind defaults to overwrite)."""
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


def _qk_btmm_kernel():
    """Per-head Q @ K^T with lane fusion via T.Kernel(1, lane_count=4)."""
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


def _vector_add_kernel():
    """T.Parallel(64) elementwise add → plena.v_add."""
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


def _fpram_buffer_kernel():
    """Per-lane FP scratch written as 1D fragment buffer indexing."""
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


def _lane_loop_fusion_kernel():
    """A pure per-lane row loop followed by per-lane matmul should share
    one by loop after lane segmentation."""
    @T.prim_func
    def k():
        with T.Kernel(1, 4, threads=128) as (bx, by):
            S_loc = T.alloc_fragment((64, 64), "float16")
            V_sh = T.alloc_shared((64, 16), "float16")
            PV_loc = T.alloc_fragment((64, 16), "float16")
            M_INIT = T.alloc_fragment((64,), "float16")
            M_OLD = T.alloc_fragment((64,), "float16")
            for row in T.serial(64):
                T.evaluate(T.call_extern(
                    "handle", "plena.fp_copy_at",
                    M_INIT[row], M_OLD[row],
                ))
            T.evaluate(T.call_extern(
                "handle", "plena.matmul",
                S_loc.data, V_sh.data, PV_loc.data,
                1, 1, 16,
                by * 64 * 64,
                by * 16,
                by * 16,
                64,
            ))
    return k


def _fpram_elementwise_kernel():
    """Element-level FP buffer assignments lower to scalar FPRAM ops."""
    @T.prim_func
    def k():
        with T.Kernel(1, 4, threads=128) as (bx, by):
            A = T.alloc_fragment((64,), "float16")
            B = T.alloc_fragment((64,), "float16")
            C = T.alloc_fragment((64,), "float16")
            D = T.alloc_fragment((64,), "float16")
            E = T.alloc_fragment((64,), "float16")
            F = T.alloc_fragment((64,), "float16")
            for row in T.serial(64):
                B[row] = A[row]
                C[row] = A[row] - B[row]
                D[row] = C[row] + B[row]
                E[row] = D[row] * A[row]
                F[row] = T.exp(E[row])
                A[row] = 1.0 / F[row]
    return k


def _row_parallel_reduce_kernel():
    """Narrow row-wise DSL patterns lower to PLENA row ops."""
    @T.prim_func
    def k(
        Q: T.Tensor((1, 64, 4, 16), "float16"),
        K: T.Tensor((1, 64, 4, 16), "float16"),
    ):
        with T.Kernel(1, 4, threads=128) as (bx, by):
            Q_sh = T.alloc_shared((64, 16), "float16")
            K_sh = T.alloc_shared((64, 16), "float16")
            S = T.alloc_fragment((64, 64), "float16")
            M = T.alloc_fragment((64,), "float16")
            T.copy(Q[0, 0, by, 0], Q_sh)
            T.copy(K[0, 0, by, 0], K_sh)
            with T.attr(0, "plena.gemm_kind", "btmm"):
                T.gemm(Q_sh, K_sh, S, transpose_B=True)
            for row in T.serial(64):
                for col in T.Parallel(64):
                    S[row, col] = S[row, col] - M[row]
                for col in T.Parallel(64):
                    S[row, col] = T.exp(S[row, col])
                for col in T.Parallel(64):
                    S[row, col] = S[row, col] * M[row]
            T.reduce_max(S, M, dim=1, clear=False)
            T.reduce_sum(S, M, dim=1, clear=False)
    return k


# ---------------------------------------------------------------------------
# TIR-text checks (cheap, run for every kernel)
# ---------------------------------------------------------------------------

def _tir_text(kernel_factory, name="k"):
    return compile_to_tir_text(kernel_factory(), name=name)


def test_mm64_emits_dma_and_matmul():
    text = _tir_text(_mm64_kernel, "mm64")
    assert 'scope="vram"' in text
    assert 'scope="mram"' in text
    assert "plena.dma_h2v_slice" in text
    assert "plena.dma_h2m_slice" in text
    assert "plena.matmul" in text
    assert "plena.dma_v2h_slice" in text
    assert "tl.tileop" not in text   # nothing tilelang-specific left


def test_mm64_drops_threadidx_and_annotations():
    text = _tir_text(_mm64_kernel, "mm64")
    # No surviving thread loops or PLENA-internal annotations.
    assert "blockIdx" not in text
    assert "threadIdx" not in text
    assert "plena.gemm_kind" not in text
    assert "plena.group" not in text
    assert "plena.sync" not in text
    # Only one matmul call, no redundant outer for-loops.
    assert text.count("plena.matmul") == 1, text


def test_btmm_kernel_drops_lane_for_loop():
    text = _tir_text(_qk_btmm_kernel, "qk_btmm")
    # The `for by in range(4)` should be GONE — all sync ops collapsed
    # into one multi-lane HW op each.
    assert "for by" not in text and "for by_o" not in text, text
    assert "plena.btmm" in text
    # Lane-fused DMA: H position extent = 4 (lane_count).
    # plena.dma_h2v_slice has args:
    #   src.data, dst.data, ndim=4, *starts(4), *extents(4)
    # The 4th extent (last) is the D extent. The 3rd extent (H position) is 4.
    assert re.search(r"plena\.dma_h2v_slice.*?, 4, 0, 0, 0, 0, 1, 64, 4, 16", text), text


def test_btmm_kernel_emits_btmm_call_with_lane_count():
    text = _tir_text(_qk_btmm_kernel, "qk_btmm")
    assert re.search(r"plena\.btmm.*?, 4\)", text), text


def test_vector_add_collapses_to_v_add():
    text = _tir_text(_vector_add_kernel, "vec_add")
    # Parallel for-loop fused away.
    assert "T.Parallel" not in text
    assert "for i" not in text
    assert "plena.v_add" in text


def test_fpram_buffers_get_scope_and_lane_indexing():
    text = _tir_text(_fpram_buffer_kernel, "fpram_buf")
    assert 'scope="fpram"' in text
    assert "plena.fp_copy_at" in text
    assert re.search(r"M_INIT\[by(_\d+)?, row\]", text), text
    assert re.search(r"M_OLD\[by(_\d+)?, row\]", text), text


def test_pure_lane_row_loop_stays_inside_by_run_before_matmul():
    text = _tir_text(_lane_loop_fusion_kernel, "lane_loop_fusion")
    by_pos = text.find("for by")
    row_pos = text.find("for row")
    matmul_pos = text.find("plena.matmul")
    assert by_pos != -1 and row_pos != -1 and matmul_pos != -1, text
    assert by_pos < row_pos < matmul_pos, text
    assert text.count("for by") == 1, text


def test_fpram_elementwise_assignments_lower_to_fp_ops():
    text = _tir_text(_fpram_elementwise_kernel, "fp_elementwise")
    for op in (
        "plena.fp_copy_at",
        "plena.fp_sub_at",
        "plena.fp_add_at",
        "plena.fp_mul_at",
        "plena.fp_exp_at",
        "plena.fp_reci_at",
    ):
        assert op in text, text
    assert "T.exp" not in text


def test_row_parallel_and_reduce_patterns_lower_to_row_ops():
    text = _tir_text(_row_parallel_reduce_kernel, "row_patterns")
    for op in (
        "plena.row_sub_fp_at",
        "plena.row_exp_at",
        "plena.row_mul_fp_at",
        "plena.row_reduce_max_at",
        "plena.row_reduce_sum_at",
    ):
        assert op in text, text
    assert "T.parallel" not in text
    assert "T.reduce" not in text
    assert re.search(
        r"for row(_\d+)? in range\(64\):\n\s+T\.call_extern"
        r"\(\"handle\", \"plena\.row_reduce_max_at\"",
        text,
    ), text


# ---------------------------------------------------------------------------
# End-to-end: compile through to ISA and assert key opcodes.
# ---------------------------------------------------------------------------

def test_mm64_isa_has_mm_opcodes():
    func = compile_func(_mm64_kernel())
    ck = compile_kernel(func, target=PlenaTarget(), name="mm64")
    isa = ck.isa_text
    assert "M_MM" in isa, isa
    assert "M_MM_WO" in isa, isa


def test_qk_btmm_isa_has_btmm_opcodes():
    func = compile_func(_qk_btmm_kernel())
    ck = compile_kernel(func, target=PlenaTarget(), name="qk_btmm")
    isa = ck.isa_text
    assert "M_BTMM" in isa, isa
    assert "M_BMM_WO" in isa, isa


def test_fpram_buffer_operands_lower_to_scalar_addresses():
    func = compile_func(_fpram_buffer_kernel())
    ck = compile_kernel(func, target=PlenaTarget(), name="fpram_buf")
    assert ck.hlir.buffers["M_INIT"].scope == "fpram"
    assert ck.hlir.buffers["M_OLD"].scope == "fpram"
    assert ck.hlir.buffers["M_INIT"].address == FPRAM_USER_BASE
    assert ck.hlir.buffers["M_OLD"].address == FPRAM_USER_BASE + 4 * 64
    assert "S_LD_FP" in ck.isa_text, ck.isa_text
    assert "S_ST_FP" in ck.isa_text, ck.isa_text


# Note: a full ISA-emit test for the vector_add kernel is not included
# yet — the backend's plena.dma_*_slice handlers require the local buffer
# to be a full mlen×mlen tile, but the per-element add kernel uses 1-D
# shared (64,) buffers. Either the backend needs a sub-tile DMA path
# or the kernel needs to allocate 2-D shared. Out of Stage-7 scope.


if __name__ == "__main__":
    test_mm64_emits_dma_and_matmul()
    test_mm64_drops_threadidx_and_annotations()
    test_btmm_kernel_drops_lane_for_loop()
    test_btmm_kernel_emits_btmm_call_with_lane_count()
    test_vector_add_collapses_to_v_add()
    test_fpram_buffers_get_scope_and_lane_indexing()
    test_pure_lane_row_loop_stays_inside_by_run_before_matmul()
    test_mm64_isa_has_mm_opcodes()
    test_qk_btmm_isa_has_btmm_opcodes()
    test_fpram_buffer_operands_lower_to_scalar_addresses()
    print("lower_to_hlir e2e tests passed")
