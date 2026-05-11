"""Unit tests for classify_lane_use.

Builds raw TIR by hand (no tilelang dependency) using ``tir.call_extern``
to encode the ``tl.tileop.*`` op names. classify_lane_use accepts both
direct-Op and call_extern forms (see ``_call_kind`` in the pass).

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_classify_lane_use
"""

from __future__ import annotations

import sys

import tvm
from tvm import tir

from tilelang_tvm_compiler.frontend.passes.classify_lane_use import (
    KIND_KEY,
    LANE_AXIS_FUNC_ATTR,
    ROLE_BTMM_LHS,
    ROLE_BTMM_OUT,
    ROLE_BTMM_RHS,
    ROLE_LANE_DMA_DST,
    ROLE_NONE,
    ROLE_PER_HEAD_LHS,
    ROLE_PER_HEAD_OUT,
    ROLE_PER_HEAD_RHS,
    run,
)


def _ii(n: int, dtype: str = "int32") -> tir.IntImm:
    return tir.IntImm(dtype, n)


def _extern(name: str, *args):
    """Build a ``Call(op=tir.call_extern, args=[StringImm(name), ...])``."""
    return tir.call_extern("handle", name, *args)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def _build_func(*,
                head_count: int = 4,
                with_btmm: bool = True,
                with_per_head_matmul: bool = True,
                with_lane_copy: bool = True,
                declare_lane_axis: bool = True) -> tir.PrimFunc:
    """Hand-build a PrimFunc shaped like a head-fused kernel.

    Mirrors what tilelang produces *after* T.gemm / T.copy lowering:
    each becomes a ``tir.call_extern("tl.tileop.gemm_py" / "copy", ...)``
    on top of ``tir.call_extern("tl.tileop.region", BufferLoad, mode,
    *extents)``.
    """
    f16 = "float16"
    rows, hlen, mlen = 64, 16, 64

    Q_hbm = tir.decl_buffer([1, rows, head_count, hlen], dtype=f16, name="Q_hbm", scope="global")
    K_hbm = tir.decl_buffer([1, rows, head_count, hlen], dtype=f16, name="K_hbm", scope="global")
    V_hbm = tir.decl_buffer([1, rows, head_count, hlen], dtype=f16, name="V_hbm", scope="global")
    O_hbm = tir.decl_buffer([1, rows, head_count, hlen], dtype=f16, name="O_hbm", scope="global")

    Q_sh   = tir.decl_buffer([rows, hlen], dtype=f16, name="Q_sh", scope="shared.dyn")
    K_sh   = tir.decl_buffer([rows, hlen], dtype=f16, name="K_sh", scope="shared.dyn")
    V_sh   = tir.decl_buffer([rows, hlen], dtype=f16, name="V_sh", scope="shared.dyn")
    S_loc  = tir.decl_buffer([rows, mlen], dtype=f16, name="S_loc", scope="local.fragment")
    PV_loc = tir.decl_buffer([rows, hlen], dtype=f16, name="PV_loc", scope="local.fragment")
    O_loc  = tir.decl_buffer([rows, hlen], dtype=f16, name="O_loc", scope="local.fragment")

    by = tir.Var("by", "int32")
    by_iv = tir.IterVar(
        dom=tvm.ir.Range.from_min_extent(_ii(0), _ii(head_count)),
        var=by,
        iter_type=tir.IterVar.ThreadIndex,
        thread_tag="blockIdx.y",
    )

    def region_full(buf):
        starts = [_ii(0)] * len(buf.shape)
        return _extern(
            "tl.tileop.region",
            tir.BufferLoad(buf, starts),
            _ii(0),
            *[_ii(int(d)) for d in buf.shape],
        )

    def region_lane_slice(hbm_buf):
        starts = [_ii(0), _ii(0), by, _ii(0)]
        return _extern(
            "tl.tileop.region",
            tir.BufferLoad(hbm_buf, starts),
            _ii(0),
            _ii(1), _ii(rows), _ii(1), _ii(hlen),
        )

    def gemm_call(A, B, C):
        return tir.Evaluate(_extern(
            "tl.tileop.gemm_py",
            region_full(A), region_full(B), region_full(C),
        ))

    def copy_call(src_region, dst_region):
        return tir.Evaluate(_extern("tl.tileop.copy", src_region, dst_region))

    body_stmts = []
    if with_lane_copy:
        body_stmts.append(copy_call(region_lane_slice(Q_hbm), region_full(Q_sh)))
        body_stmts.append(copy_call(region_lane_slice(K_hbm), region_full(K_sh)))
        body_stmts.append(copy_call(region_lane_slice(V_hbm), region_full(V_sh)))
    if with_btmm:
        body_stmts.append(tir.AttrStmt(
            _ii(0), KIND_KEY, tir.StringImm("btmm"),
            gemm_call(Q_sh, K_sh, S_loc),
        ))
    if with_per_head_matmul:
        body_stmts.append(gemm_call(S_loc, V_sh, PV_loc))
    if with_lane_copy:
        body_stmts.append(copy_call(region_full(O_loc), region_lane_slice(O_hbm)))

    body = tir.SeqStmt(body_stmts)
    for buf in [O_loc, PV_loc, S_loc, V_sh, K_sh, Q_sh]:
        body = tir.Allocate(
            buf.data, buf.dtype,
            [_ii(int(d)) for d in buf.shape],
            _ii(1, "bool"),
            body,
        )
    body = tir.AttrStmt(by_iv, "thread_extent", _ii(head_count), body)

    func = tir.PrimFunc(
        params=[Q_hbm.data, K_hbm.data, V_hbm.data, O_hbm.data],
        body=body, ret_type=None,
        buffer_map={
            Q_hbm.data: Q_hbm,
            K_hbm.data: K_hbm,
            V_hbm.data: V_hbm,
            O_hbm.data: O_hbm,
        },
    )
    if declare_lane_axis:
        func = func.with_attr(LANE_AXIS_FUNC_ATTR, "by")
    return func


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _check(name, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {name}: {actual!r}")
        return 0
    print(f"  [FAIL] {name}: got {actual!r}, expected {expected!r}")
    return 1


def test_full_kernel_classification() -> int:
    print("test_full_kernel_classification")
    func = _build_func()
    _, c = run(func)
    failures = 0
    # Q_sh / K_sh: T.copy from lane-indexed HBM slice tags them
    # lane_dma_dst FIRST. The later btmm gemm tries to retag them
    # btmm_lhs / btmm_rhs, but those are layout-compatible (both
    # COL_PACK), so the lane_dma_dst tag stays.
    failures += _check("Q_sh",   c["Q_sh"].role,   ROLE_LANE_DMA_DST)
    failures += _check("K_sh",   c["K_sh"].role,   ROLE_LANE_DMA_DST)
    failures += _check("V_sh",   c["V_sh"].role,   ROLE_LANE_DMA_DST)
    failures += _check("S_loc",  c["S_loc"].role,  ROLE_BTMM_OUT)
    failures += _check("PV_loc", c["PV_loc"].role, ROLE_PER_HEAD_OUT)
    # O_loc is the source of a lane-DMA copy → ROLE_LANE_DMA_DST.
    failures += _check("O_loc",  c["O_loc"].role,  ROLE_LANE_DMA_DST)
    # HBM params untouched.
    for name in ("Q_hbm", "K_hbm", "V_hbm", "O_hbm"):
        failures += _check(name, c[name].role, ROLE_NONE)
    return failures


def test_no_btmm_attr() -> int:
    print("test_no_btmm_attr — gemm without KIND attr is per_head")
    func = _build_func(with_btmm=False)
    _, c = run(func)
    failures = 0
    # Per-head gemm seen: S_loc=LHS, V_sh=RHS, PV_loc=OUT
    failures += _check("S_loc",  c["S_loc"].role,  ROLE_PER_HEAD_LHS)
    # V_sh was lane_dma_dst from the copy first.
    failures += _check("V_sh",   c["V_sh"].role,   ROLE_LANE_DMA_DST)
    failures += _check("PV_loc", c["PV_loc"].role, ROLE_PER_HEAD_OUT)
    return failures


def test_no_lane_axis_attr() -> int:
    print("test_no_lane_axis_attr — without plena.lane_axis attr, copies don't promote")
    func = _build_func(declare_lane_axis=False)
    _, c = run(func)
    failures = 0
    # Without lane_axis: copies don't see `by` as the lane var, so
    # the dst doesn't get lane_dma_dst. But the gemms still run.
    # Q_sh becomes btmm_lhs straight from the gemm.
    failures += _check("Q_sh",  c["Q_sh"].role,  ROLE_BTMM_LHS)
    failures += _check("K_sh",  c["K_sh"].role,  ROLE_BTMM_RHS)
    # O_loc is alloc'd but never touches a gemm; without lane_axis the
    # copies don't tag it either. The classifier only inserts entries
    # for buffers it saw — O_loc shouldn't be in the table at all.
    if "O_loc" in c and c["O_loc"].role != ROLE_NONE:
        print(f"  [FAIL] O_loc unexpectedly classified as {c['O_loc'].role!r}")
        failures += 1
    else:
        print("  [OK]   O_loc: not classified (expected)")
    return failures


def main() -> int:
    failures = 0
    failures += test_full_kernel_classification()
    failures += test_no_btmm_attr()
    failures += test_no_lane_axis_attr()
    print()
    if failures == 0:
        print("PASS — all classify_lane_use tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
