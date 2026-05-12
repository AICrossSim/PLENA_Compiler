"""Unit tests for mid_ir.passes.mark.

Coverage:
  * Dma → Marker.DMA
  * Gemm(kind="btmm") → Marker.BTMM
  * Gemm(kind="overwrite") → no marker
  * Elementwise → Marker.LANE_OP
  * Reduce → Marker.LANE_OP
  * RawStore → no marker (pass-through)
  * Inside For: nested ops still get marked
  * Idempotency: mark(mark(x)) == mark(x)

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_mark
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.mark import run as mark_run


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _slice_ref(buf):
    return ir.BufferRef(buffer=buf, indices=[ir.Slice() for _ in buf.shape])


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _wrap(body):
    return ir.MidFunc(name="t", params=[], allocs=[], body=list(body))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mark_elementwise_pure_async() -> int:
    """Elementwise with no Broadcast src lowers to v_* — can_async=True."""
    print("test_mark_elementwise_pure_async — v_add / v_exp / etc.")
    A = _mk_buf("A", [64, 16])
    B = _mk_buf("B", [64, 16])
    C = _mk_buf("C", [64, 16])
    fn = _wrap([
        # v_add: dst, srcA, srcB all same shape
        ir.Elementwise(dst=_slice_ref(C),
                       srcs=[_slice_ref(A), _slice_ref(B)],
                       op=ir.BinOp.ADD),
        # v_exp_v: unary
        ir.Elementwise(dst=_slice_ref(A),
                       srcs=[_slice_ref(A)],
                       op=ir.UnaryOp.EXP),
        # zero_v: srcs=[]
        ir.Elementwise(dst=_slice_ref(C), srcs=[], op=ir.UnaryOp.COPY),
    ])
    out = mark_run(fn)
    failures = 0
    for i, label in enumerate(["v_add", "v_exp_v", "zero_v"]):
        failures += _check(f"[{i}] {label} marker", out.body[i].marker, ir.Marker.LANE_OP)
        failures += _check(f"[{i}] {label} can_async", out.body[i].can_async, True)
    return failures


def test_mark_elementwise_with_broadcast_not_async() -> int:
    """Elementwise with a Broadcast src lowers to row_*_fp_at — per-row,
    NOT async."""
    print("test_mark_elementwise_with_broadcast_not_async — row_sub_fp_at")
    S = _mk_buf("S", [64, 64], scope="fragment")
    M = _mk_buf("M_CURR", [64], scope="fragment")
    fn = _wrap([ir.Elementwise(
        dst=_slice_ref(S),
        srcs=[
            _slice_ref(S),
            ir.Broadcast(src=ir.BufferRef(M, [ir.Slice()]), broadcast_dims=[1]),
        ],
        op=ir.BinOp.SUB,
    )])
    out = mark_run(fn)
    failures = 0
    failures += _check("marker", out.body[0].marker, ir.Marker.LANE_OP)
    failures += _check("can_async", out.body[0].can_async, False)
    return failures


def test_mark_reduce_not_async() -> int:
    """Reduce always lowers to row_reduce_*_at — per-row, NOT async."""
    print("test_mark_reduce_not_async")
    src = _mk_buf("src", [64, 64], scope="fragment")
    dst = _mk_buf("dst", [64], scope="fragment")
    fn = _wrap([ir.Reduce(
        dst=_slice_ref(dst), src=_slice_ref(src),
        op=ir.ReduceOp.MAX, axis=1,
    )])
    out = mark_run(fn)
    failures = 0
    failures += _check("marker", out.body[0].marker, ir.Marker.LANE_OP)
    failures += _check("can_async", out.body[0].can_async, False)
    return failures


def test_mark_dma_async() -> int:
    print("test_mark_dma_async — DMA always async")
    a = _mk_buf("A", [64, 16])
    b = _mk_buf("B", [64, 16])
    fn = _wrap([ir.Dma(src=_slice_ref(a), dst=_slice_ref(b))])
    out = mark_run(fn)
    failures = 0
    failures += _check("marker", out.body[0].marker, ir.Marker.DMA)
    failures += _check("can_async", out.body[0].can_async, True)
    return failures


def test_mark_gemm_btmm_async() -> int:
    print("test_mark_gemm_btmm_async — btmm async")
    Q = _mk_buf("Q", [64, 16])
    K = _mk_buf("K", [64, 16])
    S = _mk_buf("S", [64, 64], scope="fragment")
    fn = _wrap([ir.Gemm(
        a=_slice_ref(Q), b=_slice_ref(K), c=_slice_ref(S),
        kind="btmm", transpose_b=True,
    )])
    out = mark_run(fn)
    failures = 0
    failures += _check("marker", out.body[0].marker, ir.Marker.BTMM)
    failures += _check("can_async", out.body[0].can_async, True)
    return failures


def test_mark_gemm_per_head_not_async() -> int:
    print("test_mark_gemm_per_head_not_async — overwrite per-head, no marker, no async")
    A = _mk_buf("A", [64, 64], scope="fragment")
    B = _mk_buf("B", [64, 16])
    C = _mk_buf("C", [64, 16], scope="fragment")
    fn = _wrap([ir.Gemm(
        a=_slice_ref(A), b=_slice_ref(B), c=_slice_ref(C),
        kind="overwrite",
    )])
    out = mark_run(fn)
    failures = 0
    failures += _check("marker", out.body[0].marker, None)
    failures += _check("can_async", out.body[0].can_async, False)
    return failures


def test_mark_raw_store_pass_through() -> int:
    print("test_mark_raw_store_pass_through — RawStore stays unmarked")
    buf = _mk_buf("padded", [67], scope="fragment")
    fn = _wrap([ir.For(loop_var="k", extent=3, body=[
        ir.RawStore(
            dst=ir.BufferRef(buf, [{"op": "add", "args": [64, "k"]}]),
            value="<opaque>",
        ),
    ])])
    out = mark_run(fn)
    failures = 0
    # The For is preserved, body still has the RawStore unchanged.
    f = out.body[0]
    failures += _check("body type", type(f.body[0]).__name__, "RawStore")
    failures += _check(
        "RawStore has no marker attr", hasattr(f.body[0], "marker"), False,
    )
    return failures


def test_mark_inside_for() -> int:
    print("test_mark_inside_for — ops nested inside a For still get marked")
    A = _mk_buf("A", [64, 16])
    B = _mk_buf("B", [64, 16])
    fn = _wrap([ir.For(loop_var="row", extent=64, body=[
        ir.Dma(src=_slice_ref(A), dst=_slice_ref(B)),
        ir.Elementwise(dst=_slice_ref(B), srcs=[], op=ir.UnaryOp.COPY),
    ])])
    out = mark_run(fn)
    failures = 0
    body = out.body[0].body
    failures += _check("Dma marker", body[0].marker, ir.Marker.DMA)
    failures += _check("Elementwise marker", body[1].marker, ir.Marker.LANE_OP)
    return failures


def test_mark_idempotent() -> int:
    print("test_mark_idempotent — running twice yields the same markers")
    A = _mk_buf("A", [64, 16])
    fn = _wrap([ir.Dma(src=_slice_ref(A), dst=_slice_ref(A))])
    once = mark_run(fn)
    twice = mark_run(once)
    return _check(
        "marker after 2x", twice.body[0].marker, ir.Marker.DMA,
    )


def test_mark_elementwise_with_broadcast_src() -> int:
    """``S[r,c] - M_CURR[r]`` folds to Elementwise(S, [S, Broadcast(M_CURR)], SUB).
    Mark sets the outer Elementwise's marker; the Broadcast itself
    has no marker field — it's just a src-shape annotation."""
    print("test_mark_elementwise_with_broadcast_src")
    S = _mk_buf("S", [64, 64], scope="fragment")
    M = _mk_buf("M_CURR", [64], scope="fragment")
    fn = _wrap([ir.Elementwise(
        dst=_slice_ref(S),
        srcs=[
            _slice_ref(S),
            ir.Broadcast(src=ir.BufferRef(M, [ir.Slice()]), broadcast_dims=[1]),
        ],
        op=ir.BinOp.SUB,
    )])
    out = mark_run(fn)
    failures = 0
    ew = out.body[0]
    failures += _check("outer Elementwise marker", ew.marker, ir.Marker.LANE_OP)
    # Confirm the Broadcast src is preserved structurally + has no
    # marker attribute.
    failures += _check(
        "src[1] type after mark", type(ew.srcs[1]).__name__, "Broadcast",
    )
    failures += _check(
        "Broadcast has no marker attr", hasattr(ew.srcs[1], "marker"), False,
    )
    failures += _check("broadcast dims", ew.srcs[1].broadcast_dims, [1])
    return failures


def test_mark_full_kernel_shape() -> int:
    """Mimic the post-fold shape of flash_attention_min's inner body —
    one of each op kind. Verify all markers in one shot."""
    print("test_mark_full_kernel_shape — flash_attention_min slice")
    Q_hbm = _mk_buf("Q_hbm", [1, 64, 4, 16], scope="global")
    Q_sh  = _mk_buf("Q_sh",  [64, 16])
    K_sh  = _mk_buf("K_sh",  [64, 16])
    S_loc = _mk_buf("S_loc", [64, 64], scope="fragment")
    M_CURR = _mk_buf("M_CURR", [64], scope="fragment")
    O_loc = _mk_buf("O_loc", [64, 16], scope="fragment")
    PV_loc = _mk_buf("PV_loc", [64, 16], scope="fragment")
    V_sh = _mk_buf("V_sh", [64, 16])

    fn = _wrap([
        ir.Dma(src=_slice_ref(Q_hbm), dst=_slice_ref(Q_sh)),                  # → DMA
        ir.Gemm(a=_slice_ref(Q_sh), b=_slice_ref(K_sh), c=_slice_ref(S_loc),  # → BTMM
                kind="btmm", transpose_b=True),
        ir.Reduce(dst=_slice_ref(M_CURR), src=_slice_ref(S_loc),              # → LANE_OP
                  op=ir.ReduceOp.MAX, axis=1),
        ir.Gemm(a=_slice_ref(S_loc), b=_slice_ref(V_sh), c=_slice_ref(PV_loc),  # → no marker
                kind="overwrite"),
        ir.Elementwise(dst=_slice_ref(O_loc), srcs=[], op=ir.UnaryOp.COPY),    # → LANE_OP
    ])
    out = mark_run(fn)
    failures = 0
    failures += _check("[0] Dma marker",          out.body[0].marker, ir.Marker.DMA)
    failures += _check("[0] Dma can_async",       out.body[0].can_async, True)
    failures += _check("[1] btmm Gemm marker",    out.body[1].marker, ir.Marker.BTMM)
    failures += _check("[1] btmm Gemm can_async", out.body[1].can_async, True)
    failures += _check("[2] Reduce marker",       out.body[2].marker, ir.Marker.LANE_OP)
    failures += _check("[2] Reduce can_async",    out.body[2].can_async, False)
    failures += _check("[3] per-head Gemm marker",   out.body[3].marker, None)
    failures += _check("[3] per-head Gemm can_async", out.body[3].can_async, False)
    failures += _check("[4] Elementwise marker",  out.body[4].marker, ir.Marker.LANE_OP)
    # Pure elementwise (zero_v) → can async
    failures += _check("[4] Elementwise can_async", out.body[4].can_async, True)
    return failures


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_mark_dma_async()
    failures += test_mark_gemm_btmm_async()
    failures += test_mark_gemm_per_head_not_async()
    failures += test_mark_elementwise_pure_async()
    failures += test_mark_elementwise_with_broadcast_not_async()
    failures += test_mark_reduce_not_async()
    failures += test_mark_raw_store_pass_through()
    failures += test_mark_inside_for()
    failures += test_mark_idempotent()
    failures += test_mark_elementwise_with_broadcast_src()
    failures += test_mark_full_kernel_shape()
    print()
    if failures == 0:
        print("PASS — all mid_ir.mark tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
