"""Unit tests for mid_ir.passes.burn_view (pass_5b).

Coverage:
  * Buffer shape gets permuted by view_perm
  * All ref indices on that buffer permute by the same perm
  * view_perm reset to None after bake
  * Buffer with identity perm: shape unchanged, indices unchanged,
    view_perm cleared
  * Mixed perms across buffers: each baked independently
  * BHSD buffer (identity) coexists with BSHD buffer (permute) in
    same kernel
  * Conflict (mid_ir bug — pass_4b should have caught): raises
  * cluster_guard skip → no-op

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_burn_view
"""

from __future__ import annotations

import sys

from tilelang_tvm_compiler.frontend.mid_ir import ir
from tilelang_tvm_compiler.frontend.mid_ir.passes.burn_view import (
    BurnViewError,
    run as burn_run,
)


LANE = 4


def _mk_buf(name, shape, scope="shared"):
    return ir.BufferDef(name=name, shape=shape, dtype="float16", scope=scope)


def _ref(buf, indices, view_perm=None):
    return ir.BufferRef(buf, list(indices), view_perm=view_perm)


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _wrap(body, allocs=()):
    return ir.MidFunc(
        name="t", params=[], allocs=list(allocs), body=list(body),
        lane_axes=["by"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_bshd_bake_permutes_shape_and_indices() -> int:
    """Q_sh shape (4, 64, 16) with view_perm=[1,0,2] (BSHD) →
    HLIR shape (64, 4, 16); ref indices ['by_phase', :, :] →
    [:, 'by_phase', :]."""
    print("test_bshd_bake_permutes_shape_and_indices")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16])
    fn = _wrap([
        ir.Dma(
            src=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
            dst=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
        ),
    ], allocs=[Q_sh])
    out = burn_run(fn)
    failures = 0
    # Buffer shape permuted
    new_buf = out.allocs[0]
    failures += _check("Buffer shape", new_buf.shape, [64, LANE, 16])
    # Ref indices permuted
    dma = out.body[0]
    failures += _check("src indices", dma.src.indices,
                       [ir.Slice(), "by_phase", ir.Slice()])
    failures += _check("dst indices", dma.dst.indices,
                       [ir.Slice(), "by_phase", ir.Slice()])
    failures += _check("src view_perm cleared", dma.src.view_perm, None)
    failures += _check("dst view_perm cleared", dma.dst.view_perm, None)
    return failures


def test_bhsd_identity_unchanged_shape_indices() -> int:
    """S_loc with view_perm=[0,1,2] (BHSD identity): shape stays
    (4, 64, 16), indices stay; view_perm just clears.

    Use D=16 (not 64) to keep cluster_guard from no-op-ing.
    """
    print("test_bhsd_identity_unchanged_shape_indices")
    S = _mk_buf("S", [LANE, 64, 16], scope="fragment")
    fn = _wrap([
        ir.Reduce(
            dst=_ref(S, ["by_phase", 0, 0], view_perm=[0, 1, 2]),
            src=_ref(S, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[0, 1, 2]),
            op=ir.ReduceOp.MAX, axis=2,
        ),
    ], allocs=[S])
    out = burn_run(fn)
    failures = 0
    new_buf = out.allocs[0]
    failures += _check("shape unchanged", new_buf.shape, [LANE, 64, 16])
    red = out.body[0]
    failures += _check("dst indices unchanged",
                       red.dst.indices, ["by_phase", 0, 0])
    failures += _check("src indices unchanged",
                       red.src.indices, ["by_phase", ir.Slice(), ir.Slice()])
    failures += _check("dst view_perm cleared", red.dst.view_perm, None)
    failures += _check("src view_perm cleared", red.src.view_perm, None)
    return failures


def test_mixed_buffers_baked_independently() -> int:
    """Q_sh BSHD, S_loc BHSD, in same kernel — each baked own way."""
    print("test_mixed_buffers_baked_independently")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16])               # → BSHD permute
    S_loc = _mk_buf("S_loc", [LANE, 64, 64], scope="fragment")  # BHSD identity
    fn = _wrap([
        ir.Dma(
            src=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
            dst=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
        ),
        ir.Reduce(
            dst=_ref(S_loc, ["by_phase", 0, 0], view_perm=[0, 1, 2]),
            src=_ref(S_loc, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[0, 1, 2]),
            op=ir.ReduceOp.MAX, axis=2,
        ),
    ], allocs=[Q_sh, S_loc])
    out = burn_run(fn)
    failures = 0
    failures += _check("Q_sh shape", out.allocs[0].shape, [64, LANE, 16])
    failures += _check("S_loc shape unchanged", out.allocs[1].shape,
                       [LANE, 64, 64])
    return failures


def test_buffer_pointer_swap() -> int:
    """After bake, BufferRef.buffer points to the *new* permuted def
    (not the old one)."""
    print("test_buffer_pointer_swap")
    Q_sh = _mk_buf("Q_sh", [LANE, 64, 16])
    fn = _wrap([
        ir.Dma(
            src=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
            dst=_ref(Q_sh, ["by_phase", ir.Slice(), ir.Slice()],
                    view_perm=[1, 0, 2]),
        ),
    ], allocs=[Q_sh])
    out = burn_run(fn)
    new_buf = out.allocs[0]
    dma = out.body[0]
    return (_check("src.buffer is new def", dma.src.buffer is new_buf, True)
            + _check("dst.buffer is new def", dma.dst.buffer is new_buf, True))


def test_inconsistent_perms_raise() -> int:
    """Bug case: same buffer with conflicting perms (pass_4b should
    have caught it). burn_view re-verifies as defense in depth."""
    print("test_inconsistent_perms_raise")
    Q = _mk_buf("Q", [LANE, 64, 16])
    fn = _wrap([
        ir.Dma(
            src=_ref(Q, ["by_phase", ir.Slice(), ir.Slice()], view_perm=[1, 0, 2]),
            dst=_ref(Q, ["by_phase", ir.Slice(), ir.Slice()], view_perm=[0, 1, 2]),
        ),
    ], allocs=[Q])
    try:
        burn_run(fn)
    except BurnViewError as e:
        print(f"  [OK]   raised BurnViewError: {str(e)[:60]}...")
        return 0
    return 1


def test_skip_no_lane_axes() -> int:
    print("test_skip_no_lane_axes")
    Q = _mk_buf("Q", [LANE, 64, 16])
    fn = ir.MidFunc(
        name="t", params=[], allocs=[Q],
        body=[ir.Dma(src=_ref(Q, [ir.Slice()] * 3),
                     dst=_ref(Q, [ir.Slice()] * 3))],
        lane_axes=[],
    )
    out = burn_run(fn)
    return _check("shape unchanged", out.allocs[0].shape, [LANE, 64, 16])


def test_no_views_set_no_op() -> int:
    """No ref carries view_perm — nothing to bake, returns input."""
    print("test_no_views_set_no_op")
    Q = _mk_buf("Q", [LANE, 64, 16])
    fn = _wrap([
        ir.Dma(src=_ref(Q, [ir.Slice()] * 3),
               dst=_ref(Q, [ir.Slice()] * 3)),
    ], allocs=[Q])
    out = burn_run(fn)
    return _check("shape unchanged", out.allocs[0].shape, [LANE, 64, 16])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_bshd_bake_permutes_shape_and_indices()
    failures += test_bhsd_identity_unchanged_shape_indices()
    failures += test_mixed_buffers_baked_independently()
    failures += test_buffer_pointer_swap()
    failures += test_inconsistent_perms_raise()
    failures += test_skip_no_lane_axes()
    failures += test_no_views_set_no_op()
    print()
    if failures == 0:
        print("PASS — all mid_ir.burn_view tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
