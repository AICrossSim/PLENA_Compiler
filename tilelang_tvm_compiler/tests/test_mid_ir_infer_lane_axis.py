"""Unit tests for mid_ir.passes.infer_lane_axis (pass_0).

Heuristic: a lane axis is a blockIdx grid var that
  * has static int extent divisible by LANE
  * appears as a *bare* index slot in some BufferLoad

Coverage:
  * Single bare-indexed grid var → picked
  * Multiple grid vars but only one bare-indexed → that one wins
    (this is the flash_attention case: ``by`` bare, ``q_block`` only
    in arithmetic)
  * No bare-indexed grid var → no attr set
  * Multiple bare-indexed candidates → raises (ambiguous)
  * Manual override preserved
  * Grid var with extent NOT multiple of LANE → not eligible

Run:
    /home/a13247568123124/project/PLENA_Simulator/.venv-tvm/bin/python \\
        -m tilelang_tvm_compiler.tests.test_mid_ir_infer_lane_axis
"""

from __future__ import annotations

import sys

import tvm
from tvm import tir

from tilelang_tvm_compiler.frontend.mid_ir.passes.infer_lane_axis import (
    InferLaneAxisError,
    run as infer_run,
)


_LANE = 4
_LANE_ATTR = "plena.lane_axis"


def _ii(n: int) -> tir.IntImm:
    return tir.IntImm("int32", n)


def _check(label, actual, expected) -> int:
    if actual == expected:
        print(f"  [OK]   {label}: {actual!r}")
        return 0
    print(f"  [FAIL] {label}: got {actual!r}, expected {expected!r}")
    return 1


def _block_idx(name: str, extent: int, tag: str, body) -> tir.Stmt:
    var = tir.Var(name, "int32")
    iv = tir.IterVar(
        dom=tvm.ir.Range.from_min_extent(_ii(0), _ii(extent)),
        var=var, iter_type=tir.IterVar.ThreadIndex, thread_tag=tag,
    )
    return tir.AttrStmt(iv, "thread_extent", _ii(extent), body)


def _read_lane_axis(func: tir.PrimFunc):
    if func.attrs is None or _LANE_ATTR not in func.attrs:
        return None
    v = func.attrs[_LANE_ATTR]
    return str(v.value) if isinstance(v, tir.StringImm) else str(v)


def _wrap(body, attrs=None) -> tir.PrimFunc:
    f = tir.PrimFunc(params=[], body=body, ret_type=None, buffer_map={})
    if attrs:
        for k, v in attrs.items():
            f = f.with_attr(k, v)
    return f


def _scoped_with_buf_use(grid_decls, buffer_load_indices_per_buf):
    """Build a body that wraps ``grid_decls`` (outer-to-inner) around a
    BufferLoad chain that exercises bare-vs-compound indexing per
    buffer.

    grid_decls: list of (name, extent, tag, var) tuples — note we need
        to track Var identity to pass into BufferLoads below; instead
        of returning the body alone we build it inline here.
    """
    raise NotImplementedError


def _make_body_with_loads(loads):
    """Make a body of N consecutive Evaluate(BufferLoad)s wrapped in a
    trivial scope. ``loads`` is a list of BufferLoad instances.

    SeqStmt requires ``seq.size() != 1`` so for a single load we just
    return the bare Evaluate."""
    evals = [tir.Evaluate(load) for load in loads]
    if len(evals) == 1:
        return evals[0]
    return tir.SeqStmt(evals)


def _decl_buffer(name, shape):
    return tir.decl_buffer(shape, dtype="float16", name=name, scope="global")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_bare_indexed_grid_var_picked() -> int:
    """Single blockIdx ``by`` (extent=4) used bare in a BufferLoad → picked."""
    print("test_single_bare_indexed_grid_var_picked")
    by = tir.Var("by", "int32")
    Q = _decl_buffer("Q", [1, 64, 4, 16])
    load = tir.BufferLoad(Q, [_ii(0), _ii(0), by, _ii(0)])
    body = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                               _make_body_with_loads([load]))
    func = _wrap(body)
    out = infer_run(func)
    return _check("picked", _read_lane_axis(out), "by")


def _block_idx_with_var(name, extent, tag, var, body):
    """Same as _block_idx but lets caller supply the Var identity so it
    can also be referenced inside the body's BufferLoads."""
    iv = tir.IterVar(
        dom=tvm.ir.Range.from_min_extent(_ii(0), _ii(extent)),
        var=var, iter_type=tir.IterVar.ThreadIndex, thread_tag=tag,
    )
    return tir.AttrStmt(iv, "thread_extent", _ii(extent), body)


def test_q_block_only_arithmetic_not_picked() -> int:
    """flash_attention case: outer q_block (extent=2 — NOT multiple of
    LANE so doesn't qualify even shape-wise) + inner by (extent=4)
    where Q_hbm is loaded with ``q_block * 64`` and bare ``by``. Only
    by qualifies."""
    print("test_q_block_only_arithmetic_not_picked")
    q_block = tir.Var("q_block", "int32")
    by = tir.Var("by", "int32")
    Q = _decl_buffer("Q", [1, 64 * 2, 4, 16])
    # Q_hbm[0, q_block*64, by, 0] — by is bare, q_block is in q_block*64
    load = tir.BufferLoad(Q, [_ii(0), q_block * _ii(64), by, _ii(0)])
    inner = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                                _make_body_with_loads([load]))
    outer = _block_idx_with_var("q_block", 2, "blockIdx.x", q_block, inner)
    func = _wrap(outer)
    out = infer_run(func)
    return _check("picked by", _read_lane_axis(out), "by")


def test_q_block_lane_eligible_only_when_bare() -> int:
    """Even if q_block extent IS divisible by LANE (e.g. extent=8), if
    it's only used as ``q_block * 64`` it's not a lane candidate."""
    print("test_q_block_lane_eligible_only_when_bare — q_block only in arithmetic")
    q_block = tir.Var("q_block", "int32")
    by = tir.Var("by", "int32")
    Q = _decl_buffer("Q", [1, 64 * 8, 4, 16])
    load = tir.BufferLoad(Q, [_ii(0), q_block * _ii(64), by, _ii(0)])
    inner = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                                _make_body_with_loads([load]))
    outer = _block_idx_with_var("q_block", 8, "blockIdx.x", q_block, inner)
    func = _wrap(outer)
    out = infer_run(func)
    # by is bare, q_block isn't → only by qualifies
    return _check("picked by, not q_block", _read_lane_axis(out), "by")


def test_no_buffer_loads_no_attr() -> int:
    """No BufferLoad anywhere → no bare-index candidates → no attr."""
    print("test_no_buffer_loads_no_attr")
    by = tir.Var("by", "int32")
    body = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                               tir.Evaluate(_ii(0)))
    func = _wrap(body)
    out = infer_run(func)
    return _check("no attr set", _read_lane_axis(out), None)


def test_multiple_bare_candidates_raise() -> int:
    """Two grid vars both used bare AND both extent divisible by LANE
    → ambiguous; raise."""
    print("test_multiple_bare_candidates_raise")
    by = tir.Var("by", "int32")
    bx = tir.Var("bx", "int32")
    Q = _decl_buffer("Q", [4, 4, 16])
    # Q[bx, by, 0] — both bare
    load = tir.BufferLoad(Q, [bx, by, _ii(0)])
    inner = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                                _make_body_with_loads([load]))
    outer = _block_idx_with_var("bx", _LANE, "blockIdx.x", bx, inner)
    func = _wrap(outer)
    try:
        infer_run(func)
    except InferLaneAxisError as e:
        print(f"  [OK]   raised InferLaneAxisError: {str(e)[:60]}...")
        return 0
    print("  [FAIL] expected InferLaneAxisError")
    return 1


def test_manual_override_preserved() -> int:
    print("test_manual_override_preserved")
    by = tir.Var("by", "int32")
    Q = _decl_buffer("Q", [1, 64, 4, 16])
    load = tir.BufferLoad(Q, [_ii(0), _ii(0), by, _ii(0)])
    body = _block_idx_with_var("by", _LANE, "blockIdx.y", by,
                               _make_body_with_loads([load]))
    func = _wrap(body, attrs={_LANE_ATTR: "manual"})
    out = infer_run(func)
    return _check("preserved", _read_lane_axis(out), "manual")


def test_extent_not_multiple_of_lane() -> int:
    """Bare-indexed grid var, but extent=3 (not multiple of LANE=4) →
    not eligible."""
    print("test_extent_not_multiple_of_lane")
    by = tir.Var("by", "int32")
    Q = _decl_buffer("Q", [3, 16])
    load = tir.BufferLoad(Q, [by, _ii(0)])
    body = _block_idx_with_var("by", 3, "blockIdx.y", by,
                               _make_body_with_loads([load]))
    func = _wrap(body)
    out = infer_run(func)
    return _check("no attr (extent not lane-multiple)",
                  _read_lane_axis(out), None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    failures = 0
    failures += test_single_bare_indexed_grid_var_picked()
    failures += test_q_block_only_arithmetic_not_picked()
    failures += test_q_block_lane_eligible_only_when_bare()
    failures += test_no_buffer_loads_no_attr()
    failures += test_multiple_bare_candidates_raise()
    failures += test_manual_override_preserved()
    failures += test_extent_not_multiple_of_lane()
    print()
    if failures == 0:
        print("PASS — all mid_ir.infer_lane_axis tests")
        return 0
    print(f"FAIL — {failures} failed assertion(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
