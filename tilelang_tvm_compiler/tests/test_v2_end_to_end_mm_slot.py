"""End-to-end v2 tests for ``mm_slot``.

mm_slot applies M_MM/M_MM_WO over a col-slot of rhs/dst with optional
dynamic LHS row, RHS col, DST col offsets. Static-offset case is
covered first; dynamic (PrimExpr) offsets are covered by a second
test that uses a tir.Var offset.

Comparison is structural — same M_MM / M_MM_WO count + order. Scalar
address-arithmetic mnemonics (S_ADDI_INT / S_SLLI_INT / S_ADD_INT /
S_MUL_INT) are stripped: v2 builds each pair's addresses from fresh
SSA chains while legacy reuses a 5-GP allocation with per-iter
S_ADDI bumps. Same dynamic semantics, different static form.
"""

import re

import pytest
from tvm import tir

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import pre_isa_to_mir as p2m
from tilelang_tvm_compiler import mir_to_isa as m2i
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
BLEN = 4
_GP_RE = re.compile(r"\bgp\d+\b")
_ADDR_SETUP = frozenset({
    "S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT",
    "S_ADD_INT", "S_SUB_INT", "S_MUL_INT",
})


def _strip(isa: str):
    out = []
    for ln in isa.split("\n"):
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        head = s.split(None, 1)[0]
        if head in _ADDR_SETUP:
            continue
        out.append(_GP_RE.sub("gpX", s))
    return out


def _vram(name, addr, shape):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM, shape=shape,
        dtype="float16", address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _mram(name, addr, shape):
    return _hlir.Buffer(
        name=name, scope=_scope.MRAM, shape=shape,
        dtype="float16", address=addr,
    )


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=16)
    return IsaEmitterPass(shim).run(hlir)


def _count(isa, mnem):
    return sum(
        1 for ln in isa.split("\n") if ln.strip().startswith(mnem + " ")
    )


def test_mm_slot_static_offsets_v2_matches_legacy():
    """All 4 scalar args literal: lhs_row=0, rhs_col=0, dst_col=0,
    col_count=blen (= 1 oc tile).

    tiles_per_slot=1, tiles_per_mlen=16 → 16 M_MM pairs.
    """
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))

    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[0, 0, 0, BLEN],
        annotations={"intrinsic": "mm_slot_static"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_slot_static",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    expected = (BLEN // BLEN) * (MLEN // BLEN)  # 1 * 16
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == expected
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected


def test_mm_slot_wide_col_count_v2_matches_legacy():
    """col_count = 4*blen (= 4 oc tiles) → 4*16 = 64 M_MM pairs."""
    cc = 4 * BLEN
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))
    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[0, 0, 0, cc],
        annotations={"intrinsic": "mm_slot_wide"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_slot_wide",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    expected = (cc // BLEN) * (MLEN // BLEN)
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == expected
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected


def test_mm_slot_static_nonzero_offsets_v2_matches_legacy():
    """Static lhs_row, rhs_col, dst_col offsets all non-zero.

    Sanity: legacy folds the literals into S_ADDI immediates while v2
    builds (base + offset) PrimExprs that arith.simplify reduces to
    literals — final HW ops still match structurally.
    """
    cc = 2 * BLEN
    # Pick offsets that keep us in bounds.
    lhs_row_off = MLEN * MLEN  # second mlen*mlen tile of a 2-tile lhs
    rhs_col_off = 2 * BLEN
    dst_col_off = 3 * BLEN
    # Resize lhs so lhs_row_off is in range.
    lhs = _vram("lhs", 0, (2 * MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))
    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[lhs_row_off, rhs_col_off, dst_col_off, cc],
        annotations={"intrinsic": "mm_slot_static_off"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_slot_static_off",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )


def test_mm_slot_dynamic_lhs_row_offset_v2():
    """lhs_row_offset is a PrimExpr (= h * mlen * mlen).

    Legacy materialises it once to a GP and bases each oc tile off
    that GP. v2 keeps it as a symbolic PrimExpr per pair — same M_MM
    pair count, different static form (different scalar S_* prefix).

    We don't compare to legacy here (mnemonic counts differ at the
    S_* level since legacy hoists, v2 inlines per iter). Instead we
    verify the M_MM pair count is correct and the v2 path passes MIR
    verify().
    """
    cc = 2 * BLEN
    h = tir.Var("h", "int32")
    lhs_off_expr = tir.Mul(h, tir.IntImm("int32", MLEN * MLEN))
    lhs = _vram("lhs", 0, (4 * MLEN, MLEN))  # roomy enough
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))
    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[lhs_off_expr, 0, 0, cc],
        annotations={"intrinsic": "mm_slot_dynamic_lhs"},
    )
    # Wrap in a for-h loop so h is in scope as a loop_var.
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[],
        scalar_args=[],
        annotations={
            "loop_var": h, "extent": 2, "init": 0,
            "loop_kind": "unroll",
        },
        body=[op],
    )
    hlir = _hlir.HLIRModule(
        name="mm_slot_dyn",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[for_op], param_names=[],
    )
    new = _v2_emit(hlir)
    # 2 (h iters) * (cc/blen) * tiles_per_mlen pairs.
    expected = 2 * (cc // BLEN) * (MLEN // BLEN)
    assert _count(new, "M_MM") == expected
    assert _count(new, "M_MM_WO") == expected
