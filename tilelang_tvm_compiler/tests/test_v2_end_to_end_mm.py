"""End-to-end v2 tests for ``mm`` (single-tile mlen*mlen path).

Structural comparison: same HW mnemonic stream (M_MM / M_MM_WO in
same order, same count). Address-setup S_ADDI_INT etc. skipped since
v2 builds addresses from SSA chains while legacy uses literals.

The narrow-tile (mlen*hlen) and rectangular path is not covered here
— v2's ``_emit_mm`` currently rejects ``cols != mlen`` with a
"narrow-tile path not yet migrated" error. That path moves into the
general ``matmul`` family handler.
"""

import re

import pytest

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
# Both paths emit very different S_* sequences for the same address
# expression (legacy preserves S_ADDI/SLLI/SRLI/LUI chains; v2 rebuilds
# fresh PrimExpr → SSA chains). We compare only HW core ops.
_ADDR_SETUP = frozenset({
    "S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT",
    # v2 builds the result_addr = dst.base + oc*blen + orow*row_stride
    # as a multi-step PrimExpr chain. The final SSA "two non-const
    # operands" reduction emits S_ADD_INT (legacy folds it into the
    # materialiser preamble — no S_ADD_INT in legacy mm). Filter the
    # whole scalar address-arithmetic family out of the comparison.
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


def test_mm_single_tile_v2_matches_legacy():
    """mlen*mlen @ mlen*mlen → mlen*mlen.

    tiles_per_mlen = mlen/blen = 16; outer oc loop x inner orow loop
    each runs 16 iters → 256 M_MM pairs.
    """
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))

    op = _hlir.Op(
        kind="mm",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[],
        annotations={"intrinsic": "mm_single_tile_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_single_tile",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)

    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    # Sanity — both emit 256 pairs.
    tiles = MLEN // BLEN
    expected = tiles * tiles
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == expected
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected


def test_mm_narrow_tile_not_yet_supported():
    """v2 _emit_mm explicitly rejects mlen*hlen narrow-tile path.

    Once the general ``matmul`` handler is migrated, narrow MM will
    route there; for now we just assert v2 declines cleanly.
    """
    from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2Error
    HLEN = 16
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, HLEN))
    dst = _vram("dst", 8192, (MLEN, HLEN))
    op = _hlir.Op(
        kind="mm",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[],
        annotations={"intrinsic": "mm_narrow_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_narrow_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    with pytest.raises(PreIsaPassV2Error, match="narrow-tile"):
        _v2_emit(hlir)
