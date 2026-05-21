"""End-to-end v2 tests for btmm / btmv / mv.

Structural comparison: same set of HW mnemonics (M_BTMM / M_BMM_WO /
M_BTMV / M_BMV_WO / M_MV / M_MV_WO) in same order. Address-setup
S_ADDI / S_SLLI / etc. skipped since v2 builds addresses from SSA
chains while legacy uses literals.
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
HLEN = 16
LANE = 4
_GP_RE = re.compile(r"\bgp\d+\b")
_ADDR_SETUP = frozenset({"S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT"})


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
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    return IsaEmitterPass(shim).run(hlir)


def test_btmm_v2_matches_legacy():
    lhs = _vram("lhs", 0, (MLEN, LANE, HLEN))
    rhs = _mram("rhs", 4096, (MLEN, LANE, HLEN))
    dst = _vram("dst", 8192, (MLEN, LANE, MLEN))
    op = _hlir.Op(
        kind="btmm",
        buffer_args=[
            _hlir.VramRegion(parent="lhs", starts=(0, 0, 0),
                             extents=(MLEN, LANE, HLEN)),
            _hlir.MramRegion(parent="rhs", starts=(0, 0, 0),
                             extents=(MLEN, LANE, HLEN)),
            _hlir.VramRegion(parent="dst", starts=(0, 0, 0),
                             extents=(MLEN, LANE, MLEN)),
        ],
        scalar_args=[],
        annotations={"intrinsic": "btmm_test"},
    )
    hlir = _hlir.HLIRModule(
        name="btmm", buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )


def test_btmv_v2_matches_legacy():
    lhs = _vram("lhs", 0, (1, LANE, HLEN))
    rhs = _mram("rhs", 4096, (MLEN, LANE, HLEN))
    dst = _vram("dst", 8192, (1, LANE, MLEN))
    op = _hlir.Op(
        kind="btmv",
        buffer_args=[
            _hlir.VramRegion(parent="lhs", starts=(0, 0, 0),
                             extents=(1, LANE, HLEN)),
            _hlir.MramRegion(parent="rhs", starts=(0, 0, 0),
                             extents=(MLEN, LANE, HLEN)),
            _hlir.VramRegion(parent="dst", starts=(0, 0, 0),
                             extents=(1, LANE, MLEN)),
        ],
        scalar_args=[],
        annotations={"intrinsic": "btmv_test"},
    )
    hlir = _hlir.HLIRModule(
        name="btmv", buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )


def test_mv_v2_matches_legacy():
    lhs = _vram("lhs", 0, (1, HLEN))
    rhs = _mram("rhs", 4096, (MLEN, HLEN))
    dst = _vram("dst", 8192, (1, HLEN))
    op = _hlir.Op(
        kind="mv",
        buffer_args=[
            _hlir.VramRegion(parent="lhs", starts=(0, 0), extents=(1, HLEN)),
            _hlir.MramRegion(parent="rhs", starts=(0, 0), extents=(MLEN, HLEN)),
            _hlir.VramRegion(parent="dst", starts=(0, 0), extents=(1, HLEN)),
        ],
        scalar_args=[],
        annotations={"intrinsic": "mv_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mv", buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
