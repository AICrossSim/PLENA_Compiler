"""End-to-end v2 tests for the vector ops
(v_zero / v_add / v_sub / v_mul / v_exp / v_reci / v_sqrt).
"""

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


import re

_GP_RE = re.compile(r"\bgp\d+\b")


def _non_addi_lines(isa: str):
    """Non-S_ADDI ISA lines with gpN → gpX canonicalisation, so
    v2's aggressive GP reuse compares equal to legacy's separate
    GP assignments."""
    out = []
    for ln in isa.split("\n"):
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        head = s.split(None, 1)[0]
        if head == "S_ADDI_INT":
            continue
        out.append(_GP_RE.sub("gpX", s))
    return out


def _vram(name, addr):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, 1, MLEN), dtype="float16",
        address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _whole(name):
    return _hlir.VramRegion(
        parent=name, starts=(0, 0, 0, 0), extents=(1, MLEN, 1, MLEN),
    )


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=16)
    return IsaEmitterPass(shim).run(hlir)


def test_v_zero_v2_matches_legacy():
    hlir = _hlir.HLIRModule(
        name="v_zero",
        buffers={"dst": _vram("dst", 0)},
        ops=[_hlir.Op(
            kind="v_zero",
            buffer_args=[_whole("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_lines(legacy_isa) == _non_addi_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", ["v_add", "v_sub", "v_mul"])
def test_v_binary_v2_matches_legacy(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "lhs": _vram("lhs", 0),
            "rhs": _vram("rhs", MLEN * MLEN),
            "dst": _vram("dst", 2 * MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_whole("lhs"), _whole("rhs"), _whole("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_lines(legacy_isa) == _non_addi_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", ["v_exp", "v_reci", "v_sqrt"])
def test_v_unary_v2_matches_legacy(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram("src", 0),
            "dst": _vram("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_whole("src"), _whole("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_lines(legacy_isa) == _non_addi_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
