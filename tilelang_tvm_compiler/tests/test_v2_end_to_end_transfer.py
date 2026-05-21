"""End-to-end v2 tests for transfer ops:
``copy_v_to_v`` / ``v_fp_transfer_slice_v_to_fp`` /
``v_fp_transfer_slice_fp_to_v``.
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
_GP_RE = re.compile(r"\bgp\d+\b")


def _non_addi_lines(isa: str):
    """Non-S_ADDI ISA lines with gpN → gpX canonicalisation."""
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


def _vram(name, addr, shape=None):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=shape or (1, MLEN, 1, MLEN), dtype="float16",
        address=addr, cluster_dim=None, tile_layout=None,
    )


def _fpram(name, addr, shape=(MLEN,)):
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM,
        shape=shape, dtype="float16", address=addr,
    )


def _whole_vram(name):
    return _hlir.VramRegion(
        parent=name, starts=(0, 0, 0, 0), extents=(1, MLEN, 1, MLEN),
    )


def _row_vram(name, row):
    return _hlir.VramRegion(
        parent=name, starts=(0, row, 0, 0), extents=(1, 1, 1, MLEN),
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


def test_copy_v_to_v_v2_matches_legacy():
    hlir = _hlir.HLIRModule(
        name="copy_v_to_v",
        buffers={
            "src": _vram("src", 0),
            "dst": _vram("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind="copy_v_to_v",
            buffer_args=[_whole_vram("src"), _whole_vram("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_lines(legacy_isa) == _non_addi_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", [
    "v_fp_transfer_slice_v_to_fp",
    "v_fp_transfer_slice_fp_to_v",
])
def test_v_fp_transfer_slice_v2_matches_legacy(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "v": _vram("v", 0),
            "fp": _fpram("fp", 8192),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row_vram("v", row=0)],
            scalar_args=[
                _hlir.BufferElement(buffer="fp", indices=(0,)),
            ],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _non_addi_lines(legacy_isa) == _non_addi_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
