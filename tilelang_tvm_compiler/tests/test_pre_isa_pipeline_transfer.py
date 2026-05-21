"""Byte-equal verification for the migrated transfer ops:
``copy_v_to_v`` / ``v_fp_transfer_slice_v_to_fp``
/ ``v_fp_transfer_slice_fp_to_v``.
"""

import pytest

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
HLEN = 16


def _instr_lines(text: str):
    return [
        ln.strip()
        for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


def _vram_buf(name: str, addr: int) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, 1, MLEN), dtype="float16",
        address=addr, cluster_dim=None, tile_layout=None,
    )


def _fpram_buf(name: str, addr: int, shape=(MLEN,)) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM,
        shape=shape, dtype="float16", address=addr,
    )


def _whole(name: str) -> _hlir.VramRegion:
    return _hlir.VramRegion(
        parent=name, starts=(0, 0, 0, 0), extents=(1, MLEN, 1, MLEN),
    )


def _byte_equal(hlir: _hlir.HLIRModule) -> None:
    shim_legacy = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )


def test_copy_v_to_v_byte_equal():
    hlir = _hlir.HLIRModule(
        name="copy_v_to_v",
        buffers={
            "src": _vram_buf("src", 0),
            "dst": _vram_buf("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind="copy_v_to_v",
            buffer_args=[_whole("src"), _whole("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


@pytest.mark.parametrize("kind", [
    "v_fp_transfer_slice_v_to_fp", "v_fp_transfer_slice_fp_to_v",
])
def test_v_fp_transfer_slice_byte_equal(kind):
    """1-row VRAM region + 1-element FPRAM slot. ``_vram_region_iter_chunks``
    yields one chunk → exactly one S_MAP_*."""
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "v": _vram_buf("v", 0),
            "fp": _fpram_buf("fp", 8192, shape=(MLEN,)),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_hlir.VramRegion(
                parent="v", starts=(0, 0, 0, 0),
                extents=(1, 1, 1, MLEN),
            )],
            scalar_args=[_hlir.BufferElement(buffer="fp", indices=(0,))],
        )],
        param_names=[],
    )
    _byte_equal(hlir)
