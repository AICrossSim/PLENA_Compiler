"""Byte-equal verification for the migrated vector ops:
``v_zero`` / ``v_add`` / ``v_sub`` / ``v_mul`` / ``v_exp`` / ``v_reci``
/ ``v_sqrt``.

These ops all walk VRAM regions chunk-by-chunk (via the legacy
``_vram_region_iter_chunks``) and emit one HW vector instruction per
chunk. The PreIsaIR migration must produce byte-equal HW instruction
sequences to the legacy ``IsaEmitterPass``.
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
    """A simple cluster-less, tile-layout-less 4D VRAM buffer
    (1, MLEN, 1, MLEN). _vram_region_iter_chunks takes the
    row-major-flat code path for these (cluster_dim=None,
    tile_layout=None) — keeps the test setup minimal."""
    return _hlir.Buffer(
        name=name,
        scope=_scope.VRAM,
        shape=(1, MLEN, 1, MLEN),
        dtype="float16",
        address=addr,
        cluster_dim=None,
        tile_layout=None,
    )


def _whole_region(name: str) -> _hlir.VramRegion:
    return _hlir.VramRegion(
        parent=name,
        starts=(0, 0, 0, 0),
        extents=(1, MLEN, 1, MLEN),
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


def test_v_zero_byte_equal():
    hlir = _hlir.HLIRModule(
        name="v_zero_smoke",
        buffers={"dst": _vram_buf("dst", 0)},
        ops=[_hlir.Op(
            kind="v_zero",
            buffer_args=[_whole_region("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


@pytest.mark.parametrize("kind", ["v_add", "v_sub", "v_mul"])
def test_v_binary_byte_equal(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "lhs": _vram_buf("lhs", 0),
            "rhs": _vram_buf("rhs", MLEN * MLEN),
            "dst": _vram_buf("dst", 2 * MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[
                _whole_region("lhs"),
                _whole_region("rhs"),
                _whole_region("dst"),
            ],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


@pytest.mark.parametrize("kind", ["v_exp", "v_reci", "v_sqrt"])
def test_v_unary_byte_equal(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram_buf("src", 0),
            "dst": _vram_buf("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_whole_region("src"), _whole_region("dst")],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)
