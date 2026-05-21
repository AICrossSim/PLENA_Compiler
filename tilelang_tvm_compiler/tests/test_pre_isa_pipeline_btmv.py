"""Byte-equal verification for the migrated ``btmv``."""

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
        ln.strip() for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


def _vram_buf(name, addr, shape):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM, shape=shape,
        dtype="float16", address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _mram_buf(name, addr, shape):
    return _hlir.Buffer(
        name=name, scope=_scope.MRAM, shape=shape,
        dtype="float16", address=addr,
    )


def test_btmv_byte_equal():
    LANE = 4
    lhs = _vram_buf("lhs", 0, (1, LANE, HLEN))
    rhs = _mram_buf("rhs", 4096, (MLEN, LANE, HLEN))
    dst = _vram_buf("dst", 8192, (1, LANE, MLEN))

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
        name="btmv_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )
