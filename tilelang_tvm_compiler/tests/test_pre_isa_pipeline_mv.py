"""Byte-equal verification for the migrated ``mv`` (M_MV + M_MV_WO).

Exercises emit_mv's tiles-loop unroll and the destructive in-place
``S_ADDI_INT gp{m}, gp{m}, blen`` stride bump between iterations.
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
HLEN = 16
BLEN = 4


def _instr_lines(text):
    return [
        ln.strip() for ln in text.split("\n")
        if ln.strip() and not ln.strip().startswith(";")
    ]


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


def test_mv_byte_equal():
    """Single-head MV: lhs (1, hlen) vector @ rhs (mlen, hlen) matrix
    -> dst (1, hlen). emit_mv runs tiles = hlen/blen = 4 iterations."""
    lhs = _vram("lhs", 0, (1, HLEN))
    rhs = _mram("rhs", 4096, (MLEN, HLEN))
    dst = _vram("dst", 8192, (1, HLEN))

    op = _hlir.Op(
        kind="mv",
        buffer_args=[
            _hlir.VramRegion(parent="lhs", starts=(0, 0),
                             extents=(1, HLEN)),
            _hlir.MramRegion(parent="rhs", starts=(0, 0),
                             extents=(MLEN, HLEN)),
            _hlir.VramRegion(parent="dst", starts=(0, 0),
                             extents=(1, HLEN)),
        ],
        scalar_args=[],
        annotations={"intrinsic": "mv_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mv_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )
