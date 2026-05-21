"""Byte-equal verification for the migrated ``btmm`` (Q @ K^T packed-
head matmul).

Legacy path: isa_pass._emit_btmm -> ISAEmitter.emit_btmm + .emit_btmm_wo.
New path: PreIsaPass._emit_btmm decomposes those two emit_* helpers
into a stream of PreIsaOps (preloads + M_BTMM + M_BMM_WO) so the
operand PrimExprs are visible to the optimiser.
"""

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


def _vram_buf(name: str, addr: int, shape) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM, shape=shape,
        dtype="float16", address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _mram_buf(name: str, addr: int, shape) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.MRAM, shape=shape,
        dtype="float16", address=addr,
    )


def test_btmm_byte_equal():
    """Minimal btmm: lhs(M, gh, hlen) @ rhs(M, gh, hlen) -> dst(M, gh, M).
    With M=mlen=64, gh=lane_count=4, hlen=16:
      tile_elems = mlen*mlen = 4096
      dst.num_elements = 64*4*64 = 16384 -> tile_count = 16384 / 4096 = 4.
    """
    LANE = 4   # btmm_lane_count
    lhs = _vram_buf("lhs", 0, (MLEN, LANE, HLEN))
    rhs = _mram_buf("rhs", 4096, (MLEN, LANE, HLEN))
    dst = _vram_buf("dst", 8192, (MLEN, LANE, MLEN))

    op = _hlir.Op(
        kind="btmm",
        buffer_args=[
            _hlir.VramRegion(
                parent="lhs", starts=(0, 0, 0), extents=(MLEN, LANE, HLEN),
            ),
            _hlir.MramRegion(
                parent="rhs", starts=(0, 0, 0), extents=(MLEN, LANE, HLEN),
            ),
            _hlir.VramRegion(
                parent="dst", starts=(0, 0, 0), extents=(MLEN, LANE, MLEN),
            ),
        ],
        scalar_args=[],
        annotations={"intrinsic": "btmm_test"},
    )
    hlir = _hlir.HLIRModule(
        name="btmm_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op],
        param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=4, btmm_lane_count=LANE, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )
