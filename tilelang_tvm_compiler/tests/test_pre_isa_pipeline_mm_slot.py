"""Structural verification for the migrated ``mm_slot`` (slot matmul).

Legacy ``emit_slot_matmul`` runs ``tiles_per_slot * tiles_per_mlen``
(M_MM, M_MM_WO) pairs. PreIsaIR migration: outer oc loop is
per_iter, inner t loop is shared-scope so the destructive
act/out S_ADDI bumps carry across t-iters.

Static-offset path only — dynamic offset migration is a TODO.
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
BLEN_L = 4
HLEN = 16


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


def _mm_pairs(isa: str):
    mm = sum(1 for ln in isa.split("\n") if ln.strip().startswith("M_MM "))
    wo = sum(1 for ln in isa.split("\n") if ln.strip().startswith("M_MM_WO "))
    return mm, wo


def test_mm_slot_structural_equal():
    """LHS mlen*mlen tile sliced from a 2*mlen*mlen buffer at offset 0;
    RHS / DST are 64*256 wide tensors with rhs_col_offset=64,
    dst_col_offset=128, col_count=16.
    Pairs: tiles_per_slot * tiles_per_mlen = 4 * 16 = 64."""
    lhs = _vram("lhs", 0, (MLEN * 2, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, 256))
    dst = _vram("dst", 8192, (MLEN, 256))

    op = _hlir.Op(
        kind="mm_slot",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[
            0,    # lhs_row_offset
            64,   # rhs_col_offset
            128,  # dst_col_offset
            HLEN,   # col_count = 16, divisible by blen=4
        ],
        annotations={"intrinsic": "mm_slot_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_slot_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN_L, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN_L, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    legacy_mm, legacy_wo = _mm_pairs(legacy_isa)
    new_mm, new_wo = _mm_pairs(new_isa)
    assert legacy_mm == new_mm, (
        f"M_MM count differs: legacy={legacy_mm} new={new_mm}"
    )
    assert legacy_wo == new_wo, (
        f"M_MM_WO count differs: legacy={legacy_wo} new={new_wo}"
    )
    # tiles_per_slot * tiles_per_mlen = 4 * 16 = 64.
    assert new_mm == 64
    assert new_wo == 64
