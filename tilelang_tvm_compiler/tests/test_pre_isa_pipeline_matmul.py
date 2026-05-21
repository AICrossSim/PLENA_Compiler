"""Structural verification for the migrated ``matmul`` (the unified
plena.matmul HLIR op that lowers to emit_matmul_general).

This is the most complex matmul handler in the migration: 5-deep
nested LOOP_START loops (m, n_mlen, oc, orow, k) with PrimExpr
addresses referencing the loop vars + hw-shape consts
(MLEN_VAR / BLEN_VAR). The optimiser sees the full address algebra.

We use a coarse structural check (M_MM / M_MM_WO counts equal
legacy) rather than strict byte-equal: the PreIsaIR's per_iter
nested unrolls allocate GPs differently than legacy's pre-pinned
7-reg block, and the legacy emit_matmul_general(unroll_loops=True)
bakes literal addresses into S_ADDI_INTs while PreIsaIR keeps them
symbolic — so the S_ADDI count + form will be different. M_MM /
M_MM_WO counts are the right invariant: same algorithm, same
sub-tile structure.
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
BLEN = 4
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


def _counts(isa: str):
    mm = sum(1 for ln in isa.split("\n") if ln.strip().startswith("M_MM "))
    tmm = sum(1 for ln in isa.split("\n") if ln.strip().startswith("M_TMM "))
    wo = sum(1 for ln in isa.split("\n") if ln.strip().startswith("M_MM_WO "))
    return mm, tmm, wo


def test_matmul_general_structural_equal():
    """Single-tile (M=K=N=mlen) non-transpose matmul. Legacy
    emit_matmul_general(unroll_loops=True) produces 16 (M_MM, M_MM_WO)
    pairs for M_tiles=K_tiles=1, N_mlen_tiles=1, tiles_per_n_mlen=16,
    tiles_per_mlen=16. Per orow: K_tiles M_MMs then 1 M_MM_WO.
    Total: 16 orow * 16 oc * 1 K_tile = 256 M_MMs, 256 M_MM_WOs.
    """
    # 4D BSHD shape: (1, M, 1, K) for A, (1, K, 1, N) for B in row-major
    # (we use rank-4 with the M / K / N roles tagged below).
    M, K, N = MLEN, MLEN, MLEN
    lhs = _vram("lhs", 0, (1, M, 1, K))
    rhs = _mram("rhs", 4096, (1, K, 1, N))
    dst = _vram("dst", 8192, (1, M, 1, N))

    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="lhs", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, K)),
            _hlir.MramRegion(parent="rhs", starts=(0, 0, 0, 0),
                             extents=(1, K, 1, N)),
            _hlir.VramRegion(parent="dst", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, N)),
        ],
        scalar_args=[
            ("_", "M", "_", "K"),
            ("_", "K", "_", "N"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_test"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    legacy_mm, legacy_tmm, legacy_wo = _counts(legacy_isa)
    new_mm, new_tmm, new_wo = _counts(new_isa)
    # Expected pair count: tiles_per_n_mlen * tiles_per_mlen * K_tiles
    # = 16 * 16 * 1 = 256 M_MMs; 16 * 16 = 256 M_MM_WOs.
    expected_mm = 16 * 16 * 1
    expected_wo = 16 * 16
    assert legacy_mm == expected_mm, (
        f"legacy M_MM count: {legacy_mm} (expected {expected_mm})"
    )
    assert new_mm == expected_mm, (
        f"new M_MM count: {new_mm} (expected {expected_mm})"
    )
    assert legacy_wo == expected_wo
    assert new_wo == expected_wo
    assert legacy_tmm == 0 and new_tmm == 0
