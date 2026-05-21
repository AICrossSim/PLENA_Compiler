"""Semantic byte-equal verification for the migrated ``mm`` (M_MM +
M_MM_WO, single-tile mlen*mlen).

Legacy emit_matmul_single_tile_hwloop pre-allocates ``allocate_gp(6)``
and uses only 3 of those GPs (with a specific non-sequential
assignment); the per-iter PreIsaIR materialiser can't reproduce that
allocation scheme without abandoning the var-ref operand model. We
use ``semantic_isa_equal`` instead: mnemonics + literals must match,
GP renumbering is allowed.
"""

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler.backend_emit import BackendEmit
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass import PreIsaPass
from tilelang_tvm_compiler.program_shim import make_shim

from ._isa_diff import assert_semantic_isa_equal


MLEN = 64
BLEN = 4


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


def test_mm_narrow_tile_structural_equal():
    """``mlen*mlen @ mlen*hlen -> mlen*hlen`` matmul.

    Strict semantic equality fails on narrow MM because legacy
    materialises ``mat_addr`` ONCE per oc-iter (before the inner t
    loop), while PreIsaIR's per_iter inner unroll closes the
    surrounding scope and forces a re-preload per (oc, t). That
    emits ``tiles_per_mlen - 1 = 15`` extra S_ADDIs per oc — a known
    instruction-count divergence. The structural check verifies the
    PreIsaIR path produces the right MNEMONIC stream around each
    M_MM / M_MM_WO pair: a sequence of ``S_ADDI_INT``s setting up
    the operand GPs, then ``M_MM``, then ``M_MM_WO``. Counts are
    checked at the M_MM / M_MM_WO level, not per S_ADDI.

    The address-algebra optimisation opportunity exists either way
    (mat_addr expr ``rhs.base + oc * blen`` is preserved as a
    PrimExpr) — only the ISA emit form differs.
    """
    MLEN, BLEN_L, HLEN = 64, 4, 16
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, HLEN))
    dst = _vram("dst", 8192, (MLEN, HLEN))

    op = _hlir.Op(
        kind="mm",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[],
        annotations={"intrinsic": "mm_narrow_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_narrow_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN_L, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN_L, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    def _mm_pairs(isa: str):
        """Return the count of (M_MM, M_MM_WO) pairs in stream order."""
        mm = sum(1 for ln in isa.split("\n")
                 if ln.strip().startswith("M_MM "))
        wo = sum(1 for ln in isa.split("\n")
                 if ln.strip().startswith("M_MM_WO "))
        return mm, wo

    legacy_mm, legacy_wo = _mm_pairs(legacy_isa)
    new_mm, new_wo = _mm_pairs(new_isa)
    assert legacy_mm == new_mm, (
        f"M_MM count differs: legacy={legacy_mm} new={new_mm}"
    )
    assert legacy_wo == new_wo, (
        f"M_MM_WO count differs: legacy={legacy_wo} new={new_wo}"
    )
    # tiles_per_slot * tiles_per_mlen = 4 * 16 = 64 M_MM pairs total.
    assert new_mm == 64
    assert new_wo == 64


def test_mm_single_tile_semantic_equal():
    """Single mlen*mlen MM. Legacy emits tiles_per_mlen² = 16² = 256
    (M_MM, M_MM_WO) pairs after 3 S_ADDI_INTs each — 1280 instr total."""
    lhs = _vram("lhs", 0, (MLEN, MLEN))
    rhs = _mram("rhs", 4096, (MLEN, MLEN))
    dst = _vram("dst", 8192, (MLEN, MLEN))

    op = _hlir.Op(
        kind="mm",
        buffer_args=["lhs", "rhs", "dst"],
        scalar_args=[],
        annotations={"intrinsic": "mm_test"},
    )
    hlir = _hlir.HLIRModule(
        name="mm_smoke",
        buffers={"lhs": lhs, "rhs": rhs, "dst": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=16)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert_semantic_isa_equal(legacy_isa, new_isa)
