"""Structural verification for the migrated DMA family
(``dma_h2v``, ``dma_h2m``, ``dma_v2h``).

Legacy emit goes through ISAEmitter.emit_load_tile_from_hbm /
emit_hbm_tile_to_mram / emit_store_tile_to_hbm, each emitting a
multi-line setup sequence (S_ADDI_INT × N + C_SET_*_REG + the actual
H_PREFETCH_V / H_PREFETCH_M / H_STORE_V).

PreIsaIR migration: addresses + scale/stride literals become PrimExprs
referencing ``MLEN_VAR``, ``V_PREFETCH_AMOUNT_VAR``,
``V_WRITEBACK_AMOUNT_VAR``. Optimiser sees ``vram_base + idx *
(mlen * v_prefetch_amount)`` etc. Backend lowers to the same HW
mnemonics; GP renumbering is expected.

Coarse structural check: equal counts of H_PREFETCH_V / H_PREFETCH_M
/ H_STORE_V on both sides.
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


def _counts(isa: str):
    p_v = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_V"))
    p_m = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_M"))
    s_v = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_STORE_V"))
    return p_v, p_m, s_v


def _hbm(name, addr, num_elements):
    """Minimal HBM buffer."""
    return _hlir.Buffer(
        name=name, scope=_scope.HBM,
        shape=(num_elements,), dtype="float16",
        address=addr,
        hbm_offset=0, hbm_stride=MLEN,
        hbm_scale_size=MLEN * MLEN,
    )


def _vram(name, addr):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(MLEN, MLEN), dtype="float16", address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _mram(name, addr):
    return _hlir.Buffer(
        name=name, scope=_scope.MRAM,
        shape=(MLEN, MLEN), dtype="float16", address=addr,
    )


def test_dma_h2m_structural_equal():
    """Single-tile HBM -> MRAM. One H_PREFETCH_M emitted."""
    src = _hbm("src_hbm", 0, MLEN * MLEN)
    dst = _mram("dst_mram", 4096)
    op = _hlir.Op(
        kind="dma_h2m",
        buffer_args=["src_hbm", "dst_mram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2m_smoke",
        buffers={"src_hbm": src, "dst_mram": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    legacy_pv, legacy_pm, legacy_sv = _counts(legacy_isa)
    new_pv, new_pm, new_sv = _counts(new_isa)
    assert legacy_pm == new_pm == 1, (
        f"H_PREFETCH_M count differs: legacy={legacy_pm} new={new_pm}"
    )
    assert new_pv == legacy_pv == 0
    assert new_sv == legacy_sv == 0


def test_dma_h2v_structural_equal():
    """Single-tile HBM -> VRAM. Expected H_PREFETCH_V count:
    inner_count * load_amount_per_hidden = ceil(mlen/v_prefetch_amount) * 1
    For mlen=64, v_prefetch_amount=1: 64 H_PREFETCH_Vs."""
    src = _hbm("src_hbm", 0, MLEN * MLEN)
    dst = _vram("dst_vram", 4096)
    op = _hlir.Op(
        kind="dma_h2v",
        buffer_args=["src_hbm", "dst_vram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2v_smoke",
        buffers={"src_hbm": src, "dst_vram": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    legacy_pv, _, _ = _counts(legacy_isa)
    new_pv, _, _ = _counts(new_isa)
    assert legacy_pv == new_pv, (
        f"H_PREFETCH_V count differs: legacy={legacy_pv} new={new_pv}"
    )
    assert legacy_pv > 0


def test_dma_v2h_structural_equal():
    """Single-tile VRAM -> HBM. Expected H_STORE_V count matches
    H_PREFETCH_V count under symmetric mlen/v_writeback_amount setup."""
    src = _vram("src_vram", 0)
    dst = _hbm("dst_hbm", 4096, MLEN * MLEN)
    op = _hlir.Op(
        kind="dma_v2h",
        buffer_args=["src_vram", "dst_hbm"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_v2h_smoke",
        buffers={"src_vram": src, "dst_hbm": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    _, _, legacy_sv = _counts(legacy_isa)
    _, _, new_sv = _counts(new_isa)
    assert legacy_sv == new_sv, (
        f"H_STORE_V count differs: legacy={legacy_sv} new={new_sv}"
    )
    assert legacy_sv > 0
