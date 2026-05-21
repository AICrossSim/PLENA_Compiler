"""Structural verification for the migrated DMA slice family
(``dma_h2v_slice``, ``dma_h2m_slice``, ``dma_v2h_slice``).

Static-start slices only (dynamic-start path is a known TODO in the
PreIsaPass migration — legacy uses ``hbm_start_offset_reg``).

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


def _hbm_2d(name, addr, rows, cols):
    """Minimal 2D HBM buffer (e.g. an activations tensor)."""
    return _hlir.Buffer(
        name=name, scope=_scope.HBM,
        shape=(rows, cols), dtype="float16",
        address=addr,
        hbm_offset=0, hbm_stride=cols,
        hbm_scale_size=MLEN * MLEN,
    )


def _vram_2d(name, addr, rows=MLEN, cols=MLEN):
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(rows, cols), dtype="float16", address=addr,
        cluster_dim=None, tile_layout=None,
    )


def _mram_2d(name, addr, rows=MLEN, cols=MLEN):
    return _hlir.Buffer(
        name=name, scope=_scope.MRAM,
        shape=(rows, cols), dtype="float16", address=addr,
    )


def test_dma_h2v_slice_structural_equal():
    """Single mlen*mlen tile slice from a wider 2*mlen × 2*mlen HBM
    parent. Grid = 1×1×1×1 (one tile)."""
    parent = _hbm_2d("hbm", 0, 2 * MLEN, 2 * MLEN)
    dst = _vram_2d("vram", 4096)
    sl = _hlir.BufferSlice(
        parent="hbm",
        starts=(MLEN, MLEN),
        extents=(MLEN, MLEN),
    )
    op = _hlir.Op(
        kind="dma_h2v_slice",
        buffer_args=[sl, "vram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2v_slice_smoke",
        buffers={"hbm": parent, "vram": dst},
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


def test_dma_h2m_slice_structural_equal():
    """Single-tile slice into MRAM."""
    parent = _hbm_2d("hbm", 0, 2 * MLEN, 2 * MLEN)
    dst = _mram_2d("mram", 4096)
    sl = _hlir.BufferSlice(
        parent="hbm",
        starts=(0, MLEN),
        extents=(MLEN, MLEN),
    )
    op = _hlir.Op(
        kind="dma_h2m_slice",
        buffer_args=[sl, "mram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2m_slice_smoke",
        buffers={"hbm": parent, "mram": dst},
        ops=[op], param_names=[],
    )

    shim_legacy = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    _, legacy_pm, _ = _counts(legacy_isa)
    _, new_pm, _ = _counts(new_isa)
    assert legacy_pm == new_pm == 1, (
        f"H_PREFETCH_M count differs: legacy={legacy_pm} new={new_pm}"
    )


def test_dma_v2h_slice_structural_equal():
    """VRAM source → slice into wider HBM dst."""
    src = _vram_2d("vram", 0)
    parent = _hbm_2d("hbm", 4096, 2 * MLEN, 2 * MLEN)
    sl = _hlir.BufferSlice(
        parent="hbm",
        starts=(MLEN, 0),
        extents=(MLEN, MLEN),
    )
    op = _hlir.Op(
        kind="dma_v2h_slice",
        buffer_args=["vram", sl],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_v2h_slice_smoke",
        buffers={"vram": src, "hbm": parent},
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
