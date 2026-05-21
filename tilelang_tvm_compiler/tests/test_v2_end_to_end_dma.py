"""End-to-end v2 tests for DMA family — ``dma_h2v``, ``dma_h2m``,
``dma_v2h``.

Each one walks an HBM buffer's tile grid via the same
``_iter_tile_offsets`` legacy uses. Per-tile body is a fixed sequence
of ``C_SET_ADDR_REG`` + ``C_SET_SCALE_REG`` + ``C_SET_STRIDE_REG`` +
one or more ``H_PREFETCH_V`` / ``H_PREFETCH_M`` / ``H_STORE_V``.

Structural verification: equal counts of HW HBM ops on both sides
(legacy → v2). Address-setup S_ADDI_INT and similar scalar arithmetic
are skipped (different in-place vs SSA-rebuild styles).
"""

import re

import pytest

from tilelang_tvm_compiler import hlir as _hlir
from tilelang_tvm_compiler import scope as _scope
from tilelang_tvm_compiler import mir
from tilelang_tvm_compiler import pre_isa_to_mir as p2m
from tilelang_tvm_compiler import mir_to_isa as m2i
from tilelang_tvm_compiler.isa_pass import IsaEmitterPass
from tilelang_tvm_compiler.pre_isa_pass_v2 import PreIsaPassV2
from tilelang_tvm_compiler.program_shim import make_shim


MLEN = 64
BLEN = 4
HLEN = 16


def _counts(isa: str):
    """Return (H_PREFETCH_V, H_PREFETCH_M, H_STORE_V) line counts."""
    pv = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_V"))
    pm = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_M"))
    sv = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_STORE_V"))
    return pv, pm, sv


def _set_counts(isa: str):
    sa = sum(1 for ln in isa.split("\n") if ln.strip().startswith("C_SET_ADDR_REG"))
    sc = sum(1 for ln in isa.split("\n") if ln.strip().startswith("C_SET_SCALE_REG"))
    st = sum(1 for ln in isa.split("\n") if ln.strip().startswith("C_SET_STRIDE_REG"))
    return sa, sc, st


def _hbm(name, addr, num_elements,
         hbm_stride=MLEN, hbm_scale_size=MLEN * MLEN, hbm_offset=0):
    return _hlir.Buffer(
        name=name, scope=_scope.HBM,
        shape=(num_elements,), dtype="float16",
        address=addr,
        hbm_offset=hbm_offset,
        hbm_stride=hbm_stride,
        hbm_scale_size=hbm_scale_size,
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


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    return IsaEmitterPass(shim).run(hlir)


def test_dma_h2m_single_tile_v2():
    """Single-tile HBM → MRAM. One H_PREFETCH_M, one addr_reg setup."""
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)

    l_pv, l_pm, l_sv = _counts(legacy)
    n_pv, n_pm, n_sv = _counts(new)
    assert l_pm == n_pm == 1, (
        f"H_PREFETCH_M legacy={l_pm} v2={n_pm}\nv2:\n{new}"
    )
    assert n_pv == l_pv == 0
    assert n_sv == l_sv == 0
    # v2 also binds the addr_reg once and sets scale + stride once
    # (then resets at end). Sanity: at least one set of each.
    n_sa, n_sc, n_st = _set_counts(new)
    assert n_sa == 1
    assert n_sc >= 1
    assert n_st >= 1


def test_dma_h2v_single_tile_v2():
    """Single-tile HBM → VRAM. H_PREFETCH_V count = inner_count *
    load_amount_per_hidden. For mlen=64, v_prefetch=1 → 64 prefetches."""
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    l_pv, l_pm, l_sv = _counts(legacy)
    n_pv, n_pm, n_sv = _counts(new)
    assert l_pv == n_pv, (
        f"H_PREFETCH_V legacy={l_pv} v2={n_pv}\nv2:\n{new[:2000]}"
    )
    assert n_pm == l_pm == 0
    assert n_sv == l_sv == 0


def test_dma_v2h_single_tile_v2():
    """Single-tile VRAM → HBM. H_STORE_V count = inner_count *
    store_amount_per_hidden. For mlen=64, v_writeback=4 → 16 stores."""
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    l_pv, l_pm, l_sv = _counts(legacy)
    n_pv, n_pm, n_sv = _counts(new)
    assert l_sv == n_sv, (
        f"H_STORE_V legacy={l_sv} v2={n_sv}\nv2:\n{new[:2000]}"
    )
    assert n_pv == l_pv == 0
    assert n_pm == l_pm == 0


def test_dma_h2v_multi_tile_v2():
    """4-tile HBM (2 rows × 2 cols of mlen×mlen) → VRAM. Expect 4×
    the single-tile H_PREFETCH_V count."""
    rows = 2
    cols = 2
    src = _hbm(
        "src_hbm", 0, MLEN * MLEN * rows * cols,
        hbm_stride=MLEN * cols,
        hbm_scale_size=MLEN * MLEN,
    )
    src.annotations["logical_rows"] = MLEN * rows
    src.annotations["logical_cols"] = MLEN * cols
    src.annotations["row_blocks"] = rows
    src.annotations["col_blocks"] = cols
    # dst VRAM big enough for all tiles.
    dst = _hlir.Buffer(
        name="dst_vram", scope=_scope.VRAM,
        shape=(rows * cols * MLEN, MLEN), dtype="float16",
        address=4096, cluster_dim=None, tile_layout=None,
    )
    op = _hlir.Op(
        kind="dma_h2v",
        buffer_args=["src_hbm", "dst_vram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2v_multi",
        buffers={"src_hbm": src, "dst_vram": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    l_pv, l_pm, l_sv = _counts(legacy)
    n_pv, n_pm, n_sv = _counts(new)
    assert l_pv == n_pv, (
        f"H_PREFETCH_V legacy={l_pv} v2={n_pv}\nv2:\n{new[:2000]}"
    )
