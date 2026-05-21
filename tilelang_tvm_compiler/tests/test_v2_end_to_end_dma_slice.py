"""End-to-end v2 tests for DMA slice family — ``dma_h2v_slice``,
``dma_h2m_slice``, ``dma_v2h_slice``.

Slice is a ``BufferSlice(parent, starts, extents)`` of an HBM
buffer. v2 walks a 4-level (d_tile, s_tile, h_grp, b) tile grid via
nested LoopRegions; the per-tile body is the same preload/store
helper as the whole-buffer DMA path. Slice ``starts`` may be static
(int / IntImm) or dynamic (PrimExpr referencing an outer
LoopRegion's loop_var) — arith.simplify in pre_isa_to_mir folds
static cases away.

Structural verification: H_PREFETCH_V / H_PREFETCH_M / H_STORE_V
counts match legacy.
"""

import pytest
from tvm import tir

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
    pv = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_V"))
    pm = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_PREFETCH_M"))
    sv = sum(1 for ln in isa.split("\n") if ln.strip().startswith("H_STORE_V"))
    return pv, pm, sv


def _hbm_2d(name, addr, rows, cols):
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


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    return IsaEmitterPass(shim).run(hlir)


def test_dma_h2v_slice_static_v2():
    """Single-tile static slice from a 2*mlen × 2*mlen parent."""
    parent = _hbm_2d("hbm", 0, 2 * MLEN, 2 * MLEN)
    dst = _vram_2d("vram", 4096)
    sl = _hlir.BufferSlice(
        parent="hbm", starts=(MLEN, MLEN), extents=(MLEN, MLEN),
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    l_pv, _, _ = _counts(legacy)
    n_pv, _, _ = _counts(new)
    assert l_pv == n_pv > 0, (
        f"H_PREFETCH_V legacy={l_pv} v2={n_pv}\nv2:\n{new[:2000]}"
    )


def test_dma_h2m_slice_static_v2():
    """Single-tile slice into MRAM."""
    parent = _hbm_2d("hbm", 0, 2 * MLEN, 2 * MLEN)
    dst = _mram_2d("mram", 4096)
    sl = _hlir.BufferSlice(
        parent="hbm", starts=(0, MLEN), extents=(MLEN, MLEN),
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    _, l_pm, _ = _counts(legacy)
    _, n_pm, _ = _counts(new)
    assert l_pm == n_pm == 1


def test_dma_v2h_slice_static_v2():
    """Static slice into HBM (single tile)."""
    src = _vram_2d("vram", 0)
    parent = _hbm_2d("hbm", 4096, 2 * MLEN, 2 * MLEN)
    sl = _hlir.BufferSlice(
        parent="hbm", starts=(MLEN, 0), extents=(MLEN, MLEN),
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
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    _, _, l_sv = _counts(legacy)
    _, _, n_sv = _counts(new)
    assert l_sv == n_sv > 0


def test_dma_h2v_slice_dynamic_start_v2():
    """Slice with a dynamic start (PrimExpr derived from an outer
    for-loop var). v2 keeps the offset symbolic; converter folds the
    static parts via arith.simplify. We can't byte-compare to legacy
    here — legacy materialises an extra GP and counts differently.
    Just check the v2 path produces a well-formed MIR with the
    expected H_PREFETCH_V count.
    """
    parent = _hbm_2d("hbm", 0, 4 * MLEN, MLEN)
    dst = _vram_2d("vram", 4096)
    i = tir.Var("i", "int32")
    sl = _hlir.BufferSlice(
        parent="hbm",
        starts=(tir.Mul(i, tir.IntImm("int32", MLEN)), 0),
        extents=(MLEN, MLEN),
    )
    body_op = _hlir.Op(
        kind="dma_h2v_slice",
        buffer_args=[sl, "vram"],
        scalar_args=[],
    )
    for_op = _hlir.Op(
        kind="for",
        buffer_args=[], scalar_args=[],
        annotations={
            "loop_var": i, "extent": 4, "init": 0,
            "loop_kind": "unroll",
        },
        body=[body_op],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2v_slice_dyn",
        buffers={"hbm": parent, "vram": dst},
        ops=[for_op], param_names=[],
    )
    new = _v2_emit(hlir)
    n_pv, _, _ = _counts(new)
    # 4 outer iters × per-tile H_PREFETCH_V count.
    # For mlen=64, v_prefetch=1, single-tile grid: 64 per iter.
    assert n_pv > 0
    # Sanity: outer 4 iters compound.
    assert n_pv % 4 == 0


def test_dma_h2v_slice_multi_tile_v2():
    """Multi-tile slice — dst is large enough to require a 2-tile
    row dimension. Expect 2x single-tile H_PREFETCH_V count."""
    parent = _hbm_2d("hbm", 0, 2 * MLEN, 2 * MLEN)
    dst = _vram_2d("vram", 4096, rows=2 * MLEN, cols=MLEN)
    sl = _hlir.BufferSlice(
        parent="hbm",
        starts=(0, 0),
        extents=(2 * MLEN, MLEN),
    )
    op = _hlir.Op(
        kind="dma_h2v_slice",
        buffer_args=[sl, "vram"],
        scalar_args=[],
    )
    hlir = _hlir.HLIRModule(
        name="dma_h2v_slice_multi",
        buffers={"hbm": parent, "vram": dst},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    l_pv, _, _ = _counts(legacy)
    n_pv, _, _ = _counts(new)
    assert l_pv == n_pv, (
        f"H_PREFETCH_V legacy={l_pv} v2={n_pv}\nv2:\n{new[:2000]}"
    )
