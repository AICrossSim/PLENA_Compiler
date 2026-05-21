"""Byte-equal verification for the migrated ``row_*_at`` family:
``row_reduce_max_at`` / ``row_reduce_sum_at`` /
``row_exp`` / ``row_sub_fp`` / ``row_mul_fp`` / ``row_add_fp``.

These ops:
  * walk d_tiles in an unrolled loop (n_d_tiles HW ops per call);
  * use a destructive in-place stride-bump pattern on the cached
    src/dst GPs;
  * optionally arm/reset the V_MASK register for packed-head buffers.

The PreIsaIR migration exercises _PRELOAD_ADDR, _BUMP_CACHED_GP,
group_id scoping, and the cached-GP slot kind all at once.
"""

import pytest

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


def _vram_buf(name: str, addr: int) -> _hlir.Buffer:
    """A simple cluster-less 1×MLEN×1×MLEN VRAM buffer with
    tile_layout=None (single-tile path inside
    _logical_to_phys_row_offset / _tile_layout_strides)."""
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, 1, MLEN), dtype="float16",
        address=addr,
        # Single-inner-tile case: d_tiles=1, no packed-head mask.
        tile_layout=_hlir.TileLayout(
            logical_b=1, logical_s=MLEN, logical_h=1, logical_d=MLEN,
            d_tiles=1, s_tiles=1, h_groups=1,
            mlen=MLEN, lane_count=1, d_inner=MLEN,
        ),
        cluster_dim=None,
    )


def _fpram_slot(name: str, addr: int) -> _hlir.Buffer:
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM,
        shape=(1,), dtype="float16", address=addr,
    )


def _row_region(name: str, row: int) -> _hlir.VramRegion:
    """One logical row of VRAM: extents (1,1,1,MLEN), starts pick the
    row index in S."""
    return _hlir.VramRegion(
        parent=name, starts=(0, row, 0, 0), extents=(1, 1, 1, MLEN),
    )


def _byte_equal(hlir: _hlir.HLIRModule) -> None:
    shim_legacy = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=HLEN)
    legacy_isa = IsaEmitterPass(shim_legacy).run(hlir)

    shim_new = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPass(shim_new).run(hlir)
    new_isa = BackendEmit(shim_new).run(pre)

    assert _instr_lines(legacy_isa) == _instr_lines(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nnew:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", ["row_reduce_max_at", "row_reduce_sum_at"])
def test_row_reduce_byte_equal(kind):
    """Reduce: src VRAM row → FPRAM scalar accumulator."""
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram_buf("src", 0),
            "fp": _fpram_slot("fp", 1024),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row_region("src", row=5)],
            scalar_args=[_hlir.BufferElement(buffer="fp", indices=(0,))],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


def test_row_exp_byte_equal():
    hlir = _hlir.HLIRModule(
        name="row_exp",
        buffers={
            "src": _vram_buf("src", 0),
            "dst": _vram_buf("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind="row_exp",
            buffer_args=[_row_region("src", row=2), _row_region("dst", row=2)],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


@pytest.mark.parametrize("kind", ["row_add_fp", "row_sub_fp", "row_mul_fp"])
def test_row_binary_fp_byte_equal(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram_buf("src", 0),
            "dst": _vram_buf("dst", MLEN * MLEN),
            "fp": _fpram_slot("fp", 2048),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row_region("src", row=3), _row_region("dst", row=3)],
            scalar_args=[_hlir.BufferElement(buffer="fp", indices=(0,))],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


# ---------- multi-d_tile coverage (exercises _BUMP_CACHED_GP loop) ----------

def _vram_buf_wide(name: str, addr: int) -> _hlir.Buffer:
    """A 1×MLEN×1×(2*MLEN) VRAM buffer with d_tiles=2 — exercises the
    stride-bump unroll loop in row_*_at."""
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, 1, 2 * MLEN), dtype="float16",
        address=addr,
        tile_layout=_hlir.TileLayout(
            logical_b=1, logical_s=MLEN, logical_h=1, logical_d=2 * MLEN,
            d_tiles=2, s_tiles=1, h_groups=1,
            mlen=MLEN, lane_count=1, d_inner=MLEN,
        ),
        cluster_dim=None,
    )


def _row_region_wide(name: str, row: int) -> _hlir.VramRegion:
    return _hlir.VramRegion(
        parent=name, starts=(0, row, 0, 0), extents=(1, 1, 1, 2 * MLEN),
    )


def test_row_exp_multi_d_tile_byte_equal():
    """d_tiles=2 means the d_tile unroll loop fires _BUMP_CACHED_GP
    once between the two V_EXP_V ops."""
    hlir = _hlir.HLIRModule(
        name="row_exp_wide",
        buffers={
            "src": _vram_buf_wide("src", 0),
            "dst": _vram_buf_wide("dst", 4096),
        },
        ops=[_hlir.Op(
            kind="row_exp",
            buffer_args=[
                _row_region_wide("src", row=1),
                _row_region_wide("dst", row=1),
            ],
            scalar_args=[],
        )],
        param_names=[],
    )
    _byte_equal(hlir)


def test_row_mul_fp_multi_d_tile_byte_equal():
    hlir = _hlir.HLIRModule(
        name="row_mul_wide",
        buffers={
            "src": _vram_buf_wide("src", 0),
            "dst": _vram_buf_wide("dst", 4096),
            "fp": _fpram_slot("fp", 8192),
        },
        ops=[_hlir.Op(
            kind="row_mul_fp",
            buffer_args=[
                _row_region_wide("src", row=2),
                _row_region_wide("dst", row=2),
            ],
            scalar_args=[_hlir.BufferElement(buffer="fp", indices=(0,))],
        )],
        param_names=[],
    )
    _byte_equal(hlir)
