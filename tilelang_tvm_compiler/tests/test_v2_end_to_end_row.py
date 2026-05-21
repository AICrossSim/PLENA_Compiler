"""End-to-end v2 tests for the row_*_at family.

Six ops:
  * row_reduce_max_at / row_reduce_sum_at  — reduce VRAM row → FPRAM
  * row_exp                                 — unary on VRAM row
  * row_add_fp / row_sub_fp / row_mul_fp    — binary with FP scalar

Single-tile-layout setup (d_tiles=1, no packed-head mask) keeps the
test focused on the basic row-op structure. Multi-d_tile variants
exercise the unroll loop in the v2 path.
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
_GP_RE = re.compile(r"\bgp\d+\b")

# Address-setup mnemonics that legacy and v2 emit in different
# quantities (legacy uses destructive in-place stride bump; v2 builds
# each iter's address from scratch as a fresh SSA chain). Skipped
# during comparison.
_ADDR_SETUP = frozenset({"S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT"})


def _strip(isa: str):
    """Strip lines + canonicalise gp numbers. Skip address-setup
    instructions so the comparison focuses on HW ops + their
    invocation count/order."""
    out = []
    for ln in isa.split("\n"):
        s = ln.strip()
        if not s or s.startswith(";"):
            continue
        head = s.split(None, 1)[0]
        if head in _ADDR_SETUP:
            continue
        out.append(_GP_RE.sub("gpX", s))
    return out


def _vram(name, addr, d_extent=MLEN):
    """1×MLEN×1×d_extent VRAM buffer with a tile_layout that matches
    legacy's single-tile case (d_tiles = ceil(d_extent / MLEN))."""
    d_tiles = (d_extent + MLEN - 1) // MLEN
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, 1, d_extent), dtype="float16",
        address=addr,
        tile_layout=_hlir.TileLayout(
            logical_b=1, logical_s=MLEN, logical_h=1, logical_d=d_extent,
            d_tiles=d_tiles, s_tiles=1, h_groups=1,
            mlen=MLEN, lane_count=1, d_inner=MLEN,
        ),
        cluster_dim=None,
    )


def _fpram(name, addr):
    return _hlir.Buffer(
        name=name, scope=_scope.FPRAM, shape=(1,),
        dtype="float16", address=addr,
    )


def _row(name, row=0, d_extent=MLEN):
    return _hlir.VramRegion(
        parent=name, starts=(0, row, 0, 0), extents=(1, 1, 1, d_extent),
    )


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=16)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=4, btmm_lane_count=4, btmm_hlen=16)
    return IsaEmitterPass(shim).run(hlir)


@pytest.mark.parametrize("kind", ["row_reduce_max_at", "row_reduce_sum_at"])
def test_row_reduce_v2_matches_legacy(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram("src", 0),
            "fp": _fpram("fp", 1024),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row("src", row=2)],
            scalar_args=[
                _hlir.BufferElement(buffer="fp", indices=(0,)),
            ],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


def test_row_exp_v2_matches_legacy():
    hlir = _hlir.HLIRModule(
        name="row_exp",
        buffers={
            "src": _vram("src", 0),
            "dst": _vram("dst", MLEN * MLEN),
        },
        ops=[_hlir.Op(
            kind="row_exp",
            buffer_args=[_row("src", row=1), _row("dst", row=1)],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


@pytest.mark.parametrize("kind", ["row_add_fp", "row_sub_fp", "row_mul_fp"])
def test_row_binary_fp_v2_matches_legacy(kind):
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram("src", 0),
            "dst": _vram("dst", MLEN * MLEN),
            "fp": _fpram("fp", 2048),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row("src", row=3), _row("dst", row=3)],
            scalar_args=[
                _hlir.BufferElement(buffer="fp", indices=(0,)),
            ],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


def test_row_exp_multi_d_tile_v2():
    """d_tiles=2 — exercises the unroll loop."""
    hlir = _hlir.HLIRModule(
        name="row_exp_wide",
        buffers={
            "src": _vram("src", 0, d_extent=2 * MLEN),
            "dst": _vram("dst", 4096, d_extent=2 * MLEN),
        },
        ops=[_hlir.Op(
            kind="row_exp",
            buffer_args=[
                _row("src", row=1, d_extent=2 * MLEN),
                _row("dst", row=1, d_extent=2 * MLEN),
            ],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )


# -----------------------------------------------------------------
# Packed-head (col_pack) masked variants
# -----------------------------------------------------------------
# Layout: shape=(1, MLEN, LANE_COUNT, D_INNER), cluster_dim=2, with
# LANE_COUNT > 1, D_INNER = MLEN // LANE_COUNT. Picking a non-zero lane
# index (= region.starts[2]) forces _logical_to_phys_row_offset to
# emit a real mask_expr (1 << (lane % lane_count)), so the row scalar
# handler must bracket the body with C_SET_V_MASK_REG <mask> / 0.

LANE_COUNT_PACKED = 4
D_INNER_PACKED = MLEN // LANE_COUNT_PACKED  # 16 when MLEN=64


def _vram_packed(name, addr):
    """col_pack buffer: 1×MLEN×LANE_COUNT×D_INNER, cluster on H."""
    return _hlir.Buffer(
        name=name, scope=_scope.VRAM,
        shape=(1, MLEN, LANE_COUNT_PACKED, D_INNER_PACKED),
        dtype="float16", address=addr,
        tile_layout=_hlir.TileLayout(
            logical_b=1, logical_s=MLEN,
            logical_h=LANE_COUNT_PACKED, logical_d=D_INNER_PACKED,
            d_tiles=1, s_tiles=1, h_groups=1,
            mlen=MLEN, lane_count=LANE_COUNT_PACKED,
            d_inner=D_INNER_PACKED,
        ),
        cluster_dim=2,
    )


def _row_packed(name, row, lane):
    """One logical row at (b=0, s=row, h=lane, d=0..D_INNER)."""
    return _hlir.VramRegion(
        parent=name,
        starts=(0, row, lane, 0),
        extents=(1, 1, 1, D_INNER_PACKED),
    )


def _count_mask_set(isa):
    return sum(
        1 for l in isa.split("\n")
        if l.strip().startswith("C_SET_V_MASK_REG")
    )


@pytest.mark.parametrize("kind", ["row_reduce_max_at", "row_reduce_sum_at"])
def test_row_reduce_masked_v2_matches_legacy(kind):
    """Reduce on a col_pack source with lane=1 → packed-head mask."""
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram_packed("src", 0),
            "fp":  _fpram("fp", 1024),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[_row_packed("src", row=2, lane=1)],
            scalar_args=[
                _hlir.BufferElement(buffer="fp", indices=(0,)),
            ],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
    # Sanity: both bracket the body with set/reset.
    assert _count_mask_set(legacy_isa) == _count_mask_set(new_isa) == 2


def test_row_exp_masked_v2_matches_legacy():
    """row_exp with packed-head src+dst, lane=3."""
    hlir = _hlir.HLIRModule(
        name="row_exp_masked",
        buffers={
            "src": _vram_packed("src", 0),
            "dst": _vram_packed("dst", 4096),
        },
        ops=[_hlir.Op(
            kind="row_exp",
            buffer_args=[
                _row_packed("src", row=1, lane=3),
                _row_packed("dst", row=1, lane=3),
            ],
            scalar_args=[],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
    assert _count_mask_set(legacy_isa) == _count_mask_set(new_isa) == 2


@pytest.mark.parametrize("kind", ["row_add_fp", "row_sub_fp", "row_mul_fp"])
def test_row_binary_fp_masked_v2_matches_legacy(kind):
    """V_*_VF with col_pack + lane mask."""
    hlir = _hlir.HLIRModule(
        name=kind,
        buffers={
            "src": _vram_packed("src", 0),
            "dst": _vram_packed("dst", 4096),
            "fp":  _fpram("fp", 2048),
        },
        ops=[_hlir.Op(
            kind=kind,
            buffer_args=[
                _row_packed("src", row=3, lane=2),
                _row_packed("dst", row=3, lane=2),
            ],
            scalar_args=[
                _hlir.BufferElement(buffer="fp", indices=(0,)),
            ],
        )],
        param_names=[],
    )
    legacy_isa = _legacy_emit(hlir)
    new_isa = _v2_emit(hlir)
    assert _strip(legacy_isa) == _strip(new_isa), (
        f"\nlegacy:\n{legacy_isa}\nv2:\n{new_isa}"
    )
    assert _count_mask_set(legacy_isa) == _count_mask_set(new_isa) == 2
