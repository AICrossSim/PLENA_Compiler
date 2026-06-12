"""End-to-end v2 tests for the general ``matmul`` op.

5-level nested unroll: (m, n_mlen, oc, orow, k). K folds into the
systolic-array accumulator → K_tiles M_MM/M_TMM followed by one
M_MM_WO per output BLEN×BLEN tile. transpose_b is inferred from the
B-region axis order (b_N_axis < b_K_axis).

Structural comparison: same M_MM/M_TMM and M_MM_WO count + order.
Scalar address-arithmetic stripped (legacy reuses 7 GPs with per-iter
S_ADDI bumps; v2 builds fresh PrimExpr chains).
"""

import re

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
_GP_RE = re.compile(r"\bgp\d+\b")
_ADDR_SETUP = frozenset({
    "S_ADDI_INT", "S_SLLI_INT", "S_SRLI_INT", "S_LUI_INT",
    "S_ADD_INT", "S_SUB_INT", "S_MUL_INT",
})


def _strip(isa: str):
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


def _v2_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    pre = PreIsaPassV2(shim).run(hlir)
    fn = p2m.convert(pre, shim)
    mir.verify(fn)
    return m2i.emit(fn, shim)


def _legacy_emit(hlir):
    shim = make_shim(mlen=MLEN, blen=BLEN, btmm_lane_count=4, btmm_hlen=HLEN)
    return IsaEmitterPass(shim).run(hlir)


def _count(isa, mnem):
    return sum(
        1 for ln in isa.split("\n") if ln.strip().startswith(mnem + " ")
    )


def test_matmul_mlen_square_v2_matches_legacy():
    """(MLEN, MLEN) @ (MLEN, MLEN) → (MLEN, MLEN); B as (K, N) row-major.

    M_tiles=K_tiles=N_mlen_tiles=1; tiles_per_n_mlen=16; tiles_per_mlen=16.
    Total M_MM pairs = 1*1*16*16*1 = 256.
    """
    a = _vram("a", 0, (1, MLEN, 1, MLEN))
    b = _mram("b", 4096, (1, MLEN, 1, MLEN))
    c = _vram("c", 8192, (1, MLEN, 1, MLEN))
    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="a", starts=(0, 0, 0, 0),
                             extents=(1, MLEN, 1, MLEN)),
            _hlir.MramRegion(parent="b", starts=(0, 0, 0, 0),
                             extents=(1, MLEN, 1, MLEN)),
            _hlir.VramRegion(parent="c", starts=(0, 0, 0, 0),
                             extents=(1, MLEN, 1, MLEN)),
        ],
        # a axes (B, M, _, K); b axes (B, K, _, N); c axes (B, M, _, N).
        scalar_args=[
            ("_", "M", "_", "K"),
            ("_", "K", "_", "N"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_square"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_square",
        buffers={"a": a, "b": b, "c": c},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    # tiles_per_mlen^2 = 256 (M_tiles=K_tiles=N_mlen_tiles=1)
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == 256
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == 256


def test_matmul_2mlen_x_2mlen_v2_matches_legacy():
    """(2*MLEN, MLEN) @ (MLEN, 2*MLEN) → (2*MLEN, 2*MLEN).

    M_tiles=2, K_tiles=1, N_mlen_tiles=2 → 4 mlen-tile output blocks
    × 16 oc × 16 orow × 1 k = 1024 M_MMs.
    """
    M = 2 * MLEN
    N = 2 * MLEN
    K = MLEN
    a = _vram("a", 0, (1, M, 1, K))
    b = _mram("b", 4096, (1, K, 1, N))
    c = _vram("c", 8192, (1, M, 1, N))
    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="a", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, K)),
            _hlir.MramRegion(parent="b", starts=(0, 0, 0, 0),
                             extents=(1, K, 1, N)),
            _hlir.VramRegion(parent="c", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, N)),
        ],
        scalar_args=[
            ("_", "M", "_", "K"),
            ("_", "K", "_", "N"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_2m_x_2n"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_2m_x_2n",
        buffers={"a": a, "b": b, "c": c},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    expected = 2 * 2 * 16 * 16 * 1
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == expected
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected


def test_matmul_k_accumulation_v2_matches_legacy():
    """K_tiles=2: each output BLEN×BLEN tile gets 2 M_MM issuances
    feeding one M_MM_WO (K folds into systolic accumulator)."""
    M = MLEN
    K = 2 * MLEN
    N = MLEN
    a = _vram("a", 0, (1, M, 1, K))
    b = _mram("b", 4096, (1, K, 1, N))
    c = _vram("c", 8192, (1, M, 1, N))
    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="a", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, K)),
            _hlir.MramRegion(parent="b", starts=(0, 0, 0, 0),
                             extents=(1, K, 1, N)),
            _hlir.VramRegion(parent="c", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, N)),
        ],
        scalar_args=[
            ("_", "M", "_", "K"),
            ("_", "K", "_", "N"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_K2"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_K2",
        buffers={"a": a, "b": b, "c": c},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    expected_mm = 1 * 1 * 16 * 16 * 2  # K_tiles=2
    expected_wo = 1 * 1 * 16 * 16 * 1  # one per (oc, orow)
    assert _count(legacy, "M_MM") == _count(new, "M_MM") == expected_mm
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected_wo


def test_matmul_transpose_a_v2_emits_m_tmm_a():
    """A as (K, M) — transpose_a inferred from a_K_axis < a_M_axis.
    Should emit M_TMM_A, not M_MM (the backward dW = dY^T·X / dQ = dS·K
    case where the contracted axis sits on the VRAM operand). v2-only:
    the legacy IsaEmitterPass has no transpose-A path."""
    M = MLEN
    K = MLEN
    N = MLEN
    # A physical shape with K before M (axis 1 = K, axis 3 = M).
    a = _vram("a", 0, (1, K, 1, M))
    b = _mram("b", 4096, (1, K, 1, N))
    c = _vram("c", 8192, (1, M, 1, N))
    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="a", starts=(0, 0, 0, 0),
                             extents=(1, K, 1, M)),
            _hlir.MramRegion(parent="b", starts=(0, 0, 0, 0),
                             extents=(1, K, 1, N)),
            _hlir.VramRegion(parent="c", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, N)),
        ],
        scalar_args=[
            ("_", "K", "_", "M"),
            ("_", "K", "_", "N"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_transpose_a"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_transpose_a",
        buffers={"a": a, "b": b, "c": c},
        ops=[op], param_names=[],
    )
    new = _v2_emit(hlir)
    # v2 keeps the (m, oc, orow, k) nest rolled, so one M_TMM_A sits in
    # the k-loop body (vs the legacy 256 physical copies). The task
    # comment must carry the transpose_a tag, and there must be exactly
    # one M_TMM_A / one M_MM_WO drain in the rolled body.
    assert "transpose_a" in new, f"\nv2:\n{new}"
    assert _count(new, "M_TMM_A") == 1, f"\nv2:\n{new}"
    assert _count(new, "M_MM_WO") == 1
    # No plain M_MM and no M_TMM (B-side transpose).
    assert _count(new, "M_MM") == 0
    assert _count(new, "M_TMM") == 0


def test_matmul_transpose_b_v2_matches_legacy():
    """B as (N, K) — transpose_b inferred from b_N_axis < b_K_axis.
    Should emit M_TMM, not M_MM."""
    M = MLEN
    K = MLEN
    N = MLEN
    a = _vram("a", 0, (1, M, 1, K))
    # B physical shape with N before K (axis 1 = N, axis 3 = K)
    b = _mram("b", 4096, (1, N, 1, K))
    c = _vram("c", 8192, (1, M, 1, N))
    op = _hlir.Op(
        kind="matmul",
        buffer_args=[
            _hlir.VramRegion(parent="a", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, K)),
            _hlir.MramRegion(parent="b", starts=(0, 0, 0, 0),
                             extents=(1, N, 1, K)),
            _hlir.VramRegion(parent="c", starts=(0, 0, 0, 0),
                             extents=(1, M, 1, N)),
        ],
        scalar_args=[
            ("_", "M", "_", "K"),
            ("_", "N", "_", "K"),
            ("_", "M", "_", "N"),
        ],
        annotations={"intrinsic": "matmul_transpose_b"},
    )
    hlir = _hlir.HLIRModule(
        name="matmul_transpose_b",
        buffers={"a": a, "b": b, "c": c},
        ops=[op], param_names=[],
    )
    legacy = _legacy_emit(hlir)
    new = _v2_emit(hlir)
    assert _strip(legacy) == _strip(new), (
        f"\nlegacy:\n{legacy}\nv2:\n{new}"
    )
    expected = 1 * 1 * 16 * 16 * 1
    assert _count(legacy, "M_TMM") == _count(new, "M_TMM") == expected
    assert _count(legacy, "M_MM_WO") == _count(new, "M_MM_WO") == expected
    # No M_MM, only M_TMM.
    assert _count(new, "M_MM") == 0
