"""The online softmax's live scalar state must fit the scalar FP SRAM.

Flash attention folds every key tile into one running state, so the three
scalars each query row of each broadcast head carries stay live for the whole
key sweep. Sweeping all MLEN rows of all MLEN/HLEN broadcast heads at once makes
that state grow as ``3 * MLEN * (MLEN / HLEN)``, which is unbounded in the array
geometry and exceeds the FP SRAM the RTL provides. Tiling the query rows caps
it at ``3 * tile * broadcast_heads + constants`` whatever MLEN and the cache
length are.

The bound is read from the RTL rather than restated here, so widening the SRAM
in ``configuration.svh`` moves this test with it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from asm_templates.flashattn.overall import (
    FP_SRAM_CONSTANT_SLOTS,
    softmax_row_tile,
    softmax_state_slots,
)

RTL_CONFIGURATION = (
    Path(__file__).resolve().parents[4]
    / "PLENA_RTL"
    / "src"
    / "definitions"
    / "configuration.svh"
)

# The geometry the RTL is parameterised at in configuration.svh. Its untiled
# state is 6 + 3*16*2 = 102 slots, which already fits, so tiling is a no-op
# here and the bound below is not what makes this geometry work.
RTL_DEFAULT_GEOMETRY = (16, 8)

# (MLEN, HLEN) points on the geometry ladder that the decode work reports.
# Broadcast heads are MLEN / HLEN. Every one of these overflows untiled.
GEOMETRIES = [(64, 16), (128, 16), (256, 32), (512, 64), (1024, 128)]


def rtl_fp_sram_depth() -> int:
    match = re.search(
        r"localparam\s+FP_SRAM_DEPTH\s*=\s*(\d+)\s*;",
        RTL_CONFIGURATION.read_text(),
    )
    if match is None:
        raise AssertionError(f"FP_SRAM_DEPTH not declared in {RTL_CONFIGURATION}")
    return int(match.group(1))


@pytest.mark.skipif(
    not RTL_CONFIGURATION.is_file(), reason="RTL checkout not present"
)
@pytest.mark.parametrize("mlen,hlen", GEOMETRIES)
def test_tiled_softmax_state_fits_the_rtl_fp_sram(mlen: int, hlen: int) -> None:
    depth = rtl_fp_sram_depth()
    heads = mlen // hlen
    tile = softmax_row_tile(mlen, heads, depth)
    assert softmax_state_slots(tile, heads) <= depth


@pytest.mark.skipif(
    not RTL_CONFIGURATION.is_file(), reason="RTL checkout not present"
)
@pytest.mark.parametrize("mlen,hlen", GEOMETRIES)
def test_untiled_softmax_state_is_what_overflows(mlen: int, hlen: int) -> None:
    """The tiling is load-bearing at every geometry the work reports."""
    depth = rtl_fp_sram_depth()
    heads = mlen // hlen
    assert softmax_state_slots(mlen, heads) > depth


@pytest.mark.skipif(
    not RTL_CONFIGURATION.is_file(), reason="RTL checkout not present"
)
def test_the_rtl_default_geometry_fits_without_tiling() -> None:
    """The overflow is a property of the reported geometries, not of every one.

    At the geometry the RTL is actually parameterised at, the untiled state
    already fits and the row tile covers all the query rows, so the claim is
    that the lowering outgrows the SRAM as MLEN scales — not that it never fits.
    """
    depth = rtl_fp_sram_depth()
    mlen, hlen = RTL_DEFAULT_GEOMETRY
    heads = mlen // hlen
    assert softmax_state_slots(mlen, heads) <= depth
    assert softmax_row_tile(mlen, heads, depth) == mlen


def test_the_row_tile_need_not_divide_the_query_rows() -> None:
    """The emitter sizes its last tile, so the bound is the state, not divisibility.

    Rounding the tile down to a divisor of the row count would leave scalar SRAM
    unused and, for a row count with no convenient factor, collapse the tile to a
    single row: at MLEN=64/HLEN=16 the divisor rule gives 32 rows against the 42
    that fit, so the batch is split into more passes than it needs.
    """
    for mlen, hlen in GEOMETRIES:
        heads = mlen // hlen
        tile = softmax_row_tile(mlen, heads, 512)
        assert 1 <= tile <= mlen
        assert softmax_state_slots(tile, heads) <= 512
    assert softmax_row_tile(64, 4, 512) == 42


def test_row_tile_is_the_largest_that_fits() -> None:
    """A smaller tile than necessary splits the batch into more passes."""
    for mlen, hlen in GEOMETRIES:
        heads = mlen // hlen
        tile = softmax_row_tile(mlen, heads, 512)
        if tile == mlen:
            continue
        larger = next(
            (c for c in range(tile + 1, mlen + 1) if mlen % c == 0), None
        )
        assert larger is not None
        assert softmax_state_slots(larger, heads) > 512


def test_state_is_independent_of_the_cache_length() -> None:
    """The running state is per query row, so it cannot grow with the cache."""
    heads = 4
    tile = softmax_row_tile(64, heads, 512)
    assert all(
        softmax_state_slots(tile, heads) == softmax_state_slots(tile, heads)
        for _ in (128, 256, 512, 1024)
    )
    assert softmax_state_slots(tile, heads) <= 512


def test_a_row_that_cannot_fit_is_rejected() -> None:
    """One query row of every head must fit, or no tiling exists."""
    with pytest.raises(ValueError):
        softmax_row_tile(1024, 8, FP_SRAM_CONSTANT_SLOTS + 3 * 8 - 1)


def test_qkt_scores_only_the_row_tile_it_sweeps() -> None:
    """A row tile must score its own query rows, not the whole MLEN tile.

    Each row tile carries its own softmax state over the whole key sweep, so a
    QK^T that covers all MLEN query rows recomputes every other tile's rows once
    per tile. The query-block trip count is what makes the work scale with the
    tile rather than with the number of tiles.
    """
    from asm_templates.flashattn import flash_attn_asm

    mlen, blen, hlen, heads = 64, 4, 16, 4
    # An FP SRAM small enough to force two row tiles out of the MLEN rows.
    fp_sram_depth = FP_SRAM_CONSTANT_SLOTS + 3 * (mlen // 2) * heads

    row_tile = softmax_row_tile(mlen, heads, fp_sram_depth)
    assert row_tile < mlen, "this geometry must actually tile for the test to bite"

    code = flash_attn_asm(
        mlen=mlen, vlen=mlen, blen=blen, batch=1, hq=heads, hkv=1, d=hlen,
        q_len=mlen, kv_len=2 * mlen,
        alive_registers_int=list(range(1, 16)),
        alive_registers_fp=list(range(1, 8)),
        vector_sram_base_address=0, fp_sram_start_address=6,
        k_base_hbm_offset_reg=0, v_base_hbm_offset_reg=1,
        broadcast_amount=heads,
        scratch_base_address=1 << 16, output_base_address=1 << 18,
        fp_sram_depth=fp_sram_depth,
    )

    lines = [line.strip() for line in code.splitlines()]
    query_blocks = []
    for index, line in enumerate(lines):
        if not line.startswith("M_BTMM"):
            continue
        # Walk back to the two enclosing loops; the outer one counts query blocks.
        trips = []
        for previous in reversed(lines[:index]):
            if previous.startswith("C_LOOP_START"):
                trips.append(int(previous.rsplit(",", 1)[1]))
                if len(trips) == 2:
                    break
        query_blocks.append(trips[1])
    assert query_blocks, "no batched QK^T was emitted"
    expected = -(-row_tile // blen)
    assert set(query_blocks) == {expected}, (
        f"QK^T covers {set(query_blocks)} query blocks; a {row_tile}-row tile "
        f"needs {expected}, and the full tile would need {mlen // blen}"
    )
    assert expected != mlen // blen, "the two outcomes must be distinct"


def test_row_tiles_issue_exactly_ceil_rows_over_blen_query_blocks() -> None:
    """Snapping the tile to a block boundary makes the pass structure free.

    QKt issues `ceil(tile_rows / BLEN)` blocks per pass. With the tile a multiple
    of BLEN, the whole passes contribute `rows / BLEN` blocks and only the final
    short pass rounds, so the total equals the untiled `ceil(rows / BLEN)`. A tile
    that is not a block multiple rounds up once per pass instead, and the analytic
    model -- which charges `ceil(rows / BLEN)` -- would then under-count.
    """
    for blen in (4, 8, 16):
        for depth in (512, 774, 1024, 4096):
            for rows in (9, 64, 100, 229, 256, 1024):
                for heads in (1, 4, 8):
                    tile = softmax_row_tile(rows, heads, depth, blen=blen)
                    if tile > blen:
                        assert tile % blen == 0
                    blocks = sum(
                        -(-min(tile, rows - base) // blen)
                        for base in range(0, rows, tile)
                    )
                    if tile >= blen:
                        assert blocks == -(-rows // blen), (
                            f"blen {blen} depth {depth} rows {rows} heads {heads}: "
                            f"{blocks} blocks against {-(-rows // blen)}"
                        )
