"""One `M_BTMM` covers a BLEN x BLEN block per broadcast head, not a whole tile.

Measured on SimTop with `PLENA_RTL/tools/testworkloads/utils/
btmm_granularity_probe.py`: a single `M_BTMM` followed by one `M_BMM_WO` commits
`HEAD_COUNT * BLEN` VRAM rows, in `HEAD_COUNT` groups of BLEN rows separated by
MLEN, each row carrying BLEN non-zero elements. The array is partitioned into
`MLEN / HLEN` cores each running a `(BLEN, HLEN) x (HLEN, BLEN)` GEMM, so the
reduction runs over the head dimension inside one issue and the output block is
BLEN x BLEN.

The batched QK^T in `flashattn/qkt.py` therefore covers an `MLEN x MLEN` score
tile with `(MLEN / BLEN)^2` issues. These tests pin both halves — the RTL
constant that sets the granularity, and the loop nest the lowering emits.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from asm_templates.flashattn.qkt import qkt_multiply

RTL_ROOT = Path(__file__).resolve().parents[4] / "PLENA_RTL"
MCU = (
    RTL_ROOT
    / "src"
    / "basic_components"
    / "systolic_gemm_mxint"
    / "rtl"
    / "mxint_systolic_mcu.sv"
)
CONFIGURATION = RTL_ROOT / "src" / "definitions" / "configuration.svh"

# Rows the probe observed one M_BTMM + M_BMM_WO commit, and their positions,
# at the RTL's MLEN=16 BLEN=4 HLEN=8.
MEASURED_ROWS = [0, 1, 2, 3, 16, 17, 18, 19]

MLEN = 64
BLEN = 4
HEAD_DIM = 16

requires_rtl = pytest.mark.skipif(
    not MCU.is_file() or not CONFIGURATION.is_file(),
    reason="RTL checkout not present",
)


def rtl_param(name: str) -> int:
    match = re.search(
        rf"localparam\s+{name}\s*=\s*(\d+)\s*;", CONFIGURATION.read_text()
    )
    if match is None:
        raise AssertionError(f"{name} not declared in configuration.svh")
    return int(match.group(1))


def batched_qkt(mlen: int = MLEN, blen: int = BLEN) -> str:
    return qkt_multiply(
        d=HEAD_DIM,
        mlen=mlen,
        stage="prefill",
        alive_registers=list(range(1, 10)),
        q_base_address=0,
        k_base_hbm_offset_reg=0,
        q_head_index=0,
        k_head_index=0,
        s_base_address=1 << 14,
        use_batched=True,
        blen=blen,
        prefetch_k=False,
    )


@requires_rtl
def test_drain_rows_are_head_count_times_blen() -> None:
    """The RTL constant that sets the granularity, read rather than restated."""
    text = MCU.read_text()
    assert re.search(
        r"localparam\s+int\s+PH_DRAIN_ROWS\s*=\s*HEAD_COUNT\s*\*\s*BLEN\s*;", text
    ), "PH_DRAIN_ROWS is no longer HEAD_COUNT * BLEN"
    assert re.search(
        r"drain_last\s*=\s*per_head_exe\s*\?\s*\(\s*PH_DRAIN_ROWS\s*-\s*1\s*\)", text
    ), "the per-head drain no longer terminates at PH_DRAIN_ROWS"


@requires_rtl
def test_measured_rows_match_the_blen_granular_shape() -> None:
    """The probe's row pattern is BLEN-sized groups at an MLEN stride."""
    mlen, blen, hlen = rtl_param("MLEN"), rtl_param("BLEN"), rtl_param("HLEN")
    heads = mlen // hlen
    assert len(MEASURED_ROWS) == heads * blen
    assert len(MEASURED_ROWS) != heads * mlen, "the two outcomes must be distinct"
    for head in range(heads):
        group = MEASURED_ROWS[head * blen : (head + 1) * blen]
        assert group == [head * mlen + row for row in range(blen)], (
            f"head {head} rows {group} are not BLEN rows at an MLEN stride"
        )


def test_batched_qkt_covers_the_tile_with_blen_granular_issues() -> None:
    """One `M_BTMM` per BLEN x BLEN block, driven by a two-deep loop nest.

    A single issue per tile would leave all but the first BLEN query rows and
    BLEN key columns of each head stale on hardware.
    """
    code = batched_qkt()
    lines = [line.strip() for line in code.splitlines()]

    issues = [line for line in lines if line.startswith("M_BTMM")]
    drains = [line for line in lines if line.startswith("M_BMM_WO")]
    assert len(issues) == 1 and len(drains) == 1, (
        "the tile is covered by a loop nest, so one static issue and one static "
        f"drain are expected, found {issues} / {drains}"
    )

    trip_counts = [
        int(line.rsplit(",", 1)[1])
        for line in lines
        if line.startswith("C_LOOP_START")
    ]
    assert len(trip_counts) == 2, f"expected a two-deep loop nest, got {trip_counts}"
    blocks = MLEN // BLEN
    assert trip_counts == [blocks, blocks], (
        f"the nest must run {blocks} x {blocks} times to cover an "
        f"{MLEN} x {MLEN} tile, got {trip_counts}"
    )
    assert trip_counts[0] * trip_counts[1] == (MLEN // BLEN) ** 2


def test_batched_qkt_walks_blen_blocks_in_both_operands() -> None:
    """Key rows advance by BLEN * MLEN; the score column advances by BLEN.

    `M_BTMM` takes an MLEN-scaled row index, so a BLEN-wide key block is
    `BLEN * MLEN` in the operand address, while `M_BMM_WO` writes into an
    element-addressed column group that advances by BLEN.
    """
    code = batched_qkt()
    strides = [
        int(line.rsplit(",", 1)[1])
        for line in (line.strip() for line in code.splitlines())
        if line.startswith("S_ADDI_INT") and "gp0" not in line
    ]
    assert BLEN * MLEN in strides, (
        f"no BLEN * MLEN = {BLEN * MLEN} operand advance in {strides}"
    )
    assert BLEN in strides, f"no BLEN = {BLEN} result column advance in {strides}"


def test_issue_count_scales_with_the_square_of_the_block_count() -> None:
    """Halving BLEN quadruples the issues needed to cover the same tile."""
    counts = {}
    for blen in (4, 8):
        lines = [line.strip() for line in batched_qkt(blen=blen).splitlines()]
        trips = [
            int(line.rsplit(",", 1)[1])
            for line in lines
            if line.startswith("C_LOOP_START")
        ]
        counts[blen] = trips[0] * trips[1]
    assert counts[4] == (MLEN // 4) ** 2
    assert counts[8] == (MLEN // 8) ** 2
    assert counts[4] == 4 * counts[8]


def test_the_per_head_lowering_stays_blen_granular() -> None:
    """`_qkt_per_head_prefill` walks BLEN row and column blocks with `M_TMM`.

    It is the non-broadcast lowering of the same shape and must not regress to a
    whole-tile issue either.
    """
    per_head = qkt_multiply(
        d=HEAD_DIM, mlen=MLEN, stage="prefill", alive_registers=list(range(1, 10)),
        q_base_address=0, k_base_hbm_offset_reg=0, q_head_index=0, k_head_index=0,
        use_batched=False, blen=BLEN, prefetch_k=False,
    )
    assert "M_TMM" in per_head and "M_BTMM" not in per_head
    assert per_head.count("C_LOOP_START") >= 2
