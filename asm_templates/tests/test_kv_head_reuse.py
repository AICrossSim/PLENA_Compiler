"""The KV-head reuse schedule reads the packed KV row once per key tile.

The default schedule gives every KV head its own pass over the key cache, so a
packed KV row — one row carrying every head's HLEN window — crosses HBM once per
head. Hoisting the KV-head loop inside the key-tile loop fetches that row once
and picks each head's window out of the resident tile with `M_BTMM`'s
head-selector field, which is what removes the per-head re-read.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from compiler.asm_templates.flashattn import flash_attn_asm  # noqa: E402

MLEN = 64
HLEN = 16
BLEN = 4
BROADCAST = MLEN // HLEN
RATIO = BROADCAST
S_Q = 64
KV_LEN = 256
KEY_TILES = KV_LEN // MLEN


def emit(
    kv_heads: int,
    reuse: bool,
    rows_live: int = BLEN,
    packed_group_layout: bool | None = True,
) -> str:
    heads_live = RATIO * (kv_heads if reuse else 1)
    layout = {}
    if packed_group_layout is not None:
        layout["packed_group_layout"] = packed_group_layout
    if packed_group_layout:
        layout["q_group_stride"] = S_Q * MLEN
        layout["o_group_stride"] = S_Q * MLEN
    return flash_attn_asm(
        mlen=MLEN, vlen=MLEN, blen=BLEN,
        batch=1, hq=RATIO * kv_heads, hkv=kv_heads, d=HLEN,
        q_len=S_Q, kv_len=KV_LEN,
        broadcast_amount=BROADCAST,
        alive_registers_int=list(range(1, 16)),
        alive_registers_fp=list(range(1, 8)),
        vector_sram_base_address=0,
        fp_sram_start_address=6,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
        scratch_base_address=S_Q * kv_heads * MLEN,
        output_base_address=S_Q * kv_heads * MLEN + MLEN * MLEN * BROADCAST * 2,
        # Both schedules get the same query-row tile, so the KV-head axis is
        # what the comparison isolates.
        fp_sram_depth=6 + 3 * rows_live * heads_live,
        kv_head_reuse=reuse,
        **layout,
    )


class KVHeadReuseTests(unittest.TestCase):
    def test_single_head_default_retains_the_legacy_layout(self) -> None:
        implicit_legacy = emit(1, reuse=False, packed_group_layout=None)
        explicit_legacy = emit(1, reuse=False, packed_group_layout=False)
        self.assertEqual(implicit_legacy, explicit_legacy)

    def test_reuse_prefetch_count_is_independent_of_kv_head_count(self) -> None:
        counts = {
            kv_heads: emit(kv_heads, reuse=True).count("H_PREFETCH_M")
            for kv_heads in (1, 2, 4)
        }
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_default_schedule_refetches_once_per_kv_head(self) -> None:
        for kv_heads in (1, 2, 4):
            with self.subTest(kv_heads=kv_heads):
                per_head = emit(kv_heads, reuse=False).count("H_PREFETCH_M")
                reused = emit(kv_heads, reuse=True).count("H_PREFETCH_M")
                self.assertEqual(per_head, reused * kv_heads)

    def test_reuse_selects_every_head_out_of_the_resident_tile(self) -> None:
        asm = emit(4, reuse=True)
        for selector in range(4):
            self.assertIn(f"M_BTMM {selector},", asm)
        # The selector replaces the per-head element offset, so one prefetch
        # serves the whole sweep over a key tile.
        self.assertEqual(asm.count("H_PREFETCH_M"), 2 * KEY_TILES * (S_Q // BLEN))

    def test_reuse_overlaps_value_fetch_with_softmax(self) -> None:
        lines = emit(4, reuse=True).splitlines()
        k_prefetches = [
            index
            for index, line in enumerate(lines)
            if line.startswith("H_PREFETCH_M") and " a1," in line
        ]
        v_prefetches = [
            index
            for index, line in enumerate(lines)
            if line.startswith("H_PREFETCH_M") and " a2," in line
        ]
        self.assertGreaterEqual(len(k_prefetches), 2)
        first_iteration = lines[k_prefetches[0] : k_prefetches[1]]
        qkt_positions = [
            index for index, line in enumerate(first_iteration) if line.startswith("M_BTMM")
        ]
        v_positions = [
            index
            for index, line in enumerate(first_iteration)
            if line.startswith("H_PREFETCH_M") and " a2," in line
        ]
        self.assertEqual(len(qkt_positions), 4)
        self.assertEqual(len(v_positions), 1)
        self.assertLess(qkt_positions[0], v_positions[0])
        self.assertLess(v_positions[0], qkt_positions[1])
        self.assertEqual(len(k_prefetches), len(v_prefetches))

    def test_default_schedule_rereads_aligned_packed_tiles(self) -> None:
        # The default schedule retains one transfer per head, but every transfer
        # starts at the packed row base. The matrix selector chooses the HLEN
        # window, so all rereads have the same physical byte footprint.
        asm = emit(4, reuse=False)
        prefetches = re.findall(
            r"; Pipelined ([KV]) prefetch for KV head (\d+) tile (\d+) \n"
            r"S_ADDI_INT gp2, gp0, (\d+)\n",
            asm,
        )
        self.assertTrue(prefetches)
        for _role, _head, tile, element_offset in prefetches:
            self.assertEqual(int(element_offset), int(tile) * MLEN * MLEN)
        for selector in range(4):
            self.assertIn(f"M_BTMM {selector},", asm)


if __name__ == "__main__":
    unittest.main()
