from __future__ import annotations

import re
import unittest

from asm_templates.flashattn import flash_attn_asm


class FlashAttentionKvTileAddressingTest(unittest.TestCase):
    """Each key tile must be fetched from its own place in the KV cache.

    The K/V cache is (kv_len, MLEN) row-major, so key tile `t` starts at
    `t * MLEN * MLEN` elements. A prefetch address built only from the KV-head
    index makes every tile read the first one, which leaves attention covering
    `MLEN` keys no matter how long the cache is.
    """

    MLEN = 64
    BLEN = 4
    HEAD_DIM = 16
    HEADS = 4

    def _lower(self, kv_len: int) -> str:
        return flash_attn_asm(
            mlen=self.MLEN,
            vlen=self.MLEN,
            blen=self.BLEN,
            batch=1,
            hq=self.HEADS,
            hkv=1,
            d=self.HEAD_DIM,
            q_len=self.MLEN,
            kv_len=kv_len,
            alive_registers_int=list(range(1, 16)),
            alive_registers_fp=list(range(1, 8)),
            vector_sram_base_address=0,
            fp_sram_start_address=6,
            k_base_hbm_offset_reg=0,
            v_base_hbm_offset_reg=1,
            broadcast_amount=self.HEADS,
            scratch_base_address=1 << 16,
            output_base_address=1 << 18,
        )

    @staticmethod
    def _prefetch_offsets(asm: str, kind: str) -> list[int]:
        """Immediates loaded into the address register of each K or V prefetch."""
        offsets: list[int] = []
        pending: dict[int, int] = {}
        for line in asm.splitlines():
            line = line.strip()
            load = re.match(r"S_ADDI_INT gp(\d+), gp0, (\d+)", line)
            if load:
                pending[int(load.group(1))] = int(load.group(2))
                continue
            lui = re.match(r"S_LUI_INT gp(\d+), (\d+)", line)
            if lui:
                pending[int(lui.group(1))] = int(lui.group(2)) << 12
                continue
            add = re.match(r"S_ADDI_INT gp(\d+), gp\1, (\d+)", line)
            if add:
                register = int(add.group(1))
                pending[register] = pending.get(register, 0) + int(add.group(2))
                continue
            fetch = re.match(r"H_PREFETCH_M gp\d+, gp(\d+), a(\d+)", line)
            if fetch:
                register, port = int(fetch.group(1)), int(fetch.group(2))
                if (kind == "K" and port == 0) or (kind == "V" and port == 1):
                    offsets.append(pending.get(register, 0))
        return offsets

    def test_key_tiles_are_fetched_from_distinct_addresses(self):
        tiles = 4
        asm = self._lower(kv_len=tiles * self.MLEN)
        for kind in ("K", "V"):
            offsets = self._prefetch_offsets(asm, kind)
            self.assertGreaterEqual(len(offsets), tiles, f"{kind} prefetches")
            self.assertEqual(
                len(set(offsets[:tiles])),
                tiles,
                f"{kind} prefetch reuses one address across key tiles: {offsets}",
            )

    def test_key_tile_stride_is_one_cache_tile(self):
        tiles = 4
        asm = self._lower(kv_len=tiles * self.MLEN)
        expected = {t * self.MLEN * self.MLEN for t in range(tiles)}
        for kind in ("K", "V"):
            offsets = sorted(set(self._prefetch_offsets(asm, kind)))[:tiles]
            self.assertEqual(
                set(offsets),
                expected,
                f"{kind} prefetch offsets {offsets} are not whole cache tiles",
            )

    def test_a_single_key_tile_needs_only_the_base_address(self):
        asm = self._lower(kv_len=self.MLEN)
        for kind in ("K", "V"):
            self.assertEqual(set(self._prefetch_offsets(asm, kind)), {0})


if __name__ == "__main__":
    unittest.main()
