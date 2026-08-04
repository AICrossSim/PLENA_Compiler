from __future__ import annotations

import unittest

from asm_templates.lm_head import lm_head_asm, lm_head_vocab_padding


class LmHeadLoweringTest(unittest.TestCase):
    def _lower(self, **overrides: int) -> str:
        arguments = dict(
            mlen=1024,
            blen=8,
            batch=8,
            hidden_size=5120,
            vocab_size=151936,
            alive_registers=[1, 2, 3, 4, 5, 6],
            lm_head_weight_hbm_offset_reg=1,
            activation_base_address=0,
            result_base_address=1024,
        )
        arguments.update(overrides)
        return lm_head_asm(**arguments)

    def test_vocab_padding_rounds_up_to_whole_blen_tiles(self) -> None:
        self.assertEqual(lm_head_vocab_padding(151936, 8), 151936)
        self.assertEqual(lm_head_vocab_padding(151937, 8), 151944)
        self.assertEqual(lm_head_vocab_padding(1, 8), 8)

    def test_lowering_reduces_over_the_transposed_weight(self) -> None:
        """The weight is (vocab, hidden), so the reduction runs down its columns.

        `M_TMM` selects BLEN rows of the tile — BLEN columns of the transpose —
        and reduces over hidden. `M_MM` would slice the tile's columns instead
        and pair the activation's hidden dimension with the vocabulary.
        """
        asm = self._lower()
        self.assertIn("M_TMM ", asm)
        self.assertNotIn("M_MM ", asm)
        self.assertIn("M_MM_WO ", asm)
        self.assertIn("H_PREFETCH_M ", asm)

    def test_reduction_streams_every_hidden_tile_per_output_tile(self) -> None:
        mlen, blen, hidden, vocab = 1024, 8, 5120, 64
        asm = self._lower(mlen=mlen, blen=blen, batch=blen, hidden_size=hidden, vocab_size=vocab)
        # One matrix issue per (output tile x batch tile x reduction tile).
        expected = (vocab // blen) * (blen // blen) * (hidden // mlen)
        self.assertEqual(asm.count("M_TMM "), expected)

    def test_output_group_index_is_mlen_scaled(self) -> None:
        """A BLEN-wide output group is BLEN * MLEN in the matrix operand.

        The matrix SRAM returns whole MLEN-wide vectors, so the address bits
        below MLEN never reach the array and the group index is MLEN-scaled.
        The VRAM result cursor is element-addressed and advances by BLEN.
        """
        mlen, blen, hidden, vocab = 64, 4, 64, 64
        result_base = 4096
        asm = self._lower(
            mlen=mlen, blen=blen, batch=blen, hidden_size=hidden,
            vocab_size=vocab, result_base_address=result_base,
        )
        groups = mlen // blen
        for group in range(1, groups):
            self.assertIn(
                f"S_ADDI_INT gp1, gp0, {group * blen * mlen} ", asm,
                f"output group {group} is not MLEN-scaled in the matrix operand",
            )
            self.assertIn(
                f"S_ADDI_INT gp4, gp6, {group * blen} ", asm,
                f"output group {group}'s VRAM result cursor is not element-addressed",
            )

    def test_header_records_the_hbm_weight_layout(self) -> None:
        asm = self._lower(vocab_size=151937)
        self.assertIn("row_major_vocab_by_hidden", asm)
        self.assertIn("151944", asm)
        self.assertIn("7 masked entries", asm)

    def test_geometry_and_register_budget_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 6 alive registers"):
            self._lower(alive_registers=[1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "multiple of MLEN"):
            self._lower(hidden_size=5000)


if __name__ == "__main__":
    unittest.main()
