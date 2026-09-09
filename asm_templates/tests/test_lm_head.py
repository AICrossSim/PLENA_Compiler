from __future__ import annotations

import unittest

from asm_templates.lm_head import (
    HBM_WEIGHT_LAYOUT,
    lm_head_asm,
    lm_head_hidden_padding,
    lm_head_native_row_element_offset,
    lm_head_physical_batch,
    lm_head_vocab_padding,
    local_lm_head_lowering_receipt,
)
from generator.passes.code_gen import _generate_addr_reg_init, _weight_hbm_bytes


PROFILE = {
    "weight_format": "MXINT4",
    "activation_format": "MXINT4",
    "vector_format": "FP_E3M2",
    "matrix_mlen": 256,
    "block_size": 8,
    "scale_format": "E8M0",
    "scale_bits": 8,
}


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

    def test_vocab_padding_rounds_up_to_whole_mlen_prefetch_tiles(self) -> None:
        self.assertEqual(lm_head_vocab_padding(151936, 16, 256), 152064)
        self.assertEqual(lm_head_vocab_padding(151937, 16, 256), 152064)
        self.assertEqual(lm_head_vocab_padding(1, 4, 64), 64)
        with self.assertRaisesRegex(ValueError, "MLEN divisible by BLEN"):
            lm_head_vocab_padding(128, 6, 64)

    def test_hidden_padding_supports_mlen_larger_than_model_hidden(self) -> None:
        self.assertEqual(lm_head_hidden_padding(2048, 1024), 2048)
        self.assertEqual(lm_head_hidden_padding(2048, 4096), 4096)
        asm = self._lower(
            mlen=4096,
            blen=16,
            batch=1,
            hidden_size=2048,
            vocab_size=4096,
        )
        self.assertIn("padded hidden: 4096 (2048 zero columns)", asm)
        self.assertIn("Linear T: (batch, 4096)", asm)

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
        mlen, blen, hidden, vocab = 64, 8, 512, 64
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
        self.assertIn("152576", asm)
        self.assertIn("639 masked entries", asm)

    def test_batch_one_and_nonmultiple_batches_issue_padded_tiles(self) -> None:
        geometry = dict(mlen=64, blen=4, hidden_size=64, vocab_size=64)
        batch_one = self._lower(batch=1, **geometry)
        batch_five = self._lower(batch=5, **geometry)
        self.assertEqual(lm_head_physical_batch(1, 4), 4)
        self.assertEqual(lm_head_physical_batch(5, 4), 8)
        self.assertEqual(batch_one.count("M_TMM "), 16)
        self.assertEqual(batch_five.count("M_TMM "), 32)
        self.assertIn("Active rows=1, physical rows=4", batch_one)
        self.assertIn("caller must zero padded inputs", batch_one)
        self.assertIn("Active rows=5, physical rows=8", batch_five)

    def test_native_weight_row_layout_numeric_oracle(self) -> None:
        hidden = [2.0, -1.0, 0.5, 3.0]
        weights = [
            [1.0, 2.0, 3.0, 4.0],
            [-2.0, 0.25, 5.0, -1.0],
            [0.5, 0.5, 0.5, 0.5],
        ]
        flat_native = [value for row in weights for value in row]
        logits = []
        for token_id in range(len(weights)):
            start = lm_head_native_row_element_offset(token_id, len(hidden), len(weights))
            row = flat_native[start : start + len(hidden)]
            logits.append(sum(x * w for x, w in zip(hidden, row)))
        self.assertEqual(logits, [13.5, -4.75, 2.25])
        self.assertEqual(HBM_WEIGHT_LAYOUT, "row_major_vocab_by_hidden")

    def test_target_receipt_has_exact_bytes_events_and_fail_closed_boundaries(self) -> None:
        receipt = local_lm_head_lowering_receipt(
            mlen=256,
            blen=16,
            batch=1,
            hidden_size=2048,
            vocab_size=151936,
            profile=PROFILE,
            profile_id="test-mxint4",
        )
        geometry = receipt["model_geometry"]
        self.assertEqual(geometry["physical_vocab"], 152064)
        self.assertEqual(geometry["physical_batch"], 16)
        self.assertEqual(receipt["numerical_identity"]["mlen"], 256)
        self.assertEqual(
            receipt["numeric_semantics"][
                "compiler_numerical_parity_requires_same_profile_id_and_mlen"
            ],
            True,
        )
        footprint = receipt["hbm_footprint"]
        physical_elements = 152064 * 2048
        self.assertEqual(footprint["physical_weight_elements"], physical_elements)
        self.assertEqual(footprint["weight_data_bytes"], physical_elements * 4 // 8)
        self.assertEqual(footprint["scale_bytes"], physical_elements // 8)
        events = receipt["structural_events"]
        self.assertEqual(events["weight_prefetches"], (152064 // 256) * (2048 // 256))
        self.assertEqual(
            events["matrix_issues"],
            (152064 // 16) * (16 // 16) * (2048 // 256),
        )
        self.assertIsNone(events["calibrated_matrix_cycles"])
        selection = receipt["selection"]
        self.assertEqual(selection["full_logits_bf16_bytes"], 151936 * 2)
        self.assertEqual(selection["streamed_matrix_tile_bf16_bytes"], 16 * 256 * 2)
        self.assertEqual(selection["topk_state_bytes_active_batch"], 160)
        self.assertEqual(selection["argmax_state_bytes_active_batch"], 8)
        self.assertFalse(receipt["validity"]["streaming_selector_lowered"])
        self.assertFalse(receipt["validity"]["serving_compiler_valid"])
        self.assertRegex(receipt["contract_sha256"], r"^[0-9a-f]{64}$")

    def test_mxint2_profile_has_exact_subbyte_weight_footprint(self) -> None:
        profile = dict(PROFILE, weight_format="MXINT2", matrix_mlen=64)
        receipt = local_lm_head_lowering_receipt(
            mlen=64,
            blen=4,
            batch=1,
            hidden_size=64,
            vocab_size=65,
            profile=profile,
            profile_id="test-mxint2",
        )
        physical_elements = 128 * 64
        self.assertEqual(receipt["hbm_footprint"]["weight_element_bits"], 2)
        self.assertEqual(
            receipt["hbm_footprint"]["weight_data_bytes"],
            physical_elements * 2 // 8,
        )

    def test_batch_128_receipt_does_not_hide_full_logit_or_selection_state(self) -> None:
        receipt = local_lm_head_lowering_receipt(
            mlen=256,
            blen=16,
            batch=128,
            hidden_size=2048,
            vocab_size=151936,
            profile=PROFILE,
            profile_id="test-mxint4",
        )
        selection = receipt["selection"]
        self.assertEqual(selection["full_logits_bf16_bytes"], 38_895_616)
        self.assertEqual(selection["topk_state_bytes_active_batch"], 20_480)
        self.assertEqual(selection["argmax_state_bytes_active_batch"], 1_024)
        self.assertFalse(selection["full_logit_materialization_allowed"])

    def test_mlen_4096_has_distinct_numerical_identity_and_physical_k_padding(self) -> None:
        base_profile = dict(PROFILE, matrix_mlen=1024)
        wide_profile = dict(PROFILE, matrix_mlen=4096)
        base = local_lm_head_lowering_receipt(
            mlen=1024,
            blen=16,
            batch=1,
            hidden_size=2048,
            vocab_size=151936,
            profile=base_profile,
            profile_id="test-mxint4",
        )
        wide = local_lm_head_lowering_receipt(
            mlen=4096,
            blen=16,
            batch=1,
            hidden_size=2048,
            vocab_size=151936,
            profile=wide_profile,
            profile_id="test-mxint4",
        )
        self.assertEqual(base["model_geometry"]["physical_hidden_size"], 2048)
        self.assertEqual(wide["model_geometry"]["physical_hidden_size"], 4096)
        self.assertNotEqual(
            base["numerical_identity_sha256"], wide["numerical_identity_sha256"]
        )
        self.assertFalse(wide["validity"]["hidden_padding_zero_fill_lowered"])

    def test_tensor_parallel_selection_requires_deterministic_global_merge(self) -> None:
        receipt = local_lm_head_lowering_receipt(
            mlen=256,
            blen=16,
            batch=128,
            hidden_size=2048,
            vocab_size=151936,
            profile=PROFILE,
            profile_id="test-mxint4",
            tensor_parallel_ranks=4,
        )
        tp = receipt["selection"]["tensor_parallel"]
        self.assertEqual(tp["required_global_merge_candidates_per_row"], 80)
        self.assertEqual(
            tp["required_global_merge_candidate_bytes_active_batch"],
            128 * 80 * 8,
        )
        self.assertEqual(
            tp["merge_order"],
            "score_descending_then_global_token_id_ascending",
        )
        self.assertFalse(tp["global_merge_lowered"])
        self.assertFalse(receipt["validity"]["multi_tp_serving_valid"])

    def test_generator_initializes_dedicated_lm_head_hbm_base(self) -> None:
        model = {
            "hidden_size": 64,
            "intermediate_size": 128,
            "vocab_size": 65,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        }
        hardware = {
            "MLEN": 64,
            "BLEN": 4,
            "block_dim": 4,
            "wt_block_width": 32,
            "scale_width": 8,
        }
        scheduler = {
            "register_assignment": {
                "hbm_addr_reg": {"lm_head_weight_offset": 10},
            }
        }
        asm = _generate_addr_reg_init(model, hardware, scheduler)
        self.assertIn("C_SET_ADDR_REG a10", asm)
        preceding_shapes = [
            (65, 64),
            (64, 64),
            (64, 32),
            (64, 32),
            (64, 64),
            (64, 128),
            (64, 128),
            (128, 64),
        ]
        expected_head_base = sum(
            _weight_hbm_bytes(rows, cols, hardware)
            for rows, cols in preceding_shapes
        )
        expected_total = expected_head_base + _weight_hbm_bytes(128, 64, hardware)
        self.assertIn(f"Total HBM weight footprint: {expected_total} bytes", asm)

    def test_geometry_and_register_budget_are_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 6 alive registers"):
            self._lower(alive_registers=[1, 2, 3, 4])
        padded = self._lower(hidden_size=5000)
        self.assertIn("padded hidden: 5120 (120 zero columns)", padded)
        with self.assertRaisesRegex(ValueError, "MLEN divisible by BLEN"):
            local_lm_head_lowering_receipt(
                mlen=64,
                blen=6,
                batch=1,
                hidden_size=64,
                vocab_size=128,
                profile=PROFILE,
                profile_id="test-mxint4",
            )

    def test_receipt_rejects_a_profile_from_a_different_mlen(self) -> None:
        with self.assertRaisesRegex(ValueError, "MLEN differs"):
            local_lm_head_lowering_receipt(
                mlen=1024,
                blen=16,
                batch=1,
                hidden_size=2048,
                vocab_size=151936,
                profile=PROFILE,
                profile_id="test-mxint4",
            )
        unsupported = dict(PROFILE, weight_format="MXINT3", matrix_mlen=64)
        with self.assertRaisesRegex(ValueError, "unsupported local LM-head"):
            local_lm_head_lowering_receipt(
                mlen=64,
                blen=4,
                batch=1,
                hidden_size=64,
                vocab_size=128,
                profile=unsupported,
                profile_id="test-mxint3",
            )


if __name__ == "__main__":
    unittest.main()
