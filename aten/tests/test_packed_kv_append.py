from __future__ import annotations

import re
import unittest

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.memory import MatrixBlockLayout
from compiler.aten.plena.packed_kv import (
    PackedKVLayout,
    resolve_packed_kv_append,
)


class PackedKVAppendTests(unittest.TestCase):
    def test_address_contract_separates_element_and_scale_planes(self) -> None:
        for bits, plane_bytes, element_offset, transfer_bytes in (
            (2, 2048, 1040, 128),
            (4, 4096, 2080, 256),
            (8, 8192, 4160, 512),
        ):
            with self.subTest(bits=bits):
                cache = MatrixBlockLayout(
                    name="cache",
                    full_shape=(65, 64),
                    physical_shape=(128, 64),
                    block_size=64,
                    hbm_base_addr=4096,
                    hbm_row_width=256,
                    hbm_element_width=bits,
                    hbm_block_size=8,
                    hbm_scale_width=8,
                    precision_role="key",
                )
                packed = PackedKVLayout(
                    kv_heads=2,
                    head_dim=8,
                    mlen=64,
                    element_bits=bits,
                )
                address = resolve_packed_kv_append(
                    cache,
                    packed,
                    token_index=65,
                    transfer_rows=8,
                )

                self.assertEqual(address.element_plane_bytes, plane_bytes)
                self.assertEqual(address.element_offset_bytes, element_offset)
                self.assertEqual(address.scale_offset_bytes, 520)
                self.assertEqual(
                    address.element_address,
                    4096 + element_offset,
                )
                self.assertEqual(
                    address.scale_address,
                    4096 + plane_bytes + 520,
                )
                self.assertEqual(
                    address.element_transfer_bytes,
                    transfer_bytes,
                )
                self.assertEqual(address.scale_transfer_bytes, 64)

    def _compiler_and_cache(self):
        compiler = PlenaCompiler(
            mlen=64,
            blen=8,
            hbm_element_width=4,
            hbm_block_size=8,
            hbm_scale_width=8,
            hbm_v_writeback_amount=8,
        )
        cache = compiler.input(
            "K_cache",
            shape=(65, 64),
            physical_shape=(128, 64),
            hbm_element_width=4,
            hbm_block_size=8,
            hbm_scale_width=8,
            precision_role="key",
        )
        source = compiler.alloc(
            "K_new",
            1,
            16,
            strict=False,
            physical_shape=(8, 64),
        )
        packed = PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=64,
            element_bits=4,
        )
        return compiler, cache, source, packed

    def test_program_append_uses_global_plane_base_and_one_dma(self) -> None:
        compiler, cache, source, packed = self._compiler_and_cache()
        source_before = compiler[source.name].hbm_addr
        first = compiler.append_packed_kv_row(
            source,
            cache,
            token_index=65,
            packed_layout=packed,
            role="key",
        )
        second = compiler.append_packed_kv_row(
            source,
            cache,
            token_index=66,
            packed_layout=packed,
            role="key",
        )
        assembly = compiler.compile()

        self.assertEqual(first.element_offset_bytes, 2080)
        self.assertEqual(second.element_offset_bytes, 2112)
        self.assertEqual(assembly.count("H_STORE_V"), 2)
        self.assertNotIn("C_LOOP_START", assembly)
        self.assertEqual(compiler[source.name].hbm_addr, source_before)
        self.assertEqual(
            len(re.findall(r"S_ADDI_INT gp\d+, gp0, 4096", assembly)),
            2,
        )
        self.assertRegex(
            assembly,
            r"S_ADDI_INT gp\d+, gp0, 2080\s+"
            r"S_ADDI_INT gp\d+, gp0, 4096\s+"
            r"C_SET_SCALE_REG",
        )
        self.assertRegex(
            assembly,
            r"S_ADDI_INT gp\d+, gp0, 2112\s+"
            r"S_ADDI_INT gp\d+, gp0, 4096\s+"
            r"C_SET_SCALE_REG",
        )

    def test_program_append_rejects_nonsequential_or_unsafe_writes(self) -> None:
        compiler, cache, source, packed = self._compiler_and_cache()
        compiler.append_packed_kv_row(
            source,
            cache,
            token_index=65,
            packed_layout=packed,
            role="key",
        )
        with self.assertRaisesRegex(ValueError, "expected token 66"):
            compiler.append_packed_kv_row(
                source,
                cache,
                token_index=67,
                packed_layout=packed,
                role="key",
            )
        with self.assertRaisesRegex(ValueError, "precision role"):
            compiler.append_packed_kv_row(
                source,
                cache,
                token_index=66,
                packed_layout=packed,
                role="value",
            )

        short_source = compiler.alloc(
            "short",
            1,
            16,
            strict=False,
            physical_shape=(4, 64),
        )
        value_cache = compiler.input(
            "V_cache",
            shape=(65, 64),
            physical_shape=(128, 64),
            hbm_element_width=4,
            precision_role="value",
        )
        with self.assertRaisesRegex(ValueError, "padding rows"):
            compiler.append_packed_kv_row(
                short_source,
                value_cache,
                token_index=65,
                packed_layout=packed,
                role="value",
            )
        wrong_width = compiler.alloc(
            "wrong_width",
            1,
            8,
            strict=False,
            physical_shape=(8, 64),
        )
        with self.assertRaisesRegex(ValueError, "kv_heads \\* head_dim"):
            compiler.append_packed_kv_row(
                wrong_width,
                value_cache,
                token_index=65,
                packed_layout=packed,
                role="value",
            )

    def test_address_contract_rejects_capacity_overrun(self) -> None:
        cache = MatrixBlockLayout(
            name="cache",
            full_shape=(65, 64),
            physical_shape=(68, 64),
            block_size=64,
            hbm_row_width=256,
            hbm_element_width=4,
            hbm_block_size=8,
            hbm_scale_width=8,
        )
        packed = PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=64,
            element_bits=4,
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            resolve_packed_kv_append(
                cache,
                packed,
                token_index=65,
                transfer_rows=8,
            )

    def test_append_rejects_unsupported_width_and_plane_base(self) -> None:
        cache = MatrixBlockLayout(
            name="cache",
            full_shape=(65, 64),
            physical_shape=(128, 64),
            block_size=64,
            hbm_row_width=256,
            hbm_element_width=3,
            hbm_block_size=8,
            hbm_scale_width=8,
        )
        packed = PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=64,
            element_bits=3,
        )
        with self.assertRaisesRegex(ValueError, "2-, 4-, or 8-bit"):
            resolve_packed_kv_append(
                cache,
                packed,
                token_index=65,
                transfer_rows=8,
            )

        compiler, _cache, source, _packed = self._compiler_and_cache()
        with self.assertRaisesRegex(ValueError, "positive integer"):
            compiler.store_to_hbm(
                source.name,
                hbm_addr=4096,
                hbm_element_plane_bytes=0,
            )

    def test_two_appends_feed_the_existing_cached_q1_lowering(self) -> None:
        compiler = PlenaCompiler(
            mlen=64,
            blen=8,
            hbm_element_width=4,
            hbm_v_writeback_amount=8,
        )
        compiler.hlen = 8
        compiler.broadcast_amount = 8
        packed = PackedKVLayout(
            kv_heads=2,
            head_dim=8,
            mlen=64,
            element_bits=4,
        )
        key_cache = compiler.input(
            "K_cache",
            shape=(65, 64),
            physical_shape=(128, 64),
            hbm_element_width=4,
            precision_role="key",
        )
        value_cache = compiler.input(
            "V_cache",
            shape=(65, 64),
            physical_shape=(128, 64),
            hbm_element_width=4,
            precision_role="value",
        )
        key_row = compiler.alloc(
            "K_new",
            1,
            16,
            strict=False,
            physical_shape=(8, 64),
        )
        value_row = compiler.alloc(
            "V_new",
            1,
            16,
            strict=False,
            physical_shape=(8, 64),
        )

        for step, cache_length in enumerate((66, 67)):
            compiler.append_packed_kv_row(
                key_row,
                key_cache,
                token_index=cache_length - 1,
                packed_layout=packed,
                role="key",
            )
            compiler.append_packed_kv_row(
                value_row,
                value_cache,
                token_index=cache_length - 1,
                packed_layout=packed,
                role="value",
            )
            query = compiler.alloc(
                f"Q_{step}",
                1,
                128,
                strict=False,
                physical_shape=(64, 128),
            )
            output = compiler.alloc(
                f"O_{step}",
                1,
                128,
                strict=False,
                physical_shape=(64, 128),
            )
            scratch = compiler.alloc(
                f"S_{step}",
                512,
                64,
                strict=True,
            )
            compiler.flash_attention_packed_cache(
                query,
                key_cache,
                value_cache,
                num_kv_heads=2,
                group_heads=8,
                head_slot_dim=8,
                output_base_address=compiler.get_vram_addr(output.name),
                scratch_base_address=compiler.get_vram_addr(scratch.name),
                broadcast_amount=8,
                causal_mask=True,
                valid_cols=cache_length,
                cache_position=cache_length - 1,
                batch_size=1,
                rows_per_batch=64,
                query_rows_per_batch=1,
                cache_rows_per_batch=128,
            )
            compiler.free_tensor(query)
            compiler.free_tensor(output)
            compiler.free_tensor(scratch)

        assembly = compiler.compile()
        self.assertEqual(assembly.count("H_STORE_V"), 4)
        self.assertEqual(assembly.count("M_BTMM 0,"), 18)
        self.assertEqual(assembly.count("M_BTMM 1,"), 18)
        self.assertEqual(assembly.count("M_BMM_WO"), 36)
        self.assertEqual(
            len(re.findall(r"^M_MM ", assembly, flags=re.MULTILINE)),
            64,
        )
        self.assertEqual(
            len(re.findall(r"^M_MM_WO ", assembly, flags=re.MULTILINE)),
            64,
        )
        first_attention = assembly.index(
            "; PackedKV batch 0, selector 0, K block 0"
        )
        first_store = assembly.index("H_STORE_V")
        self.assertLess(first_store, first_attention)


if __name__ == "__main__":
    unittest.main()
