#!/usr/bin/env python3
"""Contract tests for generator codegen/scheduler conventions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from generator.passes import code_gen
from generator.scheduler.scheduler import generate_mem_layout


def _scheduler() -> dict:
    return {
        "memory_layout": {
            "vector_sram_addr": {
                "block1": 111,
                "block2": 222,
                "block3": 333,
                "block4": 444,
                "block5": 555,
                "k_split_scratch": 666,
                "q_scratch": 777,
                "k_scratch": 888,
                "v_scratch": 999,
            },
            "fp_sram": {
                "eps": 3,
                "hid_reciprocal": 4,
                "silu_one": 5,
                "gelu_1702": 6,
                "attn_scale": 7,
                "infinity": 8,
            },
        },
        "register_assignment": {
            "hbm_addr_reg": {
                "q_weight_offset": 1,
                "k_weight_offset": 2,
                "v_weight_offset": 3,
                "rope_params_offset": 4,
                "ffn_gate_offset": 5,
                "ffn_up_offset": 6,
                "ffn_down_offset": 7,
                "previous_activation_offset": 8,
                "lm_head_weight_offset": 9,
            }
        },
    }


class GeneratorCodegenContractTests(unittest.TestCase):
    def test_codegen_uses_batch_size_key_for_template_calls(self):
        model_info = {
            "batch_size": 4,
            "seq_len": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
        }
        hardware_config = {
            "MLEN": 64,
            "VLEN": 64,
            "BLEN": 4,
            "MATRIX_SRAM_SIZE": 1024,
            "alive_registers": [1, 2, 3],
        }
        calls: list[tuple[str, dict]] = []

        def capture(name):
            def _stub(**kwargs):
                calls.append((name, kwargs))
                return f"; {name}\n"

            return _stub

        attention_node = {
            "name": "attention",
            "operation_type": "attention",
            "dimensions": {
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
            },
        }
        ffn_node = {
            "name": "ffn",
            "operation_type": "ffn",
            "dimensions": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "activation": "silu",
            },
        }
        vit_ffn_node = {
            "name": "vit_ffn",
            "operation_type": "ffn",
            "dimensions": {
                "hidden_size": 64,
                "intermediate_size": 128,
                "activation": "gelu",
                "arch": "vit",
            },
        }
        add_node = {
            "name": "residual",
            "operation_type": "elementwise_add",
            "dimensions": {"shape": [4, 2, 64]},
        }

        with (
            patch.object(code_gen, "projection_asm", capture("projection")),
            patch.object(code_gen, "flash_attn_asm", capture("flash")),
            patch.object(code_gen, "ffn_asm", capture("ffn")),
            patch.object(code_gen, "gelu_asm", capture("gelu")),
            patch.object(code_gen, "elementwise_add_asm", capture("add")),
        ):
            code_gen._generate_attention_code(attention_node, model_info, hardware_config, _scheduler())
            code_gen._generate_ffn_code(ffn_node, model_info, hardware_config, _scheduler())
            code_gen._generate_ffn_code(vit_ffn_node, model_info, hardware_config, _scheduler())
            code_gen._generate_elementwise_add_code(add_node, model_info, hardware_config, _scheduler())

        batch_args = [
            kwargs["batch"]
            for name, kwargs in calls
            if name in {"projection", "flash", "ffn", "add"}
        ]
        gelu_batch_args = [kwargs["batch_size"] for name, kwargs in calls if name == "gelu"]

        self.assertTrue(batch_args, "test did not capture any batch-bearing template calls")
        self.assertTrue(gelu_batch_args, "test did not capture GELU batch_size")
        self.assertEqual(set(batch_args), {4})
        self.assertEqual(set(gelu_batch_args), {4})

    def test_codegen_reads_vector_sram_addresses_from_memory_layout(self):
        model_info = {
            "batch_size": 4,
            "seq_len": 2,
            "hidden_size": 64,
            "vocab_size": 1000,
        }
        hardware_config = {
            "VLEN": 64,
            "MLEN": 64,
            "BLEN": 4,
            "alive_registers": [1, 2, 3],
        }
        calls: list[tuple[str, dict]] = []

        def capture(name):
            def _stub(**kwargs):
                calls.append((name, kwargs))
                return f"; {name}\n"

            return _stub

        norm_node = {
            "name": "norm",
            "operation_type": "normalization",
            "dimensions": {
                "normalized_shape": 64,
                "norm_type": "rms_norm",
            },
        }
        add_node = {
            "name": "residual",
            "operation_type": "elementwise_add",
            "dimensions": {"shape": [4, 2, 64]},
        }
        lm_head_node = {
            "name": "lm_head",
            "operation_type": "lm_head",
            "dimensions": {"hidden_size": 64, "vocab_size": 1000},
        }

        with (
            patch.object(code_gen, "rms_norm_asm", capture("norm")),
            patch.object(code_gen, "elementwise_add_asm", capture("add")),
            patch.object(code_gen, "lm_head_asm", capture("lm_head")),
        ):
            code_gen._generate_normalization_code(norm_node, model_info, hardware_config, _scheduler())
            code_gen._generate_elementwise_add_code(add_node, model_info, hardware_config, _scheduler())
            code_gen._generate_lm_head_code(lm_head_node, model_info, hardware_config, _scheduler())

        by_name = {name: kwargs for name, kwargs in calls}
        self.assertEqual(by_name["norm"]["activation_base_address"], 111)
        self.assertEqual(by_name["norm"]["scratchpad_base_address"], 222)
        self.assertEqual(by_name["add"]["stored_activation_base_address"], 111)
        self.assertEqual(by_name["add"]["previous_activation_base_address"], 222)
        self.assertEqual(by_name["lm_head"]["activation_base_address"], 111)
        self.assertEqual(by_name["lm_head"]["result_base_address"], 222)

    def test_codegen_rejects_legacy_top_level_vector_sram_layout(self):
        model_info = {
            "batch_size": 4,
            "seq_len": 2,
            "hidden_size": 64,
        }
        hardware_config = {
            "VLEN": 64,
        }
        legacy_scheduler = {
            "vector_sram_addr": {
                "block1": 111,
                "block2": 222,
            },
            "memory_layout": {
                "fp_sram": {},
            },
            "register_assignment": {
                "hbm_addr_reg": {},
            },
        }
        norm_node = {
            "name": "norm",
            "operation_type": "normalization",
            "dimensions": {
                "normalized_shape": 64,
                "norm_type": "rms_norm",
            },
        }

        with self.assertRaises(KeyError):
            code_gen._generate_normalization_code(norm_node, model_info, hardware_config, legacy_scheduler)

    def test_scheduler_expression_errors_fail_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            layout_path = Path(tmpdir) / "mem_layout.json"
            layout_path.write_text(
                json.dumps({"vector_sram_addr": {"block1": "batch_size * missing_dim"}})
            )

            with self.assertRaises(ValueError):
                generate_mem_layout(
                    hardware_config={},
                    model_config={"batch_size": 4},
                    mem_layout_lib=str(layout_path),
                )


if __name__ == "__main__":
    unittest.main()
