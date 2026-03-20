"""
Tests for vlm_codegen_generator.py shared-context lowering.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
MODULE_ROOT = TEST_DIR.parent
ASM_LIB_ROOT = MODULE_ROOT / "asm_lib"

# Make generator/multi_model and asm_lib importable when run from any cwd.
sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(ASM_LIB_ROOT))

from vlm_codegen_handlers import LoweringResult
from vlm_codegen_env import VLMCodegenEnvironment
from vlm_codegen_generator import VLMAssemblyGenerator


def _tensor_meta(shape: list[int]) -> dict:
    return {"shape": shape, "dtype": "torch.float32", "device": "cpu"}


def _vision_mlp_nodes() -> list[dict]:
    layer_norm = {
        "name": "visual.blocks.0.norm",
        "type": "LayerNorm",
        "order": 0,
        "attrs": {"normalized_shape": [128], "eps": 1e-6},
        "in": [_tensor_meta([1, 64, 128])],
        "out": _tensor_meta([1, 64, 128]),
        "children": [],
        "weights": [],
        "in_syms": ["%0"],
        "in_sym_sources": {"%0": ""},
        "out_syms": ["%1"],
    }
    fc1 = {
        "name": "visual.blocks.0.mlp.linear_fc1",
        "type": "Linear",
        "order": 2,
        "attrs": {"in_features": 128, "out_features": 256, "bias": True},
        "in": [_tensor_meta([1, 64, 128])],
        "out": _tensor_meta([1, 64, 256]),
        "children": [],
        "weights": [
            {"name": "weight", "shape": [256, 128]},
            {"name": "bias", "shape": [256]},
        ],
        "in_syms": ["%1"],
        "in_sym_sources": {"%1": "visual.blocks.0.norm"},
        "out_syms": ["%fc1"],
    }
    fc2 = {
        "name": "visual.blocks.0.mlp.linear_fc2",
        "type": "Linear",
        "order": 3,
        "attrs": {"in_features": 256, "out_features": 128, "bias": True},
        "in": [_tensor_meta([1, 64, 256])],
        "out": _tensor_meta([1, 64, 128]),
        "children": [],
        "weights": [
            {"name": "weight", "shape": [128, 256]},
            {"name": "bias", "shape": [128]},
        ],
        "in_syms": ["%fc1"],
        "in_sym_sources": {"%fc1": "visual.blocks.0.mlp.linear_fc1"},
        "out_syms": ["%2"],
    }
    mlp = {
        "name": "visual.blocks.0.mlp",
        "type": "Qwen3VLVisionMLP",
        "order": 1,
        "attrs": {},
        "in": [_tensor_meta([1, 64, 128])],
        "out": _tensor_meta([1, 64, 128]),
        "children": [fc1, fc2],
        "weights": [],
        "in_syms": ["%1"],
        "in_sym_sources": {"%1": "visual.blocks.0.norm"},
        "out_syms": ["%2"],
    }
    return [layer_norm, mlp, fc1, fc2]


def _linear_node(name: str, *, in_features: int = 128, out_features: int = 128) -> dict:
    return {
        "name": name,
        "type": "Linear",
        "order": 0,
        "attrs": {"in_features": in_features, "out_features": out_features, "bias": False},
        "in": [_tensor_meta([1, 64, in_features])],
        "out": _tensor_meta([1, 64, out_features]),
        "children": [],
        "weights": [{"name": "weight", "shape": [out_features, in_features]}],
        "in_syms": ["%0"],
        "in_sym_sources": {"%0": "upstream"},
        "out_syms": ["%1"],
    }


def _layer_norm_nodes(*, hidden_sizes: list[int]) -> list[dict]:
    nodes: list[dict] = []
    input_sym = "%0"
    for order, hidden_size in enumerate(hidden_sizes):
        output_sym = f"%{order + 1}"
        nodes.append(
            {
                "name": f"layers.{order}.norm",
                "type": "LayerNorm",
                "order": order,
                "attrs": {"normalized_shape": [hidden_size], "eps": 1e-6},
                "in": [_tensor_meta([1, 64, hidden_size])],
                "out": _tensor_meta([1, 64, hidden_size]),
                "children": [],
                "weights": [],
                "in_syms": [input_sym],
                "in_sym_sources": {input_sym: "" if order == 0 else f"layers.{order - 1}.norm"},
                "out_syms": [output_sym],
            }
        )
        input_sym = output_sym
    return nodes


def _custom_kernel_node() -> dict:
    return {
        "name": "custom.kernel",
        "type": "CustomKernel",
        "order": 0,
        "attrs": {},
        "in": [_tensor_meta([1, 64, 128])],
        "out": _tensor_meta([1, 64, 128]),
        "children": [],
        "weights": [],
        "in_syms": ["%0"],
        "in_sym_sources": {"%0": "model_input"},
        "out_syms": ["%1"],
    }


class TestVLMCodegenSharedContext(unittest.TestCase):
    def test_qwen3_vision_mlp_uses_dedicated_op_key(self):
        env = VLMCodegenEnvironment()
        self.assertEqual(env.operation_for("Qwen3VLVisionMLP"), "vision_mlp_plena")

    def test_generate_mixed_lowering_and_shared_program(self):
        env = VLMCodegenEnvironment(hw={"mlen": 64})
        generator = VLMAssemblyGenerator(env)
        asm = generator.generate(
            _vision_mlp_nodes(),
            {
                "model_name": "unit-test",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertIn("; === Template / Partial Lowering ===", asm)
        self.assertIn("; === Covered Node Bindings ===", asm)
        self.assertIn("; === Shared-Context Lowering Summary ===", asm)
        self.assertIn("; === Reusable Type Bodies ===", asm)
        self.assertIn("; === Shared PLENA Program ===", asm)
        self.assertIn("lowering=plena_shared", asm)
        self.assertIn("call LayerNorm:", asm)
        self.assertIn("LayerNorm:", asm)
        self.assertIn("param_visual_blocks_0_mlp_linear_fc1_weight", asm)
        self.assertIn("param_visual_blocks_0_mlp_linear_fc2_weight", asm)
        self.assertIn("template=1", asm)
        self.assertIn("plena_shared=1", asm)
        self.assertIn("covered=2", asm)
        self.assertIn("; --- [Linear]  visual.blocks.0.mlp.linear_fc1", asm)
        self.assertIn("; --- [Linear]  visual.blocks.0.mlp.linear_fc2", asm)
        self.assertIn("covered_by=visual.blocks.0.mlp", asm)
        self.assertIn("input_activation=0 (template:visual.blocks.0.norm)", asm)
        self.assertIn(";   %1 -> ext_1", asm)
        self.assertIn("; registers: free_gp=", asm)

    def test_same_type_template_body_is_generated_once_and_reused(self):
        env = VLMCodegenEnvironment()
        generator = VLMAssemblyGenerator(env)
        asm = generator.generate(
            _layer_norm_nodes(hidden_sizes=[128, 128]),
            {
                "model_name": "reuse-test",
                "hidden_size": 128,
                "num_layers": 2,
                "num_attention_heads": 2,
            },
        )

        self.assertEqual(asm.count("call LayerNorm:"), 2)
        self.assertEqual(asm.count("\nLayerNorm:\n"), 1)
        self.assertIn("type_body=LayerNorm (generated once)", asm)
        self.assertIn("type_body=LayerNorm (reused)", asm)
        self.assertIn("; Layer Norm generation", asm)

    def test_same_type_with_different_signature_falls_back_to_inline(self):
        env = VLMCodegenEnvironment()
        generator = VLMAssemblyGenerator(env)
        asm = generator.generate(
            _layer_norm_nodes(hidden_sizes=[128, 256]),
            {
                "model_name": "reuse-test",
                "hidden_size": 256,
                "num_layers": 2,
                "num_attention_heads": 2,
            },
        )

        self.assertEqual(asm.count("call LayerNorm:"), 1)
        self.assertIn("type_body=LayerNorm (signature mismatch, inlined)", asm)
        self.assertEqual(asm.count("\nLayerNorm:\n"), 1)

    def test_debug_mode_omits_emitted_isa_but_keeps_context_dump(self):
        env = VLMCodegenEnvironment(hw={"mlen": 64})
        generator = VLMAssemblyGenerator(env)
        generator.debug_mode(True)

        asm = generator.generate(
            _vision_mlp_nodes(),
            {
                "model_name": "unit-test",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertIn("asm_omitted_in_debug_mode", asm)
        self.assertIn("; === Shared Context Debug Dump ===", asm)
        self.assertNotIn("; Allocate VRAM Matrix", asm)

    def test_codegen_linear_uses_qkv_registers_from_attention_projection_names(self):
        env = VLMCodegenEnvironment()
        generator = VLMAssemblyGenerator(env)

        asm = generator.generate(
            [
                _linear_node("layers.0.self_attn.q_proj"),
                _linear_node("layers.0.self_attn.k_proj"),
                _linear_node("layers.0.self_attn.v_proj"),
            ],
            {
                "model_name": "linear-regs",
                "hidden_size": 128,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertIn("weight_reg=q_weight_offset:a2", asm)
        self.assertIn("weight_reg=k_weight_offset:a3", asm)
        self.assertIn("weight_reg=v_weight_offset:a4", asm)

    def test_codegen_linear_uses_ffn_register_for_mlp_projections(self):
        env = VLMCodegenEnvironment()
        generator = VLMAssemblyGenerator(env)

        asm = generator.generate(
            [
                _linear_node("layers.0.mlp.gate_proj", out_features=256),
                _linear_node("layers.0.mlp.up_proj", out_features=256),
                _linear_node("layers.0.mlp.down_proj", in_features=256),
                _linear_node("visual.blocks.0.mlp.linear_fc1", out_features=256),
                _linear_node("visual.blocks.0.mlp.linear_fc2", in_features=256),
            ],
            {
                "model_name": "linear-regs",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertEqual(asm.count("weight_reg=ffn_weight_offset:a5"), 5)

    def test_codegen_linear_falls_back_to_default_projection_register(self):
        env = VLMCodegenEnvironment()
        generator = VLMAssemblyGenerator(env)

        asm = generator.generate(
            [_linear_node("vision.patch_merger.proj")],
            {
                "model_name": "linear-regs",
                "hidden_size": 128,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertIn("weight_reg=q_weight_offset:a2", asm)

    def test_generator_can_register_external_handler(self):
        env = VLMCodegenEnvironment()
        env.register_type("CustomKernel", "custom_kernel")
        generator = VLMAssemblyGenerator(env, auto_register_default_handlers=False)

        def custom_handler(
            codegen: VLMAssemblyGenerator,
            node: dict,
            model_info: dict,
        ) -> LoweringResult:
            del model_info
            activation_addr, _ = codegen.resolve_input_activation(node)
            codegen.bind_template_outputs(node, activation_addr)
            return codegen.wrap_template(
                "; external handler body\n",
                reuse_label=node["type"],
                comments=[f"custom_activation={activation_addr}"],
            )

        generator.register_handler("custom_kernel", custom_handler)

        asm = generator.generate(
            [_custom_kernel_node()],
            {
                "model_name": "custom-handler",
                "hidden_size": 128,
                "num_layers": 1,
                "num_attention_heads": 2,
            },
        )

        self.assertIn("call CustomKernel:", asm)
        self.assertIn("\nCustomKernel:\n", asm)
        self.assertIn("custom_activation=0", asm)
        self.assertIn("generated once", asm)


if __name__ == "__main__":
    unittest.main()
