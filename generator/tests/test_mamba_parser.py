"""Tests for LLMModelParser on Mamba-2 (selective state-space) configs.

Kept separate from test_llm_parser.py / test_vlm_parser.py the same way the VLM
tests are: Mamba-2 is a distinct architecture family with its own node sequence,
and mixing its fixtures into the attention tests would obscure which family a
regression belongs to.

Two config sources are exercised deliberately:

* ``doc/Model_Lib/mamba2-2.7b.json`` -- the real published shape, loaded through
  HuggingFace's ``AutoConfig`` so a drift between our JSON and ``Mamba2Config``
  shows up here rather than silently downstream.
* a scaled-down ``SimpleNamespace`` -- for the structural checks, so they run
  without any model download and stay readable.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.llm_parser import LLMModelParser

MODEL_LIB_CONFIG = Path(__file__).resolve().parents[2] / "doc" / "Model_Lib" / "mamba2-2.7b.json"

#: The per-layer node sequence Mamba-2 lowers to.  One residual, not two: the
#: mixer is a single sublayer, unlike attention + FFN.
EXPECTED_LAYER_SEQUENCE = [
    "normalization",
    "projection",
    "conv1d",
    "ssd_scan",
    "gated_rmsnorm",
    "projection",
    "elementwise_add",
]


def _make_small_mamba2_config(num_hidden_layers: int = 2):
    """A hardware-shaped but tiny Mamba-2 config.

    ``num_heads * head_dim == expand * hidden_size`` and
    ``num_heads % n_groups == 0`` both hold, which the parser enforces.
    ``num_heads`` is a multiple of BLEN so ``in_proj_out`` lands on a BLEN
    boundary and the projection covers every output column.
    """
    return SimpleNamespace(
        model_type="mamba2",
        architectures=["Mamba2ForCausalLM"],
        hidden_size=128,
        num_hidden_layers=num_hidden_layers,
        expand=2,
        head_dim=64,
        num_heads=4,
        state_size=64,
        n_groups=1,
        conv_kernel=4,
        chunk_size=64,
        vocab_size=1024,
        layer_norm_epsilon=1e-5,
        time_step_limit=(0.0, float("inf")),
        time_step_min=0.001,
        time_step_max=0.1,
        use_conv_bias=True,
        use_bias=False,
        rms_norm=True,
        hidden_act="silu",
        tie_word_embeddings=False,
    )


def _make_small_parser(num_hidden_layers: int = 2) -> LLMModelParser:
    parser = LLMModelParser("mock-mamba2")
    parser.config = _make_small_mamba2_config(num_hidden_layers)
    parser.model = SimpleNamespace()
    return parser


def _make_model_lib_parser() -> LLMModelParser:
    """Parser bound to the Model_Lib config, loaded via HuggingFace AutoConfig."""
    from transformers import AutoConfig

    parser = LLMModelParser(str(MODEL_LIB_CONFIG))
    parser.config = AutoConfig.from_pretrained(str(MODEL_LIB_CONFIG))
    parser.model = SimpleNamespace()
    return parser


class TestMamba2ModelLibConfig(unittest.TestCase):
    """The published mamba2-2.7b shape, as committed to doc/Model_Lib."""

    def test_config_file_exists_and_is_json(self):
        self.assertTrue(MODEL_LIB_CONFIG.is_file(), f"missing {MODEL_LIB_CONFIG}")
        with open(MODEL_LIB_CONFIG) as f:
            raw = json.load(f)
        self.assertEqual(raw["model_type"], "mamba2")
        self.assertEqual(raw["architectures"], ["Mamba2ForCausalLM"])

    def test_published_dimensions(self):
        with open(MODEL_LIB_CONFIG) as f:
            raw = json.load(f)
        self.assertEqual(raw["hidden_size"], 2560)
        self.assertEqual(raw["num_hidden_layers"], 64)
        self.assertEqual(raw["expand"], 2)
        self.assertEqual(raw["state_size"], 128)
        self.assertEqual(raw["conv_kernel"], 4)
        self.assertEqual(raw["head_dim"], 64)
        self.assertEqual(raw["n_groups"], 1)
        self.assertEqual(raw["chunk_size"], 256)
        self.assertEqual(raw["vocab_size"], 50288)

    def test_num_heads_is_consistent(self):
        """num_heads must equal expand * hidden_size / head_dim.

        A mismatch would silently corrupt d_inner, the dt slice width and the
        SSD batch axis, so it is pinned rather than derived.
        """
        with open(MODEL_LIB_CONFIG) as f:
            raw = json.load(f)
        self.assertEqual(raw["num_heads"], raw["expand"] * raw["hidden_size"] // raw["head_dim"])
        self.assertEqual(raw["num_heads"], 80)

    def test_required_mamba2_fields_present(self):
        with open(MODEL_LIB_CONFIG) as f:
            raw = json.load(f)
        for field in (
            "layer_norm_epsilon",
            "time_step_limit",
            "time_step_min",
            "time_step_max",
            "use_conv_bias",
            "use_bias",
            "rms_norm",
            "tie_word_embeddings",
            "torch_dtype",
        ):
            self.assertIn(field, raw, f"Model_Lib config is missing {field}")

    def test_loads_as_huggingface_mamba2config(self):
        from transformers import AutoConfig, Mamba2Config

        cfg = AutoConfig.from_pretrained(str(MODEL_LIB_CONFIG))
        self.assertIsInstance(cfg, Mamba2Config)
        self.assertEqual(cfg.hidden_size, 2560)
        self.assertEqual(cfg.num_heads, 80)
        self.assertEqual(cfg.n_groups, 1)
        self.assertEqual(cfg.state_size, 128)
        self.assertEqual(cfg.chunk_size, 256)
        self.assertEqual(list(cfg.time_step_limit), [0.0, float("inf")])


class TestMamba2Detection(unittest.TestCase):
    def test_model_type_selects_mamba2(self):
        self.assertTrue(_make_small_parser().is_mamba2())

    def test_architecture_string_selects_mamba2_without_model_type(self):
        parser = LLMModelParser("mock")
        parser.config = SimpleNamespace(
            architectures=["Mamba2ForCausalLM"],
            hidden_size=128,
            num_hidden_layers=1,
            expand=2,
            head_dim=64,
            num_heads=4,
            state_size=64,
            n_groups=1,
            conv_kernel=4,
            chunk_size=64,
            vocab_size=1024,
        )
        parser.model = SimpleNamespace()
        self.assertTrue(parser.is_mamba2())

    def test_llama_config_is_not_mamba2(self):
        parser = LLMModelParser("mock-llama")
        parser.config = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=256,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=512,
            head_dim=64,
            vocab_size=1024,
        )
        parser.model = SimpleNamespace()
        self.assertFalse(parser.is_mamba2())


class TestMamba2Dimensions(unittest.TestCase):
    def test_model_lib_derived_dimensions(self):
        dims = _make_model_lib_parser().extract_critical_dimensions()
        self.assertIn("mamba", dims)
        m = dims["mamba"]
        self.assertEqual(m["d_inner"], 2 * 2560)
        self.assertEqual(m["num_heads"] * m["head_dim"], m["d_inner"])
        # conv1d runs over [x, B, C]
        self.assertEqual(m["conv_dim"], 5120 + 2 * 1 * 128)
        # in_proj emits [z, x, B, C, dt]
        self.assertEqual(m["in_proj_out"], 2 * 5120 + 2 * 1 * 128 + 80)
        self.assertEqual(m["eps"], 1e-05)
        self.assertEqual(m["time_step_min"], 0.0)
        self.assertEqual(m["time_step_max"], float("inf"))
        self.assertTrue(m["use_conv_bias"])
        self.assertFalse(m["use_bias"])

    def test_small_config_dimensions(self):
        m = _make_small_parser().extract_critical_dimensions()["mamba"]
        self.assertEqual(m["d_inner"], 256)
        self.assertEqual(m["conv_dim"], 256 + 2 * 64)
        self.assertEqual(m["in_proj_out"], 512 + 128 + 4)
        self.assertEqual(m["heads_per_group"], 4)

    def test_inconsistent_num_heads_is_rejected(self):
        """A config where num_heads * head_dim != expand * hidden_size must fail
        loudly, not silently produce a wrong d_inner."""
        parser = _make_small_parser()
        parser.config.num_heads = 5  # 5 * 64 != 2 * 128
        with self.assertRaises(ValueError) as ctx:
            parser.extract_critical_dimensions()
        self.assertIn("num_heads", str(ctx.exception))

    def test_num_heads_not_divisible_by_groups_is_rejected(self):
        parser = _make_small_parser()
        parser.config.n_groups = 3  # 4 heads / 3 groups
        with self.assertRaises(ValueError) as ctx:
            parser.extract_critical_dimensions()
        self.assertIn("n_groups", str(ctx.exception))


class TestMamba2SymbolicGraph(unittest.TestCase):
    def test_layer_node_sequence(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        types = [n["operation_type"] for n in graph["nodes"]]
        self.assertEqual(types[0], "embedding")
        self.assertEqual(types[1 : 1 + len(EXPECTED_LAYER_SEQUENCE)], EXPECTED_LAYER_SEQUENCE)
        self.assertEqual(types[-2:], ["normalization", "lm_head"])

    def test_layer_node_names(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        names = [n["name"] for n in graph["nodes"]]
        self.assertEqual(
            names,
            [
                "embed_tokens",
                "layer_0_input_layernorm",
                "layer_0_in_proj",
                "layer_0_conv1d",
                "layer_0_ssd_scan",
                "layer_0_gated_rmsnorm",
                "layer_0_out_proj",
                "layer_0_residual",
                "final_layernorm",
                "lm_head",
            ],
        )
        self.assertEqual(names, graph["execution_order"])

    def test_execution_order_is_dense_and_monotonic(self):
        graph = _make_small_parser(num_hidden_layers=3).create_symbolic_graph(batch_size=1, seq_len=64)
        self.assertEqual([n["execution_order"] for n in graph["nodes"]], list(range(graph["total_nodes"])))

    def test_node_count_scales_with_layers(self):
        for num_layers in (1, 2, 5):
            graph = _make_small_parser(num_layers).create_symbolic_graph(batch_size=1, seq_len=64)
            # embed + 7 per layer + final_norm + lm_head
            self.assertEqual(graph["total_nodes"], 1 + 7 * num_layers + 1 + 1)
            self.assertEqual(len(graph["execution_order"]), graph["total_nodes"])

    def test_projection_dimensions(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        by_name = {n["name"]: n for n in graph["nodes"]}

        in_proj = by_name["layer_0_in_proj"]["dimensions"]
        self.assertEqual(in_proj["in_features"], 128)
        self.assertEqual(in_proj["out_features"], 644)  # 2*256 + 2*1*64 + 4
        # [z, x, B, C, dt] slice offsets must tile the output exactly.
        offsets = in_proj["slices"]
        self.assertEqual(list(offsets), ["z", "x", "B", "C", "dt"])
        cursor = 0
        for name in ("z", "x", "B", "C", "dt"):
            start, width = offsets[name]
            self.assertEqual(start, cursor, f"slice {name} does not abut the previous one")
            cursor += width
        self.assertEqual(cursor, in_proj["out_features"])

        out_proj = by_name["layer_0_out_proj"]["dimensions"]
        self.assertEqual(out_proj["in_features"], 256)
        self.assertEqual(out_proj["out_features"], 128)

    def test_conv1d_dimensions(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        conv = {n["name"]: n for n in graph["nodes"]}["layer_0_conv1d"]["dimensions"]
        self.assertEqual(conv["conv_dim"], 384)  # d_inner + 2 * n_groups * state_size
        self.assertEqual(conv["in_channels"], conv["out_channels"])
        self.assertEqual(conv["groups"], conv["in_channels"], "conv1d must be depthwise")
        self.assertEqual(conv["kernel_size"], 4)
        self.assertEqual(conv["padding"], 3, "causal conv pads kernel_size - 1 on the left only")
        self.assertTrue(conv["causal"])
        self.assertTrue(conv["depthwise"])
        self.assertEqual(conv["activation"], "silu")

    def test_ssd_scan_dimensions(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=128)
        ssd = {n["name"]: n for n in graph["nodes"]}["layer_0_ssd_scan"]["dimensions"]
        self.assertEqual(ssd["num_heads"], 4)
        self.assertEqual(ssd["head_dim"], 64)
        self.assertEqual(ssd["state_size"], 64)
        self.assertEqual(ssd["n_groups"], 1)
        self.assertEqual(ssd["chunk_size"], 64)
        self.assertEqual(ssd["num_chunks"], 2, "128 timesteps / chunk 64")
        self.assertEqual(ssd["time_step_min"], 0.0)
        self.assertEqual(ssd["time_step_max"], float("inf"))

    def test_gated_rmsnorm_dimensions(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        gn = {n["name"]: n for n in graph["nodes"]}["layer_0_gated_rmsnorm"]["dimensions"]
        # Mamba-2 normalises per group over d_inner / n_groups, not over hidden_size.
        self.assertEqual(gn["normalized_shape"], 256)
        self.assertEqual(gn["group_size"], 256)
        self.assertEqual(gn["norm_type"], "gated_rms_norm")
        self.assertEqual(gn["gate_activation"], "silu")

    def test_shapes_thread_through_the_layer(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        by_name = {n["name"]: n for n in graph["nodes"]}
        self.assertEqual(by_name["layer_0_in_proj"]["output_shape"], [1, 64, 644])
        self.assertEqual(by_name["layer_0_conv1d"]["output_shape"], [1, 64, 384])
        self.assertEqual(by_name["layer_0_ssd_scan"]["output_shape"], [1, 64, 256])
        self.assertEqual(by_name["layer_0_out_proj"]["output_shape"], [1, 64, 128])
        self.assertEqual(by_name["layer_0_residual"]["output_shape"], [1, 64, 128])

    def test_num_chunks_scales_with_seq_len(self):
        for seq_len, expected_chunks in ((64, 1), (128, 2), (192, 3), (200, 4)):
            graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(
                batch_size=1, seq_len=seq_len
            )
            ssd = {n["name"]: n for n in graph["nodes"]}["layer_0_ssd_scan"]["dimensions"]
            self.assertEqual(ssd["num_chunks"], expected_chunks, f"seq_len={seq_len}")
            # Node count is independent of seq_len: chunking is a lowering
            # parameter of one kernel, not graph structure.
            self.assertEqual(graph["total_nodes"], 1 + 7 + 1 + 1)

    def test_graph_carries_family_and_shape(self):
        graph = _make_small_parser(num_hidden_layers=1).create_symbolic_graph(batch_size=1, seq_len=64)
        self.assertEqual(graph["architecture_family"], "mamba2")
        shape = graph["mamba_shape"]
        self.assertEqual(shape["seq_len"], 64)
        self.assertEqual(shape["d_inner"], 256)
        # Every mixer node carries the same shape so code_gen can derive one
        # consistent VRAM map from any of them.
        for node in graph["nodes"]:
            node_shape = node["dimensions"].get("mamba_shape")
            if node_shape is not None:
                self.assertEqual(node_shape, shape)

    def test_model_lib_graph_shape(self):
        """Full mamba2-2.7b: 64 layers, correct widths, no ASM generated."""
        graph = _make_model_lib_parser().create_symbolic_graph(batch_size=1, seq_len=256)
        self.assertEqual(graph["total_nodes"], 1 + 7 * 64 + 1 + 1)
        by_name = {n["name"]: n for n in graph["nodes"]}
        self.assertEqual(by_name["layer_0_in_proj"]["dimensions"]["out_features"], 10576)
        self.assertEqual(by_name["layer_0_conv1d"]["dimensions"]["conv_dim"], 5376)
        self.assertEqual(by_name["layer_63_out_proj"]["dimensions"]["in_features"], 5120)
        self.assertEqual(by_name["lm_head"]["dimensions"]["vocab_size"], 50288)


class TestOtherFamiliesUnaffected(unittest.TestCase):
    """Guards the regression the Mamba-2 branch could most easily cause."""

    def test_llama_graph_still_has_six_nodes_per_layer(self):
        parser = LLMModelParser("mock-llama")
        parser.config = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            hidden_size=256,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=1,
            intermediate_size=512,
            head_dim=64,
            vocab_size=1024,
            rms_norm_eps=1e-5,
            hidden_act="silu",
        )
        parser.model = SimpleNamespace()
        graph = parser.create_symbolic_graph(batch_size=1, seq_len=32)
        self.assertEqual(graph["total_nodes"], 1 + 6 * 3 + 1 + 1)
        self.assertNotIn("architecture_family", graph)
        types = [n["operation_type"] for n in graph["nodes"]]
        self.assertEqual(
            types[1:7],
            ["normalization", "attention", "elementwise_add", "normalization", "ffn", "elementwise_add"],
        )


if __name__ == "__main__":
    unittest.main()
