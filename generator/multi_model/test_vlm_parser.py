"""
Tests for vlm_parser.py using small synthetic networks.
No model downloads required — all networks are built inline with torch.nn.

Run:
    uv run python -m pytest parser/test_vlm_parser.py -v
  or:
    uv run python parser/test_vlm_parser.py
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

# Make the parser package importable when run from any cwd
sys.path.insert(0, str(Path(__file__).parent))

from vlm_parser import VLMModelParser, _static_attrs, combine_traces, flatten_call_tree


# ---------------------------------------------------------------------------
# Small synthetic modules that mimic real Qwen3-VL components
# ---------------------------------------------------------------------------


class FakeRMSNorm(nn.Module):
    """Mimics Qwen3VLTextRMSNorm: has variance_epsilon + weight."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class FakeVisionRoPE(nn.Module):
    """Mimics Qwen3VLVisionRotaryEmbedding: has dim + theta + inv_freq."""

    def __init__(self, dim: int = 64, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = nn.Parameter(torch.ones(dim // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FakeTextRoPE(nn.Module):
    """Mimics Qwen3VLTextRotaryEmbedding: has mrope_section + rope_type + inv_freq."""

    def __init__(self):
        super().__init__()
        self.rope_type = "default"
        self.mrope_section = [16, 24, 24]
        self.inv_freq = nn.Parameter(torch.ones(32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MiniAttn(nn.Module):
    """Three-linear pseudo-attention (q + k + v projections)."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x) + self.k(x) + self.v(x)


class MiniMLP(nn.Module):
    """Linear → LayerNorm → Linear."""

    def __init__(self, d_in: int = 8, d_hidden: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.norm = nn.LayerNorm(d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.norm(self.fc1(x)))


class MiniBlock(nn.Module):
    """Decoder-like block: attn followed by mlp."""

    def __init__(self, d: int = 8):
        super().__init__()
        self.attn = MiniAttn(d)
        self.mlp = MiniMLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x))


class MiniTransformer(nn.Module):
    """Embedding → N × MiniBlock → FakeRMSNorm."""

    def __init__(self, vocab: int = 32, d: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([MiniBlock(d) for _ in range(n_layers)])
        self.norm = FakeRMSNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


class MiniBlockWithResidual(nn.Module):
    """
    Decoder-like block with explicit residual adds:
        x = x + attn(x)
        x = x + mlp(x)

    These bare tensor '+' ops are INVISIBLE to the hook-based tracer but
    ARE captured by the FX-based tracer as 'elementwise_add' nodes.
    """

    def __init__(self, d: int = 8):
        super().__init__()
        self.attn = MiniAttn(d)
        self.mlp = MiniMLP(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MiniTransformerWithResidual(nn.Module):
    """Embedding → N × MiniBlockWithResidual → FakeRMSNorm."""

    def __init__(self, vocab: int = 32, d: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([MiniBlockWithResidual(d) for _ in range(n_layers)])
        self.norm = FakeRMSNorm(d)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Helper — create a VLMModelParser without loading any real model
# ---------------------------------------------------------------------------


def make_parser() -> VLMModelParser:
    """Instantiate VLMModelParser with a dummy path (no download triggered)."""
    return VLMModelParser("_test_dummy_")


# ---------------------------------------------------------------------------
# Test suite: _static_attrs
# ---------------------------------------------------------------------------


class TestStaticAttrs(unittest.TestCase):
    def test_linear_with_bias(self):
        m = nn.Linear(4, 8, bias=True)
        a = _static_attrs(m)
        self.assertEqual(a["in_features"], 4)
        self.assertEqual(a["out_features"], 8)
        self.assertTrue(a["bias"])

    def test_linear_no_bias(self):
        m = nn.Linear(4, 8, bias=False)
        a = _static_attrs(m)
        self.assertFalse(a["bias"])

    def test_layer_norm(self):
        m = nn.LayerNorm([16], eps=1e-5)
        a = _static_attrs(m)
        self.assertEqual(a["normalized_shape"], [16])
        self.assertAlmostEqual(a["eps"], 1e-5)

    def test_embedding(self):
        m = nn.Embedding(100, 32)
        a = _static_attrs(m)
        self.assertEqual(a["num_embeddings"], 100)
        self.assertEqual(a["embedding_dim"], 32)

    def test_conv3d(self):
        m = nn.Conv3d(3, 16, kernel_size=(2, 14, 14), stride=(2, 14, 14))
        a = _static_attrs(m)
        self.assertEqual(a["in_channels"], 3)
        self.assertEqual(a["out_channels"], 16)
        self.assertEqual(a["kernel_size"], [2, 14, 14])
        self.assertEqual(a["stride"], [2, 14, 14])

    def test_fake_rms_norm(self):
        m = FakeRMSNorm(64, eps=1e-5)
        a = _static_attrs(m)
        self.assertEqual(a["hidden_size"], 64)
        self.assertAlmostEqual(a["eps"], 1e-5)

    def test_fake_vision_rope(self):
        m = FakeVisionRoPE(dim=64, theta=10000.0)
        a = _static_attrs(m)
        self.assertEqual(a["dim"], 64)
        self.assertEqual(a["theta"], 10000.0)
        self.assertEqual(a["freq_table_shape"], [32])  # dim // 2

    def test_fake_text_rope(self):
        m = FakeTextRoPE()
        a = _static_attrs(m)
        self.assertEqual(a["rope_type"], "default")
        self.assertEqual(a["mrope_section"], [16, 24, 24])
        self.assertEqual(a["freq_table_shape"], [32])

    def test_unknown_module_returns_empty(self):
        class Mystery(nn.Module):
            def forward(self, x):
                return x

        self.assertEqual(_static_attrs(Mystery()), {})


# ---------------------------------------------------------------------------
# Test suite: flatten_call_tree
# ---------------------------------------------------------------------------


class TestFlattenCallTree(unittest.TestCase):
    def _make_tree(self):
        """
        Build a hand-crafted tree:
            root(0) → a(1) → b(2)
            root(0) → c(3)
        """
        b = {"name": "b", "order": 2, "children": []}
        a = {"name": "a", "order": 1, "children": [b]}
        c = {"name": "c", "order": 3, "children": []}
        root = {"name": "root", "order": 0, "children": [a, c]}
        return root

    def test_length(self):
        flat = flatten_call_tree(self._make_tree())
        self.assertEqual(len(flat), 4)

    def test_sorted_by_order(self):
        flat = flatten_call_tree(self._make_tree())
        orders = [n["order"] for n in flat]
        self.assertEqual(orders, sorted(orders))

    def test_all_names_present(self):
        flat = flatten_call_tree(self._make_tree())
        self.assertEqual({n["name"] for n in flat}, {"root", "a", "b", "c"})

    def test_single_node(self):
        root = {"name": "x", "order": 0, "children": []}
        flat = flatten_call_tree(root)
        self.assertEqual(len(flat), 1)
        self.assertEqual(flat[0]["name"], "x")

    def test_deep_chain(self):
        # a(0) → b(1) → c(2) → d(3)
        d = {"name": "d", "order": 3, "children": []}
        c = {"name": "c", "order": 2, "children": [d]}
        b = {"name": "b", "order": 1, "children": [c]}
        a = {"name": "a", "order": 0, "children": [b]}
        flat = flatten_call_tree(a)
        self.assertEqual([n["name"] for n in flat], ["a", "b", "c", "d"])


# ---------------------------------------------------------------------------
# Test suite: trace_leaf_modules
# ---------------------------------------------------------------------------


class TestTraceLeafModules(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()

    # ── return type and root node ─────────────────────────────────────────

    def test_returns_dict(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertIsInstance(tree, dict)

    def test_required_keys_present(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        for key in (
            "name", "type", "order", "attrs", "in", "out", "children",
            "weights", "in_syms", "out_syms", "in_sym_sources",
        ):
            self.assertIn(key, tree, f"missing key: {key}")

    def test_root_is_top_level_module(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertEqual(tree["type"], "MiniMLP")
        self.assertEqual(tree["name"], "")   # root module has empty dotted name
        self.assertEqual(tree["order"], 0)   # first module called

    # ── children structure ────────────────────────────────────────────────

    def test_direct_children_count_mlp(self):
        # MiniMLP.forward calls fc1 → norm → fc2 sequentially
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertEqual(len(tree["children"]), 3)

    def test_children_names_mlp(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        child_names = [c["name"] for c in tree["children"]]
        self.assertEqual(child_names, ["fc1", "norm", "fc2"])

    def test_children_types_mlp(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        child_types = [c["type"] for c in tree["children"]]
        self.assertEqual(child_types, ["Linear", "LayerNorm", "Linear"])

    # ── execution order ───────────────────────────────────────────────────

    def test_execution_order_is_consecutive(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        orders = [n["order"] for n in flat]
        self.assertEqual(orders, list(range(len(flat))))

    # ── I/O shape recording ───────────────────────────────────────────────

    def test_root_input_shape(self):
        model = MiniMLP()
        x = torch.randn(2, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        self.assertEqual(len(tree["in"]), 1)
        self.assertEqual(tree["in"][0]["shape"], [2, 8])

    def test_root_output_shape(self):
        model = MiniMLP()
        x = torch.randn(2, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        self.assertIsNotNone(tree["out"])
        self.assertEqual(tree["out"]["shape"], [2, 8])

    def test_inner_linear_output_shape(self):
        model = MiniMLP()
        x = torch.randn(2, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        fc1 = tree["children"][0]   # fc1: (2,8) → (2,16)
        self.assertEqual(fc1["out"]["shape"], [2, 16])

    # ── static attrs ──────────────────────────────────────────────────────

    def test_linear_attrs_captured(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        fc1 = next(n for n in flat if n["name"] == "fc1")
        self.assertEqual(fc1["attrs"]["in_features"], 8)
        self.assertEqual(fc1["attrs"]["out_features"], 16)

    def test_layer_norm_attrs_captured(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        norm = next(n for n in flat if n["name"] == "norm")
        self.assertEqual(norm["attrs"]["normalized_shape"], [16])

    # ── nested network ────────────────────────────────────────────────────

    def test_nested_names_present(self):
        model = MiniTransformer(vocab=32, d=8, n_layers=2)
        ids = torch.randint(0, 32, (1, 4))
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        print("== Nested Call Tree ==")
        print(tree)  # for debugging if test fails
        print("======================")
        flat = flatten_call_tree(tree)
        names = {n["name"] for n in flat}
        for expected in (
            "embed",
            "blocks.0",
            "blocks.0.attn",
            "blocks.0.attn.q",
            "blocks.0.attn.k",
            "blocks.0.attn.v",
            "blocks.0.mlp",
            "blocks.1",
            "norm",
        ):
            self.assertIn(expected, names, f"expected '{expected}' in trace")

    def test_module_list_not_called(self):
        # nn.ModuleList is iterated, never called — should not appear in trace
        model = MiniTransformer(vocab=32, d=8, n_layers=2)
        ids = torch.randint(0, 32, (1, 4))
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        flat = flatten_call_tree(tree)
        names = {n["name"] for n in flat}
        self.assertNotIn("blocks", names)

    def test_total_node_count(self):
        # 1 root + 1 embed + 2*(block + attn + q,k,v + mlp + fc1,norm,fc2) + 1 rms_norm
        # = 1 + 1 + 2*9 + 1 = 21
        model = MiniTransformer(vocab=32, d=8, n_layers=2)
        ids = torch.randint(0, 32, (1, 4))
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        flat = flatten_call_tree(tree)
        self.assertEqual(len(flat), 21)

    def test_embedding_attrs_nested(self):
        model = MiniTransformer(vocab=32, d=8, n_layers=2)
        ids = torch.randint(0, 32, (1, 4))
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        flat = flatten_call_tree(tree)
        embed_node = next(n for n in flat if n["name"] == "embed")
        self.assertEqual(embed_node["attrs"]["num_embeddings"], 32)
        self.assertEqual(embed_node["attrs"]["embedding_dim"], 8)

    def test_fake_rms_norm_attrs_nested(self):
        model = MiniTransformer(vocab=32, d=8, n_layers=2)
        ids = torch.randint(0, 32, (1, 4))
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        flat = flatten_call_tree(tree)
        norm_node = next(n for n in flat if n["name"] == "norm")
        self.assertEqual(norm_node["attrs"]["hidden_size"], 8)
        self.assertIn("eps", norm_node["attrs"])

    # ── RoPE modules must appear in trace ─────────────────────────────────

    def test_vision_rope_included_in_trace(self):
        class ModelWithVisionRoPE(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = FakeVisionRoPE(dim=16)
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                _ = self.rope(x)
                return self.fc(x)

        model = ModelWithVisionRoPE()
        x = torch.randn(1, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        flat = flatten_call_tree(tree)
        names = {n["name"] for n in flat}
        self.assertIn("rope", names)
        rope_node = next(n for n in flat if n["name"] == "rope")
        self.assertEqual(rope_node["attrs"]["dim"], 16)

    def test_text_rope_included_in_trace(self):
        class ModelWithTextRoPE(nn.Module):
            def __init__(self):
                super().__init__()
                self.rope = FakeTextRoPE()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                _ = self.rope(x)
                return self.fc(x)

        model = ModelWithTextRoPE()
        x = torch.randn(1, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        flat = flatten_call_tree(tree)
        names = {n["name"] for n in flat}
        self.assertIn("rope", names)
        rope_node = next(n for n in flat if n["name"] == "rope")
        self.assertEqual(rope_node["attrs"]["rope_type"], "default")

    # ── hook cleanup ──────────────────────────────────────────────────────

    def test_hooks_removed_after_normal_run(self):
        model = MiniMLP()
        self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        for _, m in model.named_modules():
            self.assertEqual(len(m._forward_hooks), 0, f"{m}: hooks not cleaned")
            self.assertEqual(len(m._forward_pre_hooks), 0, f"{m}: pre-hooks not cleaned")

    def test_hooks_removed_after_exception(self):
        class ExplodingMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)

            def forward(self, x):
                raise RuntimeError("intentional boom")

        model = ExplodingMLP()
        with self.assertRaises(RuntimeError):
            self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        for _, m in model.named_modules():
            self.assertEqual(len(m._forward_hooks), 0)
            self.assertEqual(len(m._forward_pre_hooks), 0)

    # ── verbose flag ──────────────────────────────────────────────────────

    def test_verbose_prints_hooking_lines(self):
        model = MiniMLP()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)}, verbose=True)
        output = buf.getvalue()
        self.assertIn("hooking", output)
        # Should mention at least the 3 sub-modules
        self.assertGreaterEqual(output.count("hooking"), 3)

    def test_verbose_false_prints_nothing(self):
        model = MiniMLP()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)}, verbose=False)
        self.assertEqual(buf.getvalue(), "")

    # ── no_grad is active during trace ───────────────────────────────────

    def test_no_grad_during_trace(self):
        grad_was_enabled = []

        class WatchGrad(nn.Module):
            def forward(self, x):
                grad_was_enabled.append(torch.is_grad_enabled())
                return x

        model = WatchGrad()
        self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertEqual(grad_was_enabled, [False])


# ---------------------------------------------------------------------------
# Test suite: trace_leaf_modules — symbols and weights
# ---------------------------------------------------------------------------


class TestTraceLeafModulesSymbolsWeights(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()

    # ── weights field ────────────────────────────────────────────────────

    def test_weights_is_list(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            self.assertIsInstance(n["weights"], list)

    def test_weight_entries_have_name_and_shape(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        fc1 = next(n for n in flat if n["name"] == "fc1")
        for w in fc1["weights"]:
            self.assertIn("name", w)
            self.assertIn("shape", w)
            self.assertIsInstance(w["shape"], list)

    def test_linear_weight_shape(self):
        # nn.Linear(8, 16) with bias → weight [16, 8], bias [16]
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        fc1 = next(n for n in flat if n["name"] == "fc1")
        w_entry = next(w for w in fc1["weights"] if w["name"] == "weight")
        self.assertEqual(w_entry["shape"], [16, 8])
        b_entry = next(w for w in fc1["weights"] if w["name"] == "bias")
        self.assertEqual(b_entry["shape"], [16])

    def test_module_without_params_has_empty_weights(self):
        # MiniBlock itself has no direct parameters — only its children do
        model = MiniBlock()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertEqual(tree["weights"], [])

    def test_weights_are_direct_only(self):
        # MiniMLP root has no *direct* parameters (they belong to children)
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        self.assertEqual(tree["weights"], [])

    # ── in_syms / out_syms ───────────────────────────────────────────────

    def test_in_syms_parallel_to_in(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            self.assertEqual(len(n["in_syms"]), len(n["in"]))

    def test_in_syms_are_strings(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            for s in n["in_syms"]:
                self.assertIsInstance(s, str)
                self.assertTrue(s.startswith("%"))

    def test_out_syms_nonempty_for_tensor_output(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            self.assertGreater(len(n["out_syms"]), 0, f"{n['name']} has no out_syms")

    def test_out_syms_are_strings(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            for s in n["out_syms"]:
                self.assertIsInstance(s, str)
                self.assertTrue(s.startswith("%"))

    # ── in_sym_sources ───────────────────────────────────────────────────

    def test_in_sym_sources_keys_match_in_syms(self):
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        flat = flatten_call_tree(tree)
        for n in flat:
            for s in n["in_syms"]:
                self.assertIn(s, n["in_sym_sources"])

    def test_root_input_source_is_empty(self):
        # Tensors fed directly from the caller have no producer module
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        for s in tree["in_syms"]:
            self.assertEqual(tree["in_sym_sources"][s], "")

    def test_symbol_source_links_sequential_modules(self):
        # In fc1 → norm → fc2: fc2's input symbol should be produced by norm
        model = MiniMLP()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(2, 8)})
        fc2 = tree["children"][2]  # fc1(0) → norm(1) → fc2(2)
        for s in fc2["in_syms"]:
            self.assertEqual(fc2["in_sym_sources"][s], "norm")

    def test_symbol_reuse_across_same_tensor(self):
        # The same tensor flowing root → child must share the same symbol
        model = MiniMLP()
        x = torch.randn(2, 8)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        root_in_sym = tree["in_syms"][0]
        fc1 = tree["children"][0]  # fc1 receives the same tensor
        self.assertEqual(fc1["in_syms"][0], root_in_sym)

    def test_output_symbol_differs_from_input_symbol(self):
        # Each module produces a new (or at least different identity) tensor
        model = nn.Linear(8, 16)
        tree = self.parser.trace_leaf_modules(model, {"input": torch.randn(2, 8)})
        in_sym = tree["in_syms"][0]
        out_sym = tree["out_syms"][0]
        self.assertNotEqual(in_sym, out_sym)


# ---------------------------------------------------------------------------
# Test suite: trace_fx_module_level
# ---------------------------------------------------------------------------

# Leaf types for all FX tests: trace INTO MiniBlock* but stop at these.
_FX_LEAVES = {"MiniAttn", "MiniMLP", "Embedding", "FakeRMSNorm"}


class TestFxModuleLevelTrace(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()

    # ── return type and source tag ────────────────────────────────────────

    def test_returns_list(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        self.assertIsInstance(nodes, list)

    def test_all_nodes_tagged_fx(self):
        model = MiniMLP()
        nodes = self.parser.trace_fx_module_level(model, {"LayerNorm", "Linear"})
        for n in nodes:
            self.assertEqual(n["source"], "fx")

    def test_required_keys_present(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        for key in ("name", "type", "order", "attrs", "in", "out", "source", "children"):
            self.assertIn(key, nodes[0], f"missing key: {key}")

    # ── residual add discovery (the core motivation) ──────────────────────

    def test_hook_trace_misses_residuals(self):
        """Baseline: hook trace cannot see bare tensor + ops."""
        model = MiniBlockWithResidual()
        tree = self.parser.trace_leaf_modules(model, {"x": torch.randn(1, 4, 8)})
        types = {n["type"] for n in flatten_call_tree(tree)}
        self.assertNotIn("elementwise_add", types)

    def test_fx_trace_captures_residuals(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        types = [n["type"] for n in nodes]
        self.assertIn("elementwise_add", types)

    def test_residual_add_count_single_block(self):
        # MiniBlockWithResidual has exactly 2 residual adds
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        adds = [n for n in nodes if n["type"] == "elementwise_add"]
        self.assertEqual(len(adds), 2)

    # ── module nodes ──────────────────────────────────────────────────────

    def test_leaf_module_types_present(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        types = {n["type"] for n in nodes}
        self.assertIn("MiniAttn", types)
        self.assertIn("MiniMLP", types)

    def test_leaf_module_attrs_captured(self):
        # MiniMLP is a leaf → _static_attrs returns {} (unknown type), attrs is a dict
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        for n in nodes:
            self.assertIsInstance(n["attrs"], dict)

    def test_known_leaf_linear_attrs(self):
        # If we make Linear a leaf, its attrs should be populated
        model = MiniMLP()
        nodes = self.parser.trace_fx_module_level(model, {"Linear", "LayerNorm"})
        linear_nodes = [n for n in nodes if n["type"] == "Linear"]
        self.assertTrue(len(linear_nodes) > 0)
        for n in linear_nodes:
            self.assertIn("in_features", n["attrs"])
            self.assertIn("out_features", n["attrs"])

    # ── execution order ───────────────────────────────────────────────────

    def test_order_is_consecutive_from_zero(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        orders = [n["order"] for n in nodes]
        self.assertEqual(orders, list(range(len(nodes))))

    # ── shapes are None (no real forward pass) ────────────────────────────

    def test_in_out_are_none(self):
        model = MiniBlockWithResidual()
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        for n in nodes:
            self.assertIsNone(n["in"])
            self.assertIsNone(n["out"])

    # ── full transformer node count ───────────────────────────────────────

    def test_full_transformer_node_count(self):
        # Per layer: attn(1) + add(1) + mlp(1) + add(1) = 4
        # 2 layers + embed(1) + norm(1) = 4*2 + 2 = 10
        model = MiniTransformerWithResidual(vocab=32, d=8, n_layers=2)
        nodes = self.parser.trace_fx_module_level(model, _FX_LEAVES)
        self.assertEqual(len(nodes), 10)

    # ── error on dynamic control flow ─────────────────────────────────────

    def test_dynamic_model_raises_runtime_error(self):
        class DynamicModel(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # data-dependent branch: FX cannot trace this
                if x.sum() > 0:
                    return x * 2
                return x

        with self.assertRaises(RuntimeError):
            self.parser.trace_fx_module_level(DynamicModel(), set())


# ---------------------------------------------------------------------------
# Test suite: combine_traces
# ---------------------------------------------------------------------------


class TestCombineTraces(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()
        # hook sub-network: simple MiniMLP (4 nodes)
        self.hook_model = MiniMLP()
        self.hook_tree = self.parser.trace_leaf_modules(
            self.hook_model, {"x": torch.randn(2, 8)}
        )
        # fx sub-network: MiniBlockWithResidual (4 nodes: attn, add, mlp, add)
        self.fx_model = MiniBlockWithResidual()
        self.fx_nodes = self.parser.trace_fx_module_level(self.fx_model, _FX_LEAVES)

    def _combined(self):
        return combine_traces(self.hook_tree, self.fx_nodes)

    # ── return type ───────────────────────────────────────────────────────

    def test_returns_list(self):
        self.assertIsInstance(self._combined(), list)

    # ── source tags ───────────────────────────────────────────────────────

    def test_hook_nodes_tagged(self):
        combined = self._combined()
        hook_part = [n for n in combined if n.get("source") == "hook"]
        self.assertEqual(len(hook_part), len(flatten_call_tree(self.hook_tree)))

    def test_fx_nodes_tagged(self):
        combined = self._combined()
        fx_part = [n for n in combined if n.get("source") == "fx"]
        self.assertEqual(len(fx_part), len(self.fx_nodes))

    def test_only_hook_and_fx_sources(self):
        combined = self._combined()
        sources = {n.get("source") for n in combined}
        self.assertEqual(sources, {"hook", "fx"})

    # ── ordering ──────────────────────────────────────────────────────────

    def test_hook_nodes_come_before_fx_nodes(self):
        combined = self._combined()
        hook_orders = [n["order"] for n in combined if n["source"] == "hook"]
        fx_orders   = [n["order"] for n in combined if n["source"] == "fx"]
        self.assertLess(max(hook_orders), min(fx_orders))

    def test_global_order_is_consecutive(self):
        combined = self._combined()
        orders = [n["order"] for n in combined]
        self.assertEqual(orders, list(range(len(combined))))

    # ── total count ───────────────────────────────────────────────────────

    def test_total_node_count(self):
        # MiniMLP hook: 4 nodes  (root + fc1 + norm + fc2)
        # MiniBlockWithResidual FX: 4 nodes  (attn + add + mlp + add)
        self.assertEqual(len(self._combined()), 8)

    # ── shape info survives ───────────────────────────────────────────────

    def test_hook_root_has_real_input_shape(self):
        combined = self._combined()
        root = next(n for n in combined if n["source"] == "hook" and n["name"] == "")
        self.assertIsNotNone(root["in"])
        self.assertEqual(root["in"][0]["shape"], [2, 8])

    def test_fx_nodes_have_no_shapes(self):
        combined = self._combined()
        for n in combined:
            if n["source"] == "fx":
                self.assertIsNone(n["in"])
                self.assertIsNone(n["out"])

    # ── original nodes are not mutated ────────────────────────────────────

    def test_hook_tree_not_mutated(self):
        # The original tree root should not gain a "source" key
        self.assertNotIn("source", self.hook_tree)

    def test_fx_nodes_not_mutated(self):
        original_orders = [n["order"] for n in self.fx_nodes]
        self._combined()
        self.assertEqual([n["order"] for n in self.fx_nodes], original_orders)


if __name__ == "__main__":
    unittest.main(verbosity=2)
