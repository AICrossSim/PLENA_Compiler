"""
Tests for trace-based utilization reporting.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

TEST_DIR = Path(__file__).resolve().parent
MODULE_ROOT = TEST_DIR.parent
ASM_LIB_ROOT = MODULE_ROOT / "asm_lib"

sys.path.insert(0, str(MODULE_ROOT))
sys.path.insert(0, str(ASM_LIB_ROOT))

from utilization_report import analyse_trace_utilization, render_markdown_report
from vlm_parser import VLMModelParser


def make_parser() -> VLMModelParser:
    return VLMModelParser("_test_dummy_")


class ChainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.fc1 = nn.Linear(4, 8, bias=False)
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        y = self.fc1(x)
        return self.fc2(y)


class HalfLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TinyAttentionModel(nn.Module):
    def __init__(self, d: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.o_proj(q + k + v)


class TinyBranchVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Linear(4, 4, bias=False)
        self.language_model = nn.Linear(4, 4, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vision_tokens = self.visual_encoder(x)
        return self.language_model(vision_tokens)


class TestTraceUtilizationReport(unittest.TestCase):
    def setUp(self):
        self.parser = make_parser()

    def test_peak_memory_tracks_symbol_retirement(self):
        model = ChainModel()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        tree = self.parser.trace_leaf_modules(model, {"ids": ids})
        self.assertIs(tree, self.parser.traced_tree)
        self.assertEqual(self.parser.mode, "trace")

        report = analyse_trace_utilization(
            tree,
            hw={"tile_m": 4, "tile_k": 4, "tile_n": 4, "memory_capacity_bytes": 512},
        )

        self.assertEqual(report["summary"]["node_count_analysed"], 3)
        self.assertEqual(report["summary"]["peak_live_bytes"], 288)
        self.assertAlmostEqual(report["summary"]["memory_utilization_ratio"], 288 / 512)
        self.assertEqual(report["memory"]["final_live_bytes"], 96)

        input_symbols = [s for s in report["memory"]["symbols"] if s["is_model_input"]]
        self.assertEqual(len(input_symbols), 1)
        self.assertEqual(input_symbols[0]["bytes"], 48)

    def test_dtype_aware_symbol_bytes(self):
        model = HalfLinearModel().half()
        x = torch.randn(2, 4, dtype=torch.float16)
        tree = self.parser.trace_leaf_modules(model, {"x": x})
        self.assertIsNotNone(self.parser.flattened_traced_tree)

        report = analyse_trace_utilization(tree, hw={"memory_capacity_bytes": 128})

        fc_node = next(node for node in report["nodes"] if node["name"] == "fc")
        self.assertEqual(fc_node["input_dtype"], "torch.float16")
        self.assertEqual(fc_node["output_dtype"], "torch.float16")
        self.assertEqual(fc_node["input_bytes"], 16)
        self.assertEqual(fc_node["output_bytes"], 16)
        self.assertEqual(report["summary"]["peak_live_bytes"], 32)
        self.assertAlmostEqual(report["summary"]["memory_utilization_ratio"], 0.25)

    def test_attention_semantic_group_aggregates_projection_linears(self):
        model = TinyAttentionModel()
        x = torch.randn(2, 3, 4)
        tree = self.parser.trace_leaf_modules(model, {"x": x})

        report = analyse_trace_utilization(
            tree,
            hw={"tile_m": 4, "tile_k": 4, "tile_n": 4},
        )

        self.assertEqual(report["summary"]["node_count_analysed"], 4)
        self.assertEqual(report["summary"]["by_op_family"]["matmul"]["node_count"], 4)
        self.assertEqual(report["summary"]["by_semantic_group"]["attention"]["node_count"], 4)
        self.assertAlmostEqual(report["summary"]["overall_compute_utilization"], 0.75)

    def test_branch_analysis_separates_vision_and_text(self):
        model = TinyBranchVLM()
        x = torch.randn(2, 4)
        tree = self.parser.trace_leaf_modules(model, {"x": x})

        report = analyse_trace_utilization(
            tree,
            hw={"tile_m": 4, "tile_k": 4, "tile_n": 4, "memory_capacity_bytes": 128},
        )

        self.assertEqual(report["branch_analysis"]["vision"]["node_count"], 1)
        self.assertEqual(report["branch_analysis"]["text"]["node_count"], 1)
        self.assertGreater(report["branch_analysis"]["vision"]["peak_live_bytes"], 0)
        self.assertGreater(report["branch_analysis"]["text"]["peak_live_bytes"], 0)

        markdown = render_markdown_report(report)
        self.assertIn("## Vision Branch", markdown)
        self.assertIn("## Text Branch", markdown)


if __name__ == "__main__":
    unittest.main()
