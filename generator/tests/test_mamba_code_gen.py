"""Tests for code generation on Mamba-2 (selective state-space) models.

Mirrors test_vlm_code_gen.py: the config is scaled down to hardware-compatible
dimensions so the full generator pipeline (parser -> scheduler -> code_gen ->
assembler) runs with no model download, and the correctness bar for this pipeline
is that the emitted ASM assembles cleanly.

Shape choices, all of them load-bearing:

* ``state_size`` and ``chunk_size`` are multiples of MLEN=64 -- both are K
  dimensions of the SSD GEMMs, and ``batched_matmul_asm`` requires ``k % mlen``.
* ``num_heads`` is a multiple of BLEN=4 so ``in_proj_out`` lands on a BLEN
  boundary; otherwise ``projection_asm`` (which emits whole BLEN column blocks)
  would not cover the dt tail.
* ``num_heads * head_dim == expand * hidden_size``, which the parser enforces.
"""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from parser.llm_parser import LLMModelParser
from parser.hardware_parser import hardware_parser
from passes.code_gen import code_gen_pass
from scheduler import gen_scheduler


def _get_paths():
    compiler_root = Path(__file__).resolve().parents[2]
    return {
        "hw_config": str(compiler_root / "doc" / "configuration.svh"),
        "precision": str(compiler_root / "doc" / "precision.svh"),
        "mem_layout": str(compiler_root / "generator" / "scheduler" / "mem_layout_lib.json"),
        "reg_assign": str(compiler_root / "generator" / "scheduler" / "reg_assignment_lib.json"),
        "operation": str(compiler_root / "doc" / "operation.svh"),
    }


def _make_mamba2_config(num_hidden_layers: int = 1):
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


def _build(num_hidden_layers: int = 1, seq_len: int = 64):
    """Return (graph, model_info, hw_config, scheduler) for a Mamba-2 decoder."""
    paths = _get_paths()
    parser = LLMModelParser("mock-mamba2")
    parser.config = _make_mamba2_config(num_hidden_layers)
    parser.model = SimpleNamespace()

    graph = parser.create_symbolic_graph(batch_size=1, seq_len=seq_len)
    hw = hardware_parser(paths["hw_config"], paths["precision"])
    dims = parser.extract_critical_dimensions()
    model_config = {
        "hidden_size": dims["hidden_size"],
        "num_layers": dims["num_hidden_layers"],
        "seq_len": seq_len,
        "batch_size": 1,
        "vocab_size": dims["vocab_size"],
        # Mamba-2 has no FFN; d_inner is the analogous "wide" dimension the
        # scheduler's VRAM block sizing keys off.
        "intermediate_size": dims["mamba"]["d_inner"],
        "batch": 1,
    }
    sched = gen_scheduler(hw, model_config, paths["mem_layout"], paths["reg_assign"])
    model_info = dict(model_config, model_name="mamba2-mock", architecture="mamba2")
    return graph, model_info, hw, sched


def _codegen(num_hidden_layers: int = 1, seq_len: int = 64) -> str:
    graph, model_info, hw, sched = _build(num_hidden_layers, seq_len)
    return code_gen_pass(graph, model_info, hw, sched)


def _instruction_lines(asm: str) -> list[str]:
    """The lines the assembler will actually encode (comments/blanks dropped)."""
    out = []
    for line in asm.splitlines():
        line = line.strip()
        if not line or line[0] == ";" or line.startswith("//"):
            continue
        out.append(line)
    return out


class TestMamba2CodeGenStructure(unittest.TestCase):
    def setUp(self):
        self.asm = _codegen()

    def test_all_node_types_are_dispatched(self):
        """Every Mamba-2 node type must reach an emitter, not fall through."""
        for marker in (
            "(embedding)",
            "(normalization)",
            "(projection)",
            "(conv1d)",
            "(ssd_scan)",
            "(gated_rmsnorm)",
            "(elementwise_add)",
            "(lm_head)",
        ):
            self.assertIn(marker, self.asm, f"no code emitted for node type {marker}")

    def test_mixer_stages_present(self):
        self.assertIn("Mamba-2 causal depthwise conv1d", self.asm)
        self.assertIn("Mamba-2 chunked SSD scan", self.asm)
        self.assertIn("Mamba-2 gated RMSNorm", self.asm)
        for stage in (
            "@stage=mamba_dt",
            "@stage=mamba_chunk_cumsum",
            "@stage=mamba_intra_chunk",
            "@stage=mamba_decay_mask",
            "@stage=mamba_state_update",
            "@stage=mamba_inter_chunk",
            "@stage=mamba_skip",
        ):
            self.assertIn(stage, self.asm, f"SSD scan is missing stage {stage}")

    def test_conv1d_is_not_lowered_through_im2col(self):
        """The depthwise causal conv must not go through the dense im2col path.

        im2col hardcodes a square KxK patch, cannot express the causal left pad,
        and would build a dense GEMM against a block-diagonal weight.
        """
        self.assertNotIn("im2col", self.asm.lower())
        self.assertNotIn("V_SHFT_V", self.asm)

    def test_conv1d_uses_vector_taps_not_scalar_broadcast(self):
        """A depthwise tap varies per channel = per lane, so it must be a
        V_MUL_VV operand rather than a V_MUL_VF scalar broadcast."""
        conv_start = self.asm.index("Mamba-2 causal depthwise conv1d")
        conv_end = self.asm.index("Mamba-2 SSD scan")
        conv_body = self.asm[conv_start:conv_end]
        self.assertIn("V_MUL_VV", conv_body)

    def test_uses_the_mamba_specific_opcodes(self):
        # softplus has no software lowering (no logarithm in the ISA).
        self.assertIn("V_SOFTPLUS_V", self.asm)
        # time_step_limit clamp -- V_MAX_VV / V_MIN_VV do not exist.
        self.assertIn("V_MAX_VF", self.asm)
        self.assertIn("V_MIN_VF", self.asm)
        # per-head decay scalars: VRAM row -> FPRAM slots, for V_MUL_VF broadcast.
        self.assertIn("S_MAP_FP_V", self.asm)

    def test_ssd_pays_for_the_mram_round_trip(self):
        """Both SSD GEMM operands are activations, and MRAM is write-only from
        HBM, so each must be spilled with H_STORE_V before H_PREFETCH_M."""
        ssd_body = self.asm[self.asm.index("Mamba-2 chunked SSD scan") :]
        self.assertIn("H_STORE_V", ssd_body)
        self.assertIn("H_PREFETCH_M", ssd_body)
        self.assertIn("M_MM", ssd_body)

    def test_ssd_spill_operands_use_distinct_hbm_registers(self):
        """Sharing one address register would overlay the two GEMM operands."""
        ssd_body = self.asm[self.asm.index("Mamba-2 chunked SSD scan") :]
        stores = re.findall(r"^H_STORE_V .*?, a(\d+),", ssd_body, flags=re.MULTILINE)
        self.assertTrue(stores, "no H_STORE_V found in the SSD body")
        self.assertGreaterEqual(len(set(stores)), 2, "both operands spilled to the same HBM register")

    def test_no_unavailable_opcodes(self):
        """Guard against inventing ISA that doesn't exist."""
        for forbidden in ("V_MAX_VV", "V_MIN_VV", "V_LOG", "V_SQRT_V", "V_GATHER", "M_MM_VV"):
            self.assertNotIn(forbidden, self.asm, f"emitted non-existent opcode {forbidden}")

    def test_loop_trip_counts_are_compile_time_immediates(self):
        """C_LOOP_START takes an immediate only -- there is no register form and
        no branch to fall back on."""
        starts = [ln for ln in _instruction_lines(self.asm) if ln.startswith("C_LOOP_START")]
        self.assertTrue(starts, "no hardware loops emitted")
        for line in starts:
            trip = line.split(",")[-1].strip()
            self.assertTrue(trip.isdigit(), f"non-immediate loop trip count: {line}")
            self.assertGreater(int(trip), 0, f"degenerate loop: {line}")

    def test_loops_are_balanced(self):
        lines = _instruction_lines(self.asm)
        starts = sum(1 for ln in lines if ln.startswith("C_LOOP_START"))
        ends = sum(1 for ln in lines if ln.startswith("C_LOOP_END"))
        self.assertEqual(starts, ends, "unbalanced C_LOOP_START / C_LOOP_END")

    def test_addr_reg_init_covers_the_mamba_tensors(self):
        """The Mamba-2 HBM layout must be emitted, not the attention one."""
        header = self.asm[: self.asm.index("=== embed_tokens")]
        self.assertIn("C_SET_ADDR_REG", header)
        # 6 tensors: token table, in_proj, conv1d, out_proj, and the two SSD spills.
        self.assertEqual(header.count("C_SET_ADDR_REG"), 6)


class TestMamba2CodeGenScaling(unittest.TestCase):
    def test_instruction_count_scales_with_layers(self):
        one = len(_instruction_lines(_codegen(num_hidden_layers=1)))
        two = len(_instruction_lines(_codegen(num_hidden_layers=2)))
        three = len(_instruction_lines(_codegen(num_hidden_layers=3)))
        per_layer = two - one
        self.assertGreater(per_layer, 0)
        # Layers are emitted identically, so growth must be exactly linear.
        self.assertEqual(three - two, per_layer)

    def test_instruction_count_scales_with_seq_len(self):
        short = len(_instruction_lines(_codegen(seq_len=64)))
        long = len(_instruction_lines(_codegen(seq_len=128)))
        self.assertGreater(long, short, "doubling seq_len must cost more instructions")

    def test_chunk_count_scales_with_seq_len(self):
        self.assertEqual(_codegen(seq_len=64).count("; ================ chunk"), 1)
        self.assertEqual(_codegen(seq_len=128).count("; ================ chunk"), 2)
        self.assertEqual(_codegen(seq_len=192).count("; ================ chunk"), 3)

    def test_conv_prologue_is_independent_of_seq_len(self):
        """Only conv_kernel - 1 timesteps see the causal pad, whatever the
        sequence length; the rest share one hardware-loop body."""
        for seq_len in (64, 128):
            asm = _codegen(seq_len=seq_len)
            self.assertEqual(asm.count("fall before the sequence (zero pad)"), 3 * (384 // 64))


class TestMamba2CodeGenAssembles(unittest.TestCase):
    def _assemble(self, asm_text: str, label: str) -> int:
        from assembler import AssemblyToBinary

        paths = _get_paths()
        asm_tool = AssemblyToBinary(paths["operation"], paths["hw_config"])
        with tempfile.NamedTemporaryFile(suffix=".asm", mode="w", delete=False) as af:
            af.write(asm_text)
            asm_path = af.name
        mem_path = asm_path.replace(".asm", ".mem")
        try:
            binary = asm_tool.generate_binary(asm_path, mem_path)
            self.assertGreater(os.path.getsize(mem_path), 0, f"{label}: assembled .mem is empty")
            return len(binary)
        except ValueError as exc:
            raise AssertionError(f"{label}: assembler raised ValueError (u32 overflow): {exc}") from exc
        except KeyError as exc:
            raise AssertionError(f"{label}: assembler does not know opcode {exc}") from exc
        finally:
            for path in (asm_path, mem_path):
                if os.path.exists(path):
                    os.unlink(path)

    def test_single_layer_assembles(self):
        count = self._assemble(_codegen(num_hidden_layers=1, seq_len=64), "mamba2-1L-64")
        self.assertGreater(count, 0)

    def test_multi_chunk_assembles(self):
        """A sequence longer than chunk_size exercises the multi-chunk path."""
        count = self._assemble(_codegen(num_hidden_layers=1, seq_len=128), "mamba2-1L-128")
        self.assertGreater(count, 0)

    def test_two_layers_assemble(self):
        count = self._assemble(_codegen(num_hidden_layers=2, seq_len=64), "mamba2-2L-64")
        self.assertGreater(count, 0)


class TestMamba2UtilizationReport(unittest.TestCase):
    """The utilization pass must not report a Mamba-2 model as doing no work."""

    def test_projection_and_ssd_are_accounted(self):
        from passes.utilization_report import analyse_overall_utilization

        graph, model_info, _hw, _sched = _build(num_hidden_layers=2, seq_len=128)
        report = analyse_overall_utilization(graph, model_info, 64, 64, 64)

        self.assertGreater(report["operations"]["projection"], 0, "in_proj/out_proj not accounted")
        self.assertGreater(report["operations"]["ssd_scan"], 0, "SSD GEMMs not accounted")
        self.assertGreater(report["theoretical_FLOPS"]["ssd_scan"], 0)
        # Mamba-2 has no attention and no gated FFN.
        self.assertEqual(report["operations"]["attention"], 0)
        self.assertEqual(report["operations"]["ffn"], 0)

    def test_systolic_work_scales_with_layers_and_chunks(self):
        from passes.utilization_report import analyse_overall_utilization

        def ops(num_layers, seq_len):
            graph, model_info, _hw, _sched = _build(num_layers, seq_len)
            return analyse_overall_utilization(graph, model_info, 64, 64, 64)["operations"]

        one = ops(1, 64)
        two = ops(2, 64)
        self.assertEqual(two["projection"], 2 * one["projection"])
        self.assertEqual(two["ssd_scan"], 2 * one["ssd_scan"])

        # Doubling seq_len doubles the chunk count, hence the SSD GEMM count.
        long = ops(1, 128)
        self.assertEqual(long["ssd_scan"], 2 * one["ssd_scan"])


if __name__ == "__main__":
    unittest.main()
