"""Tests for code generation on multimodal (vision-language) models.

Verifies that the generator pipeline produces valid, assembler-compatible
ASM for SmolVLM2-shaped configs — covering vision encoder nodes (conv2d,
bidirectional attention, GELU FFN, vision projection) and text decoder
nodes (causal attention, gated FFN, embedding, lm_head).

The configs are intentionally scaled down to hardware-compatible dimensions
(num_attention_heads=4 == BLEN=4 so the GQA ratio constraint is satisfied),
allowing tests to run without any HF model download.
"""

import os
import sys
import tempfile
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


def _make_smolvlm2_config():
    """Build a hardware-compatible SmolVLM2-shaped SimpleNamespace config.

    Dimensions are reduced so that num_attention_heads / num_key_value_heads
    (the GQA ratio) equals BLEN=4, satisfying the flash_attn_asm assertion
    ``blen >= ratio``.  This avoids any HF model download while still
    exercising the full VLM code-gen path (text decoder + vision encoder).
    """
    text_cfg = SimpleNamespace(
        model_type="llama",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,   # MQA: ratio = 4 == BLEN
        intermediate_size=512,
        head_dim=64,
        vocab_size=49280,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        max_position_embeddings=8192,
    )
    vision_cfg = SimpleNamespace(
        model_type="siglip_vision_model",
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=512,
        head_dim=64,
        image_size=32,
        patch_size=16,
        norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )
    return SimpleNamespace(
        model_type="smolvlm",
        architectures=["SmolVLMForConditionalGeneration"],
        text_config=text_cfg,
        vision_config=vision_cfg,
    )


def _make_parser():
    parser = LLMModelParser("mock-smolvlm2")
    parser.config = _make_smolvlm2_config()
    parser.model = SimpleNamespace()
    return parser


def _build_text_graph_and_scheduler(seq_len: int = 32):
    """Return (graph, model_info, hw_config, scheduler) for the text decoder."""
    paths = _get_paths()
    parser = _make_parser()
    graph = parser.create_symbolic_graph(batch_size=1, seq_len=seq_len)
    hw = hardware_parser(paths["hw_config"], paths["precision"])
    dims = parser.extract_critical_dimensions()
    model_config = {
        "hidden_size": dims["hidden_size"],
        "num_layers": dims["num_hidden_layers"],
        "seq_len": seq_len,
        "batch_size": 1,
        "vocab_size": dims["vocab_size"],
        "intermediate_size": dims["ffn"]["intermediate_size"],
        "batch": 1,
    }
    sched = gen_scheduler(hw, model_config, paths["mem_layout"], paths["reg_assign"])
    model_info = dict(
        model_config,
        model_name="SmolVLM2-mock",
        architecture="smolvlm",
        batch=1,
    )
    return graph, model_info, hw, sched


def _build_vision_graph_and_scheduler():
    """Return (graph, model_info, hw_config, scheduler) for the vision encoder."""
    paths = _get_paths()
    parser = _make_parser()
    vgraph = parser.create_vision_symbolic_graph(batch_size=1)
    hw = hardware_parser(paths["hw_config"], paths["precision"])
    dims = parser.extract_critical_dimensions()
    model_config = {
        "hidden_size": dims["vision"]["hidden_size"],
        "num_layers": dims["vision"]["num_hidden_layers"],
        "seq_len": 4,
        "batch_size": 1,
        "vocab_size": 49280,
        "intermediate_size": dims["vision"]["intermediate_size"],
        "batch": 1,
    }
    sched = gen_scheduler(hw, model_config, paths["mem_layout"], paths["reg_assign"])
    model_info = dict(
        model_config,
        model_name="SmolVLM2-vision",
        architecture="smolvlm",
        batch=1,
    )
    return vgraph, model_info, hw, sched


def test_smolvlm2_codegen_no_m_mm_vv():
    """code_gen should not emit fabricated M_MM_VV instructions.

    All matrix multiplies must go through the real M_BTMM flash-attention
    path — no synthetic placeholder opcodes that would silently pass
    assembly but execute incorrectly on hardware.
    """
    graph, model_info, hw, sched = _build_text_graph_and_scheduler(seq_len=32)
    asm_output = code_gen_pass(graph, model_info, hw, sched)

    assert isinstance(asm_output, str) and len(asm_output) > 0, (
        "code_gen_pass returned empty output"
    )
    assert "M_MM_VV" not in asm_output, (
        "ASM contains forbidden M_MM_VV instruction — fabricated opcode detected"
    )
    assert "M_BTMM" in asm_output or "M_TMM" in asm_output, (
        "ASM missing M_BTMM/M_TMM — flash attention path was not emitted"
    )
    print("test_smolvlm2_codegen_no_m_mm_vv PASSED")


def test_smolvlm2_codegen_has_vision_nodes():
    """Vision encoder codegen emits expected operation bodies.

    Checks that:
    - The text decoder produces an embedding DMA section.
    - The vision encoder emits bidirectional flash attention (M_BTMM or M_TMM).
    - The GELU activation body appears in the vision FFN section.
    """
    # Text decoder: must contain embedding section header
    text_graph, text_model_info, hw, text_sched = _build_text_graph_and_scheduler(seq_len=32)
    text_asm = code_gen_pass(text_graph, text_model_info, hw, text_sched)

    assert "embed_tokens" in text_asm, (
        "Text decoder ASM missing embed_tokens section"
    )
    # code_gen wraps every node in '; === <name> (<type>) ==='
    assert "(embedding)" in text_asm, (
        "Text decoder ASM missing embedding operation marker"
    )

    # Vision encoder: must contain flash attention + GELU
    vgraph, vmodel_info, hw2, vsched = _build_vision_graph_and_scheduler()
    vision_asm = code_gen_pass(vgraph, vmodel_info, hw2, vsched)

    assert "M_BTMM" in vision_asm or "M_TMM" in vision_asm, (
        "Vision encoder ASM missing M_BTMM/M_TMM — flash attention not emitted"
    )
    assert "Flash Attention" in vision_asm, (
        "Vision encoder ASM missing '; Flash Attention' comment marker"
    )
    # GELU is emitted as a comment header by _generate_ffn_code for vit arch
    assert "gelu" in vision_asm.lower(), (
        "Vision encoder ASM missing GELU activation body"
    )
    print("test_smolvlm2_codegen_has_vision_nodes PASSED")


def test_smolvlm2_codegen_assembles():
    """Generated SmolVLM2 ASM must assemble without errors.

    Specifically asserts no ValueError (which the assembler raises on u32
    overflow — immediate values that don't fit in a 32-bit instruction word).
    Covers both the text decoder and vision encoder graphs.
    """
    from assembler import AssemblyToBinary

    paths = _get_paths()
    compiler_root = Path(__file__).resolve().parents[2]
    asm_tool = AssemblyToBinary(paths["operation"], paths["precision"])

    for label, build_fn in [
        ("text_decoder", lambda: _build_text_graph_and_scheduler(seq_len=32)),
        ("vision_encoder", _build_vision_graph_and_scheduler),
    ]:
        graph, model_info, hw, sched = build_fn()
        asm_text = code_gen_pass(graph, model_info, hw, sched)

        with tempfile.NamedTemporaryFile(suffix=".asm", mode="w", delete=False) as af:
            af.write(asm_text)
            asm_path = af.name
        mem_path = asm_path.replace(".asm", ".mem")
        try:
            asm_tool.generate_binary(asm_path, mem_path)
            mem_bytes = os.path.getsize(mem_path)
            assert mem_bytes > 0, f"{label}: assembled .mem is empty"
            print(f"  {label}: assembled OK ({mem_bytes} bytes)")
        except ValueError as exc:
            raise AssertionError(
                f"{label}: assembler raised ValueError (u32 overflow): {exc}"
            ) from exc
        finally:
            if os.path.exists(asm_path):
                os.unlink(asm_path)
            if os.path.exists(mem_path):
                os.unlink(mem_path)

    print("test_smolvlm2_codegen_assembles PASSED")


if __name__ == "__main__":
    test_smolvlm2_codegen_no_m_mm_vv()
    test_smolvlm2_codegen_has_vision_nodes()
    test_smolvlm2_codegen_assembles()
    print("All SmolVLM2 codegen tests passed.")
