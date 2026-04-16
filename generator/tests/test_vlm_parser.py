"""Tests for LLMModelParser on multimodal (vision-language) configs.

Kept separate from test_llm_parser.py so that pure LLM regression tests
stay isolated from VLM-specific fixtures (text_config + vision_config)
and so SigLIP/ViT encoder graph checks have a natural home.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.llm_parser import LLMModelParser


def _make_smolvlm2_config():
    """Build a SmolVLM2-shaped SimpleNamespace config (no HF download)."""
    # text_config: Llama-style language decoder with MQA
    text_cfg = SimpleNamespace(
        model_type="llama",
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=1,  # MQA!
        intermediate_size=8192,
        head_dim=64,
        vocab_size=49280,
        rms_norm_eps=1e-5,
        hidden_act="silu",
        max_position_embeddings=8192,
    )

    # vision_config: SigLIP encoder
    vision_cfg = SimpleNamespace(
        model_type="siglip_vision_model",
        hidden_size=1152,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=4304,
        head_dim=72,
        image_size=384,
        patch_size=14,
        norm_eps=1e-6,
        hidden_act="gelu_pytorch_tanh",
    )

    return SimpleNamespace(
        model_type="smolvlm",
        architectures=["SmolVLMForConditionalGeneration"],
        text_config=text_cfg,
        vision_config=vision_cfg,
    )


def _make_smolvlm2_parser():
    parser = LLMModelParser("mock-smolvlm2")
    parser.config = _make_smolvlm2_config()
    # Empty model: embed_tokens detection misses but the symbolic graph
    # is still built from the config alone.
    parser.model = SimpleNamespace()
    return parser


def test_smolvlm2_critical_dimensions():
    """extract_critical_dimensions should read text_config + vision_config."""
    parser = _make_smolvlm2_parser()
    dims = parser.extract_critical_dimensions()

    assert dims["hidden_size"] == 2048
    assert dims["num_hidden_layers"] == 24
    assert dims["vocab_size"] == 49280
    assert dims["attention"]["num_attention_heads"] == 32
    assert dims["attention"]["num_key_value_heads"] == 1, "MQA expected (num_kv_heads=1)"
    assert dims["attention"]["head_dim"] == 64, "SmolVLM2 uses explicit head_dim=64 (not hidden//heads)"
    assert dims["ffn"]["intermediate_size"] == 8192
    assert dims.get("vision", {}).get("hidden_size") == 1152
    assert dims.get("vision", {}).get("num_hidden_layers") == 27
    assert dims.get("vision", {}).get("head_dim") == 72


def test_smolvlm2_text_decoder_graph():
    """Text decoder symbolic graph should respect GQA/MQA projection dims."""
    parser = _make_smolvlm2_parser()
    graph = parser.create_symbolic_graph(batch_size=1, seq_len=8192)

    # embed + 24*(norm,attn,res,norm,ffn,res) + final_norm
    expected_text_nodes = 1 + 24 * 6 + 1
    assert graph["total_nodes"] == expected_text_nodes

    attn_nodes = [n for n in graph["nodes"] if n["operation_type"] == "attention"]
    assert attn_nodes, "expected at least one attention node in text decoder graph"
    attn = attn_nodes[0]
    assert attn["dimensions"]["q_proj"]["out_features"] == 32 * 64  # num_heads * head_dim
    assert attn["dimensions"]["k_proj"]["out_features"] == 1 * 64  # num_kv_heads * head_dim (MQA)
    assert attn["dimensions"]["v_proj"]["out_features"] == 1 * 64


def test_smolvlm2_vision_encoder_graph():
    """create_vision_symbolic_graph should emit a SigLIP-shaped graph."""
    parser = _make_smolvlm2_parser()
    vgraph = parser.create_vision_symbolic_graph(batch_size=1)
    assert vgraph is not None, "create_vision_symbolic_graph returned None for SmolVLM2"

    # patch_embed + 27*(norm,attn,res,norm,ffn,res) + final_norm
    expected_vision_nodes = 1 + 27 * 6 + 1
    assert vgraph["total_nodes"] == expected_vision_nodes
    assert vgraph.get("component") == "vision_encoder"

    vattn_nodes = [n for n in vgraph["nodes"] if n["operation_type"] == "attention"]
    assert vattn_nodes, "expected at least one attention node in vision graph"
    vattn = vattn_nodes[0]
    assert vattn["dimensions"]["head_dim"] == 72
    assert vattn["dimensions"]["num_key_value_heads"] == 16, "no GQA in SigLIP vision encoder"


if __name__ == "__main__":
    test_smolvlm2_critical_dimensions()
    test_smolvlm2_text_decoder_graph()
    test_smolvlm2_vision_encoder_graph()
    print("All SmolVLM2 parser tests passed.")
