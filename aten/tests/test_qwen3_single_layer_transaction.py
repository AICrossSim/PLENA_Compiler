from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from compiler.assembler.parser import parse_asm_file
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.packed_kv import PackedKVLayout
from compiler.aten.qwen3_moe_runtime import Qwen3RawBf16ExpertBankLayout


COMPILER_ROOT = Path(__file__).resolve().parents[2]


def _assemble(code: str) -> list[int]:
    assembler = AssemblyToBinary(
        str(COMPILER_ROOT / "doc" / "operation.svh"),
        str(COMPILER_ROOT / "doc" / "configuration.svh"),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".asm") as handle:
        handle.write(code)
        handle.flush()
        return [
            assembler._convert_to_binary(instruction)
            for instruction in parse_asm_file(handle.name)
        ]


def _build_transaction(
    *,
    descriptor_base: int = 0x1_0000,
    cache_position: int = 3,
    packed_layout: PackedKVLayout | None = None,
):
    compiler = PlenaCompiler(
        mlen=64,
        blen=4,
        hbm_v_prefetch_amount=4,
        hbm_v_writeback_amount=4,
    )
    layer_input = compiler.alloc(
        "layer_input", 1, 64, strict=False, physical_shape=(4, 64)
    )
    attention_norm = compiler.alloc(
        "attention_norm", 1, 64, strict=False, physical_shape=(4, 64)
    )
    q_norm = compiler.alloc(
        "q_norm", 1, 64, strict=False, physical_shape=(4, 64)
    )
    k_norm = compiler.alloc(
        "k_norm", 1, 64, strict=False, physical_shape=(4, 64)
    )
    rope_cos = compiler.alloc(
        "rope_cos", 1, 64, strict=False, physical_shape=(4, 64)
    )
    rope_sin = compiler.alloc(
        "rope_sin", 1, 64, strict=False, physical_shape=(4, 64)
    )
    post_attention_norm = compiler.alloc(
        "post_attention_norm", 1, 64, strict=False, physical_shape=(4, 64)
    )
    router = compiler.alloc(
        "router", 128, 64, strict=False, physical_shape=(128, 64)
    )

    def mx_input(name: str, role: str):
        return compiler.input(
            name,
            (64, 64),
            physical_shape=(64, 64),
            hbm_element_width=8,
            hbm_block_size=8,
            hbm_scale_width=8,
            precision_role=role,
        )

    q_weight = mx_input("q_weight", "weight")
    k_weight = mx_input("k_weight", "weight")
    v_weight = mx_input("v_weight", "weight")
    rotate_half_weight = mx_input("rotate_half_weight", "weight")
    o_weight = mx_input("o_weight", "weight")
    k_cache = mx_input("k_cache", "key")
    v_cache = mx_input("v_cache", "value")
    layout = Qwen3RawBf16ExpertBankLayout.canonical(
        descriptor_base, hidden=64, intermediate=64
    )
    packed = packed_layout or PackedKVLayout(
        kv_heads=1,
        head_dim=64,
        mlen=64,
        element_bits=8,
        scale_bits=8,
        block_size=8,
    )
    output, receipt = compiler.qwen3_tiny_single_layer_decode_transaction_v0(
        layer_input,
        attention_norm,
        q_norm,
        k_norm,
        rope_cos,
        rope_sin,
        post_attention_norm,
        router,
        q_weight,
        k_weight,
        v_weight,
        rotate_half_weight,
        o_weight,
        k_cache,
        v_cache,
        packed_layout=packed,
        descriptor_layout=layout,
        cache_position=cache_position,
    )
    return compiler, output, receipt


def test_compiler_generates_complete_single_layer_transaction_and_receipt():
    compiler, output, receipt = _build_transaction()
    compiler.emit("C_BREAK\n")
    assembly = compiler.compile()
    binary = _assemble(assembly)

    assert output.shape == (1, 64)
    assert binary
    assert receipt["scope"] == "tiny_single_layer_q_len1_decode_validation"
    assert receipt["packedkv_attention_prefix_executed"] is True
    assert receipt["compiler_lowering_valid"] is True
    assert receipt["compiler_generated_binary_emulator_parity"] is False
    assert receipt["tiny_single_layer_transaction_parity"] is False
    assert receipt["full_model_compiler_valid"] is False
    assert receipt["target_geometry_valid"] is False
    assert receipt["publication_rankable"] is False
    assert receipt["expected_assignment_count"] == 8
    assert receipt["packedkv_layout_id"].startswith("PACKED_KV-")
    assert receipt["append"]["key"][0]["token_index"] == 3
    assert receipt["append"]["value"][0]["token_index"] == 3
    assert receipt["append"]["key"][0]["transfer_rows"] == 4
    assert set(receipt["vram_boundaries"]) == {
        "layer_input",
        "attention_normalized",
        "attention_normalized_padded",
        "q_projected",
        "k_projected",
        "v_projected",
        "q_normalized",
        "k_normalized",
        "q_rope",
        "k_rope",
        "attention_output",
        "o_projected",
        "attention_residual",
        "post_attention_normalized",
        "router_logits",
        "expert_combined",
        "output",
    }

    opcodes = [line.split(maxsplit=1)[0] for line in assembly.splitlines() if line]
    assert opcodes.count("V_QWEN3_RMSNORM_BF16") == 4
    assert opcodes.count("H_STORE_V") == 2
    assert opcodes.count("V_ROUTER_LINEAR_BF16") == 1
    assert opcodes.count("V_TOPK") == 1
    assert opcodes.count("V_QWEN3_EXPERT_COMBINE_BF16") == 1
    ordered = (
        "@qwen3_stage=attention_rmsnorm",
        "@qwen3_stage=q_rmsnorm",
        "@qwen3_stage=k_rmsnorm",
        "PackedKV batch 0, selector 0",
        "@stage=post_attention_rmsnorm",
        "@stage=router_bf16",
        "@stage=topk8_runtime",
        "@stage=dispatch_runtime_expert_id",
    )
    positions = [assembly.index(marker) for marker in ordered]
    assert positions == sorted(positions)


def test_single_layer_transaction_rejects_unsealed_cache_position_and_layout():
    with pytest.raises(ValueError, match="cache_position=3"):
        _build_transaction(cache_position=4)
    with pytest.raises(ValueError, match="one 64-wide MXFP8 PackedKV head"):
        _build_transaction(
            packed_layout=PackedKVLayout(
                kv_heads=1,
                head_dim=64,
                mlen=64,
                element_bits=4,
                scale_bits=8,
                block_size=8,
            )
        )


def test_single_layer_transaction_rejects_expert_image_overlap():
    with pytest.raises(ValueError, match="overlaps HBM tensor"):
        _build_transaction(descriptor_base=0)
