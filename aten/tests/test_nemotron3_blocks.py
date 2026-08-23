from aten.nemotron3.blocks import (
    NemotronAttentionShape,
    NemotronAttentionWeights,
    allocate_nemotron_gqa_decode_cache,
    NemotronMoeConstants,
    NemotronMoeShape,
    NemotronMoeWeights,
    emit_nemotron_attention_block,
    emit_nemotron_moe_block,
)
from aten.plena import PlenaCompiler, reserve_expert_weight_table


def _program() -> tuple[PlenaCompiler, object]:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=0,
        ),
        name="hidden",
    )
    return prog, hidden


def _weight(prog: PlenaCompiler, name: str, rows: int, cols: int, *, bf16=False):
    return prog.input(
        name,
        shape=(rows, cols),
        physical_shape=(rows, cols),
        real_data_ratio=2.0 if bf16 else None,
    )


def test_connected_nemotron_gqa_consumes_projected_qkv() -> None:
    prog, hidden = _program()
    output = emit_nemotron_attention_block(
        prog,
        hidden,
        shape=NemotronAttentionShape(
            hidden=64,
            query_heads=2,
            kv_heads=1,
            head_dim=64,
        ),
        weights=NemotronAttentionWeights(
            q=_weight(prog, "w_q", 64, 128),
            k=_weight(prog, "w_k", 64, 64),
            v=_weight(prog, "w_v", 64, 64),
            out=_weight(prog, "w_o", 128, 64),
        ),
        rows=1,
    )
    assembly = prog.compile()

    assert output.shape == (1, 64)
    assert "nemotron_attention_q" in assembly
    assert "nemotron_attention_k_scratch0" in assembly
    assert assembly.count("VRAM Sub Projection T To") == 2


def test_four_token_nemotron_gqa_uses_persistent_bf16_cache() -> None:
    prog, hidden = _program()
    shape = NemotronAttentionShape(
        hidden=64,
        query_heads=2,
        kv_heads=1,
        head_dim=64,
    )
    weights = NemotronAttentionWeights(
        q=_weight(prog, "cache_w_q", 64, 128),
        k=_weight(prog, "cache_w_k", 64, 64),
        v=_weight(prog, "cache_w_v", 64, 64),
        out=_weight(prog, "cache_w_o", 128, 64),
    )
    cache = allocate_nemotron_gqa_decode_cache(
        prog,
        shape=shape,
        max_tokens=4,
    )

    outputs = [
        emit_nemotron_attention_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            rows=1,
            name=f"gqa_token{token}",
            cache=cache,
            token_index=token,
        )
        for token in range(4)
    ]
    assembly = prog.compile()

    assert all(output.shape == (1, 64) for output in outputs)
    assert assembly.count("DECODE_CACHE_APPEND nemotron_gqa_cache_k_head0") == 4
    assert assembly.count("DECODE_CACHE_APPEND nemotron_gqa_cache_v_head0") == 4
    assert assembly.count("H_STORE_V") >= 8
    pv_sections = assembly.split("; === Compute PV = P @ V")
    assert len(pv_sections) > 1
    for section in pv_sections[1:]:
        prefetch = next(
            line for line in section.splitlines() if line.startswith("H_PREFETCH_M")
        )
        assert prefetch.endswith(", 1, 1")
    assert cache.persistent_bytes == 2 * 64 * 64 * 2
    assert cache.keys[0].prefix(4).shape == (4, 64)


def test_nemotron_gqa_cache_rejects_an_out_of_range_token() -> None:
    prog, hidden = _program()
    shape = NemotronAttentionShape(64, 2, 1, 64)
    cache = allocate_nemotron_gqa_decode_cache(prog, shape=shape, max_tokens=4)
    weights = NemotronAttentionWeights(
        q=_weight(prog, "range_w_q", 64, 128),
        k=_weight(prog, "range_w_k", 64, 64),
        v=_weight(prog, "range_w_v", 64, 64),
        out=_weight(prog, "range_w_o", 128, 64),
    )

    import pytest

    with pytest.raises(ValueError, match="exceeds cache capacity"):
        emit_nemotron_attention_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            cache=cache,
            token_index=4,
        )


def test_two_chunk_gqa_prefill_appends_cache_with_shifted_causal_mask() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden_chunks = [
        prog.load_batch(
            prog.input(
                f"prefill_hidden_{chunk}",
                shape=(16, 64),
                physical_shape=(64, 64),
                prestaged_vram_addr=chunk * 4096,
            )
        )
        for chunk in range(2)
    ]
    shape = NemotronAttentionShape(64, 2, 1, 64)
    cache = allocate_nemotron_gqa_decode_cache(prog, shape=shape, max_tokens=32)
    weights = NemotronAttentionWeights(
        q=_weight(prog, "prefill_w_q", 64, 128),
        k=_weight(prog, "prefill_w_k", 64, 64),
        v=_weight(prog, "prefill_w_v", 64, 64),
        out=_weight(prog, "prefill_w_o", 128, 64),
    )
    outputs = [
        emit_nemotron_attention_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            rows=16,
            name=f"prefill_chunk{chunk}",
            cache=cache,
            token_index=chunk * 16,
            causal=True,
        )
        for chunk, hidden in enumerate(hidden_chunks)
    ]
    assembly = prog.compile()

    assert all(output.shape == (16, 64) for output in outputs)
    assert assembly.count("DECODE_CACHE_APPEND nemotron_gqa_cache_k_head0") == 32
    assert assembly.count("DECODE_CACHE_APPEND nemotron_gqa_cache_v_head0") == 32
    assert "diagonal_offset=0" in assembly
    assert "diagonal_offset=16" in assembly
    assert cache.keys[0].prefix(32).shape == (32, 64)


def test_connected_nemotron_moe_executes_routed_and_shared_relu2() -> None:
    prog, hidden = _program()
    correction = prog.load_batch(
        prog.input(
            "correction",
            shape=(4, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=256,
        ),
        name="correction",
    )
    zero = prog.fp_var("zero_row", 64)
    routed_scale = prog.fp_var("routed_scale", 2)
    output = emit_nemotron_moe_block(
        prog,
        hidden,
        shape=NemotronMoeShape(
            hidden=64,
            intermediate=64,
            shared_intermediate=64,
            num_experts=4,
            top_k=2,
        ),
        weights=NemotronMoeWeights(
            router=_weight(prog, "router", 64, 4, bf16=True),
            routed_up=reserve_expert_weight_table(
                prog, name="expert_up", num_experts=4, rows=64, cols=64
            ),
            routed_down=reserve_expert_weight_table(
                prog, name="expert_down", num_experts=4, rows=64, cols=64
            ),
            shared_up=_weight(prog, "shared_up", 64, 64),
            shared_down=_weight(prog, "shared_down", 64, 64),
        ),
        correction_bias=correction,
        constants=NemotronMoeConstants(
            zero_row=zero,
            routed_scale=routed_scale,
        ),
        rows=1,
    )
    assembly = prog.compile()

    assert output.shape == (1, 64)
    assert sum(line.startswith("V_TOPK ") for line in assembly.splitlines()) == 1
    assert "relu2 nemotron_moe_pair0" in assembly
    assert "relu2 nemotron_moe_pair1" in assembly
    assert "relu2 nemotron_moe_shared" in assembly
    assert "nemotron_moe_combine" in assembly


def test_multi_token_nemotron_moe_reuses_topk_scale_per_token() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden = prog.load_batch(
        prog.input(
            "multi_hidden",
            shape=(16, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=0,
        )
    )
    correction = prog.load_batch(
        prog.input(
            "multi_correction",
            shape=(4, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=4096,
        )
    )
    zero = prog.fp_var("multi_zero_row", 64)
    routed_scale = prog.fp_var("multi_routed_scale", 2)
    output = emit_nemotron_moe_block(
        prog,
        hidden,
        shape=NemotronMoeShape(64, 64, 64, 4, 2),
        weights=NemotronMoeWeights(
            router=_weight(prog, "multi_router", 64, 4, bf16=True),
            routed_up=reserve_expert_weight_table(
                prog, name="multi_expert_up", num_experts=4, rows=64, cols=64
            ),
            routed_down=reserve_expert_weight_table(
                prog,
                name="multi_expert_down",
                num_experts=4,
                rows=64,
                cols=64,
            ),
            shared_up=_weight(prog, "multi_shared_up", 64, 64),
            shared_down=_weight(prog, "multi_shared_down", 64, 64),
        ),
        correction_bias=correction,
        constants=NemotronMoeConstants(zero, routed_scale),
        rows=16,
    )
    assembly = prog.compile()

    assert output.shape == (16, 64)
    assert sum(line.startswith("V_TOPK ") for line in assembly.splitlines()) == 16
    assert assembly.count("FPVar Mul") == 16
