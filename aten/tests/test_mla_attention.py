"""Attention codegen guards required by Kimi K3 MLA."""

from __future__ import annotations

import pytest

from compiler.aten.plena import PlenaCompiler
from aten.kimi3.blocks import (
    AttnResConstants,
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    MlaBlockShape,
    MlaBlockWeights,
    MlaNormConstants,
    allocate_mla_decode_cache,
    emit_kimi_attn_res,
    emit_mla_residual_block,
    emit_kimi_latent_moe_residual_block,
)
from compiler.aten.plena import reserve_expert_weight_table
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants


def test_mha_supports_wider_qk_than_value_heads() -> None:
    """MLA scores are 192-wide while each value/output head is 128-wide."""
    prog = PlenaCompiler(mlen=64, blen=4)
    q_input = prog.input(
        "Q",
        shape=(64, 192),
        physical_shape=(64, 192),
        prestaged_vram_addr=0,
    )
    k_input = prog.input("K", shape=(64, 192), physical_shape=(64, 192))
    v_input = prog.input("V", shape=(64, 128), physical_shape=(64, 128))

    out = prog.flash_attention(
        prog.load_batch(q_input, name="Q"),
        k_input,
        v_input,
        seq_len=1,
        kv_seq_len=1,
    )
    asm = prog.compile()

    assert out.shape == (1, 128)
    assert out.physical_shape == (64, 128)
    assert "Compute PV = P @ V[k_idx=0]" in asm
    pv_asm = asm.split("Compute PV = P @ V[k_idx=0]", 1)[1]
    assert pv_asm.index("C_SET_SCALE_REG") < pv_asm.index("H_PREFETCH_M")


def test_mha_releases_internal_score_and_pv_scratch() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    q = prog.load_batch(
        prog.input(
            "Q_reuse",
            shape=(1, 192),
            physical_shape=(64, 192),
            prestaged_vram_addr=0,
        )
    )
    k = prog.input("K_reuse", shape=(1, 192), physical_shape=(64, 192))
    v = prog.input("V_reuse", shape=(1, 128), physical_shape=(64, 128))

    first = prog.flash_attention(q, k, v, seq_len=1, kv_seq_len=1)
    first_addr = prog.get_vram_addr(first.name)
    prog.free_tensor(first)
    second = prog.flash_attention(q, k, v, seq_len=1, kv_seq_len=1)

    assert prog.get_vram_addr(second.name) == first_addr
    assert "S" not in prog.vram_matrices
    assert "PV" not in prog.vram_matrices


def test_online_softmax_true_zeros_reused_output_vram() -> None:
    """Attention output reset must overwrite NaNs rather than multiply by zero."""
    prog = PlenaCompiler(mlen=64, blen=4)
    q = prog.load_batch(
        prog.input(
            "Q_true_zero",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=0,
        )
    )
    k = prog.input("K_true_zero", shape=(1, 64), physical_shape=(64, 64))
    v = prog.input("V_true_zero", shape=(1, 64), physical_shape=(64, 64))

    prog.flash_attention(q, k, v, seq_len=1, kv_seq_len=1)
    asm = prog.compile()
    init = asm.split("=== Init Online Softmax", 1)[1].split(
        "=== Online Softmax Block", 1
    )[0]

    assert "S_MAP_V_FP" in init
    assert "V_MUL_VF" not in init


def test_plain_bf16_hbm_inputs_allocate_their_full_two_byte_footprint() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    first = prog.input(
        "bf16_first",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=2.0,
    )
    second = prog.input(
        "bf16_second",
        shape=(64, 64),
        physical_shape=(64, 64),
        real_data_ratio=2.0,
    )

    assert first.hbm_size == 64 * 64 * 2
    assert second.hbm_addr == first.hbm_addr + first.hbm_size


def test_plain_bf16_vector_load_selects_the_kv_decoder() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    weight = prog.input(
        "norm_weight",
        shape=(1, 128),
        physical_shape=(4, 128),
        real_data_ratio=2.0,
    )

    prog.load_batch(
        weight,
        storage_precision=2,
        hbm_precision=1,
    )

    assert "H_PREFETCH_V" in prog.compile()
    assert ", 1, 1" in prog.compile() or ", 0, 1, 0" in prog.compile()


def test_plain_bf16_vector_load_advances_hbm_addresses_in_bytes() -> None:
    from compiler.asm_templates.preload_act import preload_act_asm

    assembly = preload_act_asm(
        vlen=64,
        preload_len=4,
        batch=64,
        hidden_size=128,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=1,
        stride_size=128,
        storage_precision=2,
        hbm_precision=1,
    )

    assert "S_ADDI_INT gp2, gp0, 256" in assembly
    assert "S_ADDI_INT gp1, gp1, 128" in assembly


def test_kimi_attn_res_emits_depth_softmax_and_weighted_sum() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    prefix = prog.load_batch(
        prog.input(
            "attnres_prefix",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=0,
        ),
        name="attnres_prefix",
    )
    block = prog.load_batch(
        prog.input(
            "attnres_block",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=4096,
        ),
        name="attnres_block",
    )
    score_weight = prog.load_batch(
        prog.input(
            "attnres_score_weight",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=8192,
        ),
        name="attnres_score_weight",
    )
    prog.fp_var("attnres_zero", 1)
    eps = prog.fp_var("attnres_eps", 1)
    reciprocal = prog.fp_var("attnres_reciprocal", 1)

    output = emit_kimi_attn_res(
        prog,
        (block,),
        prefix,
        score_weight=score_weight,
        constants=AttnResConstants(eps.address, reciprocal.address),
        rows=1,
        name="layer1_attnres",
    )
    asm = prog.compile()

    assert output.shape == prefix.shape
    assert "Kimi AttnRes depth softmax: rows=1, depth=2" in asm
    assert "S_MAX_FP" in asm
    assert "S_EXP_FP" in asm
    assert "layer1_attnres_candidate0_weighted" in asm
    assert "layer1_attnres_candidate1_weighted" in asm
    assert not prog.register_allocator.used_gp
    assert not prog.register_allocator.used_fp


def test_mha_rejects_mismatched_q_and_k_head_dimensions() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    q_input = prog.input(
        "Q",
        shape=(64, 192),
        physical_shape=(64, 192),
        prestaged_vram_addr=0,
    )
    k_input = prog.input("K", shape=(64, 128), physical_shape=(64, 128))
    v_input = prog.input("V", shape=(64, 128), physical_shape=(64, 128))

    with pytest.raises(ValueError, match="Q/K head dimensions must match"):
        prog.flash_attention(
            prog.load_batch(q_input, name="Q"),
            k_input,
            v_input,
            seq_len=1,
            kv_seq_len=1,
        )


def test_vram_copy_region_uses_a_real_move_and_supports_column_packing() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    src_input = prog.input(
        "src",
        shape=(1, 192),
        physical_shape=(64, 192),
        prestaged_vram_addr=0,
    )
    src = prog.load_batch(src_input, name="src")
    dst = prog.alloc("dst", 1, 192, strict=False, physical_shape=(64, 192))

    prog.vram_copy_region(
        dst,
        src,
        num_rows=1,
        num_cols=64,
        dst_col_offset=128,
        src_col_offset=64,
    )
    asm = prog.compile()

    assert "VRAM copy region" in asm
    assert "V_ADD_VF" in asm
    assert "V_MUL_VF" not in asm


def test_connected_mla_block_uses_projection_outputs_and_keeps_residual() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    shape = MlaBlockShape(
        hidden=64,
        q_lora=64,
        kv_lora=64,
        qk_nope=64,
        qk_rope=64,
        v_head=64,
        heads=1,
    )
    hidden = prog.load_batch(
        prog.input(
            "hidden",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=0,
        ),
        name="hidden",
    )
    cos = prog.load_batch(
        prog.input(
            "cos",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=256,
        ),
        name="cos",
    )
    sin = prog.load_batch(
        prog.input(
            "sin",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=512,
        ),
        name="sin",
    )

    def weight(name: str, rows: int, cols: int):
        return prog.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    weights = MlaBlockWeights(
        q_a=weight("w_q_a", 64, 64),
        q_b=weight("w_q_b", 64, 128),
        kv_a=weight("w_kv_a", 64, 128),
        kv_b=weight("w_kv_b", 64, 128),
        out=weight("w_o", 64, 64),
        q_rope_rotate=weight("w_q_rot", 64, 64),
        k_rope_rotate=weight("w_k_rot", 64, 64),
        gate=weight("w_gate", 64, 64),
    )
    prog.fp_var("mla_zero", 1)
    mla_eps = prog.fp_var("mla_eps", 1)
    mla_reci = prog.fp_var("mla_reci", 1)
    mla_one = prog.fp_var("mla_one", 1)
    mla_neg_one = prog.fp_var("mla_neg_one", 1)
    out = emit_mla_residual_block(
        prog,
        hidden,
        shape=shape,
        weights=weights,
        cos=cos,
        sin=sin,
        norms=MlaNormConstants(
            mla_eps.address,
            mla_reci.address,
            mla_eps.address,
            mla_reci.address,
            mla_eps.address,
            mla_reci.address,
            gate_one=mla_one,
            gate_neg_one=mla_neg_one,
        ),
        rows=1,
        name="layer0_mla",
    )
    asm = prog.compile()

    assert out.shape == (1, 64)
    assert prog.get_vram_addr(out.name) != prog.get_vram_addr(hidden.name)
    assert "layer0_mla_q_a" in asm
    assert "layer0_mla_q_b" in asm
    assert "layer0_mla_kv_a" in asm
    assert "layer0_mla_kv_b" in asm
    assert asm.count("H_STORE_V") >= 2
    assert "Compute PV = P @ V[k_idx=0]" in asm
    assert "layer0_mla_out" in asm
    assert "layer0_mla_gate" in asm
    assert "layer0_mla_residual" in asm


def test_four_token_mla_keeps_only_compressed_history_persistent() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    shape = MlaBlockShape(
        hidden=64,
        q_lora=64,
        kv_lora=64,
        qk_nope=64,
        qk_rope=64,
        v_head=64,
        heads=2,
    )
    hidden = prog.load_batch(
        prog.input(
            "cache_hidden",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=0,
        )
    )
    cos = prog.load_batch(
        prog.input(
            "cache_cos",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=256,
        )
    )
    sin = prog.load_batch(
        prog.input(
            "cache_sin",
            shape=(1, 64),
            physical_shape=(4, 64),
            prestaged_vram_addr=512,
        )
    )

    def weight(name: str, rows: int, cols: int):
        return prog.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    weights = MlaBlockWeights(
        q_a=weight("cache_w_q_a", 64, 64),
        q_b=weight("cache_w_q_b", 64, 256),
        kv_a=weight("cache_w_kv_a", 64, 128),
        kv_b=weight("cache_w_kv_b", 64, 256),
        out=weight("cache_w_o", 128, 64),
        q_rope_rotate=weight("cache_w_q_rot", 64, 64),
        k_rope_rotate=weight("cache_w_k_rot", 64, 64),
        gate=None,
    )
    prog.fp_var("cache_zero", 1)
    eps = prog.fp_var("cache_eps", 1)
    reciprocal = prog.fp_var("cache_reciprocal", 1)
    norms = MlaNormConstants(
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
    )
    cache = allocate_mla_decode_cache(prog, shape=shape, max_tokens=4)

    for token in range(4):
        emit_mla_residual_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            cos=cos,
            sin=sin,
            norms=norms,
            cache=cache,
            token_index=token,
            name=f"mla_token{token}",
        )
    assembly = prog.compile()

    cache.assert_compressed_only()
    cache.assert_hbm_contract(prog)
    assert len(cache.persistent_backings) == 1
    assert len(cache.scratch_backings) == 2
    assert cache.logical_persistent_bytes == 4 * 128 * 2
    assert cache.theoretical_expanded_cache_bytes == 4 * 2 * (128 + 64) * 2
    assert assembly.count("MLA_COMPRESSED_CACHE_APPEND") == 4
    assert assembly.count("MLA_RECONSTRUCTED_HEAD_TILE") == 8
    assert "kimi_mla_cache_reconstructed_k_scratch" in assembly
    assert "kimi_mla_cache_reconstructed_v_scratch" in assembly
    assert "mla_token0_k_scratch0" not in assembly
    pv_sections = assembly.split("; === Compute PV = P @ V")
    assert len(pv_sections) > 1
    for section in pv_sections[1:]:
        prefetch = next(
            line for line in section.splitlines() if line.startswith("H_PREFETCH_M")
        )
        assert prefetch.endswith(", 1, 1")


def test_five_chunk_mla_prefill_grows_compressed_history_past_one_tile() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    shape = MlaBlockShape(
        hidden=64,
        q_lora=64,
        kv_lora=64,
        qk_nope=64,
        qk_rope=64,
        v_head=64,
        heads=1,
    )
    hidden_chunks = [
        prog.load_batch(
            prog.input(
                f"prefill_hidden_{chunk}",
                shape=(16, 64),
                physical_shape=(64, 64),
                prestaged_vram_addr=chunk * 12288,
            )
        )
        for chunk in range(5)
    ]
    cos_chunks = [
        prog.load_batch(
            prog.input(
                f"prefill_cos_{chunk}",
                shape=(16, 64),
                physical_shape=(64, 64),
                prestaged_vram_addr=chunk * 12288 + 4096,
            )
        )
        for chunk in range(5)
    ]
    sin_chunks = [
        prog.load_batch(
            prog.input(
                f"prefill_sin_{chunk}",
                shape=(16, 64),
                physical_shape=(64, 64),
                prestaged_vram_addr=chunk * 12288 + 8192,
            )
        )
        for chunk in range(5)
    ]

    def weight(name: str, rows: int, cols: int):
        return prog.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    cache = allocate_mla_decode_cache(prog, shape=shape, max_tokens=80)
    eps = prog.fp_var("prefill_eps", 1)
    reciprocal = prog.fp_var("prefill_reciprocal", 1)
    weights = MlaBlockWeights(
        q_a=weight("prefill_q_a", 64, 64),
        q_b=weight("prefill_q_b", 64, 128),
        kv_a=weight("prefill_kv_a", 64, 128),
        kv_b=weight("prefill_kv_b", 64, 128),
        out=weight("prefill_out", 64, 64),
        q_rope_rotate=weight("prefill_q_rot", 64, 64),
        k_rope_rotate=weight("prefill_k_rot", 64, 64),
    )
    norms = MlaNormConstants(
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
    )
    outputs = [
        emit_mla_residual_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            cos=cos,
            sin=sin,
            norms=norms,
            rows=16,
            name=f"prefill_chunk{chunk}",
            cache=cache,
            token_index=chunk * 16,
            causal=True,
        )
        for chunk, (hidden, cos, sin) in enumerate(
            zip(hidden_chunks, cos_chunks, sin_chunks, strict=True)
        )
    ]
    assembly = prog.compile()

    assert all(output.shape == (16, 64) for output in outputs)
    assert assembly.count("MLA_COMPRESSED_CACHE_APPEND") == 5
    assert assembly.count("DECODE_CACHE_APPEND kimi_mla_cache_compressed") == 80
    assert (
        assembly.count(
            "DECODE_CACHE_SCRATCH_ROW kimi_mla_cache_reconstructed_k_scratch"
        )
        == 240
    )
    assert (
        assembly.count(
            "DECODE_CACHE_SCRATCH_ROW kimi_mla_cache_reconstructed_v_scratch"
        )
        == 240
    )
    assert assembly.count("diagonal_offset=0") == 2
    assert "diagonal_offset=16" in assembly
    assert "diagonal_offset=32" in assembly
    assert "diagonal_offset=48" in assembly
    assert (
        "Allocate VRAM Matrix prefill_chunk4_kv_b_head0: "
        "logical=(80, 128) physical=(128, 128)" in assembly
    )
    assert (
        "Allocate VRAM Matrix prefill_chunk4_k_head0: "
        "logical=(80, 128) physical=(128, 128)" in assembly
    )
    cache.assert_hbm_contract(prog)


def test_mla_cache_hbm_contract_rejects_an_expanded_cache_allocation() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    shape = MlaBlockShape(
        hidden=64,
        q_lora=64,
        kv_lora=64,
        qk_nope=64,
        qk_rope=64,
        v_head=64,
        heads=4,
    )
    cache = allocate_mla_decode_cache(prog, shape=shape, max_tokens=4)
    prog.input(
        "kimi_mla_cache_expanded_k",
        shape=(4, shape.heads * shape.qk_head),
        physical_shape=(64, shape.heads * shape.qk_head),
        real_data_ratio=2.0,
    )

    with pytest.raises(AssertionError, match="unexpected=.*expanded_k"):
        cache.assert_hbm_contract(prog)


def test_single_token_mla_reuses_one_ephemeral_hbm_head_scratch() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden = prog.load_batch(
        prog.input(
            "ephemeral_hidden",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=0,
        )
    )
    cos = prog.alloc("ephemeral_cos", 1, 64, strict=False, physical_shape=(64, 64))
    sin = prog.alloc("ephemeral_sin", 1, 64, strict=False, physical_shape=(64, 64))
    shape = MlaBlockShape(
        hidden=64,
        q_lora=64,
        kv_lora=64,
        qk_nope=64,
        qk_rope=64,
        v_head=64,
        heads=4,
    )

    def weight(name: str, rows: int, cols: int):
        return prog.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    weights = MlaBlockWeights(
        q_a=weight("ephemeral_q_a", 64, 64),
        q_b=weight("ephemeral_q_b", 64, 512),
        kv_a=weight("ephemeral_kv_a", 64, 128),
        kv_b=weight("ephemeral_kv_b", 64, 512),
        out=weight("ephemeral_out", 256, 64),
        q_rope_rotate=weight("ephemeral_q_rotate", 64, 64),
        k_rope_rotate=weight("ephemeral_k_rotate", 64, 64),
        gate=None,
    )
    eps = prog.fp_var("ephemeral_eps", 1)
    reciprocal = prog.fp_var("ephemeral_reciprocal", 1)
    norms = MlaNormConstants(
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
        eps.address,
        reciprocal.address,
    )

    emit_mla_residual_block(
        prog,
        hidden,
        shape=shape,
        weights=weights,
        cos=cos,
        sin=sin,
        norms=norms,
        add_residual=False,
        name="ephemeral_mla",
    )
    assembly = prog.compile()

    assert assembly.count("MLA_RECONSTRUCTED_HEAD_TILE") == shape.heads
    assert (
        assembly.count(
            "DECODE_CACHE_OVERWRITE_VALID ephemeral_mla_reconstructed_k_scratch"
        )
        == shape.heads
    )
    assert (
        assembly.count(
            "DECODE_CACHE_OVERWRITE_VALID ephemeral_mla_reconstructed_v_scratch"
        )
        == shape.heads
    )
    assert "ephemeral_mla_k_scratch0" not in assembly
    assert "ephemeral_mla_v_scratch0" not in assembly
    assert "ephemeral_mla_reconstructed_k_scratch" not in prog._inputs
    assert "ephemeral_mla_reconstructed_v_scratch" not in prog._inputs


def test_connected_kimi_latent_moe_emits_every_real_stage() -> None:
    prog = PlenaCompiler(mlen=64, blen=4)
    hidden = prog.load_batch(
        prog.input(
            "moe_hidden",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=0,
        ),
        name="moe_hidden",
    )
    correction = prog.load_batch(
        prog.input(
            "router_correction",
            shape=(1, 64),
            physical_shape=(64, 64),
            prestaged_vram_addr=4096,
        ),
        name="router_correction",
    )
    shape = KimiLatentMoeShape(
        hidden=64,
        routed_hidden=64,
        intermediate=64,
        shared_intermediate=64,
        num_experts=4,
        top_k=2,
    )

    def weight(name: str, rows: int, cols: int):
        return prog.input(name, shape=(rows, cols), physical_shape=(rows, cols))

    weights = KimiLatentMoeWeights(
        router=weight("router", 64, 4),
        routed_down=weight("latent_down", 64, 64),
        routed_up=weight("latent_up", 64, 64),
        routed_gate=reserve_expert_weight_table(
            prog, name="expert_gate", num_experts=4, rows=64, cols=64
        ),
        routed_up_expert=reserve_expert_weight_table(
            prog, name="expert_up", num_experts=4, rows=64, cols=64
        ),
        routed_down_expert=reserve_expert_weight_table(
            prog, name="expert_down", num_experts=4, rows=64, cols=64
        ),
        shared=(
            weight("shared_gate", 64, 64),
            weight("shared_up", 64, 64),
            weight("shared_down", 64, 64),
        ),
    )
    zero = prog.fp_var("zero", 1)
    one = prog.fp_var("one", 4)
    neg_one = prog.fp_var("neg_one", 4)
    beta = prog.fp_var("beta", 4)
    neg_two_over_beta = prog.fp_var("neg_two_over_beta", 4)
    linear_beta = prog.fp_var("linear_beta", 4)
    neg_two_over_linear_beta = prog.fp_var("neg_two_over_linear_beta", 4)
    zero_row = prog.fp_var("zero_row", 64)
    norm_eps = prog.fp_var("norm_eps", 1)
    norm_reci = prog.fp_var("norm_reci", 1)
    routed_norm_eps = prog.fp_var("routed_norm_eps", 1)
    routed_norm_reci = prog.fp_var("routed_norm_reci", 1)
    constants = KimiLatentMoeConstants(
        situ=KimiSituFPConstants(
            zero=zero,
            one=one,
            neg_one=neg_one,
            beta=beta,
            neg_two_over_beta=neg_two_over_beta,
            linear_beta=linear_beta,
            neg_two_over_linear_beta=neg_two_over_linear_beta,
        ),
        zero_row=zero_row,
        norm_eps=norm_eps.address,
        norm_reciprocal_hidden=norm_reci.address,
        routed_norm_eps=routed_norm_eps.address,
        routed_norm_reciprocal_hidden=routed_norm_reci.address,
    )

    out = emit_kimi_latent_moe_residual_block(
        prog,
        hidden,
        shape=shape,
        weights=weights,
        correction_bias=correction,
        constants=constants,
        rows=1,
        name="layer1_moe",
    )
    asm = prog.compile()

    assert out.shape == (1, 64)
    assert "weight_mode=sigmoid_normalized" in asm
    assert "correction_bias=True" in asm
    assert any(
        line.startswith("C_SET_TOPK_REG gp") and line.endswith(", 1")
        for line in asm.splitlines()
    )
    assert "layer1_moe_latent_down" in asm
    assert "kimi_situ" in asm
    assert "layer1_moe_latent_up" in asm
    assert "layer1_moe_shared" in asm
    assert "layer1_moe_combine" in asm
    assert "layer1_moe_residual" in asm
    assert "[kimi_k3] loop Top-2 routed expert pairs" in asm
    assert "layer1_moe_loop_pair_gate: compact tile-major dynamic projection" in asm
    assert "layer1_moe_pair1_gate" not in asm
    assert "; @stage=non_moe layer1_moe complete" in asm
    assert "\n@stage=" not in asm
