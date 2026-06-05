from compiler.asm_templates.normalization_asm import layer_norm_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.gelu_asm import gelu_asm
from compiler.asm_templates.elementwise_add_vram_asm import elementwise_add_vram_asm
from compiler.asm_templates._imm import load_large_int_str
from compiler.asm_templates._imm import addi_large_int_str
from compiler.asm_templates.preload_addr_reg import preload_addr_reg_asm
from compiler.asm_templates.reset_reg_asm import reset_reg_asm
from compiler.asm_templates.flashattn.encoder_mha import flash_attn_encoder_mha_asm
from compiler.asm_templates.flashattn.reset import reset_vssram_code


def build_mlp_block(
    *,
    mlen,
    blen,
    vlen,
    batch,
    hidden_size,
    inter_dim,
    w1_hbm_offset_reg,
    w2_hbm_offset_reg,
    activation_base,
    mlp_inter_base,
    mlp_out_base,
    scratch_base,
    gelu_one_fp_slot,
    gelu_1702_fp_slot,
    fc1_bias_base=None,
    fc2_bias_base=None,
    include_gelu=True,
):
    """Emit the MLP sub-block: proj-up -> GELU -> proj-down."""
    if vlen <= 0:
        raise ValueError("build_mlp_block requires vlen > 0")
    if batch <= 0:
        raise ValueError("build_mlp_block requires batch > 0")
    if hidden_size <= 0 or inter_dim <= 0:
        raise ValueError("build_mlp_block requires positive hidden_size and inter_dim")
    if (batch * inter_dim) % vlen != 0:
        raise ValueError(
            f"build_mlp_block requires batch*inter_dim ({batch * inter_dim}) divisible by vlen ({vlen})"
        )
    if (batch * hidden_size) % vlen != 0:
        raise ValueError(
            f"build_mlp_block requires batch*hidden_size ({batch * hidden_size}) divisible by vlen ({vlen})"
        )

    asm = ""

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=hidden_size,
        vlen=vlen,
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w1_hbm_offset_reg,
        activation_base_address=activation_base,
        result_base_address=mlp_inter_base,
        out_features=inter_dim,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

    if fc1_bias_base is not None:
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(batch * inter_dim) // vlen,
            alive_registers=[10, 11],
            dst_base_address=mlp_inter_base,
            src_base_address=fc1_bias_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])

    if include_gelu:
        asm += gelu_asm(
            const_one_fp_address=gelu_one_fp_slot,
            const_1702_fp_address=gelu_1702_fp_slot,
            alive_registers=[5, 6, 7],
            activation_base_address=mlp_inter_base,
            scratchpad_base_address=scratch_base,
            vlen=vlen,
            batch_size=batch,
            hidden_dim=inter_dim,
        )
        asm += reset_reg_asm(alive_registers=[5, 6, 7])

    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=batch,
        hidden_size=inter_dim,
        vlen=vlen,
        alive_registers=[5, 6, 7, 8, 9, 10],
        w_base_hbm_offset_reg=w2_hbm_offset_reg,
        activation_base_address=mlp_inter_base,
        result_base_address=mlp_out_base,
        out_features=hidden_size,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7, 8, 9, 10])

    if fc2_bias_base is not None:
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(batch * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=mlp_out_base,
            src_base_address=fc2_bias_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])

    return asm


def _pack_seq_major_to_chunk_major(
    *,
    seq_len,
    num_chunks,
    chunk_size,
    src_base,
    dst_base,
    comment,
):
    """Pack [seq_len, num_chunks, chunk_size] -> [num_chunks, seq_len, chunk_size]."""
    asm = f"; {comment}\n"

    asm += reset_vssram_code(
        reset_start_address=dst_base,
        vect_dim=chunk_size,
        per_stride_dim=seq_len * num_chunks,
        reset_stride=chunk_size,
        reset_amount=1,
        alive_registers_int=[10, 11, 12],
    )

    # Register map:
    # - gp10/gp11: current dst/src vector pointers
    # - gp12/gp13: chunk-base dst/src pointers for outer loop
    # - gp14/gp15: outer/inner loop counters
    # - gp9: temp register for large immediate adds
    dst_reg = 10
    src_reg = 11
    dst_chunk_base_reg = 12
    src_chunk_base_reg = 13
    outer_loop_reg = 14
    inner_loop_reg = 15
    temp_reg = 9

    asm += load_large_int_str(dst_chunk_base_reg, dst_base)
    asm += load_large_int_str(src_chunk_base_reg, src_base)

    if num_chunks > 0 and seq_len > 0:
        asm += f"C_LOOP_START gp{outer_loop_reg}, {num_chunks}\n"
        asm += f"S_ADDI_INT gp{dst_reg}, gp{dst_chunk_base_reg}, 0\n"
        asm += f"S_ADDI_INT gp{src_reg}, gp{src_chunk_base_reg}, 0\n"

        asm += f"C_LOOP_START gp{inner_loop_reg}, {seq_len}\n"
        asm += f"V_ADD_VV gp{dst_reg}, gp{dst_reg}, gp{src_reg}, 0\n"
        asm += addi_large_int_str(dst_reg, dst_reg, chunk_size, temp_reg)
        asm += addi_large_int_str(src_reg, src_reg, num_chunks * chunk_size, temp_reg)
        asm += f"C_LOOP_END gp{inner_loop_reg}\n"

        asm += addi_large_int_str(dst_chunk_base_reg, dst_chunk_base_reg, seq_len * chunk_size, temp_reg)
        asm += addi_large_int_str(src_chunk_base_reg, src_chunk_base_reg, chunk_size, temp_reg)
        asm += f"C_LOOP_END gp{outer_loop_reg}\n"

    asm += reset_reg_asm(alive_registers=[9, 10, 11, 12, 13, 14, 15])
    return asm


def build_encoder_layer_asm(
    *,
    mlen,
    blen,
    vlen,
    batch,
    s_q,
    s_kv,
    s_kv_valid=None,
    hq,
    hkv,
    h_qkv,
    hidden_size,
    inter_dim,
    x_base,
    attn_base,
    residual_base,
    mlp_inter_base,
    mlp_out_base,
    scratch_base,
    k_hbm_offset,
    v_hbm_offset,
    out_hbm_offset,
    w1_hbm_offset,
    w2_hbm_offset,
    ln_eps_fp_slot,
    ln_reci_hid_fp_slot,
    gelu_one_fp_slot,
    gelu_1702_fp_slot,
    attn_scale_fp_slot,
    attn_ninf_fp_slot,
    flash_temp_fp_start,
    q_base,
    wq_hbm_offset,
    q_bias_base=None,
    ln1_affine_weight_base=None,
    ln1_affine_bias_base=None,
    ln2_affine_weight_base=None,
    ln2_affine_bias_base=None,
    out_bias_base=None,
    fc1_bias_base=None,
    fc2_bias_base=None,
    debug_stage0_snapshot_base=None,
    debug_ln1_snapshot_base=None,
    debug_qproj_snapshot_base=None,
    debug_attn_snapshot_base=None,
    debug_outproj_snapshot_base=None,
    include_final_residual=True,
    include_gelu=True,
):
    """Emit one SigLIP encoder layer in SRAM-resident pipeline form."""
    if vlen <= 0:
        raise ValueError("build_encoder_layer_asm requires vlen > 0")
    if s_q <= 0:
        raise ValueError("build_encoder_layer_asm requires s_q > 0")
    if hidden_size <= 0:
        raise ValueError("build_encoder_layer_asm requires hidden_size > 0")
    if hidden_size % vlen != 0:
        raise ValueError(
            f"build_encoder_layer_asm requires hidden_size ({hidden_size}) divisible by vlen ({vlen})"
        )
    if (s_q * hidden_size) % vlen != 0:
        raise ValueError(
            f"build_encoder_layer_asm requires s_q*hidden_size ({s_q * hidden_size}) divisible by vlen ({vlen})"
        )

    asm = "; SigLIP Encoder Layer (ASM) Test\n"
    asm += "; LayerNorm1 -> FlashAttn -> Residual -> LayerNorm2 -> MLP -> Residual\n"

    # Shape legend used below:
    # - S = s_q (query sequence length)
    # - H = hidden_size
    # - V = vlen
    # - NB = H // V (number of V-sized hidden blocks)
    # - D = h_qkv (per-head hidden, padded to mlen when needed)
    # - HQ = hq
    # Memory layouts used in this block:
    # - chunk-major: [NB, S, V]  (flat index: ((b * S) + t) * V + j)
    # - token-major: [S, NB, V]  (flat index: ((t * NB) + b) * V + j)
    # - head-major Q: [HQ, S, D] (flat index: ((h * S) + t) * D + j)
    # Stage outputs in execution order:
    # - Stage 0 input residual snapshot at residual_base: chunk-major [NB, S, V]
    # - Stage 1 LN1 output at x_base: chunk-major [NB, S, V]
    # - Stage 2 Q build: q_base chunk-major [NB, S, V]
    # - Stage 3 flash-attn output at attn_base: chunk-major [NB, S, V]
    # - Stage 4 out projection workspace: project attn_base -> q_base
    # - Stage 5 first residual add at q_base: chunk-major [NB, S, V]
    # - Stage 6 residual snapshot for final add at residual_base: chunk-major [NB, S, V]
    # - Stage 7 LN2 output at q_base: chunk-major [NB, S, V]
    # - Stage 8 MLP output at mlp_out_base: chunk-major [NB, S, V]
    # - Stage 9 final output at mlp_out_base: chunk-major [NB, S, V]

    asm += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2, 3, 4, 5],
        available_registers=[1, 2, 3, 4, 5],
        addr_reg_val=[k_hbm_offset, v_hbm_offset, w1_hbm_offset, w2_hbm_offset, wq_hbm_offset],
    )

    # Stage 0: Save X residual in chunk-major layout before LN1.
    # Shape after Stage 0 (residual_base): chunk-major [NB, S, V].
    # x_base holds X in chunk-major [hidden//vlen, s_q, vlen].
    # We keep this copy in the same chunk-major layout so the attention output
    # can be added after out projection without a token-major roundtrip.
    asm += reset_vssram_code(
        reset_start_address=residual_base,
        vect_dim=vlen,
        per_stride_dim=hidden_size // vlen,
        reset_stride=hidden_size,
        reset_amount=s_q,
        alive_registers_int=[10, 11, 12],
    )
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=residual_base,
        src_base_address=x_base,
    )

    if debug_stage0_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_stage0_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_stage0_snapshot_base,
            src_base_address=residual_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 1: LayerNorm1 in-place on X.
    # Shape after Stage 1 (x_base): chunk-major [NB, S, V].
    asm += layer_norm_asm(
        _eps_offset=ln_eps_fp_slot,
        reci_hid_offset=ln_reci_hid_fp_slot,
        alive_registers=[5, 6, 7],
        activation_base_address=x_base,
        scratchpad_base_address=scratch_base,
        vlen=vlen,
        batch_size=s_q,
        hidden_dim=hidden_size,
        affine_weight_base_address=ln1_affine_weight_base,
        affine_bias_base_address=ln1_affine_bias_base,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7])

    if debug_ln1_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_ln1_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_ln1_snapshot_base,
            src_base_address=x_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 2: Build flash-attn Q on-chip from LN1 output.
    # LN1 chunk-major X at x_base -> projection_asm with WQ -> q_base,
    # optional bias add from q_bias_base. The projection buffer is then used
    # directly by flash-attn.
    # Shape after Stage 2:
    # - q_base: chunk-major [NB, S, V]
    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=s_q,
        hidden_size=hidden_size,
        vlen=vlen,
        alive_registers=[4, 5, 6, 7, 8, 9],
        w_base_hbm_offset_reg=5,
        activation_base_address=x_base,
        result_base_address=q_base,
        out_features=hidden_size,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[4, 5, 6, 7, 8, 9])

    if q_bias_base is not None:
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=q_base,
            src_base_address=q_bias_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])

    if debug_qproj_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_qproj_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_qproj_snapshot_base,
            src_base_address=q_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 3: Flash attention (encoder-focused MHA).
    # Shape after Stage 3 (attn_base/o_old_base): chunk-major [NB, S, V].
    asm += "; Flash attention block\n"
    alive_int = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    alive_fp = [1, 2, 3, 4, 5, 6]
    asm += flash_attn_encoder_mha_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch,
        hq=hq,
        hkv=hkv,
        d=h_qkv,
        q_len=s_q,
        kv_len=s_kv,
        kv_valid_len=s_kv_valid,
        alive_registers_int=alive_int,
        alive_registers_fp=alive_fp,
        vector_sram_base_address=q_base,
        fp_sram_start_address=flash_temp_fp_start,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
        attn_scale_fp_address=attn_scale_fp_slot,
        inf_fp_address=attn_ninf_fp_slot,
        causal_mask=False,
    )

    if debug_attn_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_attn_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_attn_snapshot_base,
            src_base_address=attn_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 4: Out projection on attention output before residual add.
    # flash-attn output is already chunk-major [NB, S, V], so projection can
    # consume attn_base directly with no repack.
    asm += preload_addr_reg_asm(
        addr_reg_to_set=[2],
        available_registers=[2],
        addr_reg_val=[out_hbm_offset],
    )
    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=s_q,
        hidden_size=hidden_size,
        vlen=vlen,
        alive_registers=[4, 5, 6, 7, 8, 9],
        w_base_hbm_offset_reg=2,
        activation_base_address=attn_base,
        result_base_address=q_base,
        out_features=hidden_size,
        scratch_base_address=scratch_base,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[4, 5, 6, 7, 8, 9])
    if out_bias_base is not None:
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=q_base,
            src_base_address=out_bias_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=q_base,
        src_base_address=residual_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    if debug_outproj_snapshot_base is not None:
        asm += reset_vssram_code(
            reset_start_address=debug_outproj_snapshot_base,
            vect_dim=vlen,
            per_stride_dim=hidden_size // vlen,
            reset_stride=hidden_size,
            reset_amount=s_q,
            alive_registers_int=[10, 11, 12],
        )
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=debug_outproj_snapshot_base,
            src_base_address=q_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 5: Save the first residual output for the final MLP residual add.
    # Shape after Stage 5 (residual_base): chunk-major [NB, S, V].
    asm += reset_vssram_code(
        reset_start_address=residual_base,
        vect_dim=vlen,
        per_stride_dim=hidden_size // vlen,
        reset_stride=hidden_size,
        reset_amount=s_q,
        alive_registers_int=[10, 11, 12],
    )
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q * hidden_size) // vlen,
        alive_registers=[10, 11],
        dst_base_address=residual_base,
        src_base_address=q_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11, 12])

    # Stage 6: LayerNorm2 in-place.
    # Shape after Stage 6 (q_base): chunk-major [NB, S, V].
    asm += layer_norm_asm(
        _eps_offset=ln_eps_fp_slot,
        reci_hid_offset=ln_reci_hid_fp_slot,
        alive_registers=[5, 6, 7],
        activation_base_address=q_base,
        scratchpad_base_address=scratch_base,
        vlen=vlen,
        batch_size=s_q,
        hidden_dim=hidden_size,
        affine_weight_base_address=ln2_affine_weight_base,
        affine_bias_base_address=ln2_affine_bias_base,
    )
    asm += reset_reg_asm(alive_registers=[5, 6, 7])

    # Stage 7: MLP block.
    # Shape after Stage 7 (mlp_out_base): chunk-major [NB, S, V].
    asm += build_mlp_block(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        batch=s_q,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        w1_hbm_offset_reg=3,
        w2_hbm_offset_reg=4,
        activation_base=q_base,
        mlp_inter_base=mlp_inter_base,
        mlp_out_base=mlp_out_base,
        scratch_base=scratch_base,
        gelu_one_fp_slot=gelu_one_fp_slot,
        gelu_1702_fp_slot=gelu_1702_fp_slot,
        fc1_bias_base=fc1_bias_base,
        fc2_bias_base=fc2_bias_base,
        include_gelu=include_gelu,
    )

    if include_final_residual:
        # Stage 8: Residual 2, mlp_out += residual.
        # Shape after Stage 8 (mlp_out_base): chunk-major [NB, S, V].
        asm += elementwise_add_vram_asm(
            vlen=vlen,
            num_vectors=(s_q * hidden_size) // vlen,
            alive_registers=[10, 11],
            dst_base_address=mlp_out_base,
            src_base_address=residual_base,
        )
        asm += reset_reg_asm(alive_registers=[10, 11])

    return asm
