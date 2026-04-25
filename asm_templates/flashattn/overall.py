"""Main Flash Attention assembly code generation - orchestrates all components.

Supports arbitrary GQA ratios (``ratio = hq // hkv``):

* ``ratio == blen``: single-pass per kv-tile (the original code path).
* ``ratio > blen`` and ``ratio % blen == 0``: multi-pass over Q-head blocks of
  size ``blen``.  K/V are shared across passes within a kv-tile; only the Q
  offset is advanced.  Used for hypothetical wide-GQA (e.g. ratio=8 on
  blen=4 hardware).
* ``ratio < blen``: best-effort single-pass with ``effective_blen = ratio``.
  M_BTMM still emits its full ``blen``-wide block, but the inner softmax/PV/O
  loop only consumes the first ``ratio`` head slots.  Useful for SigLIP-style
  MHA (ratio=1) where the redundant slots are inert.
* Other ratios (``ratio % blen != 0`` and ``ratio > blen``): the dispatch in
  ``_generate_attention_code`` falls back to the compositional skeleton.
"""

from ..reset_reg_asm import reset_fpreg_asm, reset_reg_asm, reset_vmask_asm
from .online_softmax import online_softmax_code
from .output import computing_o_code, computing_row_wise_scaling_code
from .pv import computing_pv_code
from .qkt import qkt_multiply
from .reset import reset_fpsram_code, reset_kv_prefetch, reset_vssram_code

IMM2_BOUND = 2**18 - 1


def flash_attn_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    hq: int,
    hkv: int,
    d: int,
    q_len: int,
    kv_len: int,
    alive_registers_int: list[int],
    alive_registers_fp: list[int],
    vector_sram_base_address: int,
    fp_sram_start_address: int,
    k_base_hbm_offset_reg: int,
    v_base_hbm_offset_reg: int,
) -> str:
    """
    Args:
    vector_sram_base_address: the base address of the vector SRAM
    fp_sram_start_address: the start address of the fp SRAM
    k_base_hbm_offset_reg: the offset register of the k base address in HBM
    v_base_hbm_offset_reg: the offset register of the v base address in HBM
    Description:
        Multi-loops over kv heads then two flash-attn loops, with an inner Q-head
        loop per kv head.  The Q-head loop is broken into ``num_q_passes`` passes
        of up to ``blen`` heads each so that one M_BTMM consumes ``blen`` Q heads
        in parallel.  Per-head softmax/PV/O bookkeeping is unchanged.
    """
    # Iteration Settings
    q_seq_iteration_number = (q_len + mlen - 1) // mlen
    k_seq_iteration_number = (kv_len + mlen - 1) // mlen
    q_index_2_kv_index_ratio = hq // hkv

    stage = "decode" if q_len == 1 else "prefill"
    br = min(mlen, q_len)
    bc = min(mlen, kv_len)

    # Pack as many of the kv-group's Q heads into a single M_BTMM as possible.
    # When ratio < blen the array still emits ``blen`` head slots; we just
    # ignore the redundant tail (heads_in_pass < blen).  When ratio > blen we
    # require ratio % blen == 0 so passes are full.
    if q_index_2_kv_index_ratio == 0:
        raise ValueError(
            f"flash_attn_asm: invalid GQA ratio hq={hq}, hkv={hkv} (must give ratio >= 1)"
        )
    if q_index_2_kv_index_ratio > blen and q_index_2_kv_index_ratio % blen != 0:
        raise ValueError(
            f"flash_attn_asm: ratio={q_index_2_kv_index_ratio} > blen={blen} requires "
            f"ratio % blen == 0 for multi-pass (got remainder "
            f"{q_index_2_kv_index_ratio % blen}). Caller should fall back to the "
            f"compositional skeleton for this configuration."
        )

    if q_index_2_kv_index_ratio >= blen:
        heads_per_pass = blen
        num_q_passes = q_index_2_kv_index_ratio // blen
    else:
        # ratio < blen: one pass with redundant blen tail; only ``ratio`` heads
        # are consumed downstream.
        heads_per_pass = q_index_2_kv_index_ratio
        num_q_passes = 1

    # Memory Layout:
    # -- FP SRAM --
    # Defalt 0 - zero
    # 1 - infinity
    # fp_sram_start_address - 1 - qk_scale
    # per head dimension * q_index_2_kv_index_ratio level {
    # - m old (MLEN) - 0
    # - m res (MLEN) - 1
    # - l old (MLEN) - 2
    # }

    print("=" * 5, "VSRAM Memory Layout", "=" * 5)
    # -- Vector SRAM --
    # Q  (q_len, hq, d) - Q is stored with shape [seq_len, num_q_heads, head_dim]
    q_base_address = vector_sram_base_address
    print(f"Q Base Address: {q_base_address}")
    # tmp S (MLEN, MLEN, blen) and also tmp P. M_BTMM emits blen tile slots
    # regardless of heads_per_pass, so we still allocate blen-sized scratch.
    s_tile_count = max(blen, q_index_2_kv_index_ratio)
    s_base_address = q_base_address + q_len * hq * d  # Q size = seq_len * num_q_heads * head_dim
    print(f"S Base Address: {s_base_address}")
    # PV (s_tile_count, mlen, mlen)
    pv_base_address = s_base_address + mlen * mlen * s_tile_count
    print(f"PV Base Address: {pv_base_address}")
    # O_Old (q_len, HEAD_DIM * Hq * batch)
    o_old_base_address = pv_base_address + mlen * mlen * s_tile_count
    print(f"O_Old Base Address: {o_old_base_address}")
    print(
        f"GQA: hq={hq}, hkv={hkv}, ratio={q_index_2_kv_index_ratio}, blen={blen}, "
        f"num_q_passes={num_q_passes}, heads_per_pass={heads_per_pass}"
    )

    generated_code = (
        f"; Flash Attention Generation (ratio={q_index_2_kv_index_ratio}, blen={blen}, "
        f"passes={num_q_passes}, heads_per_pass={heads_per_pass})\n"
    )
    generated_code += reset_kv_prefetch(
        hkv=hkv,
        d=d,
        mlen=mlen,
        kv_len=kv_len,
        batch=batch,
        alive_registers_int=alive_registers_int[0:1],
    )

    # loop over kv heads
    for kv_head_index in range(hkv):
        # loop over per kv head kv_len // MLEN
        for _ in range(k_seq_iteration_number):
            print(f" Computing {q_index_2_kv_index_ratio} Q heads for KV head {kv_head_index} in GQA mode")

            # Reset m_fp_sram_start_address for each iteration.  The full
            # ``ratio`` head state lives in FP SRAM regardless of how many
            # passes process it.
            m_fp_sram_start_address = fp_sram_start_address

            # Reset m old for every q_index_2_kv_index_ratio q heads with -inf
            generated_code += reset_fpsram_code(
                reset_start_address=m_fp_sram_start_address,
                per_stride_dim=br,
                stride_dist=3 * br,
                reset_amount=q_index_2_kv_index_ratio,
                reset_val_address=2,
                alive_registers_fp=alive_registers_fp[0:1],
                alive_registers_int=alive_registers_int[0:4],
            )

            # Reset l with zeros
            generated_code += reset_fpsram_code(
                reset_start_address=m_fp_sram_start_address + 2 * br,
                per_stride_dim=br,
                stride_dist=3 * br,
                reset_amount=q_index_2_kv_index_ratio,
                reset_val_address=0,
                alive_registers_fp=alive_registers_fp[0:1],
                alive_registers_int=alive_registers_int[0:4],
            )

            # Reset O_old with zeros
            generated_code += reset_vssram_code(
                reset_start_address=o_old_base_address,
                vect_dim=vlen,
                per_stride_dim=d,
                reset_stride=q_index_2_kv_index_ratio * br,
                reset_amount=q_index_2_kv_index_ratio,
                alive_registers_int=alive_registers_int[0:3],
            )

            # # loop over per q_index_2_kv_index_ratio q heads (q_len // MLEN), compute q_index_2_kv_index_ratio heads in parallel.
            for _ in range(q_seq_iteration_number):
                # Multi-pass over Q-head blocks of size ``heads_per_pass``.
                # Each pass emits one qkt_multiply (= one M_BTMM) followed by
                # ``heads_per_pass`` softmax/PV/O bodies.
                for q_pass_idx in range(num_q_passes):
                    # Index of the first Q head this pass covers (within the
                    # whole-model Q-head numbering).
                    pass_q_head_base = (
                        kv_head_index * q_index_2_kv_index_ratio + q_pass_idx * heads_per_pass
                    )
                    # Index of the first Q head within the kv-group (used for
                    # FP-state addressing — m/l/o slots are laid out per
                    # ratio-head block per kv head).
                    pass_q_head_in_group = q_pass_idx * heads_per_pass

                    # FP-state base for the heads handled in this pass.
                    pass_m_fp_sram_start = (
                        fp_sram_start_address + pass_q_head_in_group * 3 * br
                    )
                    stored_m_fp_res_address = pass_m_fp_sram_start + br

                    generated_code += (
                        f"; Flash-attn pass {q_pass_idx + 1}/{num_q_passes} "
                        f"(kv_head={kv_head_index}, q_heads={pass_q_head_base}.."
                        f"{pass_q_head_base + heads_per_pass - 1})\n"
                    )

                    # Compute S = QKT result for this pass's Q heads.
                    # qkt_multiply offsets internally by q_head_index * d for
                    # the Q address and q_head_index * mlen * mlen for the S
                    # address; passing the absolute Q-head index keeps that
                    # bookkeeping consistent across passes.
                    generated_code += qkt_multiply(
                        d=d,
                        mlen=mlen,
                        stage=stage,
                        alive_registers=alive_registers_int[0:2],
                        q_base_address=q_base_address,
                        k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                        q_head_index=pass_q_head_base,
                        k_head_index=kv_head_index,
                        s_base_address=s_base_address,
                    )
                    generated_code += reset_reg_asm(alive_registers_int[0:2])

                    # Now S holds (blen, br, bc) tiles starting at
                    # s_base_address + pass_q_head_base * br * bc.  We consume
                    # ``heads_per_pass`` of them.
                    for inner_q_head_index in range(heads_per_pass):
                        absolute_q_head = pass_q_head_base + inner_q_head_index
                        in_group_q_head = pass_q_head_in_group + inner_q_head_index

                        # Per Q head level online softmax.  S tile for this
                        # head sits at s_base_address + absolute_q_head * br * bc.
                        generated_code += online_softmax_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=alive_registers_int[0:5],
                            alive_registers_fp=alive_registers_fp[0:5],
                            s_address=s_base_address + absolute_q_head * br * bc,
                            m_start_address=pass_m_fp_sram_start
                            + inner_q_head_index * 3 * br,
                            qk_scale_address=1,
                        )
                        generated_code += reset_fpreg_asm(alive_registers_fp[0:6])
                        generated_code += reset_reg_asm(alive_registers_int[0:6])

                        # Compute PV = P @ V into the packed output region.
                        # Output layout: each row is VLEN elements with heads
                        # packed [head0:d][head1:d]...[headN:d].  We pass the
                        # in-kv-group head index as q_head_index for PV's
                        # internal P-tile addressing (which assumes a per-pass
                        # block of blen heads), then steer the write with the
                        # absolute head_offset within the row.
                        generated_code += computing_pv_code(
                            head_dim=d,
                            blen=blen,
                            mlen=mlen,
                            vlen=vlen,
                            stage=stage,
                            alive_registers=alive_registers_int[0:6],
                            p_base_address=s_base_address + pass_q_head_base * br * bc,
                            v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                            q_head_index=inner_q_head_index,
                            v_head_index=kv_head_index,
                            output_base_address=pv_base_address,
                            head_offset=in_group_q_head,
                        )

                        generated_code += reset_reg_asm(alive_registers_int[0:6])
                        # V_MASK selects the in-group head's columns within
                        # the packed row.
                        generated_code += reset_vmask_asm(
                            alive_registers_int[0], 1 << in_group_q_head
                        )
                        generated_code += computing_o_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=alive_registers_int[0:4],
                            alive_registers_fp=alive_registers_fp[0:1],
                            m_res_base_address=stored_m_fp_res_address,
                            pv_base_address=pv_base_address,
                            o_old_base_address=o_old_base_address,
                            head_dim=d,
                            q_head_num=hq,
                        )
                        stored_m_fp_res_address += 3 * br

                # After processing all Q heads for this tile, apply 1/l
                # scaling for each head.  With packed output format, each row
                # has all heads: [h0:d][h1:d]...; V_MASK selects this head's
                # elements when scaling.
                for scale_head_index in range(q_index_2_kv_index_ratio):
                    # Reset registers and set V_MASK for this head
                    generated_code += reset_reg_asm(alive_registers_int[0:3])
                    generated_code += reset_fpreg_asm(alive_registers_fp[0:1])
                    generated_code += reset_vmask_asm(alive_registers_int[0], 1 << scale_head_index)

                    # l_old address for this head: fp_sram_start_address + head_index * 3 * br + 2 * br
                    l_old_base_address = fp_sram_start_address + scale_head_index * 3 * br + 2 * br

                    # Output is at o_old_base_address with packed format
                    # V_MASK selects the correct head's elements within each row
                    generated_code += computing_row_wise_scaling_code(
                        mlen=mlen,
                        stage=stage,
                        alive_registers_int=alive_registers_int[0:3],
                        alive_registers_fp=alive_registers_fp[0:1],
                        o_old_base_address=o_old_base_address,
                        l_old_base_address=l_old_base_address,
                        o_row_stride=hq * d,  # Row stride is total width of all heads
                        use_mask=True,
                    )
    return generated_code
