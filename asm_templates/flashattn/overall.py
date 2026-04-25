"""Main Flash Attention assembly code generation - orchestrates all components."""

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
    attn_scale_fp_address: int = 5,
    inf_fp_address: int = 0,
) -> str:
    """
    Args:
    vector_sram_base_address: the base address of the vector SRAM
    fp_sram_start_address: the start address of the fp SRAM
    k_base_hbm_offset_reg: the offset register of the k base address in HBM
    v_base_hbm_offset_reg: the offset register of the v base address in HBM
    attn_scale_fp_address: FP SRAM slot holding 1/sqrt(head_dim) for QK
        scaling. Defaults to 5 to match
        ``mem_layout_lib.json::fp_sram::attn_scale``. The scheduler-provided
        slot is forwarded by the caller; previously this was hardcoded to 1
        (the eps slot) and produced catastrophically mis-scaled attention
        logits once fp_sram.bin was seeded per the JSON convention.
    inf_fp_address: FP SRAM slot holding the -inf sentinel used to seed the
        running-max state at the start of each kv tile. Defaults to 0 to match
        ``mem_layout_lib.json::fp_sram::infinity``. Previously hardcoded to 2
        (the hid_reciprocal slot ~= 0.016), which made the running-max start
        at a positive value and corrupted the entire flash softmax accumulation.
    Description:
        This part of asm takes the multi-loops, looping over kv head, then two loops for the flash atten, with small loops over q head per kv head within the inner loop.
    """
    # Iteration Settings
    q_seq_iteration_number = (q_len + mlen - 1) // mlen
    k_seq_iteration_number = (kv_len + mlen - 1) // mlen
    q_index_2_kv_index_ratio = hq // hkv

    stage = "decode" if q_len == 1 else "prefill"
    br = min(mlen, q_len)
    bc = min(mlen, kv_len)

    # Assemptions
    # In current Version (Not Complete)
    assert blen == q_index_2_kv_index_ratio, "Blen must be equal to q_index_2_kv_index_ratio in current version"

    # Memory Layout:
    # -- FP SRAM --
    # ``inf_fp_address`` (default 0) - infinity sentinel for running-max init.
    # ``attn_scale_fp_address`` (default 5) - 1/sqrt(head_dim) (QK scale).
    # Both slot indices are forwarded by the caller from
    # ``scheduler["memory_layout"]["fp_sram"]`` so mem_layout_lib.json is the
    # single source of truth for FPRAM constant placement; this template no
    # longer hardcodes addresses 1 and 2.
    # ``fp_sram_start_address`` onwards holds, for each q head in the kv-group:
    # - m old (br)
    # - m res (br)
    # - l old (br)

    print("=" * 5, "VSRAM Memory Layout", "=" * 5)
    # -- Vector SRAM --
    # Q  (q_len, hq, d) - Q is stored with shape [seq_len, num_q_heads, head_dim]
    q_base_address = vector_sram_base_address
    print(f"Q Base Address: {q_base_address}")
    # tmp S (MLEN, MLEN, blen) and also tmp P.
    s_base_address = q_base_address + q_len * hq * d  # Q size = seq_len * num_q_heads * head_dim
    print(f"S Base Address: {s_base_address}")
    # PV (q_index_2_kv_index_ratio, mlen, mlen)
    pv_base_address = s_base_address + mlen * mlen * q_index_2_kv_index_ratio
    print(f"PV Base Address: {pv_base_address}")
    # O_Old (q_len, HEAD_DIM * Hq * batch)
    o_old_base_address = pv_base_address + mlen * mlen * q_index_2_kv_index_ratio
    print(f"O_Old Base Address: {o_old_base_address}")

    generated_code = "; Flash Attention Generation \n"
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

            # Reset m_fp_sram_start_address for each iteration
            m_fp_sram_start_address = fp_sram_start_address

            # Reset m old for every q_index_2_kv_index_ratio q heads with -inf.
            # ``inf_fp_address`` is forwarded by the caller from
            # ``scheduler["memory_layout"]["fp_sram"]["infinity"]`` (default 0
            # per mem_layout_lib.json).  The previous hardcoded value 2 was
            # the hid_reciprocal slot (~ 0.016 once seeded), which left the
            # running-max init positive and broke flash-softmax accumulation.
            generated_code += reset_fpsram_code(
                reset_start_address=m_fp_sram_start_address,
                per_stride_dim=br,
                stride_dist=3 * br,
                reset_amount=q_index_2_kv_index_ratio,
                reset_val_address=inf_fp_address,
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
                # Compute S = QKT result for this Q head
                # Q layout: (batch, s_q, num_q_heads, h_qkv) -> qkt_multiply adds q_head_index * d internally
                # Q row stride = (hq * d) / mlen = total elements per token / mlen
                stored_m_fp_res_address = m_fp_sram_start_address + br
                generated_code += qkt_multiply(
                    d=d,
                    mlen=mlen,
                    stage=stage,
                    alive_registers=alive_registers_int[0:2],
                    q_base_address=q_base_address + kv_head_index * q_index_2_kv_index_ratio * d,
                    k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                    q_head_index=kv_head_index * q_index_2_kv_index_ratio,
                    k_head_index=kv_head_index,
                    s_base_address=s_base_address + kv_head_index * br * bc,
                )
                generated_code += reset_reg_asm(alive_registers_int[0:2])

                # Now the S is in expected to stored in (blen, br, bc) in vsram

                for inner_q_head_index in range(q_index_2_kv_index_ratio):
                    # Per Q head level online softmax.  ``attn_scale_fp_address``
                    # is forwarded by the caller from
                    # ``scheduler["memory_layout"]["fp_sram"]["attn_scale"]``
                    # so this template no longer hardcodes the QK-scale slot.
                    generated_code += online_softmax_code(
                        mlen=mlen,
                        stage=stage,
                        alive_registers_int=alive_registers_int[0:5],
                        alive_registers_fp=alive_registers_fp[0:5],
                        s_address=s_base_address + inner_q_head_index * br * bc,
                        m_start_address=m_fp_sram_start_address,
                        qk_scale_address=attn_scale_fp_address,
                    )
                    # P is stored in s_base_address + inner_q_head_index * mlen * mlen, taking (blen, mlen, mlen) as a block
                    m_fp_sram_start_address += br * 3
                    generated_code += reset_fpreg_asm(alive_registers_fp[0:6])
                    generated_code += reset_reg_asm(alive_registers_int[0:6])

                    # Compute PV = P @ V and write directly to packed output
                    # Output layout: each row is VLEN elements with heads packed
                    # [head0: d][head1: d][...][headN: d]
                    generated_code += computing_pv_code(
                        head_dim=d,
                        blen=blen,
                        mlen=mlen,
                        vlen=vlen,  # VLEN = total width of all heads = num_q_heads * head_dim
                        stage=stage,
                        alive_registers=alive_registers_int[0:6],
                        p_base_address=s_base_address,
                        v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                        q_head_index=inner_q_head_index,
                        v_head_index=kv_head_index,
                        output_base_address=pv_base_address,
                        head_offset=inner_q_head_index,  # This head's position within the row
                    )

                    generated_code += reset_reg_asm(alive_registers_int[0:6])
                    generated_code += reset_vmask_asm(alive_registers_int[0], 1 << inner_q_head_index)
                    # Use VLEN-aligned address - V_MASK selects the correct head slot
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

                # After processing all Q heads for this tile, apply 1/l scaling for each head
                # With packed output format, each row has all heads: [h0:d][h1:d][h2:d][h3:d]
                # We use V_MASK to select only this head's elements when scaling

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
