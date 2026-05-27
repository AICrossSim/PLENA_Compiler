"""Main Flash Attention assembly code generation - orchestrates all components."""

from .._imm import load_large_int_str as _load_large_int
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
    causal_mask: bool = True,
    kv_valid_len: int | None = None,
    debug_tile_trace_base: int | None = None,
    debug_trace_head_index: int = 0,
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
    kv_valid_total = kv_len if kv_valid_len is None else max(0, min(kv_valid_len, kv_len))

    stage = "decode" if q_len == 1 else "prefill"
    br = min(mlen, q_len)
    bc = min(mlen, kv_len)

    # Batched path (M_BTMM) when ratio == blen — maximum throughput.
    # Per-head path (M_TMM) for all other ratios — avoids M_BTMM over-read
    # panic when ratio < blen (the extra blen-ratio "dummy" heads would read
    # Q past the kv-group allocation).
    use_batched = q_index_2_kv_index_ratio == blen

    # d_padded: pad head_dim to a multiple of mlen so that per-head VRAM
    # addresses are always vlen-aligned (contract requirement).
    # For d already a multiple of mlen (e.g. d=128), d_padded == d.
    d_padded = ((d + mlen - 1) // mlen) * mlen
    # Per-head Q tile size = one VRAM row per token per head.
    # kv_tile_size: number of KV tokens loaded per H_PREFETCH_M call.
    kv_tile_size = min(kv_len, mlen)
    mask_fp_sram_base = fp_sram_start_address + q_index_2_kv_index_ratio * 3 * br
    trace_tile_stride = 2 * br + br * d_padded

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
    # Per-head path (ratio != blen): Q is stored in head-major padded layout
    # [hq, q_len, d_padded] so each head's rows are contiguous for M_TMM.
    # Batched path (ratio == blen): legacy seq-major layout [q_len, hq, d].
    q_base_address = vector_sram_base_address
    print(f"Q Base Address: {q_base_address}")
    if use_batched:
        q_size = q_len * hq * d  # seq-major, no padding
    else:
        q_size = hq * q_len * d_padded  # head-major, d padded to mlen
    # tmp S (MLEN, MLEN, s_tile_count) and also tmp P.
    # Batched path: M_BMM_WO writes blen tiles; allocate blen tiles even though
    # only ratio are consumed by softmax/PV (harmless dead writes).
    # Per-head path: only 1 S tile needed at a time (reused per head).
    s_tile_count = blen if use_batched else 1
    s_base_address = q_base_address + q_size
    print(f"S Base Address: {s_base_address}")
    # PV (q_index_2_kv_index_ratio, mlen, mlen)
    pv_base_address = s_base_address + mlen * mlen * s_tile_count
    print(f"PV Base Address: {pv_base_address}")
    # O_Old (q_len, HEAD_DIM_PADDED * Hq * batch)
    o_old_base_address = pv_base_address + mlen * mlen * q_index_2_kv_index_ratio
    print(f"O_Old Base Address: {o_old_base_address}")

    generated_code = "; Flash Attention Generation \n"
    generated_code += reset_kv_prefetch(
        hkv=hkv,
        d=d_padded,
        mlen=mlen,
        kv_len=kv_len,
        batch=batch,
        alive_registers_int=alive_registers_int[0:1],
    )

    # loop over kv heads
    for kv_head_index in range(hkv):
        # loop over per kv head kv_len // MLEN
        # NOTE: For head-major padded K/V in HBM, each H_PREFETCH_M call loads
        # exactly one KV tile (kv_tile_size * d_padded = mlen*mlen elements).
        # kv_seq_tile_idx tracks the KV-sequence tile within this head.
        for kv_seq_tile_idx in range(k_seq_iteration_number):
            print(f" Computing {q_index_2_kv_index_ratio} Q heads for KV head {kv_head_index} in GQA mode")

            # HBM element offset for K (and V) for this (head, tile):
            #   head-major: kv_head * kv_tile_size * d_padded + tile * kv_tile_size * d_padded
            # With kv_tile_size=mlen, this simplifies to (kv_head*tiles + tile)*mlen*d_padded.
            kv_hbm_offset = (kv_head_index * k_seq_iteration_number + kv_seq_tile_idx) * kv_tile_size * d_padded
            valid_k_cols = max(0, min(kv_tile_size, kv_valid_total - kv_seq_tile_idx * kv_tile_size))

            # Reset m_fp_sram_start_address for each iteration
            m_fp_sram_start_address = fp_sram_start_address

            if kv_seq_tile_idx == 0:
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

                # Reset l_old to zero (use hardware f0, not FP SRAM slot 0 which is -inf)
                generated_code += reset_fpsram_code(
                    reset_start_address=m_fp_sram_start_address + 2 * br,
                    per_stride_dim=br,
                    stride_dist=3 * br,
                    reset_amount=q_index_2_kv_index_ratio,
                    reset_val_address=0,
                    alive_registers_fp=alive_registers_fp[0:1],
                    alive_registers_int=alive_registers_int[0:4],
                    use_zero_reg=True,
                )

            # Reset O_old with zeros (only on the first KV-sequence tile, and for per-head
            # only on the first kv_head so the whole interleaved buffer is zeroed once).
            _reset_o = kv_seq_tile_idx == 0 and (use_batched or kv_head_index == 0)
            if _reset_o:
                generated_code += reset_vssram_code(
                    reset_start_address=o_old_base_address,
                    vect_dim=vlen,
                    per_stride_dim=(hq * d_padded if not use_batched else q_index_2_kv_index_ratio) * br // vlen,
                    reset_stride=vlen,
                    reset_amount=1,
                    alive_registers_int=alive_registers_int[0:3],
                )

            # # loop over per q_index_2_kv_index_ratio q heads (q_len // MLEN), compute q_index_2_kv_index_ratio heads in parallel.
            for _ in range(q_seq_iteration_number):
                # Reuse the same FP SRAM scratch region for each Q tile.
                # Accumulating this base across Q tiles can exceed FP SRAM bounds.
                m_fp_sram_start_address = fp_sram_start_address
                stored_m_fp_res_address = m_fp_sram_start_address + br

                if use_batched:
                    # --- Batched path: M_BTMM computes all ratio heads at once ---
                    # Q layout: (batch, s_q, num_q_heads, h_qkv) seq-major
                    generated_code += qkt_multiply(
                        d=d,
                        mlen=mlen,
                        stage=stage,
                        alive_registers=alive_registers_int[0:2],
                        q_base_address=q_base_address + kv_head_index * q_index_2_kv_index_ratio * d,
                        k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                        q_head_index=kv_head_index * q_index_2_kv_index_ratio,
                        k_head_index=kv_head_index,
                        s_base_address=s_base_address,
                        s_head_offset=0,
                        use_batched=True,
                        blen=blen,
                    )
                    generated_code += reset_reg_asm(alive_registers_int[0:2])

                for inner_q_head_index in range(q_index_2_kv_index_ratio):
                    if not use_batched:
                        # --- Per-head path: M_TMM computes one head's QKT ---
                        # Q layout: head-major padded [hq, q_len, d_padded].
                        # Pre-compute per-head VRAM base so all addresses are vlen-aligned.
                        abs_q_head = kv_head_index * q_index_2_kv_index_ratio + inner_q_head_index
                        q_head_base = q_base_address + abs_q_head * q_len * d_padded
                        generated_code += qkt_multiply(
                            d=d_padded,
                            mlen=mlen,
                            stage=stage,
                            alive_registers=alive_registers_int[0:9],
                            q_base_address=q_head_base,
                            k_base_hbm_offset_reg=k_base_hbm_offset_reg,
                            q_head_index=0,
                            k_head_index=kv_head_index,
                            s_base_address=s_base_address,
                            s_head_offset=0,
                            use_batched=False,
                            blen=blen,
                            q_len=q_len,
                            k_hbm_head_stride=k_seq_iteration_number * kv_tile_size * d_padded,
                        )
                        generated_code += reset_reg_asm(alive_registers_int[0:9])

                    # Per Q head level online softmax.  ``attn_scale_fp_address``
                    # is forwarded by the caller from
                    # ``scheduler["memory_layout"]["fp_sram"]["attn_scale"]``
                    # so this template no longer hardcodes the QK-scale slot.
                    #
                    # For batched path: S tiles are at s_base + head * br * bc.
                    # For per-head path: S tile is always at s_base (offset 0).
                    s_softmax_addr = s_base_address + (inner_q_head_index * br * bc if use_batched else 0)
                    # Final-tile padded-KV handling (per-head path):
                    # build a vector mask row [0...0,-inf...-inf] and add it to each
                    # score row so padded KV columns become -inf before softmax.
                    if not use_batched and valid_k_cols < kv_tile_size:
                        generated_code += reset_fpsram_code(
                            reset_start_address=mask_fp_sram_base,
                            per_stride_dim=mlen,
                            stride_dist=mlen,
                            reset_amount=1,
                            reset_val_address=0,
                            alive_registers_fp=alive_registers_fp[0:1],
                            alive_registers_int=alive_registers_int[0:4],
                            use_zero_reg=True,
                        )
                        generated_code += f"S_LD_FP f{alive_registers_fp[5]}, gp0, {inf_fp_address} \n"
                        for col_idx in range(valid_k_cols, kv_tile_size):
                            generated_code += f"S_ST_FP f{alive_registers_fp[5]}, gp0, {mask_fp_sram_base + col_idx} \n"

                        generated_code += _load_large_int(alive_registers_int[0], pv_base_address)
                        generated_code += f"S_MAP_V_FP gp{alive_registers_int[0]}, gp0, {mask_fp_sram_base} \n"
                        generated_code += _load_large_int(alive_registers_int[1], s_softmax_addr)
                        generated_code += f"C_LOOP_START gp{alive_registers_int[2]}, {mlen} \n"
                        generated_code += (
                            f"V_ADD_VV gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, gp{alive_registers_int[0]}, 0 \n"
                        )
                        generated_code += (
                            f"S_ADDI_INT gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, {mlen} \n"
                        )
                        generated_code += f"C_LOOP_END gp{alive_registers_int[2]} \n"

                    generated_code += online_softmax_code(
                        mlen=mlen,
                        stage=stage,
                        alive_registers_int=alive_registers_int[0:5],
                        alive_registers_fp=alive_registers_fp[0:5],
                        s_address=s_softmax_addr,
                        m_start_address=m_fp_sram_start_address,
                        qk_scale_address=attn_scale_fp_address,
                        causal_mask=causal_mask,
                    )
                    # P is stored in s_base_address (per-head) or
                    # s_base_address + inner_q_head_index * mlen * mlen (batched)
                    m_fp_sram_start_address += br * 3
                    generated_code += reset_fpreg_asm(alive_registers_fp[0:6])
                    generated_code += reset_reg_asm(alive_registers_int[0:6])

                    # Compute PV = P @ V and write to PV accumulator.
                    # For per-head: head_offset=0 (one head at a time, no packing).
                    # V HBM offset uses kv_hbm_offset for head-major padded V
                    # (pass v_head_index=1, v_hbm_head_stride=kv_hbm_offset).
                    generated_code += computing_pv_code(
                        head_dim=d_padded,
                        blen=blen,
                        mlen=mlen,
                        vlen=vlen,
                        stage=stage,
                        alive_registers=alive_registers_int[0:6],
                        p_base_address=s_base_address,
                        v_base_hbm_offset_reg=v_base_hbm_offset_reg,
                        q_head_index=inner_q_head_index if use_batched else 0,
                        v_head_index=kv_head_index,
                        output_base_address=pv_base_address,
                        head_offset=inner_q_head_index if use_batched else 0,
                        v_hbm_offset=kv_hbm_offset if not use_batched else None,
                        v_hbm_head_stride=None,
                    )

                    generated_code += reset_reg_asm(alive_registers_int[0:6])
                    if use_batched:
                        generated_code += reset_vmask_asm(alive_registers_int[0], 1 << inner_q_head_index)
                        o_base_for_head = o_old_base_address
                    else:
                        # Interleaved (seq-major) output: head h at o_base + h * d_padded,
                        # token stride = hq * d_padded  -> same format as seq-major [s_q, hq, d_padded]
                        abs_q_head_o = kv_head_index * q_index_2_kv_index_ratio + inner_q_head_index
                        o_base_for_head = o_old_base_address + abs_q_head_o * d_padded
                    generated_code += computing_o_code(
                        mlen=mlen,
                        stage=stage,
                        alive_registers_int=alive_registers_int[0:4],
                        alive_registers_fp=alive_registers_fp[0:1],
                        m_res_base_address=stored_m_fp_res_address,
                        pv_base_address=pv_base_address,
                        o_old_base_address=o_base_for_head,
                        head_dim=d_padded,
                        q_head_num=hq,
                        use_mask=use_batched,
                        o_row_stride=None if use_batched else hq * d_padded,
                    )
                    stored_m_fp_res_address += 3 * br

                # Optional debug trace: dump per-tile m/l/O state for one selected head.
                # Layout per KV tile at debug_tile_trace_base + tile_idx * trace_tile_stride:
                #   [m_old(br), l_old(br), O_old(br * d_padded)]
                if debug_tile_trace_base is not None and not use_batched:
                    selected_kv_head = debug_trace_head_index // q_index_2_kv_index_ratio
                    selected_rel_head = debug_trace_head_index % q_index_2_kv_index_ratio
                    if kv_head_index == selected_kv_head and selected_rel_head == 0:
                        trace_tile_base = debug_tile_trace_base + kv_seq_tile_idx * trace_tile_stride
                        trace_o_base = trace_tile_base + 2 * br
                        selected_o_base = o_old_base_address + debug_trace_head_index * d_padded
                        selected_m_base = fp_sram_start_address + selected_rel_head * 3 * br
                        selected_l_base = selected_m_base + 2 * br

                        generated_code += reset_vssram_code(
                            reset_start_address=trace_tile_base,
                            vect_dim=br,
                            per_stride_dim=2,
                            reset_stride=br,
                            reset_amount=1,
                            alive_registers_int=alive_registers_int[0:3],
                        )
                        generated_code += reset_vssram_code(
                            reset_start_address=trace_o_base,
                            vect_dim=d_padded,
                            per_stride_dim=br,
                            reset_stride=d_padded,
                            reset_amount=1,
                            alive_registers_int=alive_registers_int[0:3],
                        )

                        # Copy m_old and l_old from FP SRAM into VRAM via S_MAP_V_FP.
                        generated_code += _load_large_int(alive_registers_int[0], pv_base_address)
                        generated_code += f"S_MAP_V_FP gp{alive_registers_int[0]}, gp0, {selected_m_base} \n"
                        generated_code += _load_large_int(alive_registers_int[1], trace_tile_base)
                        generated_code += (
                            f"V_ADD_VV gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, gp{alive_registers_int[0]}, 0 \n"
                        )
                        generated_code += f"S_MAP_V_FP gp{alive_registers_int[0]}, gp0, {selected_l_base} \n"
                        generated_code += _load_large_int(alive_registers_int[1], trace_tile_base + br)
                        generated_code += (
                            f"V_ADD_VV gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, gp{alive_registers_int[0]}, 0 \n"
                        )

                        # Copy O_old rows for selected head into contiguous debug region.
                        generated_code += _load_large_int(alive_registers_int[0], selected_o_base)
                        generated_code += _load_large_int(alive_registers_int[1], trace_o_base)
                        generated_code += f"C_LOOP_START gp{alive_registers_int[2]}, {br} \n"
                        generated_code += (
                            f"V_ADD_VV gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, gp{alive_registers_int[0]}, 0 \n"
                        )
                        generated_code += (
                            f"S_ADDI_INT gp{alive_registers_int[0]}, gp{alive_registers_int[0]}, {hq * d_padded} \n"
                        )
                        generated_code += (
                            f"S_ADDI_INT gp{alive_registers_int[1]}, gp{alive_registers_int[1]}, {d_padded} \n"
                        )
                        generated_code += f"C_LOOP_END gp{alive_registers_int[2]} \n"
                        generated_code += reset_reg_asm(alive_registers_int[0:3])

                # Apply final 1/l normalization only on the last KV tile.
                # Intermediate tiles must keep O_old in accumulator form.
                if kv_seq_tile_idx == k_seq_iteration_number - 1:
                    for scale_head_index in range(q_index_2_kv_index_ratio):
                        generated_code += reset_reg_asm(alive_registers_int[0:3])
                        generated_code += reset_fpreg_asm(alive_registers_fp[0:1])
                        if use_batched:
                            generated_code += reset_vmask_asm(alive_registers_int[0], 1 << scale_head_index)
                            o_base_for_scale = o_old_base_address
                        else:
                            abs_q_head_s = kv_head_index * q_index_2_kv_index_ratio + scale_head_index
                            o_base_for_scale = o_old_base_address + abs_q_head_s * d_padded

                        l_old_base_address = fp_sram_start_address + scale_head_index * 3 * br + 2 * br

                        generated_code += computing_row_wise_scaling_code(
                            mlen=mlen,
                            stage=stage,
                            alive_registers_int=alive_registers_int[0:3],
                            alive_registers_fp=alive_registers_fp[0:1],
                            o_old_base_address=o_base_for_scale,
                            l_old_base_address=l_old_base_address,
                            o_row_stride=hq * d_padded,
                            use_mask=use_batched,
                        )
    return generated_code
