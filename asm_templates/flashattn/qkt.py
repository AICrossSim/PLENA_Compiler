"""QKT multiplication assembly code generation for Flash Attention."""

from .._imm import load_large_int as _load_large_int

IMM2_BOUND = 2**18 - 1


def qkt_multiply(
    d: int,
    mlen: int,
    stage: str,
    alive_registers: list[int],
    q_base_address: int,
    k_base_hbm_offset_reg: int,
    q_head_index: int,
    k_head_index: int,
    s_base_address: int = 0,
    s_head_offset: int = 0,
    use_batched: bool = True,
    blen: int = 4,
) -> str:
    """
    Args:
        mlen: the number of rows in the first matrix.
        (mlen // hlen): the number of Q heads that could process with the same K head.
        hlen is assumed to be equal to d
        d: the head dimension
        alive_registers: the list of alive registers.  When *use_batched* is
            True the first 2 are used (existing behaviour).  When False the
            first 9 are required for the per-head M_TMM triple loop.
        q_base_address: the base address of the query.
        k_base_hbm_offset_reg: the HBM offset register for key prefetch.
        q_head_index: absolute Q-head index (used for VRAM addressing).
        k_head_index: KV-head index (used for HBM prefetch offset).
        s_base_address: scratch VRAM base for S tiles.
        s_head_offset: relative head offset (0..ratio-1) for S writeback.
        use_batched: True  → emit M_BTMM + M_BMM_WO  (ratio == blen path).
                     False → emit per-head M_TMM + M_MM_WO loop.
        blen: hardware systolic block length (needed for M_TMM loop counts).
    Description:
        This part of asm code gen template is used to compute QKT result.

        **Batched path** (use_batched=True):
        Assuming Q is in dim of (1, MLEN, MLEN//HLEN, HLEN) for prefill and
        (1, 1, MLEN//HLEN, HLEN) for decode, K is in dim of (1, MLEN, 1, HLEN).
        M_BTMM processes ``broadcast_amount`` heads simultaneously.

        **Per-head path** (use_batched=False):
        Uses M_TMM (non-batched transposed matmul) in a triple-nested loop
        to compute S = Q_head @ K^T for a single Q-head.  K is loaded once
        per KV-tile (H_PREFETCH_M stays outside the per-head loop in the
        caller).  This avoids the M_BTMM over-read panic when ratio < blen.
    """
    q_base_register = alive_registers[0]
    k_base_register = alive_registers[1]
    s_base_register = q_base_register
    generated_code = "; QKT Per KV Head Multiplication \n"

    # Prefetch K from HBM (shared by both batched and per-head paths)
    generated_code += f"S_ADDI_INT gp{q_base_register}, gp0, {q_base_address + q_head_index * d} \n"
    generated_code += f"S_ADDI_INT gp{k_base_register}, gp0, {k_head_index * d} \n"

    # Use stride_en=0 for contiguous prefetch to avoid 64-byte alignment issues
    # When stride < 64 elements, strided access causes unaligned HBM reads
    # Parameter order: rd, rs1, rs2, rstride(stride_en), funct1(scale_en)
    generated_code += f"H_PREFETCH_M gp0, gp{k_base_register}, a{k_base_hbm_offset_reg}, 0, 1 \n"

    if use_batched:
        # --- Batched path: M_BTMM processes blen heads simultaneously ---
        # s_head_offset is the relative head offset (0..ratio-1) for S writeback.
        # q_head_index is absolute and used only for Q VRAM read addressing above.
        if stage == "prefill":
            generated_code += f"M_BTMM 0, gp{q_base_register}, gp0 \n"
            assert s_base_address + s_head_offset * mlen * mlen < IMM2_BOUND, "S base address is too large"
            generated_code += f"S_ADDI_INT gp{s_base_register}, gp0, {s_base_address + s_head_offset * mlen * mlen} \n"
            generated_code += f"M_BMM_WO gp{s_base_register}, 0 \n"
        else:
            generated_code += f"M_BTMV 0, gp{q_base_register}, gp0 \n"
            assert s_base_address + s_head_offset * mlen < IMM2_BOUND, "S base address is too large"
            generated_code += f"S_ADDI_INT gp{s_base_register}, gp0, {s_base_address + s_head_offset * mlen} \n"
            generated_code += f"M_BMV_WO gp{s_base_register}, 0 \n"
    else:
        # --- Per-head path: M_TMM / M_TMV for a single Q-head ---
        # K is already in MSRAM at address 0 (prefetched above).
        # Compute S = Q_head @ K^T using non-batched matmul.
        #
        # M_TMM reads blen VRAM rows (stride mlen) and blen MSRAM cols
        # (transposed), producing [blen, blen] into m_accum.
        # Triple loop covers the full [mlen, mlen] S tile:
        #   outer  – S column blocks  (mlen // blen)
        #   middle – S row blocks     (mlen // blen)
        #   inner  – K-dim accumulate (d // blen)

        s_addr = s_base_address + s_head_offset * (mlen * mlen if stage == "prefill" else mlen)
        q_addr = q_base_address + q_head_index * d

        if stage == "prefill":
            generated_code += _qkt_per_head_prefill(
                d=d,
                mlen=mlen,
                blen=blen,
                q_vram_base=q_addr,
                k_msram_base=0,
                s_vram_base=s_addr,
                alive_registers=alive_registers,
            )
        else:
            generated_code += _qkt_per_head_decode(
                d=d,
                mlen=mlen,
                blen=blen,
                q_vram_base=q_addr,
                k_msram_base=0,
                s_vram_base=s_addr,
                alive_registers=alive_registers,
            )

    return generated_code


def _qkt_per_head_prefill(
    d: int,
    mlen: int,
    blen: int,
    q_vram_base: int,
    k_msram_base: int,
    s_vram_base: int,
    alive_registers: list[int],
) -> str:
    """Emit M_TMM triple loop for one head's QKT (prefill).

    Computes S[mlen, mlen] = Q[mlen, d] @ K^T[d, mlen] using:
        outer  : mlen/blen  column blocks of S (= row-block indices of K^T)
        middle : mlen/blen  row blocks of S    (= row-block indices of Q)
        inner  : d/blen     accumulation over the key dimension

    K is already prefetched to MSRAM at *k_msram_base* (typically 0).
    M_TMM transposes the MSRAM tile internally, so we pass K's address
    directly and it computes Q_tile @ K_tile^T.
    """
    # Register allocation
    gp_q = alive_registers[0]
    gp_k = alive_registers[1]
    gp_s = alive_registers[2]
    gp_loop_outer = alive_registers[3]
    gp_loop_middle = alive_registers[4]
    gp_loop_inner = alive_registers[5]
    gp_q_row_base = alive_registers[6]
    gp_k_col_base = alive_registers[7]
    gp_s_col_base = alive_registers[8]

    tiles_per_mlen = mlen // blen
    num_k_blocks = d // blen if d >= blen else 1

    # Strides:
    # Q rows advance by blen*mlen VRAM elements per row-block.
    q_row_stride = blen * mlen
    # K^T column blocks advance by blen*mlen in MSRAM (blen rows of K =
    # blen cols of K^T).  M_TMM selects cols via
    # mat_offset = (m_addr % mlen^2) / mlen.
    k_col_stride = blen * mlen

    lines: list[str] = []
    lines.append("; QKT per-head M_TMM triple loop (prefill)")

    # Initialise outer-loop base registers
    lines.extend(_load_large_int(gp_k_col_base, k_msram_base))
    lines.extend(_load_large_int(gp_s_col_base, s_vram_base))

    # Outer loop: S column blocks (mlen // blen)
    lines.append(f"C_LOOP_START gp{gp_loop_outer}, {tiles_per_mlen}")

    # Reset Q row base and S write pointer for this column strip
    lines.extend(_load_large_int(gp_q_row_base, q_vram_base))
    lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s_col_base}, 0")

    # Middle loop: S row blocks (mlen // blen)
    lines.append(f"C_LOOP_START gp{gp_loop_middle}, {tiles_per_mlen}")

    # Set Q and K pointers for inner accumulation
    lines.append(f"S_ADDI_INT gp{gp_q}, gp{gp_q_row_base}, 0")
    lines.append(f"S_ADDI_INT gp{gp_k}, gp{gp_k_col_base}, 0")

    # Inner loop: accumulate over K dimension (d // blen blocks)
    lines.append(f"C_LOOP_START gp{gp_loop_inner}, {num_k_blocks}")
    lines.append(f"M_TMM 0, gp{gp_q}, gp{gp_k}")
    # K advances to next blen-col block of K^T; Q stays (full-row read).
    lines.append(f"S_ADDI_INT gp{gp_k}, gp{gp_k}, {blen * mlen}")
    lines.append(f"C_LOOP_END gp{gp_loop_inner}")

    # Write accumulated [blen, blen] S tile
    lines.append(f"M_MM_WO gp{gp_s}, gp0, 0")

    # Advance Q to next row block, S to next row block
    lines.append(f"S_ADDI_INT gp{gp_q_row_base}, gp{gp_q_row_base}, {q_row_stride}")
    lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {q_row_stride}")

    lines.append(f"C_LOOP_END gp{gp_loop_middle}")

    # Advance K column base and S column base for next column block
    lines.append(f"S_ADDI_INT gp{gp_k_col_base}, gp{gp_k_col_base}, {k_col_stride}")
    lines.append(f"S_ADDI_INT gp{gp_s_col_base}, gp{gp_s_col_base}, {blen}")

    lines.append(f"C_LOOP_END gp{gp_loop_outer}")

    return "\n".join(lines) + "\n"


def _qkt_per_head_decode(
    d: int,
    mlen: int,
    blen: int,
    q_vram_base: int,
    k_msram_base: int,
    s_vram_base: int,
    alive_registers: list[int],
) -> str:
    """Emit M_TMV loop for one head's QKT (decode — single query token).

    Computes S[1, mlen] = Q[1, d] @ K^T[d, mlen].
    M_TMV computes [1, mlen] @ transpose(msram_slice)[mlen, blen] = [1, blen],
    accumulated into v_accum.  Loop over d/blen blocks of K^T columns.
    """
    gp_q = alive_registers[0]
    gp_k = alive_registers[1]
    gp_s = alive_registers[2]
    gp_loop = alive_registers[3]

    num_k_col_blocks = mlen // blen

    lines: list[str] = []
    lines.append("; QKT per-head M_TMV loop (decode)")

    lines.extend(_load_large_int(gp_q, q_vram_base))
    lines.extend(_load_large_int(gp_k, k_msram_base))
    lines.extend(_load_large_int(gp_s, s_vram_base))

    # Loop over K^T column blocks — each M_TMV produces blen elements of S
    lines.append(f"C_LOOP_START gp{gp_loop}, {num_k_col_blocks}")
    lines.append(f"M_TMV 0, gp{gp_q}, gp{gp_k}")
    lines.append(f"M_MV_WO gp{gp_s}, 0")
    lines.append(f"S_ADDI_INT gp{gp_k}, gp{gp_k}, {blen * mlen}")
    lines.append(f"S_ADDI_INT gp{gp_s}, gp{gp_s}, {blen}")
    lines.append(f"C_LOOP_END gp{gp_loop}")

    return "\n".join(lines) + "\n"
