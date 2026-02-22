"""
im2col ASM template for PLENA.

Generates assembly that transforms a NCHW input tensor in HBM into an im2col
matrix in VRAM, entirely on-chip.  Each output row m (0..M-1) corresponds to
one (oh, ow) output position and contains the flattened K*K patch across all
C_in channels -- i.e. length C_in*K*K.

Algorithm per output row m:
    oh = m // OW,  ow = m % OW
    accum = zeros(VLEN)
    for c in 0 .. C_in-1:
        for kr in 0 .. K-1:
            hbm_off = c*H*W + (oh+kr)*W + ow      # start of K contiguous elements
            tmp = load_from_hbm(hbm_off)            # VLEN elements, first K useful
            tmp = tmp * mask_vec                    # zero out positions K..VLEN-1
            shift_amt = c*K*K + kr*K
            tmp = right_shift(tmp, shift_amt)       # place K elements at target position
            accum += tmp
    store accum -> output_vram[m]

The mask vector [1,1,..,1, 0,..,0] (K ones followed by VLEN-K zeros) must be
preloaded into VRAM at mask_vec_vram_addr by the host before execution.
"""

PREFETCH_V_AMOUNT = 4  # H_PREFETCH_V always loads this many VRAM rows


def im2col_asm(
    mlen: int,
    vlen: int,
    C_in: int,
    H: int,
    W: int,
    K: int,
    OH: int,
    OW: int,
    M: int,
    alive_registers: list[int],
    input_hbm_base_addr_reg: int,
    mask_vec_vram_addr: int,
    scratch_vram_addr: int,
    output_vram_base: int,
    W_padded: int = None,
) -> str:
    """
    Generate PLENA assembly for on-chip im2col.

    Args:
        mlen:  Matrix tile size (typically 64).
        vlen:  Vector length / VRAM row width (typically 64).
        C_in:  Number of input channels.
        H:     Input height.
        W:     Input width.
        K:     Kernel size (square K x K).
        OH:    Output height  = (H - K) / stride + 1.
        OW:    Output width   = (W - K) / stride + 1.
        M:     Total output positions = B * OH * OW.
        alive_registers:
            List of available GP register indices (need at least 6).
            [0] = accumulator VRAM address register
            [1] = scratch VRAM base register
            [2] = shift amount register
            [3] = HBM offset register
            [4] = mask VRAM address register
            [5] = temp register
        input_hbm_base_addr_reg:
            HBM address register index (a?) pointing to input base.
        mask_vec_vram_addr:
            VRAM address where the [1..1, 0..0] mask vector lives.
        scratch_vram_addr:
            VRAM base address for scratch rows (needs PREFETCH_V_AMOUNT * vlen
            elements = 4 * 64 = 256 slots).
        output_vram_base:
            VRAM base address for the M im2col output rows.

    Returns:
        Generated assembly code as a string.
    """
    # ── register allocation ──────────────────────────────────────────
    acc_reg = alive_registers[0]       # points to current accumulator VRAM row
    scratch_reg = alive_registers[1]   # points to scratch VRAM area
    shift_reg = alive_registers[2]     # holds shift amount (GP integer)
    hbm_off_reg = alive_registers[3]   # holds HBM offset for H_PREFETCH_V
    mask_reg = alive_registers[4]      # holds mask vector VRAM address
    temp_reg = alive_registers[5]      # general temp

    K_col = C_in * K * K  # number of columns in im2col output row
    assert K_col <= vlen, (
        f"im2col row length {K_col} exceeds VLEN {vlen}; "
        "tiled im2col not yet supported"
    )

    # W_padded: each input row is stored with W_padded elements in HBM so
    # that row starts are 64-element aligned (required by H_PREFETCH_V).
    if W_padded is None:
        W_padded = W

    lines: list[str] = []
    lines.append("; ============================================================")
    lines.append("; im2col: transform NCHW input in HBM -> im2col matrix in VRAM")
    lines.append(f";   input shape : (1, {C_in}, {H}, {W})  in HBM (W_padded={W_padded})")
    lines.append(f";   kernel      : {K}x{K},  OH={OH}, OW={OW}")
    lines.append(f";   output      : ({M}, {K_col}) in VRAM starting at {output_vram_base}")
    lines.append("; ============================================================")

    # ── set up stride for H_PREFETCH_V ───────────────────────────────
    # Stride = W_padded so consecutive prefetched rows map to consecutive
    # HBM rows.  H_PREFETCH_V requires the base element address to be a
    # multiple of 64; using W_padded=64 guarantees this.
    lines.append("")
    lines.append("; -- configure HBM stride for H_PREFETCH_V --")
    lines.append(f"S_ADDI_INT gp{temp_reg}, gp0, {W_padded}")
    lines.append(f"C_SET_STRIDE_REG gp{temp_reg}")

    # ── set scale offset (total input tensor size in elements) ────────
    input_tensor_size = C_in * H * W_padded
    lines.append(f"S_ADDI_INT gp{temp_reg}, gp0, {input_tensor_size}")
    lines.append(f"C_SET_SCALE_REG gp{temp_reg}")

    # ── set mask register address (constant across all rows) ─────────
    lines.append(f"S_ADDI_INT gp{mask_reg}, gp0, {mask_vec_vram_addr}")

    # ── main loop: iterate over M output positions ───────────────────
    for m in range(M):
        oh = m // OW
        ow = m % OW
        out_vram_addr = output_vram_base + m * vlen

        lines.append("")
        lines.append(f"; ---- output row m={m}  (oh={oh}, ow={ow}) ----")

        # Point accumulator register at the output VRAM row.
        lines.append(f"S_ADDI_INT gp{acc_reg}, gp0, {out_vram_addr}")

        # Zero the accumulator row: multiply the scratch row by f0 (=0.0)
        # and write to the accumulator address.
        # First we need *something* in the scratch slot; we can just use
        # the mask vector as a source and multiply by f0.
        lines.append(f"V_MUL_VF gp{acc_reg}, gp{mask_reg}, f0, 0")

        # Set scratch base address.
        lines.append(f"S_ADDI_INT gp{scratch_reg}, gp0, {scratch_vram_addr}")

        for c in range(C_in):
            for kr in range(K):
                # HBM offset for this (c, kr, ow) combination.
                # Input is stored with row-stride W_padded so that each row
                # start is 64-element aligned (H_PREFETCH_V requirement):
                #   input[0, c, oh+kr, ow]  ->  (c*H + oh+kr)*W_padded + ow
                hbm_offset = (c * H + (oh + kr)) * W_padded + ow

                # Target position inside the im2col row.
                shift_amount = c * K * K + kr * K

                # Load from HBM into scratch VRAM.
                # H_PREFETCH_V loads PREFETCH_V_AMOUNT rows.  We only use
                # scratch slot 0 (at scratch_vram_addr).
                lines.append(
                    f"S_ADDI_INT gp{hbm_off_reg}, gp0, {hbm_offset}"
                )
                lines.append(
                    f"H_PREFETCH_V gp{scratch_reg}, gp{hbm_off_reg}, "
                    f"a{input_hbm_base_addr_reg}, 1, 0"
                )

                # Mask: zero out elements K..VLEN-1
                # scratch[0] = scratch[0] * mask_vec  (element-wise)
                lines.append(
                    f"V_MUL_VV gp{scratch_reg}, gp{scratch_reg}, "
                    f"gp{mask_reg}, 0"
                )

                # Shift right by shift_amount positions.
                # V_SHFT_V rd, rs1, rs2  -- rs2 is a GP register holding
                # the integer shift amount.
                if shift_amount > 0:
                    lines.append(
                        f"S_ADDI_INT gp{shift_reg}, gp0, {shift_amount}"
                    )
                    lines.append(
                        f"V_SHFT_V gp{scratch_reg}, gp{scratch_reg}, "
                        f"gp{shift_reg}"
                    )

                # Accumulate: accum += shifted scratch row.
                lines.append(
                    f"V_ADD_VV gp{acc_reg}, gp{acc_reg}, "
                    f"gp{scratch_reg}, 0"
                )

    lines.append("")
    lines.append("; ============================================================")
    lines.append("; im2col complete")
    lines.append("; ============================================================")

    return "\n".join(lines) + "\n"
