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
    fp_one_reg: int = 1,  # FP register holding 1.0 (must be in fp_preload[fp_one_reg])
    fp_sram_precious_slots: list = None,  # fp_sram slots to save before mask construction
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
    acc_reg = alive_registers[0]  # points to current accumulator VRAM row
    scratch_reg = alive_registers[1]  # points to scratch VRAM area
    shift_reg = alive_registers[2]  # holds shift amount (GP integer)
    hbm_off_reg = alive_registers[3]  # holds HBM offset for H_PREFETCH_V
    mask_reg = alive_registers[4]  # holds mask vector VRAM address
    temp_reg = alive_registers[5]  # general temp

    K_col = C_in * K * K  # number of columns in im2col output row
    num_tiles = (K_col + vlen - 1) // vlen
    if num_tiles > 1:
        assert vlen % K == 0, (
            f"multi-tile im2col with V_SHFT_V requires K ({K}) | VLEN ({vlen}); "
            "K-groups would cross tile boundaries and V_SHFT_V cannot left-shift"
        )

    # W_padded: each input row is stored with W_padded elements in HBM so
    # that row starts are 64-element aligned (required by H_PREFETCH_V).
    if W_padded is None:
        W_padded = W

    # Compute save f_regs for precious fp_sram slots (mirrors im2col_asm_no_shift).
    # Mask construction zeroes fp_sram[0..VLEN-1] then writes mask values, corrupting
    # any pre-loaded values in those slots. Hardware supports f_regs 0..7 only.
    _MAX_FREG = 7
    if fp_sram_precious_slots is None:
        fp_sram_precious_slots = [1, 2, 3, 4, 5]
    used_fregs = {0, fp_one_reg}
    save_fregs = [f for f in range(1, _MAX_FREG + 1) if f not in used_fregs]
    assert len(save_fregs) >= len(fp_sram_precious_slots), (
        f"Not enough free f_regs ({len(save_fregs)}) for {len(fp_sram_precious_slots)} precious fp_sram slots"
    )
    save_fregs = save_fregs[: len(fp_sram_precious_slots)]

    lines: list[str] = []
    lines.append("; ============================================================")
    lines.append("; im2col (with V_SHFT_V): NCHW input in HBM -> im2col matrix in VRAM")
    lines.append(f";   input shape : (1, {C_in}, {H}, {W})  in HBM (W_padded={W_padded})")
    lines.append(f";   kernel      : {K}x{K},  OH={OH}, OW={OW}")
    lines.append(f";   output      : ({M}, {K_col}) in VRAM starting at {output_vram_base}")
    lines.append(f"; Requires: f0=0.0 (hw const), fp_preload[{fp_one_reg}]=1.0")
    lines.append("; ============================================================")

    # ── save precious fp_sram slots before mask construction overwrites them ─
    if fp_sram_precious_slots:
        lines.append("")
        lines.append("; -- Save precious fp_sram slots before mask construction --")
        for slot, sv in zip(fp_sram_precious_slots, save_fregs):
            lines.append(f"S_LD_FP f{sv}, gp0, {slot}")

    # ── build mask vector [1.0]*K + [0.0]*(VLEN-K) into VRAM via FP_SRAM ─────
    lines.append("")
    lines.append(f"; -- Setup: load 1.0 from FP_SRAM[{fp_one_reg}] into f{fp_one_reg} --")
    lines.append(f"S_LD_FP f{fp_one_reg}, gp0, {fp_one_reg}")

    lines.append("")
    lines.append(f"; -- Setup: write mask FP_SRAM[0..{K - 1}]=1.0, [{K}..{vlen - 1}]=0.0 --")
    for kc in range(K):
        lines.append(f"S_ST_FP f{fp_one_reg}, gp0, {kc}")
    for kc in range(K, vlen):
        lines.append(f"S_ST_FP f0, gp0, {kc}")

    lines.append("")
    lines.append(f"; -- Map FP_SRAM -> VRAM mask row at addr {mask_vec_vram_addr} --")
    lines.append(f"S_ADDI_INT gp{temp_reg}, gp0, {mask_vec_vram_addr}")
    lines.append(f"S_MAP_V_FP gp{temp_reg}, gp0, 0")

    lines.append("")
    lines.append("; -- Restore FP_SRAM[0..K-1] to zeros (rest already zero) --")
    for kc in range(K):
        lines.append(f"S_ST_FP f0, gp0, {kc}")

    if fp_sram_precious_slots:
        lines.append("")
        lines.append("; -- Restore precious fp_sram slots --")
        for slot, sv in zip(fp_sram_precious_slots, save_fregs):
            lines.append(f"S_ST_FP f{sv}, gp0, {slot}")

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

    # ── main loop: iterate over tiles × M output positions ───────────
    # VRAM is column-block-major (matches no_shift template):
    #   vram_addr(m, t) = output_vram_base + t*M*vlen + m*vlen
    for tile_t in range(num_tiles):
        tile_start = tile_t * vlen
        tile_end = min(tile_start + vlen, K_col)

        lines.append("")
        lines.append(f"; ==== TILE t={tile_t}  global cols [{tile_start}..{tile_end - 1}] ====")

        for m in range(M):
            oh = m // OW
            ow = m % OW
            out_vram_addr = output_vram_base + tile_t * M * vlen + m * vlen

            lines.append("")
            lines.append(f"; ---- tile={tile_t} output row m={m}  (oh={oh}, ow={ow}) ----")

            # Point accumulator register at the output VRAM row.
            lines.append(f"S_ADDI_INT gp{acc_reg}, gp0, {out_vram_addr}")

            # Zero the accumulator row: multiply mask vector by f0.
            lines.append(f"V_MUL_VF gp{acc_reg}, gp{mask_reg}, f0, 0")

            # Set scratch base address.
            lines.append(f"S_ADDI_INT gp{scratch_reg}, gp0, {scratch_vram_addr}")

            for c in range(C_in):
                for kr in range(K):
                    # Global column position of this K-group in the im2col row.
                    g = c * K * K + kr * K

                    # Skip K-groups not contributing to this tile.
                    # (assertion above guarantees K|vlen for num_tiles>1, so
                    #  K-groups never straddle tile boundaries.)
                    if g + K <= tile_start or g >= tile_end:
                        continue

                    local_shift = g - tile_start

                    # HBM offset for this (c, kr, ow) combination.
                    hbm_offset = (c * H + (oh + kr)) * W_padded + ow

                    # Load K contiguous elements from HBM into scratch row.
                    lines.append(f"S_ADDI_INT gp{hbm_off_reg}, gp0, {hbm_offset}")
                    lines.append(f"H_PREFETCH_V gp{scratch_reg}, gp{hbm_off_reg}, a{input_hbm_base_addr_reg}, 1, 0")

                    # Mask: zero out elements K..VLEN-1
                    lines.append(f"V_MUL_VV gp{scratch_reg}, gp{scratch_reg}, gp{mask_reg}, 0")

                    # Shift right to local position within tile.
                    if local_shift > 0:
                        lines.append(f"S_ADDI_INT gp{shift_reg}, gp0, {local_shift}")
                        lines.append(f"V_SHFT_V gp{scratch_reg}, gp{scratch_reg}, gp{shift_reg}")

                    # Accumulate: accum += shifted scratch row.
                    lines.append(f"V_ADD_VV gp{acc_reg}, gp{acc_reg}, gp{scratch_reg}, 0")

    lines.append("")
    lines.append("; ============================================================")
    lines.append("; im2col complete")
    lines.append("; ============================================================")

    return "\n".join(lines) + "\n"
