"""
im2col ASM template for PLENA — documented-instructions-only variant.

Replaces V_SHFT_V (opcode 0x32, not formally documented/supported) with:
  V_MUL_VV   — multiply scratch row by a basis vector (isolates one element)
  V_RED_SUM  — reduce to scalar (= the single non-zero element)
  S_ST_FP    — write scalar to FP_SRAM at the target column position
  S_MAP_V_FP — flush the completed im2col row from FP_SRAM → VRAM

Algorithm per output row m:
  oh = m // OW,  ow = m % OW
  for c in 0..C_in-1:
    for kr in 0..K-1:
      scratch = H_PREFETCH_V(hbm_off)         # K real elements at positions 0..K-1
      for kc in 0..K-1:
        temp = scratch * e_{kc}               # V_MUL_VV; e_{kc}[i]=1 if i==kc else 0
        f_ex = 0                              # S_ADD_FP f_ex, f0, f0  (f0 is HW const 0)
        f_ex += sum(temp)                     # V_RED_SUM = scratch[kc]
        FP_SRAM[c*K*K + kr*K + kc] = f_ex    # S_ST_FP
  VRAM[out_row_m] = FP_SRAM[0..K_col-1]      # S_MAP_V_FP

HBM alignment: identical to im2col_asm.py — use W_padded=64 and OW=1 so that
  hbm_off = (c*H + oh+kr) * W_padded + ow   (ow=0) is always 64-element-aligned.

Instruction counts vs V_SHFT_V variant (~4× more instructions per row):
  V_SHFT_V path:  C_in * K * ~4  instructions per output row
  no-shift path:  C_in * K * (2 + K*4) instructions per output row (for K=4: ~11×)

Requires:
  f0 = 0.0  (hardware constant — guaranteed by emulator, writing is a no-op)
  fp_preload[1] = 1.0  (loaded into f1 via S_LD_FP at runtime; fp_reg starts all-zero)
"""

PREFETCH_V_AMOUNT = 4   # H_PREFETCH_V always loads this many VRAM rows


def im2col_asm_no_shift(
    mlen: int,
    vlen: int,
    C_in: int,
    H: int,
    W: int,
    K: int,
    OH: int,
    OW: int,
    M: int,
    alive_registers: list,
    input_hbm_base_addr_reg: int,
    basis_vram_base: int,
    scratch_vram_addr: int,
    temp_vram_addr: int,
    output_vram_base: int,
    W_padded: int = None,
    fp_one_reg: int = 1,   # FP register holding 1.0 (must be pre-loaded via fp_preload)
    fp_ex_reg: int = 2,    # FP register used as V_RED_SUM accumulator
    fp_sram_precious_slots: list = None,  # fp_sram slots to save before and restore after im2col
) -> str:
    """
    Generate PLENA assembly for on-chip im2col without V_SHFT_V.

    Args:
        mlen:   Matrix tile size (typically 64).
        vlen:   Vector length (typically 64).
        C_in:   Number of input channels.
        H:      Input height.
        W:      Input width.
        K:      Kernel size (K×K).
        OH:     Output height.
        OW:     Output width.
        M:      Total output positions = OH * OW.
        alive_registers:
            List of 5 GP register indices:
              [0] scratch_reg — VRAM addr of H_PREFETCH_V landing area
              [1] temp_reg   — VRAM addr of V_MUL_VV result (1 row)
              [2] off_reg    — HBM element offset
              [3] out_reg    — output VRAM row pointer
              [4] basis_reg  — basis-vector VRAM pointer (reused per kc)
        input_hbm_base_addr_reg:
            Index of 'a' register pre-loaded with input HBM base address.
        basis_vram_base:
            VRAM base for K basis vectors (each vlen elements wide).
        scratch_vram_addr:
            VRAM addr of scratch area (PREFETCH_V_AMOUNT * vlen elements).
        temp_vram_addr:
            VRAM addr of temp mul-result row (vlen elements).
        output_vram_base:
            VRAM base for M im2col output rows.
        W_padded:
            Input row width in HBM (padded for 64-element alignment). Default=W.
        fp_one_reg:
            FP register index holding 1.0.
        fp_ex_reg:
            FP register index used as V_RED_SUM accumulator (zeroed before each use).

    Returns:
        Assembly code string.
    """
    scratch_reg = alive_registers[0]
    temp_reg    = alive_registers[1]
    off_reg     = alive_registers[2]
    out_reg     = alive_registers[3]
    basis_reg   = alive_registers[4]

    K_col = C_in * K * K

    assert K <= vlen, (
        f"K={K} > vlen={vlen}; basis-vector extraction requires K <= vlen"
    )
    num_tiles = (K_col + vlen - 1) // vlen

    if W_padded is None:
        W_padded = W

    # Compute save f_regs for precious fp_sram slots.
    # im2col zeroes fp_sram[0..K-1] during setup and writes fp_sram[0..K_col-1]
    # during the main loop, corrupting any pre-loaded values in those slots.
    # Hardware supports f_regs 0..7 only.
    _MAX_FREG = 7
    if fp_sram_precious_slots is None:
        fp_sram_precious_slots = [1, 2, 3, 4, 5]
    used_fregs = {0, fp_one_reg, fp_ex_reg}
    # Collect available f_regs in [1..MAX_FREG] (skip f0 — hw const)
    save_fregs = [f for f in range(1, _MAX_FREG + 1) if f not in used_fregs]
    assert len(save_fregs) >= len(fp_sram_precious_slots), (
        f"Not enough free f_regs ({len(save_fregs)}) for "
        f"{len(fp_sram_precious_slots)} precious fp_sram slots"
    )
    save_fregs = save_fregs[:len(fp_sram_precious_slots)]

    lines = []
    lines.append("; ============================================================")
    lines.append("; im2col (no-shift): NCHW input in HBM -> im2col matrix in VRAM")
    lines.append(f";   input  : (1,{C_in},{H},{W})  W_padded={W_padded}")
    lines.append(f";   kernel : {K}x{K}  OH={OH}  OW={OW}")
    lines.append(f";   output : ({M},{K_col}) @ VRAM base {output_vram_base}")
    lines.append("; ISA: H_PREFETCH_V V_MUL_VV V_RED_SUM S_ST_FP S_MAP_V_FP")
    lines.append("; Requires: f0=0.0 (hw const), fp_preload[1]=1.0")
    lines.append("; ============================================================")

    # ------------------------------------------------------------------
    # One-time setup: build K basis vectors e_0..e_{K-1} in VRAM.
    #   e_kc[i] = 1.0 if i == kc else 0.0
    #
    # Strategy (uses only S_ST_FP and S_MAP_V_FP):
    #   1. Zero FP_SRAM[0..K-1] with f0 (hardware constant 0.0).
    #   2. For each kc: write 1.0 at position kc, S_MAP_V_FP to VRAM,
    #      restore 0.0 at position kc.
    #
    # Note: positions K..63 in FP_SRAM are uninitialized but harmless —
    # scratch rows have zeros at positions K..63 (HBM padding), so
    # scratch[i] * e_kc[i] = 0 for i >= K regardless of e_kc[i].
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Load 1.0 into fp_one_reg from FP_SRAM BEFORE zeroing FP_SRAM.
    # fp_preload[fp_one_reg] must be set to 1.0 by the caller.
    # (fp_reg starts all-zero; fp_preload goes into FP_SRAM, not fp_reg.)
    # ------------------------------------------------------------------
    if fp_sram_precious_slots:
        lines.append("")
        lines.append("; -- Save precious fp_sram slots before im2col corrupts them --")
        for slot, sv in zip(fp_sram_precious_slots, save_fregs):
            lines.append(f"S_LD_FP f{sv}, gp0, {slot}")

    lines.append("")
    lines.append(f"; -- Setup: load 1.0 from FP_SRAM[{fp_one_reg}] into f{fp_one_reg} --")
    lines.append(f"; (requires fp_preload[{fp_one_reg}] = 1.0)")
    lines.append(f"S_LD_FP f{fp_one_reg}, gp0, {fp_one_reg}")

    lines.append("")
    lines.append("; -- Setup: zero FP_SRAM[0..K-1] for basis construction --")
    for kc in range(K):
        lines.append(f"S_ST_FP f0, gp0, {kc}")

    lines.append("")
    lines.append(f"; -- Setup: build {K} basis vectors in VRAM --")
    for kc in range(K):
        basis_vram_addr = basis_vram_base + kc * vlen
        lines.append(f"; e_{kc}: 1.0 at pos {kc}")
        lines.append(f"S_ST_FP f{fp_one_reg}, gp0, {kc}")
        lines.append(f"S_ADDI_INT gp{basis_reg}, gp0, {basis_vram_addr}")
        lines.append(f"S_MAP_V_FP gp{basis_reg}, gp0, 0")  # FP_SRAM[0..63] -> VRAM
        lines.append(f"S_ST_FP f0, gp0, {kc}")             # restore 0.0

    # Pin scratch and temp VRAM addresses (constant across all rows)
    lines.append("")
    lines.append("; -- Setup: pin scratch/temp VRAM pointers --")
    lines.append(f"S_ADDI_INT gp{scratch_reg}, gp0, {scratch_vram_addr}")
    lines.append(f"S_ADDI_INT gp{temp_reg}, gp0, {temp_vram_addr}")

    # Configure HBM stride register (= W_padded) and scale (= total input size)
    # Required by H_PREFETCH_V in stride mode (mode=1).
    lines.append("")
    lines.append("; -- Setup: HBM stride and scale --")
    lines.append(f"S_ADDI_INT gp{basis_reg}, gp0, {W_padded}")
    lines.append(f"C_SET_STRIDE_REG gp{basis_reg}")
    input_tensor_elems = C_in * H * W_padded
    lines.append(f"S_ADDI_INT gp{basis_reg}, gp0, {input_tensor_elems}")
    lines.append(f"C_SET_SCALE_REG gp{basis_reg}")

    # ------------------------------------------------------------------
    # Main loop: process each tile (K_col columns split into num_tiles
    # tiles of vlen columns each), then each of the M output positions.
    #
    # VRAM layout is column-block-major:
    #   tile t of output row m → vram_addr = output_vram_base + t*M*vlen + m*vlen
    #
    # For K_col <= vlen (num_tiles == 1) this produces identical ASM to the
    # single-tile implementation.
    # ------------------------------------------------------------------
    for tile_t in range(num_tiles):
        tile_start = tile_t * vlen
        tile_end   = min(tile_start + vlen, K_col)
        tile_width = tile_end - tile_start   # < vlen only for the last partial tile

        lines.append("")
        lines.append(
            f"; ==== TILE t={tile_t}  global cols [{tile_start}..{tile_end - 1}] ===="
        )

        for m in range(M):
            oh = m // OW
            ow = m % OW

            # Column-block-major VRAM address for (row m, tile t):
            #   col-block t base  = output_vram_base + t * M * vlen
            #   row m within block = + m * vlen
            out_vram_addr = output_vram_base + tile_t * M * vlen + m * vlen

            lines.append("")
            lines.append(
                f"; ==== tile={tile_t} output row m={m}  oh={oh}  ow={ow}  "
                f"vram={out_vram_addr} ===="
            )
            lines.append(f"S_ADDI_INT gp{out_reg}, gp0, {out_vram_addr}")

            # Zero FP_SRAM slots that will NOT be written this tile.
            # For full tiles every slot [0..vlen-1] gets written, so no zeroing.
            # For the last partial tile, zero slots [tile_width..vlen-1] so that
            # S_MAP_V_FP does not flush stale values from a previous iteration.
            if tile_width < vlen:
                for pos in range(tile_width, vlen):
                    lines.append(f"S_ST_FP f0, gp0, {pos}")

            for c in range(C_in):
                for kr in range(K):
                    # Global im2col column for (c, kr, kc) = c*K*K + kr*K + kc.
                    # Only extract kc values whose global column falls in
                    # [tile_start, tile_end) for this tile.
                    contributing_kcs = [
                        kc for kc in range(K)
                        if tile_start <= c * K * K + kr * K + kc < tile_end
                    ]
                    if not contributing_kcs:
                        continue

                    # HBM offset for input[0, c, oh+kr, ow]
                    hbm_offset = (c * H + oh + kr) * W_padded + ow

                    lines.append(f"; (c={c}, kr={kr})  hbm_off={hbm_offset}")
                    lines.append(f"S_ADDI_INT gp{off_reg}, gp0, {hbm_offset}")
                    lines.append(
                        f"H_PREFETCH_V gp{scratch_reg}, gp{off_reg}, "
                        f"a{input_hbm_base_addr_reg}, 1, 0"
                    )

                    for kc in contributing_kcs:
                        # Local FP_SRAM slot = global column index minus tile_start
                        local_pos  = c * K * K + kr * K + kc - tile_start
                        basis_addr = basis_vram_base + kc * vlen

                        lines.append(
                            f"S_ADDI_INT gp{basis_reg}, gp0, {basis_addr}"
                        )
                        lines.append(
                            f"V_MUL_VV gp{temp_reg}, gp{scratch_reg}, gp{basis_reg}, 0"
                        )
                        lines.append(f"S_ADD_FP f{fp_ex_reg}, f0, f0")
                        lines.append(
                            f"V_RED_SUM f{fp_ex_reg}, gp{temp_reg}, 0, 0"
                        )
                        lines.append(f"S_ST_FP f{fp_ex_reg}, gp0, {local_pos}")

            # Flush FP_SRAM[0..vlen-1] into VRAM at (row m, tile t)
            lines.append(f"S_MAP_V_FP gp{out_reg}, gp0, 0")

    if fp_sram_precious_slots:
        lines.append("")
        lines.append("; -- Restore precious fp_sram slots overwritten by im2col --")
        for slot, sv in zip(fp_sram_precious_slots, save_fregs):
            lines.append(f"S_ST_FP f{sv}, gp0, {slot}")

    lines.append("")
    lines.append("; ============================================================")
    lines.append("; im2col (no-shift) complete")
    lines.append("; ============================================================")

    return "\n".join(lines) + "\n"
