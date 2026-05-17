"""
im2col ASM template -- uses V_MUL_VV + V_RED_SUM basis-vector extraction
instead of undocumented V_SHFT_V.

Algorithm per output row m:
  for each (c, kr, kc): extract input[c, oh+kr, ow+kc] via basis vector,
  accumulate into FP_SRAM, then S_MAP_V_FP to VRAM.

Requires: f0=0.0 (hw const), fp_preload[fp_one_reg]=1.0.
"""

from ._imm import load_large_int as _load_large_int_list

PREFETCH_V_AMOUNT = 4  # H_PREFETCH_V always loads this many VRAM rows


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
    W_padded: int | None = None,
    fp_one_reg: int = 1,  # FP register holding 1.0 (must be pre-loaded via fp_preload)
    fp_ex_reg: int = 2,  # FP register used as V_RED_SUM accumulator
    fp_sram_precious_slots: list | None = None,  # fp_sram slots to save before and restore after im2col
    stride: int = 1,  # convolution stride (patch_size for patch-embedding Conv2d)
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
        stride:
            Convolution stride (default 1).  For patch-embedding Conv2d with
            kernel_size == stride == patch_size, the pixel row is
            ``oh * stride + kr`` and the pixel column is ``ow * stride + kc``.
            The HBM load is aligned to 64-element boundaries by rounding
            the pixel column down; the basis-vector index is offset by the
            intra-64 residual so the correct element is extracted.

    Returns:
        Assembly code string.
    """
    scratch_reg = alive_registers[0]
    temp_reg = alive_registers[1]
    off_reg = alive_registers[2]
    out_reg = alive_registers[3]
    basis_reg = alive_registers[4]

    K_col = C_in * K * K

    # With stride > 1 the pixel column is ``ow * stride + kc``, which may
    # exceed a single VLEN load.  The assertion below checks the per-ow
    # maximum: intra_col + K - 1 < vlen, where intra_col = (ow*stride) % 64.
    max_intra_col = max((ow * stride) % 64 for ow in range(OW)) if OW > 0 else 0
    assert max_intra_col + K <= vlen, (
        f"im2col_asm_no_shift: intra_col ({max_intra_col}) + K ({K}) = "
        f"{max_intra_col + K} > VLEN ({vlen}); the K kernel elements must fit "
        f"within one 64-aligned HBM load"
    )
    assert K <= vlen, f"K={K} > vlen={vlen}; basis-vector extraction requires K <= vlen"
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
        f"Not enough free f_regs ({len(save_fregs)}) for {len(fp_sram_precious_slots)} precious fp_sram slots"
    )
    save_fregs = save_fregs[: len(fp_sram_precious_slots)]

    lines = []
    lines.append("; ============================================================")
    lines.append("; im2col (no-shift): NCHW input in HBM -> im2col matrix in VRAM")
    lines.append(f";   input  : (1,{C_in},{H},{W})  W_padded={W_padded}")
    lines.append(f";   kernel : {K}x{K}  OH={OH}  OW={OW}")
    lines.append(f";   output : ({M},{K_col}) @ VRAM base {output_vram_base}")
    lines.append("; ISA: H_PREFETCH_V V_MUL_VV V_RED_SUM S_ST_FP S_MAP_V_FP")
    lines.append("; Requires: f0=0.0 (hw const), fp_preload[1]=1.0")
    lines.append("; ============================================================")

    # Build basis vectors e_pos in VRAM (e_pos[i] = 1.0 if i==pos else 0.0).
    # With stride > 1, the basis position for (ow, kc) is
    #   (ow * stride) % 64 + kc
    # so we need basis vectors for every unique position index that appears.
    # Collect the full set once, build only those that are needed.
    _basis_positions: set[int] = set()
    for ow in range(OW):
        intra_col = (ow * stride) % 64
        for kc in range(K):
            _basis_positions.add(intra_col + kc)
    _basis_positions_sorted = sorted(_basis_positions)
    _num_basis = len(_basis_positions_sorted)
    # Map position → VRAM address for quick lookup in the main loop.
    _basis_pos_to_vram: dict[int, int] = {}
    for idx, pos in enumerate(_basis_positions_sorted):
        _basis_pos_to_vram[pos] = basis_vram_base + idx * vlen

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
    max_basis_pos = _basis_positions_sorted[-1] if _basis_positions_sorted else K - 1
    lines.append(f"; -- Setup: zero FP_SRAM[0..{max_basis_pos}] for basis construction --")
    for pos in range(max_basis_pos + 1):
        lines.append(f"S_ST_FP f0, gp0, {pos}")

    lines.append("")
    lines.append(f"; -- Setup: build {_num_basis} basis vectors in VRAM (stride={stride}) --")
    for pos in _basis_positions_sorted:
        basis_vram_addr = _basis_pos_to_vram[pos]
        lines.append(f"; e_{pos}: 1.0 at pos {pos}")
        lines.append(f"S_ST_FP f{fp_one_reg}, gp0, {pos}")
        lines.extend(_load_large_int_list(basis_reg, basis_vram_addr))
        lines.append(f"S_MAP_V_FP gp{basis_reg}, gp0, 0")  # FP_SRAM[0..63] -> VRAM
        lines.append(f"S_ST_FP f0, gp0, {pos}")  # restore 0.0

    # Pin scratch and temp VRAM addresses (constant across all rows)
    lines.append("")
    lines.append("; -- Setup: pin scratch/temp VRAM pointers --")
    lines.extend(_load_large_int_list(scratch_reg, scratch_vram_addr))
    lines.extend(_load_large_int_list(temp_reg, temp_vram_addr))

    # HBM stride=W_padded, scale=total input size (for H_PREFETCH_V stride mode)
    lines.append("")
    lines.append("; -- Setup: HBM stride and scale --")
    lines.extend(_load_large_int_list(basis_reg, W_padded))
    lines.append(f"C_SET_STRIDE_REG gp{basis_reg}")
    input_tensor_elems = C_in * H * W_padded
    lines.extend(_load_large_int_list(basis_reg, input_tensor_elems))
    lines.append(f"C_SET_SCALE_REG gp{basis_reg}")

    # Main loop: tiles × output positions. VRAM is column-block-major:
    #   vram_addr(m, t) = output_vram_base + t*M*vlen + m*vlen
    for tile_t in range(num_tiles):
        tile_start = tile_t * vlen
        tile_end = min(tile_start + vlen, K_col)
        tile_width = tile_end - tile_start  # < vlen only for the last partial tile

        lines.append("")
        lines.append(f"; ==== TILE t={tile_t}  global cols [{tile_start}..{tile_end - 1}] ====")

        for m in range(M):
            oh = m // OW
            ow = m % OW

            out_vram_addr = output_vram_base + tile_t * M * vlen + m * vlen

            lines.append("")
            lines.append(f"; ==== tile={tile_t} output row m={m}  oh={oh}  ow={ow}  vram={out_vram_addr} ====")
            lines.extend(_load_large_int_list(out_reg, out_vram_addr))

            # Zero unwritten FP_SRAM slots for partial last tile
            if tile_width < vlen:
                for pos in range(tile_width, vlen):
                    lines.append(f"S_ST_FP f0, gp0, {pos}")

            # Pixel column for this output position (stride-aware).
            pixel_col = ow * stride
            # Align the HBM column offset to a 64-element boundary so that
            # H_PREFETCH_V never sees a non-aligned address.
            aligned_col = (pixel_col // 64) * 64
            intra_col = pixel_col % 64  # offset within the aligned load

            for c in range(C_in):
                for kr in range(K):
                    # kc values whose global column falls in this tile
                    contributing_kcs = [kc for kc in range(K) if tile_start <= c * K * K + kr * K + kc < tile_end]
                    if not contributing_kcs:
                        continue

                    # Pixel row for this (oh, kr) pair (stride-aware).
                    pixel_row = oh * stride + kr
                    # HBM offset — 64-aligned; intra_col handled by basis vector.
                    hbm_offset = (c * H + pixel_row) * W_padded + aligned_col

                    lines.append(f"; (c={c}, kr={kr})  hbm_off={hbm_offset}  intra_col={intra_col}")
                    lines.extend(_load_large_int_list(off_reg, hbm_offset))
                    lines.append(f"H_PREFETCH_V gp{scratch_reg}, gp{off_reg}, a{input_hbm_base_addr_reg}, 1, 0")

                    for kc in contributing_kcs:
                        local_pos = c * K * K + kr * K + kc - tile_start
                        # Basis vector at (intra_col + kc) extracts the
                        # correct element from the aligned load.
                        basis_addr = _basis_pos_to_vram[intra_col + kc]

                        lines.extend(_load_large_int_list(basis_reg, basis_addr))
                        lines.append(f"V_MUL_VV gp{temp_reg}, gp{scratch_reg}, gp{basis_reg}, 0")
                        lines.append(f"S_ADD_FP f{fp_ex_reg}, f0, f0")
                        lines.append(f"V_RED_SUM f{fp_ex_reg}, gp{temp_reg}, 0, 0")
                        lines.append(f"S_ST_FP f{fp_ex_reg}, gp0, {local_pos}")

            lines.append(f"S_MAP_V_FP gp{out_reg}, gp0, 0")  # flush to VRAM

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
