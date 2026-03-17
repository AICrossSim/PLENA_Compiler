"""CSR-dense matrix multiplication assembly template for PLENA.

Generates PLENA assembly for multiplying a CSR-compressed weight matrix against
dense activations already loaded in Vector SRAM.

Key design principle
--------------------
PLENA resolves sparsity at **compile time** using the CsrLayout metadata.  The
assembly template walks row_tile_starts / col_tile_indices and
emits only the H_PREFETCH_M + M_MM pairs that actually correspond to non-zero
MLENxMLEN tiles.  Zero tiles are simply omitted from the instruction stream,
which directly reduces HBM bandwidth and systolic-array compute cycles.

Memory convention
---------------------------------------------
  * Weight tiles are stored contiguously in HBM in row-major order of non-zero
    tiles (same file written by csr_memory_map.compress_weight_to_csr).
  * The HBM base address is supplied through an hbm_addr_reg (a0–a7).
  * Matrix SRAM layout: each H_PREFETCH_M writes one MLEN×MLEN tile at a
    Matrix SRAM address that is a multiple of MLEN*MLEN (4096 for MLEN=64).
  * Vector SRAM activations are in the [h//VLEN, batch, VLEN] layout used by
    all existing templates.
  * Output (M_MM_WO) is written to Vector SRAM at result_base_address in the
    same [out_features//VLEN, batch, VLEN] layout.

H_PREFETCH_M format
------------------------------------
  H_PREFETCH_M rd, rs1, a<n>, rstride, precision
    rd       - Matrix SRAM destination address (gp register value)
    rs1      - HBM element offset (gp register value, added to hbm_addr_reg)
    a<n>     - HBM base address register
    rstride  - 1 = use STRIDE_REG for column-major stride; 0 = row-major
    precision- 0 = weight precision (WT_MX)"""

from __future__ import annotations

IMM2_BOUND = 2 ** 18    # maximum value for S_ADDI_INT immediate field


def _emit_imm(reg: int, val: int) -> list[str]:
    """
    Emit the shortest instruction sequence to load an integer constant into a GP
    register.  Mirrors the logic in preload_addr_reg.py.

    For val <= 2^18-1 a single S_ADDI_INT from gp0 suffices.
    Larger values require S_LUI_INT (loads bits [31:12]) followed by S_ADDI_INT
    (adds bits [11:0]).
    """
    if val < IMM2_BOUND:
        return [f"S_ADDI_INT gp{reg}, gp0, {val}"]
    else:
        upper = val >> 12
        lower = val & 0xFFF
        insns = [f"S_LUI_INT gp{reg}, {upper}"]
        if lower:
            insns.append(f"S_ADDI_INT gp{reg}, gp{reg}, {lower}")
        return insns


def csr_dense_mm_asm(
    layout,                     # CsrLayout from csr_memory_map.compress_weight_to_csr
    mlen            : int,
    blen            : int,
    batch           : int,
    in_features     : int,
    out_features    : int,
    alive_registers : list[int],
    w_hbm_addr_reg  : int,
    act_base_vram   : int,
    result_base_vram: int,
) -> str:
    """
    Generate PLENA assembly for CSR-compressed weight × dense activation GEMM.

    The weight matrix W has shape (out_features, in_features).  It has been
    compressed to tile-level CSR at MLEN granularity and written to HBM by
    compress_weight_to_csr().

    This function emits exactly the H_PREFETCH_M + M_MM pairs for non-zero tiles
    and skips zero tiles entirely.  The activation is assumed to have been
    prefetched into Vector SRAM before this template is called (use
    preload_act_asm for that).

    Parameters
    ----------
    layout           : CsrLayout produced by compress_weight_to_csr().
    mlen             : hardware MLEN (must match configuration.svh).
    blen             : hardware BLEN (tile batch-dimension).
    batch            : batch size.
    in_features      : K dimension of the weight matrix.
    out_features     : M dimension of the weight matrix.
    alive_registers  : list of at least 6 free GP register indices.
                       [0] w_msram_addr  – current Matrix SRAM write/read addr
                       [1] w_hbm_off    – current HBM element offset
                       [2] act_addr     – current Vector SRAM activation ptr
                       [3] result_addr  – current Vector SRAM output ptr
                       [4] tmp          – scratch for scale/stride setup
                       [5] acc_step     – scratch for accumulation stepping
    w_hbm_addr_reg   : index of the HBM address register (0–7) that holds the
                       base address of the weight tensor in HBM.
    act_base_vram    : Vector SRAM address of the first activation element.
    result_base_vram : Vector SRAM address for the output.

    Returns
    -------
    str : PLENA assembly text ready for the assembler.

    Constraints
    -----------
    * out_features and in_features must both be multiples of mlen.
    * batch must be a multiple of blen.
    * The layout must have been generated for the same (out_features, in_features)
      at the same mlen tile granularity.
    """
    assert out_features % mlen == 0, (
        f"out_features={out_features} must be divisible by mlen={mlen}"
    )
    assert in_features % mlen == 0, (
        f"in_features={in_features} must be divisible by mlen={mlen}"
    )
    assert batch % blen == 0, (
        f"batch={batch} must be divisible by blen={blen}"
    )
    assert len(alive_registers) >= 6, "Need at least 6 free registers"

    num_row_tiles = out_features // mlen   # number of MLEN-row bands in W
    num_col_tiles = in_features  // mlen   # number of MLEN-col bands in W

    # Verify layout matches the requested dimensions
    assert layout.tile_shape == (num_row_tiles, num_col_tiles), (
        f"CsrLayout tile_shape {layout.tile_shape} does not match "
        f"({num_row_tiles}, {num_col_tiles}) for the given out/in_features"
    )

    # -------------------------------------------------------------------------
    # Register aliases
    # -------------------------------------------------------------------------
    w_msram_addr  = alive_registers[0]  # Matrix SRAM destination (for H_PREFETCH_M rd)
    w_hbm_off     = alive_registers[1]  # HBM element offset      (for H_PREFETCH_M rs1)
    act_addr      = alive_registers[2]  # Vector SRAM activation pointer
    result_addr   = alive_registers[3]  # Vector SRAM output pointer
    tmp           = alive_registers[4]  # scratch for scale / stride
    acc_step      = alive_registers[5]  # scratch for accumulation stepping

    lines: list[str] = []
    lines.append("; ---- CSR-Dense GEMM (sparse W × dense A) ----")
    lines.append(f"; W: ({out_features}, {in_features})  "
                 f"A: ({in_features}, {batch})  "
                 f"nnz_tiles: {layout.nnz_tiles} / {num_row_tiles * num_col_tiles}")
    lines.append(f"; Density: {layout.density:.2%}  "
                 f"tile_elem_bytes: {layout.tile_elem_bytes}")

    # -------------------------------------------------------------------------
    # Set up SCALE_REG and STRIDE_REG once.
    #
    # C_SET_SCALE_REG: tells the HBM controller how many HBM element bytes
    #   correspond to the full weight tensor (used for correct scale addressing).
    #   Value = out_features * in_features (total element count in the tensor).
    #   This must be set before the first H_PREFETCH_M for weight tiles.
    #   We set it to the *full* dense tensor size so the controller computes
    #   scale addresses relative to the dense layout; the non-zero-only HBM
    #   storage is addressed explicitly via w_hbm_off per tile.
    #
    # C_SET_STRIDE_REG: for rstride=1 mode, H_PREFETCH_M reads rows with this
    #   column stride.  We use rstride=0 (row-major dense tile), so STRIDE_REG
    #   is not needed for the prefetch itself.  We set it to in_features so
    #   it matches the dense row stride of the weight matrix.
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append("; --- Global setup: scale and stride registers ---")

    # Scale register = total elements in the full (dense) weight matrix.
    total_weight_elems = layout.nnz_tiles * mlen * mlen
    lines.extend(_emit_imm(tmp, total_weight_elems))
    lines.append(f"C_SET_SCALE_REG gp{tmp}")

    # Stride register = in_features (column stride across a weight row in HBM).
    lines.extend(_emit_imm(tmp, in_features))
    lines.append(f"C_SET_STRIDE_REG gp{tmp}")

    # -------------------------------------------------------------------------
    # Initialise result pointer
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append("; --- Initialise result/output pointer ---")
    lines.extend(_emit_imm(result_addr, result_base_vram))

    # -------------------------------------------------------------------------
    # Main loop: iterate over tile-rows of W (each of MLEN output rows)
    # -------------------------------------------------------------------------
    # HBM tile index: non-zero tiles are stored contiguously starting at offset 0.
    # Each tile occupies layout.tile_elem_bytes in hbm_ele.mem.
    hbm_tile_idx = 0   # global index into the contiguous non-zero tile stream

    for r in range(num_row_tiles):
        nz_start = layout.row_tile_starts[r]
        nz_end   = layout.row_tile_starts[r + 1]
        nnz_in_row = nz_end - nz_start

        # Output rows for this tile-row: [r*mlen .. (r+1)*mlen-1]
        # The M_MM_WO result lands at result base + r * mlen * batch entries.
        #
        # Vector SRAM result address for this tile-row:
        out_vram_row_base = result_base_vram + r * mlen * batch

        lines.append("")
        lines.append(f"; === Tile-row {r}: output rows [{r*mlen}..{(r+1)*mlen-1}] "
                     f"— {nnz_in_row}/{num_col_tiles} non-zero K-tiles ===")

        if nnz_in_row == 0:
            # Entire output tile-row is zero: write zeros directly to Vector SRAM.
            # The accumulator never fires for this row.  Skip without comment to
            # reduce instruction count; the output region is already zeroed if
            # the caller ran a reset pass.
            lines.append(f"; (all K-tiles zero — skipping tile-row {r})")
            continue

        # -----------------------------------------------------------------
        # Step 1: Pre-load all non-zero weight tiles for this output band
        #         into Matrix SRAM.
        #
        #  H_PREFETCH_M format:
        #    H_PREFETCH_M rd, rs1, a<n>, rstride, precision
        #      rd        = Matrix SRAM destination (SRAM word address)
        #                  Must be a multiple of MLEN*MLEN (= 4096 for MLEN=64).
        #                  We use a rolling pointer that wraps around a two-tile
        #                  double-buffer (addresses 0 and MLEN*MLEN) so that a
        #                  future H_PREFETCH_M can overlap with M_MM.
        #                  For simplicity here we pre-load all tiles sequentially
        #                  and then compute — this matches the projection_asm
        #                  pattern exactly.
        #      rs1       = HBM element byte offset for THIS tile within the
        #                  weight tensor's element data.  Non-zero tiles are
        #                  stored contiguously, so offset = global_tile_idx *
        #                  tile_elem_bytes.
        #      a<n>      = w_hbm_addr_reg (base of weight HBM region)
        #      rstride=0 = row-major (no STRIDE_REG needed per tile)
        #      precision=0 = weight precision (WT_MX)
        # -----------------------------------------------------------------
        lines.append(f"; Pre-load {nnz_in_row} non-zero weight tile(s) into Matrix SRAM")

        msram_tile_base  = 0          # Matrix SRAM always starts at 0 per projection convention
        msram_tile_stride = mlen * mlen  # MLEN×MLEN words per tile

        for local_idx, col_tile in enumerate(
            layout.col_tile_indices[nz_start:nz_end]
        ):
            # HBM offset of this specific non-zero tile (byte address within the
            # element data section of hbm_ele.mem, relative to the weight base).
            hbm_off = hbm_tile_idx * layout.tile_elem_bytes

            # Matrix SRAM destination for this tile.
            # Consecutive non-zero tiles go into consecutive MSRAM slots.
            msram_dst = msram_tile_base + local_idx * msram_tile_stride

            lines.extend(_emit_imm(w_hbm_off, hbm_off))
            lines.extend(_emit_imm(w_msram_addr, msram_dst))
            lines.append(
                f"H_PREFETCH_M gp{w_msram_addr}, gp{w_hbm_off}, "
                f"a{w_hbm_addr_reg}, 0, 0"
            )

            hbm_tile_idx += 1   # advance global non-zero tile counter

        # -----------------------------------------------------------------
        # Step 2: Accumulate M_MM for each batch slice and each non-zero
        #         K-tile.
        #
        # For each batch chunk of BLEN rows we run the inner K-loop over
        # the non-zero tiles only.
        #
        # Matrix SRAM pointer (w_msram_addr) walks through the pre-loaded
        # tiles.  Vector SRAM activation pointer (act_addr) walks through
        # the in_features dimension in MLEN steps (same as projection_asm).
        # -----------------------------------------------------------------
        lines.append(f"; Accumulate M_MM for tile-row {r}")

        for b_chunk in range(batch // blen):
            # Activation Vector SRAM base for this batch chunk:
            # activations are stored as [hidden // VLEN, batch, VLEN].
            # The inner K-loop strides act_addr by mlen * batch each M_MM.
            act_vram_chunk_base = act_base_vram + b_chunk * mlen * blen

            # Result Vector SRAM destination for this (tile-row, batch-chunk):
            res_vram = out_vram_row_base + b_chunk * mlen * blen
            lines.extend(_emit_imm(result_addr, res_vram))

            # Reset weight SRAM pointer to the first pre-loaded tile for this row
            lines.extend(_emit_imm(w_msram_addr, msram_tile_base))

            for local_idx, col_tile in enumerate(
                layout.col_tile_indices[nz_start:nz_end]
            ):
                # Activation address for K-tile col_tile:
                # K-tile c occupies rows [c*mlen .. (c+1)*mlen-1] of the weight.
                # In the activation [hidden//VLEN, batch, VLEN] layout the
                # corresponding VRAM address is:
                #    act_base + col_tile * (mlen // VLEN) * batch * vlen
                #             + b_chunk * blen
                # For VLEN = MLEN (both 64) this simplifies to:
                #    act_base + col_tile * batch * mlen + b_chunk * blen
                act_k_addr = (
                    act_base_vram
                    + col_tile * batch * mlen
                    + b_chunk * blen
                )
                msram_tile_addr = msram_tile_base + local_idx * msram_tile_stride

                lines.extend(_emit_imm(act_addr, act_k_addr))
                lines.extend(_emit_imm(w_msram_addr, msram_tile_addr))
                lines.append(f"M_MM 0, gp{w_msram_addr}, gp{act_addr}")

            # Write accumulated result for this (tile-row, batch-chunk) to VSRAM
            lines.extend(_emit_imm(result_addr, res_vram))
            lines.append(f"M_MM_WO gp{result_addr}, gp0, 0")

    lines.append("")
    lines.append("; ---- CSR-Dense GEMM complete ----")
    return "\n".join(lines) + "\n"
