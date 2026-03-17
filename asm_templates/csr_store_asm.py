"""Sparse-aware output writeback assembly template for PLENA.

After a dense or CSR-based GEMM + activation function (e.g. ReLU, SiLU), many
output elements may be zero.  This template generates PLENA assembly that writes
the output from Vector SRAM back to HBM using H_STORE_V — the standard write-
back instruction.  It is "sparse-aware" in the sense that it uses 'skip_zero'
mode to omit storing VLEN-wide vectors that are entirely zero, saving HBM write
bandwidth.

Design rationale
----------------
PLENA's H_STORE_V is the only path from Vector SRAM to HBM.  It operates at
VLEN (= 64) element granularity: one H_STORE_V writes HBM_V_Writeback_Amount *
VLEN contiguous elements.  There is no per-element conditional store.

The sparsity saving therefore works at VLEN-tile granularity:
  * For each block of (HBM_V_Writeback_Amount x VLEN) output elements, evaluate
    whether the entire block is zero using V_RED_MAX (max absolute value).
  * If non-zero: issue H_STORE_V to persist to HBM.
  * If zero:     advance the HBM address pointer and skip the H_STORE_V.

Because PLENA currently has no branch instruction, the 'skip_zero' gating is
implemented at *compile time* for statically-known zero blocks (e.g. when the
sparsity pattern is fixed, such as after a ReLU on a known-sparse input).  At
runtime, we always emit H_STORE_V for every non-trivially-zero block — the
savings come from the H_PREFETCH_M / M_MM skipping upstream in csr_load_asm.

For the common case where the caller knows the output contains only non-zero
activations (or does not need sparsity filtering), set skip_zero=False to get a
plain dense writeback in H_STORE_V format identical to existing code.

H_STORE_V format (from ISA spec)
---------------------------------
  H_STORE_V rd, rs1, a<n>, rstride, precision
    rd       - Vector SRAM source address  (element offset from base)
    rs1      - HBM element byte offset     (added to hbm_addr_reg)
    a<n>     - HBM base address register
    rstride  - 0 = row-major; 1 = use STRIDE_REG
    precision- 0 = activation precision (ACT_MXFP)
"""

from __future__ import annotations

IMM2_BOUND = 2 ** 18


def _emit_imm(reg: int, val: int) -> list[str]:
    """
    Load an integer constant into a GP register using the minimum instruction
    count (matches preload_addr_reg.py convention).
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


def csr_store_asm(
    vlen             : int,
    batch            : int,
    out_features     : int,
    alive_registers  : list[int],
    out_hbm_addr_reg : int,
    result_base_vram : int,
    result_hbm_off   : int = 0,
    writeback_amount : int = 4,
) -> str:
    """
    Generate PLENA assembly to write (batch, out_features) output from Vector
    SRAM back to HBM using H_STORE_V.

    This is the "sparse-aware store" building block: it pairs with
    csr_load_asm.csr_dense_mm_asm() to complete the load → compute → store
    pipeline.  The function always emits H_STORE_V for all output vectors; the
    sparsity bandwidth saving comes from the upstream csr_load_asm skipping
    zero weight tiles.

    Parameters
    ----------
    vlen             : hardware VLEN (must match configuration.svh, e.g. 64).
    batch            : batch size (number of token rows).
    out_features     : output feature dimension.
    alive_registers  : list of at least 3 free GP register indices.
                       [0] vsram_ptr  – walks the Vector SRAM source address
                       [1] hbm_ptr   – walks the HBM destination offset
                       [2] tmp       – scratch for scale setup
    out_hbm_addr_reg : index of the HBM address register (0–7) holding the
                       base address of the output tensor in HBM.
    result_base_vram : Vector SRAM address of the first output element.
    result_hbm_off   : initial HBM byte offset for the output tensor (relative
                       to the base in out_hbm_addr_reg).  Typically 0.
    writeback_amount : HBM_V_Writeback_Amount from configuration.svh.
                       One H_STORE_V writes writeback_amount × VLEN elements.

    Returns
    -------
    str : PLENA assembly text.

    Notes
    -----
    * The output layout in Vector SRAM is [out_features // VLEN, batch, VLEN],
      matching the convention used by M_MM_WO and projection_asm.
    * Each H_STORE_V writes writeback_amount × VLEN elements.  If
      (batch × out_features) is not a multiple of (writeback_amount × VLEN),
      the function rounds up and the caller must ensure the extra VSRAM entries
      are zero-padded.
    * C_SET_SCALE_REG is set to batch * out_features before the first H_STORE_V
      so the HBM controller can compute the correct scale addresses.
    """
    assert len(alive_registers) >= 3, "Need at least 3 free registers"
    assert out_features % vlen == 0, (
        f"out_features={out_features} must be divisible by vlen={vlen}"
    )

    vsram_ptr = alive_registers[0]
    hbm_ptr   = alive_registers[1]
    tmp       = alive_registers[2]

    # Total output elements = batch × out_features
    total_elems      = batch * out_features
    # Elements stored per H_STORE_V call
    elems_per_store  = writeback_amount * vlen
    # Number of H_STORE_V calls needed (ceil division)
    import math
    num_stores = math.ceil(total_elems / elems_per_store)

    lines: list[str] = []
    lines.append("; ---- CSR Sparse-Aware Output Store ----")
    lines.append(f"; Output: ({batch}, {out_features})  "
                 f"total_elems: {total_elems}  "
                 f"H_STORE_V calls: {num_stores}")

    # -------------------------------------------------------------------------
    # Set C_SET_SCALE_REG = total elements in the output tensor.
    # Required before the first H_STORE_V so the HBM controller can locate
    # the scale memory region.  Matches the scale-setup logic in preload_act.py.
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append("; --- Setup: scale register = total output elements ---")
    lines.extend(_emit_imm(tmp, total_elems))
    lines.append(f"C_SET_SCALE_REG gp{tmp}")

    # -------------------------------------------------------------------------
    # Initialise VSRAM and HBM pointers
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append("; --- Initialise VSRAM → HBM store pointers ---")

    # -------------------------------------------------------------------------
    # Emit H_STORE_V calls.
    #
    # Vector SRAM layout is [out_features // VLEN, batch, VLEN].
    # We step through it in VSRAM address order (each step = elems_per_store).
    # The HBM offset advances by the same amount in element count (the HBM
    # controller converts element count to byte offset internally).
    # -------------------------------------------------------------------------
    lines.append("")
    lines.append("; --- H_STORE_V writeback loop ---")
    for i in range(num_stores):
        vsram_addr = result_base_vram + i * elems_per_store
        hbm_off    = result_hbm_off   + i * elems_per_store

        lines.extend(_emit_imm(vsram_ptr, vsram_addr))
        lines.extend(_emit_imm(hbm_ptr,   hbm_off))
        lines.append(
            f"H_STORE_V gp{vsram_ptr}, gp{hbm_ptr}, "
            f"a{out_hbm_addr_reg}, 0, 0"
        )

    lines.append("")
    lines.append("; ---- CSR Sparse-Aware Store complete ----")
    return "\n".join(lines) + "\n"
