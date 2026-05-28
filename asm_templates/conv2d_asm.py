"""CONV_2D assembly template.

Emits a single `CONV_2D` instruction in place of the existing
`im2col_asm` + `projection_asm` sequence for SmolVLM2's vision-encoder
patch embedding. Kernel/stride/Cin/Cout are encoded in the new
single-instruction opcode (decoder-stub level — see
`doc/CONV2D_AND_SYNTH_FYP.md` in the RTL repo).

Format::

    CONV_2D a{hbm_in_addr_reg}, a{hbm_wt_addr_reg}, gp{vram_out_reg}

Operand count is kept to 3 (rd, rs1, rs2) to fit the existing assembler
parser. Kernel/stride/Cin/Cout are baked into the conv2d_engine's
parameters at synthesis time, not encoded per-instruction.
"""


def conv2d_asm(
    hbm_in_addr_reg: int,
    hbm_wt_addr_reg: int,
    vram_out_reg: int,
) -> str:
    """Emit a single CONV_2D instruction.

    Args:
        hbm_in_addr_reg: index of the address register holding the
            HBM base address of the input feature map (NCHW layout).
        hbm_wt_addr_reg: index of the address register holding the
            HBM base address of the weight tile.
        vram_out_reg: index of the general-purpose register holding
            the VRAM output base address.

    Returns:
        A single-line assembly string ending with newline.
    """
    return (
        f"; CONV_2D (replaces im2col + projection for patch embed)\n"
        f"CONV_2D a{hbm_in_addr_reg}, a{hbm_wt_addr_reg}, gp{vram_out_reg}\n"
    )
