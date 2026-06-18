"""FYP draft (LOCAL-ONLY): native CONV_2D engine instruction emit.

This is a latency-model stub, not a functional kernel. When
``CONV_USE_CONV2D_INSTR=1`` the conv2d backend replaces the ~250K-line im2col +
systolic-GEMM lowering of the SmolVLM2/SigLIP patch embedding with a tiny
sequence of ``CONV_2D`` instructions. In the simulator each ``CONV_2D`` simply
advances the modeled clock by the cycles a dedicated conv MAC engine would take
(see ``conv2d_engine_cycles``); no real convolution is performed (the output
tensor is left as-allocated). The sole purpose is to score the engine's modeled
latency against im2col and im2col-with-shift for the thesis.

Mirrors the RTL ``kev/fyp-experiment`` opcode ``CONV_2D = 0x35``. The instruction
encodes ``CONV_2D gp{out}, {cycles}`` — the output-base register plus the modeled
cycle count carried in the 22-bit immediate field.
"""

import math

# CONV_2D carries its per-instruction cycle count in the 22-bit immediate field
# (same width as C_LOOP_START / S_LUI_INT). Latencies above this are split across
# several engine-pass instructions.
_IMM_BITS = 22
_MAX_CYCLE_IMM = (1 << _IMM_BITS) - 1


def conv2d_engine_cycles(M: int, C_out: int, c_out_par: int) -> int:
    """Modeled cycle cost of the whole patch-embed conv on the RTL MAC engine.

    The engine produces ``C_OUT_PAR`` output channels per cycle, each a full
    ``K*K*C_in``-length dot product evaluated by a combinational MAC tree. For
    ``M`` patch positions and ``C_out`` output channels that is
    ``M * ceil(C_out / C_OUT_PAR)`` cycles — independent of the systolic tile
    size, which is the point of the comparison.
    """
    return M * math.ceil(C_out / max(1, c_out_par))


def conv2d_instr_asm(
    *,
    M: int,
    C_out: int,
    c_out_par: int,
    out_vram_addr: int,
    out_reg: int,
) -> tuple[str, int]:
    """Emit the CONV_2D engine-pass sequence; return (asm, modeled_cycles)."""
    total = conv2d_engine_cycles(M, C_out, c_out_par)

    lines = [
        f"; === CONV_2D engine pass (FYP stub): M={M} C_out={C_out} "
        f"C_OUT_PAR={c_out_par} -> {total} modeled cycles ===",
    ]
    # Load the output VRAM base into gp{out_reg} (the engine's write target). The
    # simulator stub ignores the address, but it keeps the instruction's rd field
    # semantically meaningful and matches the RTL three-operand intent.
    if out_vram_addr <= 262143:
        lines.append(f"S_ADDI_INT gp{out_reg}, gp0, {out_vram_addr}")
    else:
        lines.append(f"S_LUI_INT gp{out_reg}, {out_vram_addr >> 12}")
        lines.append(f"S_ADDI_INT gp{out_reg}, gp{out_reg}, {out_vram_addr & 0xFFF}")

    # One CONV_2D per engine pass; split the modeled cycle count to fit the imm.
    remaining = max(1, total)
    while remaining > 0:
        chunk = min(remaining, _MAX_CYCLE_IMM)
        lines.append(f"CONV_2D gp{out_reg}, {chunk}")
        remaining -= chunk

    return "\n".join(lines) + "\n", total
