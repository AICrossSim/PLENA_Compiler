"""Helpers for loading large integer immediates into GP registers.

S_ADDI_INT only encodes 18-bit immediates (0..2^18-1). Anything larger needs
S_LUI_INT (which writes `imm << 12` into the destination) optionally followed
by an S_ADDI_INT for the lower 12 bits.

These helpers were previously duplicated in projection_asm, ffn_asm and
preload_act with subtly different signatures and thresholds; this module is
the single source of truth.
"""

from __future__ import annotations

IMM2_BOUND = 1 << 18  # S_ADDI_INT supports values 0..2^18-1


def load_large_int(reg: int, value: int) -> list[str]:
    """Return ASM lines that load `value` into gp{reg}.

    For values < 2^18, emits a single S_ADDI_INT from gp0.
    For values >= 2^18, emits S_LUI_INT (sets gp{reg} = imm << 12) followed
    by an S_ADDI_INT for the low 12 bits when non-zero.
    """
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{reg}, gp0, {value}"]
    upper = value >> 12
    lower = value & 0xFFF
    lines = [f"S_LUI_INT gp{reg}, {upper}"]
    if lower:
        lines.append(f"S_ADDI_INT gp{reg}, gp{reg}, {lower}")
    return lines


def addi_large_int(dest_reg: int, src_reg: int, value: int, temp_reg: int) -> list[str]:
    """Return ASM lines for `gp{dest_reg} = gp{src_reg} + value`.

    For values < 2^18 a single S_ADDI_INT is enough. Larger values are first
    materialised into `gp{temp_reg}` via load_large_int, then added.
    """
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{dest_reg}, gp{src_reg}, {value}"]
    lines = load_large_int(temp_reg, value)
    lines.append(f"S_ADD_INT gp{dest_reg}, gp{src_reg}, gp{temp_reg}")
    return lines


def load_large_int_str(reg: int, value: int) -> str:
    """String variant: each instruction terminated with a newline."""
    return "".join(line + "\n" for line in load_large_int(reg, value))


def addi_large_int_str(dest_reg: int, src_reg: int, value: int, temp_reg: int) -> str:
    """String variant of addi_large_int."""
    return "".join(line + "\n" for line in addi_large_int(dest_reg, src_reg, value, temp_reg))
