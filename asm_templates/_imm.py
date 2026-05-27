"""Helpers for loading and adding large integer immediates with GP registers.

S_ADDI_INT only encodes 18-bit immediates (0..2^18-1). Anything larger needs
S_LUI_INT (which writes `imm << 12` into the destination) optionally followed
by an S_ADDI_INT for the lower 12 bits. Relative adds can either use a caller-
provided temporary register or a register-safe chunked ADDI fallback.

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

    Always produces at least 3 instructions so the GP register write completes
    before any downstream consumer (C_SET_ADDR, H_PREFETCH, H_STORE) reads it.
    Shorter sequences are padded with S_ADDI_INT gp0, gp0, 0 (NOP).
    """
    if value < IMM2_BOUND:
        lines = [f"S_ADDI_INT gp{reg}, gp0, {value}"]
    else:
        upper = value >> 12
        lower = value & 0xFFF
        lines = [f"S_LUI_INT gp{reg}, {upper}"]
        if lower:
            lines.append(f"S_ADDI_INT gp{reg}, gp{reg}, {lower}")

    # Pad to at least 3 instructions so GP write settles before consumer reads
    while len(lines) < 3:
        lines.append("S_ADDI_INT gp0, gp0, 0")
    return lines


def add_large_int(dest_reg: int, src_reg: int, value: int, temp_reg: int | None = None) -> list[str]:
    """Return ASM lines for `gp{dest_reg} = gp{src_reg} + value`.

    For values < 2^18 a single S_ADDI_INT is enough.

    When `temp_reg` is supplied, larger values are materialised into that
    register via load_large_int, then added with S_ADD_INT. The temp register
    must not alias src_reg, because loading the immediate would clobber the
    source value before the add.

    When `temp_reg` is omitted, larger values are emitted as a sequence of
    bounded S_ADDI_INT chunks. This fallback is less compact but is safe in a
    compiler-wide legalization pass because it requires no scratch register.
    """
    if value < 0:
        raise ValueError(f"large immediate helpers only support non-negative values, got {value}")
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{dest_reg}, gp{src_reg}, {value}"]

    if temp_reg is not None and temp_reg != src_reg:
        lines = load_large_int(temp_reg, value)
        lines.append(f"S_ADD_INT gp{dest_reg}, gp{src_reg}, gp{temp_reg}")
        return lines

    lines: list[str] = []
    remaining = value
    source = src_reg
    chunk = IMM2_BOUND - 1
    while remaining:
        step = min(remaining, chunk)
        lines.append(f"S_ADDI_INT gp{dest_reg}, gp{source}, {step}")
        remaining -= step
        source = dest_reg
    return lines


def addi_large_int(dest_reg: int, src_reg: int, value: int, temp_reg: int) -> list[str]:
    """Backward-compatible alias for add_large_int with an explicit temp register."""
    return add_large_int(dest_reg, src_reg, value, temp_reg=temp_reg)


def load_large_int_str(reg: int, value: int) -> str:
    """String variant: each instruction terminated with a newline."""
    return "".join(line + "\n" for line in load_large_int(reg, value))


def add_large_int_str(dest_reg: int, src_reg: int, value: int, temp_reg: int | None = None) -> str:
    """String variant of add_large_int."""
    return "".join(line + "\n" for line in add_large_int(dest_reg, src_reg, value, temp_reg=temp_reg))


def addi_large_int_str(dest_reg: int, src_reg: int, value: int, temp_reg: int) -> str:
    """String variant of addi_large_int."""
    return add_large_int_str(dest_reg, src_reg, value, temp_reg=temp_reg)
