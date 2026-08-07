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

import re

IMM2_BOUND = 1 << 18  # S_ADDI_INT supports values 0..2^18-1

_ADDI_PATTERN = re.compile(
    r"^\s*S_ADDI_INT\s+gp(?P<dest>\d+)\s*,\s*gp(?P<source>\d+)\s*,\s*(?P<imm>\d+)\s*$"
)


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


CHUNK_LIMIT = 256  # chunked fallback ceiling; larger expansions fail loudly


def add_large_int(dest_reg: int, src_reg: int, value: int, temp_reg: int | None = None) -> list[str]:
    """Return ASM lines for `gp{dest_reg} = gp{src_reg} + value`.

    For values < 2^18 a single S_ADDI_INT is enough.

    When `temp_reg` is supplied, larger values are materialised into that
    register via load_large_int, then added with S_ADD_INT. The temp register
    must not alias src_reg, because loading the immediate would clobber the
    source value before the add.

    Without a temp register, a non-aliasing destination serves as its own
    temporary: the immediate is materialised into dest_reg and added to
    src_reg, since dest_reg is overwritten either way. Only the aliasing case
    (dest_reg == src_reg) falls back to bounded S_ADDI_INT chunks; that
    expansion is capped at CHUNK_LIMIT instructions so a pathological
    immediate fails loudly instead of flooding the program.
    """
    if value < 0:
        raise ValueError(f"large immediate helpers only support non-negative values, got {value}")
    if value < IMM2_BOUND:
        return [f"S_ADDI_INT gp{dest_reg}, gp{src_reg}, {value}"]

    if temp_reg is not None and temp_reg != src_reg:
        lines = load_large_int(temp_reg, value)
        lines.append(f"S_ADD_INT gp{dest_reg}, gp{src_reg}, gp{temp_reg}")
        return lines

    if dest_reg != src_reg:
        lines = load_large_int(dest_reg, value)
        lines.append(f"S_ADD_INT gp{dest_reg}, gp{src_reg}, gp{dest_reg}")
        return lines

    chunk = IMM2_BOUND - 1
    chunk_count = -(-value // chunk)
    if chunk_count > CHUNK_LIMIT:
        raise ValueError(
            f"chunked immediate add of {value} to gp{src_reg} needs "
            f"{chunk_count} instructions (limit {CHUNK_LIMIT}); provide a "
            "temp register or a non-aliasing destination"
        )
    lines = []
    remaining = value
    source = src_reg
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


def legalize_immediates(assembly: str) -> str:
    """Rewrite every `S_ADDI_INT` whose immediate exceeds the encoding field.

    The templates build addresses and strides from raw text, and several of
    those strides are `MLEN * MLEN`, which reaches exactly `2**18` at MLEN=512
    and overflows the 18-bit immediate. Legalising the emitted text covers every
    template at once, including ones that do not yet route their arithmetic
    through the helpers above, so a geometry cannot fail on an encoding limit
    that has a mechanical fix.

    `gp0` sources become a wide load; a non-aliasing destination becomes a
    wide load into the destination plus one add; only an aliasing
    destination falls back to the chunked relative add. All three forms need
    no scratch register and so are safe to apply after register allocation.
    """
    output: list[str] = []
    for line in assembly.splitlines():
        match = _ADDI_PATTERN.match(line)
        if match is None:
            output.append(line)
            continue
        dest, source, value = (
            int(match.group("dest")),
            int(match.group("source")),
            int(match.group("imm")),
        )
        if value < IMM2_BOUND:
            output.append(line)
            continue
        replacement = (
            load_large_int(dest, value)
            if source == 0
            else add_large_int(dest, source, value)
        )
        output.extend(replacement)
    return "\n".join(output) + ("\n" if assembly.endswith("\n") else "")
